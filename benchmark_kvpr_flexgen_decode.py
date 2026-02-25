#!/usr/bin/env python3
import argparse
import time

import torch


def _set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_cache(method, model, batch_size, max_cache_len, dtype, recompute_len, gpu_cache_len, gpu_cache_ratio):
    if method == "kvpr":
        from kvpr_flexgen.kvpr_cache_patch import KVPROffloadedStaticCache

        return KVPROffloadedStaticCache(
            config=model.config,
            max_batch_size=batch_size,
            max_cache_len=max_cache_len,
            device=model.device,
            dtype=dtype,
            recompute_len=recompute_len,
        )
    if method == "offloaded":
        from transformers.cache_utils import OffloadedStaticCache

        return OffloadedStaticCache(
            config=model.config,
            max_batch_size=batch_size,
            max_cache_len=max_cache_len,
            device=model.device,
            dtype=dtype,
        )
    if method == "flexgen":
        from kvpr_flexgen.flexgen_cache_patch import FlexGenSplitStaticCache

        return FlexGenSplitStaticCache(
            config=model.config,
            max_batch_size=batch_size,
            max_cache_len=max_cache_len,
            device=model.device,
            dtype=dtype,
            gpu_cache_len=gpu_cache_len,
            gpu_cache_ratio=gpu_cache_ratio,
        )
    return None


def _apply_attention_patch(method):
    if method == "kvpr":
        from kvpr_flexgen.kvpr_attention_patch_llama import apply_kvpr_llama_attention_patch

        apply_kvpr_llama_attention_patch()
    elif method == "flexgen":
        from kvpr_flexgen.flexgen_attention_patch_llama import apply_flexgen_llama_attention_patch

        apply_flexgen_llama_attention_patch()


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--method", type=str, choices=["baseline", "offloaded", "kvpr", "flexgen"], default="offloaded")
    parser.add_argument("--recompute-len", type=int, default=0)
    parser.add_argument("--gpu-cache-len", type=int, default=None)
    parser.add_argument("--gpu-cache-ratio", type=float, default=None)
    parser.add_argument("--use-heuristics", action="store_true")
    parser.add_argument("--gpu-mem-bytes", type=int, default=None)
    parser.add_argument("--model-weights-bytes", type=int, default=None)
    parser.add_argument("--reserve-bytes", type=int, default=0)
    parser.add_argument("--dtype", type=str, choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--attn-implementation", type=str, default="sdpa")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-result", type=str, default=None)
    args = parser.parse_args()

    _set_seed(args.seed)
    _apply_attention_patch(args.method)

    from transformers import AutoModelForCausalLM

    device = torch.device(args.device)
    if args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, attn_implementation=args.attn_implementation
    )
    model.to(device)
    model.eval()

    def estimate_model_weights_bytes(m):
        total = 0
        for p in m.parameters():
            total += p.numel() * p.element_size()
        return total

    def get_gpu_total_mem_bytes(dev):
        if dev.type != "cuda":
            return None
        free_b, total_b = torch.cuda.mem_get_info(dev)
        return total_b

    vocab_size = model.config.vocab_size
    prompt_len = args.prompt_len
    batch_size = args.batch_size
    max_cache_len = prompt_len + 1

    input_ids = torch.randint(0, vocab_size, (batch_size, prompt_len), device=device, dtype=torch.long)

    recompute_len = min(args.recompute_len, prompt_len)

    heuristics_used = {}
    if args.use_heuristics:
        from kvpr_flexgen import utils as kv_utils

        if args.method == "kvpr":
            params = kv_utils.kvpr_params_from_metrics()
            recompute_len = kv_utils.kvpr_recompute_len_bandwidth(
                total_len=prompt_len, batch_size=batch_size, params=params
            )
            recompute_len = min(recompute_len, prompt_len)
            heuristics_used = {
                "method": "kvpr",
                "cpu_to_gpu_gbps": params.cpu_to_gpu_gbps,
                "gpu_tflops": params.gpu_tflops,
                "recompute_len": recompute_len,
            }
        elif args.method == "flexgen":
            if args.gpu_mem_bytes is None:
                args.gpu_mem_bytes = get_gpu_total_mem_bytes(device)
            if args.model_weights_bytes is None:
                args.model_weights_bytes = estimate_model_weights_bytes(model)
            if args.gpu_mem_bytes is None:
                raise ValueError("Unable to infer GPU memory size for flexgen heuristics.")
            args.gpu_cache_len = kv_utils.flexgen_split_len_by_gpu_mem(
                total_len=prompt_len,
                batch_size=batch_size,
                gpu_mem_bytes=args.gpu_mem_bytes,
                num_kv_heads=model.config.num_key_value_heads or model.config.num_attention_heads,
                head_dim=getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads),
                model_weights_bytes=args.model_weights_bytes or 0,
                reserve_bytes=args.reserve_bytes,
                bytes_per_elem=2 if model.dtype == torch.float16 else 4,
            )
            heuristics_used = {
                "method": "flexgen",
                "gpu_mem_bytes": args.gpu_mem_bytes,
                "model_weights_bytes": args.model_weights_bytes or 0,
                "reserve_bytes": args.reserve_bytes,
                "gpu_cache_len": args.gpu_cache_len,
            }

    cache = _make_cache(
        args.method,
        model,
        batch_size,
        max_cache_len,
        model.dtype,
        recompute_len,
        args.gpu_cache_len,
        args.gpu_cache_ratio,
    )

    def run_step(step_input, past_key_values):
        return model(
            input_ids=step_input,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

    # Warmup
    for _ in range(args.warmup):
        with torch.no_grad():
            _ = run_step(input_ids, cache)
            next_token = input_ids[:, -1:]
            _ = run_step(next_token, cache)
        _sync(device)
    if cache is not None and hasattr(cache, "reset"):
        cache.reset()

    # Prefill to set KV cache
    with torch.no_grad():
        _ = run_step(input_ids, cache)
    past = cache

    # 1-token decode latency
    next_token = input_ids[:, -1:]
    with torch.no_grad():
        _sync(device)
        t0 = time.time()
        _ = run_step(next_token, past)
        _sync(device)
        decode_time = time.time() - t0

    print(f"method: {args.method}")
    if args.use_heuristics:
        print(f"heuristics: {heuristics_used}")
    print(f"prompt_len: {prompt_len}")
    print(f"decode_time_1token_s: {decode_time:.6f}")

    if args.save_result:
        import json
        from pathlib import Path

        out = {
            "method": args.method,
            "attn_implementation": args.attn_implementation,
            "batch_size": batch_size,
            "prompt_len": prompt_len,
            "recompute_len": recompute_len,
            "gpu_cache_len": args.gpu_cache_len,
            "decode_time_1token_s": decode_time,
            "heuristics": heuristics_used if args.use_heuristics else None,
        }
        path = Path(args.save_result)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
