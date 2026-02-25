#!/usr/bin/env python3
import argparse
import json

from kvpr_flexgen import utils as kv_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-len", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--method", type=str, choices=["kvpr", "flexgen"], required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--gpu-mem-bytes", type=int, default=None)
    parser.add_argument("--model-weights-bytes", type=int, default=0)
    parser.add_argument("--reserve-bytes", type=int, default=0)
    parser.add_argument("--num-kv-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=32, help="Number of decoder layers (included in KV bytes)")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--activation-multiplier", type=float, default=1.0)
    parser.add_argument("--ffn-multiplier", type=float, default=None)
    parser.add_argument("--bytes-per-elem", type=int, default=2)
    args = parser.parse_args()

    if args.model is not None:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(args.model)
        args.num_kv_heads = cfg.num_key_value_heads or cfg.num_attention_heads
        args.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        args.num_layers = cfg.num_hidden_layers
        args.hidden_size = cfg.hidden_size
        if args.ffn_multiplier is None:
            args.ffn_multiplier = cfg.intermediate_size / cfg.hidden_size - 1.0
    if args.ffn_multiplier is None:
        args.ffn_multiplier = 0.0

    if args.method == "kvpr":
        params = kv_utils.kvpr_params_from_metrics()
        recompute_len = kv_utils.kvpr_recompute_len_bandwidth(
            total_len=args.total_len, batch_size=args.batch_size, params=params
        )
        out = {
            "method": "kvpr",
            "total_len": args.total_len,
            "batch_size": args.batch_size,
            "cpu_to_gpu_gbps": params.cpu_to_gpu_gbps,
            "gpu_tflops": params.gpu_tflops,
            "recompute_len": recompute_len,
            "hw_metrics": kv_utils._load_hw_metrics(),
        }
    else:
        if args.gpu_mem_bytes is None:
            raise ValueError("--gpu-mem-bytes is required for flexgen")
        gpu_cache_len = kv_utils.flexgen_split_len_by_gpu_mem(
            total_len=args.total_len,
            batch_size=args.batch_size,
            gpu_mem_bytes=args.gpu_mem_bytes,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            activation_multiplier=args.activation_multiplier,
            ffn_multiplier=args.ffn_multiplier or 0.0,
            model_weights_bytes=args.model_weights_bytes,
            reserve_bytes=args.reserve_bytes,
            bytes_per_elem=args.bytes_per_elem,
        )
        out = {
            "method": "flexgen",
            "total_len": args.total_len,
            "batch_size": args.batch_size,
            "gpu_mem_bytes": args.gpu_mem_bytes,
            "model_weights_bytes": args.model_weights_bytes,
            "reserve_bytes": args.reserve_bytes,
            "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim,
            "num_layers": args.num_layers,
            "hidden_size": args.hidden_size,
            "activation_multiplier": args.activation_multiplier,
            "ffn_multiplier": args.ffn_multiplier or 0.0,
            "bytes_per_elem": args.bytes_per_elem,
            "gpu_cache_len": gpu_cache_len,
            "hw_metrics": kv_utils._load_hw_metrics(),
        }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
