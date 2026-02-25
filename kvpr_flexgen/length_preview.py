#!/usr/bin/env python3
import argparse
import json

from kvpr_flexgen import utils as kv_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-len", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--method", type=str, choices=["kvpr", "flexgen"], required=True)
    parser.add_argument("--gpu-mem-bytes", type=int, default=None)
    parser.add_argument("--model-weights-bytes", type=int, default=0)
    parser.add_argument("--reserve-bytes", type=int, default=0)
    parser.add_argument("--num-kv-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=32, help="Number of decoder layers (included in KV bytes)")
    parser.add_argument("--bytes-per-elem", type=int, default=2)
    args = parser.parse_args()

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
            "bytes_per_elem": args.bytes_per_elem,
            "gpu_cache_len": gpu_cache_len,
        }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
