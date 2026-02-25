#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import torch


def measure_cpu_to_gpu_gbps(
    size_mb: int,
    iters: int,
    device: torch.device,
):
    size_bytes = size_mb * 1024 * 1024
    numel = size_bytes // 2  # fp16
    src = torch.empty(numel, dtype=torch.float16, device="cpu", pin_memory=True)
    dst = torch.empty(numel, dtype=torch.float16, device=device)

    torch.cuda.synchronize(device)
    start = time.time()
    for _ in range(iters):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize(device)
    end = time.time()
    total_bytes = size_bytes * iters
    gbps = total_bytes / (end - start) / 1e9
    return gbps


def measure_gpu_tflops_gemm(m: int, n: int, k: int, iters: int, device: torch.device):
    a = torch.randn((m, k), device=device, dtype=torch.float16)
    b = torch.randn((k, n), device=device, dtype=torch.float16)

    torch.cuda.synchronize(device)
    start = time.time()
    for _ in range(iters):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize(device)
    end = time.time()

    flops = 2.0 * m * n * k * iters
    tflops = flops / (end - start) / 1e12
    return tflops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--size-mb", type=int, default=512)
    parser.add_argument("--copy-iters", type=int, default=50)
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--gemm-iters", type=int, default=50)
    parser.add_argument("--out", type=str, default="kvpr_flexgen/hw_metrics.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This microbench requires a CUDA device.")

    cpu_to_gpu_gbps = measure_cpu_to_gpu_gbps(args.size_mb, args.copy_iters, device)
    gpu_tflops = measure_gpu_tflops_gemm(args.m, args.n, args.k, args.gemm_iters, device)

    metrics = {
        "cpu_to_gpu_gbps": cpu_to_gpu_gbps,
        "gpu_tflops": gpu_tflops,
        "copy_size_mb": args.size_mb,
        "copy_iters": args.copy_iters,
        "gemm_shape": [args.m, args.n, args.k],
        "gemm_iters": args.gemm_iters,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
