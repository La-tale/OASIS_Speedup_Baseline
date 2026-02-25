from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import json


@dataclass
class KVPRHeuristicParams:
    # Approximate hardware characteristics
    cpu_to_gpu_gbps: float = 30.0
    gpu_tflops: float = 100.0
    bytes_per_elem: int = 2
    head_dim: int = 128
    num_kv_heads: int = 32
    # Safety factor to avoid overly aggressive recompute
    safety: float = 0.9


def kv_per_token_bytes(num_kv_heads: int, head_dim: int, bytes_per_elem: int) -> int:
    # K + V
    return 2 * num_kv_heads * head_dim * bytes_per_elem


def _load_hw_metrics(path: str = "kvpr_flexgen/hw_metrics.json") -> Optional[dict]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def kvpr_params_from_metrics(
    default: Optional[KVPRHeuristicParams] = None,
    metrics_path: str = "kvpr_flexgen/hw_metrics.json",
) -> KVPRHeuristicParams:
    params = default or KVPRHeuristicParams()
    metrics = _load_hw_metrics(metrics_path)
    if not metrics:
        return params
    if "cpu_to_gpu_gbps" in metrics:
        params.cpu_to_gpu_gbps = float(metrics["cpu_to_gpu_gbps"])
    if "gpu_tflops" in metrics:
        params.gpu_tflops = float(metrics["gpu_tflops"])
    return params


def kvpr_recompute_len_ratio(total_len: int, ratio: float) -> int:
    """Simple heuristic: recompute first ratio of tokens."""
    return max(0, min(total_len, int(total_len * ratio)))


def kvpr_recompute_len_bandwidth(
    total_len: int,
    batch_size: int,
    params: KVPRHeuristicParams,
) -> int:
    """
    Heuristic inspired by KVPR: choose recompute length to balance CPU->GPU KV copy with recompute cost.
    This is intentionally simple and depends on rough bandwidth/compute inputs.
    """
    kv_bytes_per_token = kv_per_token_bytes(params.num_kv_heads, params.head_dim, params.bytes_per_elem)
    # Approximate KV transfer time per token
    t_transfer = (batch_size * kv_bytes_per_token) / (params.cpu_to_gpu_gbps * 1e9)

    # Approximate recompute time per token (very rough: 4 * hidden^2 flops is common for attention proj + MLP)
    # We can't know hidden size here; assume head_dim * num_kv_heads * 2 as proxy.
    hidden_proxy = params.head_dim * params.num_kv_heads
    flops_per_token = 4 * hidden_proxy * hidden_proxy
    t_recompute = flops_per_token / (params.gpu_tflops * 1e12)

    if t_recompute <= 0:
        return 0

    # If recompute is cheaper than transfer, recompute more.
    ratio = min(1.0, max(0.0, (t_transfer / t_recompute) * params.safety))
    return kvpr_recompute_len_ratio(total_len, ratio)


def flexgen_split_len_ratio(total_len: int, gpu_ratio: float) -> int:
    """Split KV by sequence length ratio."""
    return max(0, min(total_len, int(total_len * gpu_ratio)))


def flexgen_split_len_by_gpu_mem(
    total_len: int,
    batch_size: int,
    gpu_mem_bytes: int,
    num_kv_heads: int,
    head_dim: int,
    num_layers: int,
    model_weights_bytes: int = 0,
    reserve_bytes: int = 0,
    bytes_per_elem: int = 2,
    safety: float = 0.9,
) -> int:
    """Compute GPU-resident KV length given a GPU memory budget for KV."""
    available_bytes = gpu_mem_bytes - model_weights_bytes - reserve_bytes
    if available_bytes < 0:
        return 0
    per_token_bytes = (
        kv_per_token_bytes(num_kv_heads, head_dim, bytes_per_elem) * batch_size * max(1, num_layers)
    )
    if per_token_bytes <= 0:
        return 0
    max_tokens = int((available_bytes * safety) // per_token_bytes)
    return max(0, min(total_len, max_tokens))
