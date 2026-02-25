# KVPR/FlexGen Patches for Transformers 4.45.2

This document summarizes what was implemented, where it lives, and how to use it for forward-only benchmarking.

## Goals
- Add KVPR-style partial KV recomputation for Llama.
- Add an approximate FlexGen split-cache mechanism for Llama (Option B: CPU KV copied to GPU before attention).
- Provide a forward-only benchmarking path (no `generate`) that selects baseline/kvpr/flexgen behavior.
- Provide simple heuristics to pick recompute/split lengths.

## Repository Layout
- Patch package: `transformers-4.45.2/kvpr_flexgen/`
  - KVPR cache: `kvpr_cache_patch.py`
  - FlexGen cache: `flexgen_cache_patch.py`
  - KVPR Llama attention patches: `kvpr_attention_patch_llama.py`
  - FlexGen Llama attention patches: `flexgen_attention_patch_llama.py`
  - Heuristics: `utils.py`
- Benchmark: `transformers-4.45.2/benchmark_kvpr_flexgen_forward.py`

## What Was Implemented

### 1) KVPR Partial Recompute Cache
File: `kvpr_flexgen/kvpr_cache_patch.py`

- Class: `KVPROffloadedStaticCache`
- Behavior:
  - KV cache stored on CPU for positions >= `recompute_len`
  - Activation cache stored for positions < `recompute_len`
  - During decode (`hidden_states.shape[1] == 1`), it recomputes K/V for the recompute region using `k_proj`/`v_proj`
  - CPU KV cache stores only the non-recompute region to reduce bandwidth

Implementation details (conceptual):
- Per-layer KV is stored offloaded (CPU) except the first layer which stays on GPU, matching the baseline
  OffloadedStaticCache pattern.
- For the recompute region, activations are stored once during prefill and reused at decode time to rebuild K/V.
- A small on-device staging buffer is used for each layer to assemble the full KV for attention.
Note:
- There is no explicit "activation prefetch priority" scheduling in this implementation. Prefetching uses a CUDA
  stream as in OffloadedStaticCache, but recompute/KV transfer overlap is not explicitly orchestrated.

### 2) KVPR Llama Attention Patch
File: `kvpr_flexgen/kvpr_attention_patch_llama.py`

- Patches the following Llama attention classes:
  - `LlamaAttention` (eager)
  - `LlamaFlashAttention2`
  - `LlamaSdpaAttention`
- When cache is KVPR-aware (`hasattr(past_key_value, "kvpr_recompute")`), the patch calls:
  - `past_key_value.update(..., hidden_states, k_proj, v_proj)`
- This enables recompute on decode.

### 3) FlexGen Split Cache (Approx)
File: `kvpr_flexgen/flexgen_cache_patch.py`

- Class: `FlexGenSplitStaticCache`
- Behavior:
  - Split KV cache by sequence length:
    - `[0, gpu_cache_len)` on GPU
    - `[gpu_cache_len, max_cache_len)` on CPU
  - Before each attention call, CPU KV is copied to GPU and a full KV is assembled in a staging buffer.
  - Attention is computed fully on GPU (approximate FlexGen Option B).

Implementation details (conceptual):
- KV is persisted split (GPU part + CPU part) in dedicated buffers per layer.
- Each forward pass creates a full KV view on GPU by copying the CPU slice into a GPU staging buffer.
- This retains the memory savings of split storage but avoids changing the attention kernel.

### 4) FlexGen Llama Attention Patch
File: `kvpr_flexgen/flexgen_attention_patch_llama.py`

- Patches `LlamaAttention`, `LlamaFlashAttention2`, `LlamaSdpaAttention`
- Calls `past_key_value.update(...)` as usual (no recompute)
- Enables FlexGen cache usage for Llama.

### 5) Forward-Only Benchmark
File: `benchmark_kvpr_flexgen_forward.py`

- Uses `model.forward` with `past_key_values` and `use_cache=True`
- Performs:
  - Prefill (prompt)
  - Decode loop (gen_len steps, 1 token per step)
- Patch selection via `--method {baseline,kvpr,flexgen}`
- Attention implementation via `--attn-implementation` (default: `flash_attention_2`)

## Why Eager Was Used Initially (and Why Flash Works Now)
In Transformers 4.45.2, `LlamaFlashAttention2` refuses `StaticCache` usage (it raises a ValueError).  
Since KVPR/FlexGen caches subclass `StaticCache`, the base Flash Attention path is incompatible by default.

We patched `LlamaFlashAttention2.forward` to remove that restriction and route the cache update to KVPR/FlexGen caches.
That enables flash attention in practice, assuming `flash-attn` is installed and compatible.

## Heuristics (utils.py)
File: `kvpr_flexgen/utils.py`

Provided:
- `kvpr_recompute_len_ratio(total_len, ratio)`
- `kvpr_recompute_len_bandwidth(total_len, batch_size, params)`
  - Rough balance of transfer vs recompute cost.
- `flexgen_split_len_ratio(total_len, gpu_ratio)`
- `flexgen_split_len_by_gpu_mem(total_len, batch_size, gpu_mem_bytes, num_kv_heads, head_dim, bytes_per_elem)`

These are simple, deterministic heuristics you can refine with paper-specific formulas later.

### Heuristic Logic (Detailed)
- KVPR recompute length:
  - `kvpr_recompute_len_ratio`: picks `recompute_len = total_len * ratio`. This is a simple fixed-ratio policy.
  - `kvpr_recompute_len_bandwidth`: balances CPU→GPU KV transfer time vs GPU recompute time.
    - Transfer time per token is estimated from KV bytes and `cpu_to_gpu_gbps`.
    - Recompute time per token is estimated from a proxy compute cost (roughly quadratic in hidden size),
      scaled by `gpu_tflops`.
    - If recompute is cheaper than transfer, it increases `recompute_len`; otherwise it decreases it.
    - A `safety` factor dampens the result to avoid overly aggressive recompute.

- FlexGen split length:
  - `flexgen_split_len_ratio`: picks `gpu_cache_len = total_len * gpu_ratio`.
  - `flexgen_split_len_by_gpu_mem`: computes maximum GPU-resident KV length given a KV memory budget.
    - Per-token KV bytes = `2 * num_kv_heads * head_dim * bytes_per_elem * batch_size * num_layers`.
    - Available KV bytes = `gpu_mem_bytes - model_weights_bytes - reserve_bytes`.
    - `gpu_cache_len = floor(available_bytes / per_token_bytes)` with a safety factor.

### When Are Heuristics Used?
They are not enabled automatically. You must call these functions explicitly and pass the resulting
`recompute_len` / `gpu_cache_len` into your benchmark or training script.

### Hardware Microbench
File: `kvpr_flexgen/microbench_hw.py`

This script measures:
- CPU→GPU copy bandwidth (GB/s)
- GPU GEMM throughput (TFLOPS)

It stores results in `kvpr_flexgen/hw_metrics.json`. `utils.py` reads that file (if present)
to populate `KVPRHeuristicParams` via `kvpr_params_from_metrics()`.

Usage:
```bash
PYTHONPATH=src python kvpr_flexgen/microbench_hw.py --device cuda:0
```

### utils.py Usage
- `kvpr_params_from_metrics()`:
  - If `kvpr_flexgen/hw_metrics.json` exists, it loads `cpu_to_gpu_gbps` and `gpu_tflops`.
  - Otherwise it falls back to default values in `KVPRHeuristicParams`.
- In the benchmark, `--use-heuristics` triggers these functions automatically.

## Example Usage

### KVPR (Flash Attention)
```bash
PYTHONPATH=src python benchmark_kvpr_flexgen_forward.py \
  --model meta-llama/Llama-2-7b-hf \
  --batch-size 1 \
  --prompt-len 256 \
  --gen-len 64 \
  --method kvpr \
  --recompute-len 128 \
  --attn-implementation flash_attention_2
```

### FlexGen (Split Cache)
```bash
PYTHONPATH=src python benchmark_kvpr_flexgen_forward.py \
  --model meta-llama/Llama-2-7b-hf \
  --batch-size 1 \
  --prompt-len 256 \
  --gen-len 64 \
  --method flexgen \
  --gpu-cache-ratio 0.5 \
  --attn-implementation flash_attention_2
```

## Known Caveats
- Flash Attention requires `flash-attn` installed. If missing, use `--attn-implementation eager`.
- FlexGen implementation is approximate (Option B). It still copies CPU KV to GPU before attention.
- KVPR is only wired for Llama attention right now.
