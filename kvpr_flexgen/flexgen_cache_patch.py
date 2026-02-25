from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from transformers.cache_utils import StaticCache
from transformers.configuration_utils import PretrainedConfig


class FlexGenSplitStaticCache(StaticCache):
    """
    Approximate FlexGen-style cache:
    - Split KV cache between GPU and CPU by sequence length.
    - Before attention, CPU portion is copied to GPU and full attention is computed on GPU.
    """

    flexgen_split = True

    def __init__(
        self,
        config: PretrainedConfig,
        max_batch_size: int,
        max_cache_len: Optional[int],
        device: Union[str, torch.device],
        dtype: Optional[torch.dtype] = None,
        offload_device: Union[str, torch.device] = torch.device("cpu"),
        gpu_cache_len: Optional[int] = None,
        gpu_cache_ratio: Optional[float] = None,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len
        self.device = torch.device(device)
        self.offload_device = torch.device(offload_device)
        self.dtype = dtype if dtype is not None else torch.float32

        if gpu_cache_len is None:
            if gpu_cache_ratio is None:
                gpu_cache_ratio = 0.5
            gpu_cache_len = int(self.max_cache_len * float(gpu_cache_ratio))

        self.gpu_cache_len = max(0, min(int(gpu_cache_len), self.max_cache_len))
        self.cpu_cache_len = self.max_cache_len - self.gpu_cache_len

        head_dim = config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        self.head_dim = head_dim
        num_key_value_heads = (
            config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        )

        gpu_cache_shape = (max_batch_size, num_key_value_heads, self.gpu_cache_len, head_dim)
        cpu_cache_shape = (max_batch_size, num_key_value_heads, self.cpu_cache_len, head_dim)
        full_cache_shape = (max_batch_size, num_key_value_heads, self.max_cache_len, head_dim)
        self.full_cache_shape = full_cache_shape

        self.gpu_key_cache: List[torch.Tensor] = []
        self.gpu_value_cache: List[torch.Tensor] = []
        self.cpu_key_cache: List[torch.Tensor] = []
        self.cpu_value_cache: List[torch.Tensor] = []

        for _ in range(config.num_hidden_layers):
            gpu_k = torch.zeros(gpu_cache_shape, dtype=self.dtype, device=self.device)
            gpu_v = torch.zeros(gpu_cache_shape, dtype=self.dtype, device=self.device)
            cpu_k = torch.zeros(cpu_cache_shape, dtype=self.dtype, device=self.offload_device, pin_memory=True)
            cpu_v = torch.zeros(cpu_cache_shape, dtype=self.dtype, device=self.offload_device, pin_memory=True)
            torch._dynamo.mark_static_address(gpu_k)
            torch._dynamo.mark_static_address(gpu_v)
            torch._dynamo.mark_static_address(cpu_k)
            torch._dynamo.mark_static_address(cpu_v)
            self.gpu_key_cache.append(gpu_k)
            self.gpu_value_cache.append(gpu_v)
            self.cpu_key_cache.append(cpu_k)
            self.cpu_value_cache.append(cpu_v)

        # Device staging buffers for full KV.
        self._device_key_cache: List[torch.Tensor] = []
        self._device_value_cache: List[torch.Tensor] = []
        for _ in range(2):
            key_cache = torch.zeros(full_cache_shape, dtype=self.dtype, device=self.device)
            value_cache = torch.zeros(full_cache_shape, dtype=self.dtype, device=self.device)
            torch._dynamo.mark_static_address(key_cache)
            torch._dynamo.mark_static_address(value_cache)
            self._device_key_cache.append(key_cache)
            self._device_value_cache.append(value_cache)

        self._seen_tokens = 0
        self._prefetch_stream = torch.cuda.Stream() if self.device.type == "cuda" else None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if self._prefetch_stream is not None:
            torch.cuda.default_stream(self.device).wait_stream(self._prefetch_stream)

        k_out = self._device_key_cache[layer_idx & 1]
        v_out = self._device_value_cache[layer_idx & 1]

        # Assemble full KV on GPU from split storage.
        if self.gpu_cache_len > 0:
            k_out[:, :, : self.gpu_cache_len, :].copy_(self.gpu_key_cache[layer_idx], non_blocking=True)
            v_out[:, :, : self.gpu_cache_len, :].copy_(self.gpu_value_cache[layer_idx], non_blocking=True)
        if self.cpu_cache_len > 0:
            k_out[:, :, self.gpu_cache_len :, :].copy_(self.cpu_key_cache[layer_idx], non_blocking=True)
            v_out[:, :, self.gpu_cache_len :, :].copy_(self.cpu_value_cache[layer_idx], non_blocking=True)

        cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        if cache_position is None:
            k_out.copy_(key_states)
            v_out.copy_(value_states)
        else:
            try:
                k_out.index_copy_(2, cache_position, key_states)
                v_out.index_copy_(2, cache_position, value_states)
            except NotImplementedError:
                k_out[:, :, cache_position] = key_states
                v_out[:, :, cache_position] = value_states

        # Persist split KV caches.
        if cache_position is not None:
            gpu_mask = cache_position < self.gpu_cache_len
            cpu_mask = ~gpu_mask

            if gpu_mask.any():
                gpu_pos = cache_position[gpu_mask]
                k_gpu = key_states.index_select(2, gpu_pos)
                v_gpu = value_states.index_select(2, gpu_pos)
                self.gpu_key_cache[layer_idx].index_copy_(2, gpu_pos, k_gpu)
                self.gpu_value_cache[layer_idx].index_copy_(2, gpu_pos, v_gpu)

            if cpu_mask.any():
                cpu_pos = cache_position[cpu_mask] - self.gpu_cache_len
                cpu_pos = cpu_pos.to(self.offload_device)
                k_cpu = key_states.index_select(2, cache_position[cpu_mask]).to(self.offload_device)
                v_cpu = value_states.index_select(2, cache_position[cpu_mask]).to(self.offload_device)
                self.cpu_key_cache[layer_idx].index_copy_(2, cpu_pos, k_cpu)
                self.cpu_value_cache[layer_idx].index_copy_(2, cpu_pos, v_cpu)

        self._prefetch_layer(layer_idx + 1)
        return k_out, v_out

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self._seen_tokens

    def get_max_length(self) -> Optional[int]:
        return self.max_cache_len

    def reset(self) -> None:
        self._seen_tokens = 0
        for layer_idx in range(len(self.gpu_key_cache)):
            self.gpu_key_cache[layer_idx].zero_()
            self.gpu_value_cache[layer_idx].zero_()
            self.cpu_key_cache[layer_idx].zero_()
            self.cpu_value_cache[layer_idx].zero_()

    def _prefetch_layer(self, layer_idx: int) -> None:
        if layer_idx >= len(self.gpu_key_cache):
            return
        if self._prefetch_stream is not None:
            with torch.cuda.stream(self._prefetch_stream):
                self._prefetch_layer_in_context(layer_idx)
        else:
            self._prefetch_layer_in_context(layer_idx)

    def _prefetch_layer_in_context(self, layer_idx: int) -> None:
        k_out = self._device_key_cache[layer_idx & 1]
        v_out = self._device_value_cache[layer_idx & 1]
        if self.gpu_cache_len > 0:
            k_out[:, :, : self.gpu_cache_len, :].copy_(self.gpu_key_cache[layer_idx], non_blocking=True)
            v_out[:, :, : self.gpu_cache_len, :].copy_(self.gpu_value_cache[layer_idx], non_blocking=True)
        if self.cpu_cache_len > 0:
            k_out[:, :, self.gpu_cache_len :, :].copy_(self.cpu_key_cache[layer_idx], non_blocking=True)
            v_out[:, :, self.gpu_cache_len :, :].copy_(self.cpu_value_cache[layer_idx], non_blocking=True)
