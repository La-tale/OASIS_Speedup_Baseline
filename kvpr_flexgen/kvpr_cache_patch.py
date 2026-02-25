from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from transformers.cache_utils import StaticCache
from transformers.configuration_utils import PretrainedConfig


class KVPROffloadedStaticCache(StaticCache):
    """
    Offloaded static cache with partial KV recomputation (KVPR-style).

    - Stores KV cache on CPU for positions >= recompute_len.
    - Stores activations for positions < recompute_len, and recomputes KV at decode time.
    """

    kvpr_recompute = True

    def __init__(
        self,
        config: PretrainedConfig,
        max_batch_size: int,
        max_cache_len: Optional[int],
        device: Union[str, torch.device],
        dtype: Optional[torch.dtype] = None,
        offload_device: Union[str, torch.device] = torch.device("cpu"),
        recompute_len: int = 0,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len
        self.device = torch.device(device)
        self.offload_device = torch.device(offload_device)
        self.dtype = dtype if dtype is not None else torch.float32
        self.recompute_len = max(0, int(recompute_len))
        self.config = config

        head_dim = config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        self.head_dim = head_dim
        num_key_value_heads = (
            config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        )

        offloaded_cache_shape = (
            max_batch_size,
            num_key_value_heads,
            self.max_cache_len - self.recompute_len,
            head_dim,
        )
        cache_shape = (max_batch_size, num_key_value_heads, self.max_cache_len, head_dim)
        recomputed_activations_shape = (max_batch_size, self.recompute_len, config.hidden_size)

        self.cache_shape = cache_shape
        self.recomputed_activations_shape = recomputed_activations_shape

        # Offloaded CPU tensors.
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        if self.recompute_len > 0:
            self.activations: List[torch.Tensor] = []

        for i in range(config.num_hidden_layers):
            # First layer on device, others offloaded.
            layer_device = self.device if i == 0 else self.offload_device
            key_cache, value_cache = self._create_key_value_cache_tensors(offloaded_cache_shape, layer_device)
            self.key_cache.append(key_cache)
            self.value_cache.append(value_cache)
            if self.recompute_len > 0:
                self.activations.append(
                    self._create_recomputed_activations(recomputed_activations_shape, layer_device)
                )

        # On-device staging buffers.
        self._device_key_cache: List[torch.Tensor] = []
        self._device_value_cache: List[torch.Tensor] = []
        for _ in range(2):
            key_cache, value_cache = self._create_key_value_cache_tensors(offloaded_cache_shape, self.device)
            self._device_key_cache.append(key_cache)
            self._device_value_cache.append(value_cache)

        if self.recompute_len > 0:
            self._device_recomputed_activations: List[torch.Tensor] = []
            for _ in range(2):
                self._device_recomputed_activations.append(
                    self._create_recomputed_activations(recomputed_activations_shape, self.device)
                )

        # For backwards compatibility.
        self._seen_tokens = 0

        # Prefetch stream (cuda only).
        self._prefetch_stream = torch.cuda.Stream() if self.device.type == "cuda" else None

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        return tensor.view(bsz, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2).contiguous()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
        hidden_states: Optional[torch.Tensor] = None,
        k_proj=None,
        v_proj=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k_out = torch.zeros(self.cache_shape, device=self.device, dtype=key_states.dtype)
        v_out = torch.zeros(self.cache_shape, device=self.device, dtype=value_states.dtype)

        if self.recompute_len > 0 and hidden_states is not None:
            if layer_idx == 0:
                if hidden_states.shape[1] > 1:
                    self.activations[layer_idx] = hidden_states[:, : self.recompute_len, :]
            else:
                if hidden_states.shape[1] > 1:
                    self.activations[layer_idx].copy_(hidden_states[:, : self.recompute_len, :])

        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
            if self.recompute_len > 0:
                recomputed_activations = self.activations[0]
        else:
            if self._prefetch_stream is not None:
                torch.cuda.default_stream(self.device).wait_stream(self._prefetch_stream)
            if self.recompute_len > 0:
                recomputed_activations = self._device_recomputed_activations[layer_idx & 1]

        if self.recompute_len > 0 and hidden_states is not None and hidden_states.shape[1] == 1:
            if k_proj is None or v_proj is None:
                raise ValueError("k_proj/v_proj must be provided for KVPR recompute.")

            recomputed_key_states = self._shape(k_proj(recomputed_activations), -1, self.max_batch_size)
            recomputed_value_states = self._shape(v_proj(recomputed_activations), -1, self.max_batch_size)
            k_out[:, :, : self.recompute_len, :].copy_(recomputed_key_states)
            v_out[:, :, : self.recompute_len, :].copy_(recomputed_value_states)

            if layer_idx == 0:
                k_out[:, :, self.recompute_len :, :].copy_(self.key_cache[0])
                v_out[:, :, self.recompute_len :, :].copy_(self.value_cache[0])
            else:
                k_out[:, :, self.recompute_len :, :].copy_(self._device_key_cache[layer_idx & 1])
                v_out[:, :, self.recompute_len :, :].copy_(self._device_value_cache[layer_idx & 1])
        elif self.recompute_len == 0:
            if layer_idx == 0:
                k_out = self.key_cache[0]
                v_out = self.value_cache[0]
            else:
                k_out = self._device_key_cache[layer_idx & 1]
                v_out = self._device_value_cache[layer_idx & 1]

        self._prefetch_layer(layer_idx + 1)

        cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        if cache_position is None:
            k_out.copy_(key_states)
            v_out.copy_(value_states)

            if layer_idx == 0:
                self.key_cache[layer_idx].copy_(key_states.to(self.offload_device))
                self.value_cache[layer_idx].copy_(value_states.to(self.offload_device))
        else:
            try:
                k_out.index_copy_(2, cache_position, key_states)
                v_out.index_copy_(2, cache_position, value_states)
            except NotImplementedError:
                k_out[:, :, cache_position] = key_states
                v_out[:, :, cache_position] = value_states

            if layer_idx != 0:
                seq_len = cache_position.shape[0]
                if self.recompute_len > 0:
                    if seq_len > 1:
                        source_position = torch.arange(self.recompute_len, cache_position.shape[0])
                        source_position = source_position.to(self.offload_device)
                        key_states_cpu = key_states.to(self.offload_device)
                        value_states_cpu = value_states.to(self.offload_device)
                        target_position = source_position - self.recompute_len
                    else:
                        source_position = cache_position.to(self.offload_device)
                        key_states_cpu = key_states.to(self.offload_device)
                        value_states_cpu = value_states.to(self.offload_device)
                        target_position = source_position - self.recompute_len
                else:
                    source_position = cache_position.to(self.offload_device)
                    key_states_cpu = key_states.to(self.offload_device)
                    value_states_cpu = value_states.to(self.offload_device)
                    target_position = source_position

                try:
                    if seq_len > 1:
                        self.key_cache[layer_idx].index_copy_(
                            2, target_position, key_states_cpu[:, :, source_position, :]
                        )
                        self.value_cache[layer_idx].index_copy_(
                            2, target_position, value_states_cpu[:, :, source_position, :]
                        )
                    else:
                        self.key_cache[layer_idx].index_copy_(2, target_position, key_states_cpu)
                        self.value_cache[layer_idx].index_copy_(2, target_position, value_states_cpu)
                except NotImplementedError:
                    self.key_cache[layer_idx][:, :, cache_position] = key_states_cpu
                    self.value_cache[layer_idx][:, :, cache_position] = value_states_cpu

        return k_out, v_out

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self._seen_tokens

    def get_max_length(self) -> Optional[int]:
        return self.max_cache_len

    def reset(self) -> None:
        self._seen_tokens = 0
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()
            if self.recompute_len > 0:
                self.activations[layer_idx].zero_()

    def _create_key_value_cache_tensors(
        self, shape: Tuple[int, ...], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        is_cpu_device = device == torch.device("cpu")
        key_cache = torch.zeros(shape, dtype=self.dtype, device=device, pin_memory=is_cpu_device)
        value_cache = torch.zeros(shape, dtype=self.dtype, device=device, pin_memory=is_cpu_device)
        torch._dynamo.mark_static_address(key_cache)
        torch._dynamo.mark_static_address(value_cache)
        return key_cache, value_cache

    def _create_recomputed_activations(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        is_cpu_device = device == torch.device("cpu")
        recomputed_activations = torch.zeros(shape, dtype=self.dtype, device=device, pin_memory=is_cpu_device)
        torch._dynamo.mark_static_address(recomputed_activations)
        return recomputed_activations

    def _prefetch_layer(self, layer_idx: int) -> None:
        if layer_idx >= len(self.key_cache):
            return
        if self._prefetch_stream is not None:
            with torch.cuda.stream(self._prefetch_stream):
                self._prefetch_layer_in_context(layer_idx)
        else:
            self._prefetch_layer_in_context(layer_idx)

    def _prefetch_layer_in_context(self, layer_idx: int) -> None:
        self._device_key_cache[layer_idx & 1].copy_(self.key_cache[layer_idx], non_blocking=True)
        self._device_value_cache[layer_idx & 1].copy_(self.value_cache[layer_idx], non_blocking=True)
        if self.recompute_len > 0:
            self._device_recomputed_activations[layer_idx & 1].copy_(
                self.activations[layer_idx], non_blocking=True
            )

