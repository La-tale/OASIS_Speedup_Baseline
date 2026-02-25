from __future__ import annotations

from typing import Optional, Tuple

import math
import torch
import torch.nn.functional as F

from transformers.models.llama import modeling_llama


_ORIG_LLAMA_ATTENTION_FORWARD = None
_ORIG_LLAMA_FLASH_ATTENTION_FORWARD = None
_ORIG_LLAMA_SDPA_ATTENTION_FORWARD = None


def _kvpr_llama_attention_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                                  output_attentions=False, use_cache=False, cache_position=None,
                                  position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, **kwargs):
    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = modeling_llama.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        if hasattr(past_key_value, "kvpr_recompute"):
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs, hidden_states, self.k_proj, self.v_proj
            )
        else:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = modeling_llama.repeat_kv(key_states, self.num_key_value_groups)
    value_states = modeling_llama.repeat_kv(value_states, self.num_key_value_groups)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def _kvpr_llama_flash_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
):
    # Same as LlamaFlashAttention2.forward, but allows our cache types and passes KVPR recompute args.
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = modeling_llama.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        if hasattr(past_key_value, "kvpr_recompute"):
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs, hidden_states, self.k_proj, self.v_proj
            )
        else:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = modeling_llama._flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        position_ids=position_ids,
        dropout=dropout_rate,
        sliding_window=getattr(self, "sliding_window", None),
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
        is_causal=self.is_causal,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    attn_weights = None
    return attn_output, attn_weights, past_key_value


def _kvpr_llama_sdpa_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    # Derived from LlamaSdpaAttention.forward, with KVPR cache update support.
    if output_attentions:
        return _kvpr_llama_attention_forward(
            self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = modeling_llama.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        if hasattr(past_key_value, "kvpr_recompute"):
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs, hidden_states, self.k_proj, self.v_proj
            )
        else:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = modeling_llama.repeat_kv(key_states, self.num_key_value_groups)
    value_states = modeling_llama.repeat_kv(value_states, self.num_key_value_groups)

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=False if attention_mask is not None else True,
    )

    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)
    return attn_output, None, past_key_value


def apply_kvpr_llama_attention_patch():
    global _ORIG_LLAMA_ATTENTION_FORWARD, _ORIG_LLAMA_FLASH_ATTENTION_FORWARD, _ORIG_LLAMA_SDPA_ATTENTION_FORWARD
    if _ORIG_LLAMA_ATTENTION_FORWARD is None:
        _ORIG_LLAMA_ATTENTION_FORWARD = modeling_llama.LlamaAttention.forward
    modeling_llama.LlamaAttention.forward = _kvpr_llama_attention_forward

    if _ORIG_LLAMA_FLASH_ATTENTION_FORWARD is None:
        _ORIG_LLAMA_FLASH_ATTENTION_FORWARD = modeling_llama.LlamaFlashAttention2.forward
    modeling_llama.LlamaFlashAttention2.forward = _kvpr_llama_flash_attention_forward

    if _ORIG_LLAMA_SDPA_ATTENTION_FORWARD is None:
        _ORIG_LLAMA_SDPA_ATTENTION_FORWARD = modeling_llama.LlamaSdpaAttention.forward
    modeling_llama.LlamaSdpaAttention.forward = _kvpr_llama_sdpa_attention_forward


def undo_kvpr_llama_attention_patch():
    global _ORIG_LLAMA_ATTENTION_FORWARD, _ORIG_LLAMA_FLASH_ATTENTION_FORWARD, _ORIG_LLAMA_SDPA_ATTENTION_FORWARD
    if _ORIG_LLAMA_ATTENTION_FORWARD is not None:
        modeling_llama.LlamaAttention.forward = _ORIG_LLAMA_ATTENTION_FORWARD
        _ORIG_LLAMA_ATTENTION_FORWARD = None
    if _ORIG_LLAMA_FLASH_ATTENTION_FORWARD is not None:
        modeling_llama.LlamaFlashAttention2.forward = _ORIG_LLAMA_FLASH_ATTENTION_FORWARD
        _ORIG_LLAMA_FLASH_ATTENTION_FORWARD = None
    if _ORIG_LLAMA_SDPA_ATTENTION_FORWARD is not None:
        modeling_llama.LlamaSdpaAttention.forward = _ORIG_LLAMA_SDPA_ATTENTION_FORWARD
        _ORIG_LLAMA_SDPA_ATTENTION_FORWARD = None
