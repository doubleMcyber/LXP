from __future__ import annotations

import inspect
from typing import Any, Optional

import torch
from torch import nn
from transformers import AutoModelForCausalLM


def _normalize_kv_cache(past_key_values: Any) -> Any:
    if past_key_values is None:
        return None
    # If it's a DynamicCache or similar non-iterable cache object, return it directly.
    # Otherwise, if it's already a tuple, or can be converted to one via legacy methods, do so.
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    if isinstance(past_key_values, (tuple, list)):
        return tuple(past_key_values)
    return past_key_values


def _kv_cache_seq_len(kv_cache: Any) -> int:
    if kv_cache is None:
        return 0
    if hasattr(kv_cache, "get_seq_length"):
        return kv_cache.get_seq_length()
    if not isinstance(kv_cache, (tuple, list)) or not kv_cache:
        return 0
    first_layer = kv_cache[0]
    if not isinstance(first_layer, (tuple, list)) or not first_layer:
        return 0
    key_tensor = first_layer[0]
    return int(key_tensor.shape[-2])


def _kv_cache_layer_count(kv_cache: Any) -> int:
    if kv_cache is None:
        return 0
    if isinstance(kv_cache, (tuple, list)):
        return len(kv_cache)
    try:
        return len(kv_cache)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        return 0


def _move_kv_cache_to_device(kv_cache: Any, device: torch.device) -> Any:
    if kv_cache is None:
        return None
    if hasattr(kv_cache, "to"):
        return kv_cache.to(device)

    if isinstance(kv_cache, (tuple, list)):
        moved_layers = []
        for layer_cache in kv_cache:
            if isinstance(layer_cache, (tuple, list)):
                moved_layer = tuple(
                    tensor.to(device) if torch.is_tensor(tensor) else tensor
                    for tensor in layer_cache
                )
                moved_layers.append(moved_layer)
            else:
                moved_layers.append(layer_cache.to(device) if hasattr(layer_cache, "to") else layer_cache)
        return tuple(moved_layers)
    return kv_cache


def _is_kv_cache_compatible(kv_cache: Any, actor_model: AutoModelForCausalLM) -> bool:
    if kv_cache is None:
        return False
    if not isinstance(kv_cache, (tuple, list)) or len(kv_cache) == 0:
        # Unknown cache object: keep it if it comes from the same architecture.
        return False

    cfg = actor_model.config
    expected_layers = getattr(cfg, "num_hidden_layers", None)
    if isinstance(expected_layers, int) and len(kv_cache) != expected_layers:
        return False

    first_layer = kv_cache[0]
    if not isinstance(first_layer, (tuple, list)) or len(first_layer) == 0:
        return False
    key_tensor = first_layer[0]
    if not torch.is_tensor(key_tensor) or key_tensor.dim() < 4:
        return False

    expected_heads = getattr(cfg, "num_key_value_heads", None)
    if expected_heads is None:
        expected_heads = getattr(cfg, "num_attention_heads", None)
    if isinstance(expected_heads, int) and key_tensor.shape[1] != expected_heads:
        return False

    expected_head_dim = getattr(cfg, "head_dim", None)
    if expected_head_dim is None:
        hidden_size = getattr(cfg, "hidden_size", None)
        num_heads = getattr(cfg, "num_attention_heads", None)
        if isinstance(hidden_size, int) and isinstance(num_heads, int) and num_heads > 0:
            expected_head_dim = hidden_size // num_heads
    if isinstance(expected_head_dim, int) and key_tensor.shape[-1] != expected_head_dim:
        return False

    return True


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class TransformerBlockDynamics(nn.Module):
    def __init__(self, transformer_block: nn.Module, rotary_emb: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.transformer_block = transformer_block
        self.rotary_emb = rotary_emb
        self._accepted_args = set(inspect.signature(transformer_block.forward).parameters.keys())
        self._attention_mask: Optional[torch.Tensor] = None
        self._position_ids: Optional[torch.LongTensor] = None
        self._cache_position: Optional[torch.LongTensor] = None

    def set_context(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> None:
        self._attention_mask = attention_mask
        self._position_ids = position_ids
        self._cache_position = cache_position

    def _extract_hidden_states(self, block_output: Any) -> torch.Tensor:
        if isinstance(block_output, tuple):
            return block_output[0]
        if hasattr(block_output, "hidden_states"):
            return block_output.hidden_states
        if hasattr(block_output, "last_hidden_state"):
            return block_output.last_hidden_state
        if torch.is_tensor(block_output):
            return block_output
        raise TypeError("Unsupported transformer block output type for ODE dynamics")

    def forward(self, t: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        del t
        kwargs: dict[str, Any] = {}
        accepts_hidden_kw = "hidden_states" in self._accepted_args
        if accepts_hidden_kw:
            kwargs["hidden_states"] = hidden_states
        if "attention_mask" in self._accepted_args and self._attention_mask is not None:
            kwargs["attention_mask"] = self._attention_mask
        if "position_ids" in self._accepted_args and self._position_ids is not None:
            kwargs["position_ids"] = self._position_ids
        if "cache_position" in self._accepted_args and self._cache_position is not None:
            kwargs["cache_position"] = self._cache_position
        if "position_embeddings" in self._accepted_args and self.rotary_emb is not None:
            kwargs["position_embeddings"] = self.rotary_emb(hidden_states, self._position_ids)
        if "past_key_value" in self._accepted_args:
            kwargs["past_key_value"] = None
        if "past_key_values" in self._accepted_args:
            kwargs["past_key_values"] = None
        if "use_cache" in self._accepted_args:
            kwargs["use_cache"] = False
        if "output_attentions" in self._accepted_args:
            kwargs["output_attentions"] = False
        if "output_hidden_states" in self._accepted_args:
            kwargs["output_hidden_states"] = False

        if accepts_hidden_kw:
            block_output = self.transformer_block(**kwargs)
        else:
            block_output = self.transformer_block(hidden_states, **kwargs)
        transformed = self._extract_hidden_states(block_output)
        return transformed - hidden_states
