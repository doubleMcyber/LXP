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
    layers = getattr(kv_cache, "layers", None)
    if layers is not None:
        return len(layers)
    key_cache = getattr(kv_cache, "key_cache", None)
    if key_cache is not None:
        return len(key_cache)
    try:
        return len(kv_cache)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        return 0


def _kv_cache_layer_key_value(kv_cache: Any, layer_index: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Return the (key, value) tensors for a given layer of a cache.

    Supports legacy tuple-of-tuples caches and modern cache objects exposing
    either ``layers`` (e.g. ``DynamicCache``) or parallel ``key_cache`` /
    ``value_cache`` lists (e.g. hybrid caches like ``Qwen3_5DynamicCache``).
    """
    if kv_cache is None:
        return None
    layers = getattr(kv_cache, "layers", None)
    if layers is not None:
        if layer_index < 0 or layer_index >= len(layers):
            return None
        layer = layers[layer_index]
        key_tensor = getattr(layer, "keys", None)
        value_tensor = getattr(layer, "values", None)
        if not torch.is_tensor(key_tensor) or not torch.is_tensor(value_tensor):
            return None
        return key_tensor, value_tensor
    key_cache = getattr(kv_cache, "key_cache", None)
    value_cache = getattr(kv_cache, "value_cache", None)
    if key_cache is not None and value_cache is not None:
        if layer_index < 0 or layer_index >= min(len(key_cache), len(value_cache)):
            return None
        key_tensor = key_cache[layer_index]
        value_tensor = value_cache[layer_index]
        if not torch.is_tensor(key_tensor) or not torch.is_tensor(value_tensor):
            return None
        return key_tensor, value_tensor
    if isinstance(kv_cache, (tuple, list)):
        if layer_index < 0 or layer_index >= len(kv_cache):
            return None
        layer = kv_cache[layer_index]
        if not isinstance(layer, (tuple, list)) or len(layer) < 2:
            return None
        key_tensor, value_tensor = layer[0], layer[1]
        if not torch.is_tensor(key_tensor) or not torch.is_tensor(value_tensor):
            return None
        return key_tensor, value_tensor
    return None


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
    status, _ = _kv_cache_compatibility_status(kv_cache, actor_model)
    return status == "transferred"


def _kv_cache_compatibility_status(
    kv_cache: Any, actor_model: AutoModelForCausalLM
) -> tuple[str, str]:
    if kv_cache is None:
        return "not_provided", "no_cache_provided"

    is_legacy = isinstance(kv_cache, (tuple, list))
    is_object_cache = (
        getattr(kv_cache, "layers", None) is not None
        or getattr(kv_cache, "key_cache", None) is not None
    )
    if not is_legacy and not is_object_cache:
        return "unsupported_cache_type", type(kv_cache).__name__

    layer_count = _kv_cache_layer_count(kv_cache)
    if layer_count == 0:
        return "unsupported_cache_type", type(kv_cache).__name__

    cfg = actor_model.config
    expected_layers = getattr(cfg, "num_hidden_layers", None)
    if isinstance(expected_layers, int) and layer_count != expected_layers:
        return (
            "unsupported_architecture_mismatch",
            f"layer_count_mismatch: expected {expected_layers}, got {layer_count}",
        )

    # For hybrid/state-space caches (e.g. Qwen3_5DynamicCache), some layers are
    # linear-attention with no standard key/value tensors. Trust the cache as
    # long as the layer count matches the receiver model; the model's own
    # forward will handle layer-type dispatch.
    first_pair = _kv_cache_layer_key_value(kv_cache, 0)
    if first_pair is None:
        if is_object_cache:
            return "transferred", "compatible_object_cache"
        return "invalid_cache", "first_layer_missing_key_value_tensors"
    first_key_tensor, first_value_tensor = first_pair
    if first_key_tensor.dim() < 4 or first_value_tensor.dim() < 4:
        return "invalid_cache", "first_layer_key_value_missing_or_rank_too_low"
    if first_key_tensor.shape != first_value_tensor.shape:
        return (
            "invalid_cache",
            f"first_layer_key_value_shape_mismatch: key={tuple(first_key_tensor.shape)}, "
            f"value={tuple(first_value_tensor.shape)}",
        )

    expected_heads = getattr(cfg, "num_key_value_heads", None)
    if expected_heads is None:
        expected_heads = getattr(cfg, "num_attention_heads", None)
    if isinstance(expected_heads, int) and first_key_tensor.shape[1] != expected_heads:
        return (
            "unsupported_architecture_mismatch",
            f"key_value_head_mismatch: expected {expected_heads}, got {first_key_tensor.shape[1]}",
        )

    expected_head_dim = getattr(cfg, "head_dim", None)
    if expected_head_dim is None:
        hidden_size = getattr(cfg, "hidden_size", None)
        num_heads = getattr(cfg, "num_attention_heads", None)
        if isinstance(hidden_size, int) and isinstance(num_heads, int) and num_heads > 0:
            expected_head_dim = hidden_size // num_heads
    if isinstance(expected_head_dim, int) and first_key_tensor.shape[-1] != expected_head_dim:
        return (
            "unsupported_architecture_mismatch",
            f"head_dim_mismatch: expected {expected_head_dim}, got {first_key_tensor.shape[-1]}",
        )

    first_shape = tuple(first_key_tensor.shape)
    for layer_index in range(1, layer_count):
        layer_pair = _kv_cache_layer_key_value(kv_cache, layer_index)
        if layer_pair is None:
            return "invalid_cache", f"layer_{layer_index}_missing_key_value_tensors"
        key_tensor, value_tensor = layer_pair
        if key_tensor.dim() < 4 or value_tensor.dim() < 4:
            return "invalid_cache", f"layer_{layer_index}_key_value_missing_or_rank_too_low"
        if key_tensor.shape != value_tensor.shape:
            return (
                "invalid_cache",
                f"layer_{layer_index}_key_value_shape_mismatch: key={tuple(key_tensor.shape)}, "
                f"value={tuple(value_tensor.shape)}",
            )
        if tuple(key_tensor.shape) != first_shape:
            return (
                "unsupported_architecture_mismatch",
                f"layer_{layer_index}_shape_mismatch: expected {first_shape}, got {tuple(key_tensor.shape)}",
            )

    return "transferred", "compatible"


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
