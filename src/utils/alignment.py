"""
LXP Geometric Alignment Utilities
---------------------------------
This module handles the mathematical transformations required to make the latent space
of one model (e.g., Qwen-2B) intelligible to another model (e.g., Qwen-0.8B).

Key Concepts:
- Orthogonal Procrustes: Finding the optimal rotation matrix (Q) that aligns two vector spaces without distorting their internal geometric relationships.
- Semantic Anchors: Using specific tokens (math symbols, numbers) as reference points to compute the cross-covariance matrix via Singular Value Decomposition (SVD).
- Adaptive Projection: Dynamically scaling and clipping transferred tensors to prevent representation drift and out-of-distribution values in the receiver model.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
import re
import weakref
from typing import Any, Optional

import torch


HiddenStateInput = torch.Tensor | Sequence[torch.Tensor]
LayerWeightInput = Sequence[float] | torch.Tensor | None
AlignmentState = dict[str, Any]

# Alignment-state tensors live on CPU (for disk caching) while activations live on the
# accelerator; without memoization every apply_alignment call re-uploads the same static
# matrices host->device. Keyed by id() with weakref.finalize eviction (tensors cannot be
# WeakKeyDictionary keys — weakref equality dereferences into elementwise ==), and kept
# outside the state dicts themselves because those are shared by reference across
# benchmark variant states.
_DEVICE_RESIDENT_TENSORS: dict[int, dict[tuple[str, torch.dtype], torch.Tensor]] = {}


def _device_resident(
    tensor: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if tensor.device == device and tensor.dtype == dtype:
        return tensor
    if tensor.requires_grad:
        return tensor.to(device=device, dtype=dtype)
    cache_id = id(tensor)
    per_tensor = _DEVICE_RESIDENT_TENSORS.get(cache_id)
    if per_tensor is None:
        per_tensor = {}
        _DEVICE_RESIDENT_TENSORS[cache_id] = per_tensor
        weakref.finalize(tensor, _DEVICE_RESIDENT_TENSORS.pop, cache_id, None)
    key = (str(device), dtype)
    cached = per_tensor.get(key)
    if cached is None:
        cached = tensor.to(device=device, dtype=dtype)
        per_tensor[key] = cached
    return cached
_ALNUM_ANCHOR_PATTERN = re.compile(r"^[A-Za-z0-9]+$")
_SIMPLE_SYMBOL_ANCHOR_PATTERN = re.compile(r"^[+\-*/=<>()[\]{}.,:;?!%^_]+$")
_MIXED_SEMANTIC_ANCHOR_PATTERN = re.compile(r"^[A-Za-z0-9+\-*/=<>()[\]{}.,:;?!%^_]+$")
_PREFERRED_SEMANTIC_ANCHORS: tuple[str, ...] = (
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "+",
    "-",
    "*",
    "/",
    "=",
    ">",
    "<",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    ".",
    ",",
    ":",
    ";",
    "?",
    "!",
    "%",
    "x",
    "y",
    "true",
    "false",
    "yes",
    "no",
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "for",
    "from",
    "if",
    "in",
    "is",
    "it",
    "not",
    "of",
    "on",
    "or",
    "the",
    "to",
    "we",
    "you",
    "with",
    "this",
    "that",
    "these",
    "those",
    "add",
    "sum",
    "math",
    "set",
    "zero",
    "one",
    "two",
    "three",
    "ten",
    "plus",
    "minus",
    "times",
    "equal",
    "less",
    "more",
    "value",
    "number",
    "count",
    "mean",
    "max",
    "min",
    "proof",
    "logic",
    "fact",
    "rule",
    "step",
    "reason",
    "think",
    "answer",
    "result",
    "code",
    "data",
    "text",
    "token",
    "model",
)


def get_preferred_semantic_anchors() -> tuple[str, ...]:
    return _PREFERRED_SEMANTIC_ANCHORS


def _is_printable_anchor(text: str) -> bool:
    if not text or text != text.strip():
        return False
    if len(text) > 16:
        return False
    return all(32 <= ord(char) <= 126 for char in text)


def _semantic_anchor_sort_key(text: str) -> tuple[int, int, str, str]:
    if _ALNUM_ANCHOR_PATTERN.fullmatch(text):
        category = 0
    elif _SIMPLE_SYMBOL_ANCHOR_PATTERN.fullmatch(text):
        category = 1
    elif _MIXED_SEMANTIC_ANCHOR_PATTERN.fullmatch(text):
        category = 2
    else:
        category = 3
    return category, len(text), text.casefold(), text


def _exact_single_token_id(tokenizer: object, text: str) -> int | None:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        return None
    token_id = int(token_ids[0])
    if token_id in getattr(tokenizer, "all_special_ids", ()):
        return None
    decoded = tokenizer.decode(
        [token_id],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if decoded != text:
        return None
    return token_id


def _shared_exact_single_token_ids(
    tokenizer_a: object, tokenizer_b: object
) -> dict[str, tuple[int, int]]:
    shared_token_ids: dict[str, tuple[int, int]] = {}
    tokenizer_a_vocab = getattr(tokenizer_a, "get_vocab")()
    for token_id in tokenizer_a_vocab.values():
        if token_id in getattr(tokenizer_a, "all_special_ids", ()):
            continue
        decoded = tokenizer_a.decode(
            [int(token_id)],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not _is_printable_anchor(decoded):
            continue
        exact_a = _exact_single_token_id(tokenizer_a, decoded)
        if exact_a is None:
            continue
        exact_b = _exact_single_token_id(tokenizer_b, decoded)
        if exact_b is None:
            continue
        shared_token_ids[decoded] = (exact_a, exact_b)
    return shared_token_ids


def _normalize_layer_weights(
    layer_count: int, layer_weights: LayerWeightInput
) -> torch.Tensor:
    if layer_count <= 0:
        raise ValueError("layer_count must be positive")
    if layer_weights is None:
        return torch.full((layer_count,), 1.0 / layer_count, dtype=torch.float32)

    weights = torch.as_tensor(layer_weights, dtype=torch.float32).flatten()
    if weights.numel() != layer_count:
        raise ValueError(
            f"Expected {layer_count} layer weights, received {weights.numel()}"
        )
    if not torch.isfinite(weights).all():
        raise ValueError("layer_weights must contain only finite values")
    if torch.any(weights <= 0):
        raise ValueError("layer_weights must be strictly positive")

    return weights / weights.sum()


def resolve_shared_semantic_anchor_ids(
    tokenizer_a: object, tokenizer_b: object, *, anchor_count: int = 100
) -> tuple[tuple[str, ...], torch.LongTensor, torch.LongTensor]:
    if anchor_count <= 0:
        raise ValueError("anchor_count must be positive")

    selected_strings: list[str] = []
    selected_ids_a: list[int] = []
    selected_ids_b: list[int] = []
    seen: set[str] = set()

    for anchor in _PREFERRED_SEMANTIC_ANCHORS:
        exact_a = _exact_single_token_id(tokenizer_a, anchor)
        exact_b = _exact_single_token_id(tokenizer_b, anchor)
        if exact_a is None or exact_b is None:
            continue
        selected_strings.append(anchor)
        selected_ids_a.append(exact_a)
        selected_ids_b.append(exact_b)
        seen.add(anchor)
        if len(selected_strings) == anchor_count:
            break

    if len(selected_strings) < anchor_count:
        shared_token_ids = _shared_exact_single_token_ids(tokenizer_a, tokenizer_b)
        sorted_candidates = sorted(shared_token_ids, key=_semantic_anchor_sort_key)
        for anchor in sorted_candidates:
            if anchor in seen:
                continue
            selected_strings.append(anchor)
            exact_a, exact_b = shared_token_ids[anchor]
            selected_ids_a.append(exact_a)
            selected_ids_b.append(exact_b)
            if len(selected_strings) == anchor_count:
                break

    if len(selected_strings) < anchor_count:
        raise ValueError(
            f"Expected at least {anchor_count} shared single-token semantic anchors, "
            f"found {len(selected_strings)}"
        )

    return (
        tuple(selected_strings),
        torch.tensor(selected_ids_a, dtype=torch.long),
        torch.tensor(selected_ids_b, dtype=torch.long),
    )


def _flatten_hidden_states(hidden_states: torch.Tensor) -> torch.Tensor:
    if hidden_states.dim() < 2:
        raise ValueError("hidden_states must have at least 2 dimensions")
    if hidden_states.dim() == 2:
        return hidden_states
    return hidden_states.reshape(-1, hidden_states.shape[-1])


def _normalize_hidden_state_layers(
    hidden_states: HiddenStateInput, *, name: str
) -> tuple[torch.Tensor, ...]:
    if isinstance(hidden_states, torch.Tensor):
        return (hidden_states,)
    if not isinstance(hidden_states, Sequence):
        raise TypeError(f"{name} must be a tensor or a sequence of tensors")

    layers = tuple(hidden_states)
    if not layers:
        raise ValueError(f"{name} must contain at least one tensor")

    for layer in layers:
        if not isinstance(layer, torch.Tensor):
            raise TypeError(f"{name} must contain only tensors")
    return layers


def _pair_anchor_states(
    sender_hidden_states: torch.Tensor, receiver_hidden_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    sender = _flatten_hidden_states(sender_hidden_states).to(torch.float32)
    receiver = _flatten_hidden_states(receiver_hidden_states).to(torch.float32)
    receiver = receiver.to(sender.device)

    pair_count = min(sender.shape[0], receiver.shape[0])
    if pair_count == 0:
        raise ValueError("sender/receiver hidden states are empty")

    return sender[:pair_count], receiver[:pair_count]


def _prepare_alignment_layers(
    sender_hidden_states: HiddenStateInput,
    receiver_hidden_states: HiddenStateInput,
    *,
    layer_weights: LayerWeightInput = None,
    center: bool = False,
) -> tuple[
    tuple[tuple[torch.Tensor, torch.Tensor], ...],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    sender_layers = _normalize_hidden_state_layers(
        sender_hidden_states, name="sender_hidden_states"
    )
    receiver_layers = _normalize_hidden_state_layers(
        receiver_hidden_states, name="receiver_hidden_states"
    )
    if len(sender_layers) != len(receiver_layers):
        raise ValueError("sender and receiver must provide the same number of layers")

    normalized_weights = _normalize_layer_weights(len(sender_layers), layer_weights)
    paired_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
    sender_means: list[torch.Tensor] = []
    receiver_means: list[torch.Tensor] = []

    for sender_layer, receiver_layer in zip(sender_layers, receiver_layers):
        sender, receiver = _pair_anchor_states(sender_layer, receiver_layer)
        if paired_layers and sender.shape[-1] != paired_layers[0][0].shape[-1]:
            raise ValueError("all sender layers must share the same hidden dimension")
        if paired_layers and receiver.shape[-1] != paired_layers[0][1].shape[-1]:
            raise ValueError("all receiver layers must share the same hidden dimension")
        sender_mean = sender.mean(dim=0, keepdim=True)
        receiver_mean = receiver.mean(dim=0, keepdim=True)
        sender_means.append(sender_mean)
        receiver_means.append(receiver_mean)
        if center:
            paired_layers.append((sender - sender_mean, receiver - receiver_mean))
        else:
            paired_layers.append((sender, receiver))

    sender_anchor_mean = torch.zeros_like(sender_means[0])
    receiver_anchor_mean = torch.zeros_like(receiver_means[0])
    for weight, sender_mean, receiver_mean in zip(
        normalized_weights, sender_means, receiver_means
    ):
        sender_anchor_mean = sender_anchor_mean + (sender_mean * float(weight.item()))
        receiver_anchor_mean = receiver_anchor_mean + (
            receiver_mean * float(weight.item())
        )

    return (
        tuple(paired_layers),
        normalized_weights,
        sender_anchor_mean,
        receiver_anchor_mean,
    )


def _stack_weighted_alignment_rows(
    paired_layers: Sequence[tuple[torch.Tensor, torch.Tensor]],
    normalized_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weighted_sender_rows: list[torch.Tensor] = []
    weighted_receiver_rows: list[torch.Tensor] = []
    for layer_index, (sender_layer, receiver_layer) in enumerate(paired_layers):
        weight_scale = math.sqrt(float(normalized_weights[layer_index].item()))
        weighted_sender_rows.append(sender_layer * weight_scale)
        weighted_receiver_rows.append(receiver_layer * weight_scale)
    return (
        torch.cat(weighted_sender_rows, dim=0),
        torch.cat(weighted_receiver_rows, dim=0),
    )


def _compute_cross_covariance_from_pairs(
    paired_layers: Sequence[tuple[torch.Tensor, torch.Tensor]],
    normalized_weights: torch.Tensor,
) -> torch.Tensor:
    mean_cross_covariance: torch.Tensor | None = None
    for layer_index, (sender_layer, receiver_layer) in enumerate(paired_layers):
        layer_cross_covariance = sender_layer.transpose(0, 1) @ receiver_layer
        if mean_cross_covariance is None:
            mean_cross_covariance = torch.zeros_like(layer_cross_covariance)
        elif layer_cross_covariance.shape != mean_cross_covariance.shape:
            raise ValueError("all layer pairs must share the same hidden dimensions")
        mean_cross_covariance = mean_cross_covariance + (
            layer_cross_covariance
            * float(normalized_weights[layer_index].item())
        )

    if mean_cross_covariance is None:
        raise ValueError("sender and receiver hidden states are empty")

    return mean_cross_covariance


def _safe_svd(
    matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if matrix.device.type != "mps":
        return torch.linalg.svd(matrix, full_matrices=False)

    cpu_matrix = matrix.detach().float().cpu()
    u_cpu, singular_values_cpu, vh_cpu = torch.linalg.svd(
        cpu_matrix,
        full_matrices=False,
    )
    return (
        u_cpu.to(device=matrix.device, dtype=matrix.dtype),
        singular_values_cpu.to(device=matrix.device, dtype=matrix.dtype),
        vh_cpu.to(device=matrix.device, dtype=matrix.dtype),
    )


def _pairwise_distance_distortion(
    left: torch.Tensor,
    right: torch.Tensor,
) -> float:
    # Keep diagnostic geometry metrics backend-safe by running them on CPU.
    # MPS does not currently implement torch.pdist, and these summaries are
    # inexpensive relative to the alignment solve itself.
    left_rows = left.detach().float().reshape(left.shape[0], -1).cpu()
    right_rows = right.detach().float().reshape(right.shape[0], -1).cpu()
    pair_count = min(left_rows.shape[0], right_rows.shape[0])
    if pair_count < 2:
        return 0.0
    left_distances = torch.pdist(left_rows[:pair_count], p=2)
    right_distances = torch.pdist(right_rows[:pair_count], p=2)
    denominator = right_distances.abs().clamp_min(1e-6)
    return float(((left_distances - right_distances).abs() / denominator).mean().item())


def _cosine_structure_error(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    max_rows: int = 128,
) -> float:
    left_rows = left.detach().float().reshape(left.shape[0], -1).cpu()
    right_rows = right.detach().float().reshape(right.shape[0], -1).cpu()
    pair_count = min(left_rows.shape[0], right_rows.shape[0], max_rows)
    if pair_count < 2:
        return 0.0
    left_rows = torch.nn.functional.normalize(left_rows[:pair_count], dim=-1)
    right_rows = torch.nn.functional.normalize(right_rows[:pair_count], dim=-1)
    left_gram = left_rows @ left_rows.transpose(0, 1)
    right_gram = right_rows @ right_rows.transpose(0, 1)
    return float((left_gram - right_gram).abs().mean().item())


def build_adaptive_projection_state(
    source_hidden_states: torch.Tensor,
    target_hidden_states: torch.Tensor,
    *,
    strength: float = 0.15,
    clip_std_multiplier: float = 4.0,
) -> AlignmentState:
    source = _flatten_hidden_states(source_hidden_states).float()
    target = _flatten_hidden_states(target_hidden_states).float().to(source.device)
    source_mean = source.mean(dim=0, keepdim=True)
    source_std = source.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    target_mean = target.mean(dim=0, keepdim=True)
    target_std = target.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    if strength <= 0.0:
        scale = torch.ones_like(source_std)
    else:
        raw_ratio = (target_std / source_std).clamp(0.5, 2.0)
        scale = 1.0 + (strength * (raw_ratio - 1.0))
    return {
        "enabled": True,
        "strength": float(strength),
        "clip_std_multiplier": float(clip_std_multiplier),
        "source_mean": source_mean.detach().cpu(),
        "source_std": source_std.detach().cpu(),
        "target_mean": target_mean.detach().cpu(),
        "target_std": target_std.detach().cpu(),
        "scale": scale.detach().cpu(),
    }


def apply_adaptive_projection(
    hidden_states: torch.Tensor,
    projection_state: Optional[AlignmentState],
) -> torch.Tensor:
    if not projection_state or not bool(projection_state.get("enabled", False)):
        return hidden_states

    source_mean = _device_resident(
        projection_state["source_mean"],
        device=hidden_states.device,
        dtype=torch.float32,
    )
    target_mean = _device_resident(
        projection_state["target_mean"],
        device=hidden_states.device,
        dtype=torch.float32,
    )
    target_std = _device_resident(
        projection_state["target_std"],
        device=hidden_states.device,
        dtype=torch.float32,
    )
    scale = _device_resident(
        projection_state["scale"],
        device=hidden_states.device,
        dtype=torch.float32,
    )
    clip_std_multiplier = float(projection_state.get("clip_std_multiplier", 4.0))

    projected = hidden_states.float()
    projected = (projected - source_mean) * scale + source_mean
    clip_low = target_mean - (clip_std_multiplier * target_std)
    clip_high = target_mean + (clip_std_multiplier * target_std)
    projected = torch.maximum(torch.minimum(projected, clip_high), clip_low)
    return projected.to(dtype=hidden_states.dtype)


def compute_cross_covariance(
    sender_hidden_states: HiddenStateInput,
    receiver_hidden_states: HiddenStateInput,
    *,
    layer_weights: LayerWeightInput = None,
) -> torch.Tensor:
    paired_layers, normalized_weights, _, _ = _prepare_alignment_layers(
        sender_hidden_states,
        receiver_hidden_states,
        layer_weights=layer_weights,
        center=False,
    )
    return _compute_cross_covariance_from_pairs(paired_layers, normalized_weights)


def compute_orthogonal_mapping(
    sender_hidden_states: HiddenStateInput,
    receiver_hidden_states: HiddenStateInput,
    *,
    layer_weights: LayerWeightInput = None,
) -> torch.Tensor:
    cross_covariance = compute_cross_covariance(
        sender_hidden_states,
        receiver_hidden_states,
        layer_weights=layer_weights,
    )
    u, _, vh = _safe_svd(cross_covariance)
    return u @ vh


def compute_ridge_mapping(
    source_hidden_states: HiddenStateInput,
    target_hidden_states: HiddenStateInput,
    *,
    regularization: float = 1e-4,
    layer_weights: LayerWeightInput = None,
) -> torch.Tensor:
    if regularization < 0:
        raise ValueError("regularization must be non-negative")

    paired_layers, normalized_weights, _, _ = _prepare_alignment_layers(
        source_hidden_states,
        target_hidden_states,
        layer_weights=layer_weights,
        center=False,
    )
    source, target = _stack_weighted_alignment_rows(paired_layers, normalized_weights)

    gram = source.transpose(0, 1) @ source
    rhs = source.transpose(0, 1) @ target
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return torch.linalg.solve(gram + (regularization * identity), rhs)

def compute_alignment_state(
    sender_hidden_states: HiddenStateInput,
    receiver_hidden_states: HiddenStateInput,
    *,
    layer_weights: LayerWeightInput = None,
    strategy: str = "orthogonal",
    center: bool = False,
    use_bias: bool = False,
    regularization: float = 1e-4,
    residual_alpha: float = 1.0,
    residual_max_norm_ratio: Optional[float] = None,
    adaptive_projection_strength: float = 0.0,
    adaptive_projection_clip_std_multiplier: float = 4.0,
) -> AlignmentState:
    """
    Calculates the full mathematical transformation mapping required to bridge two distinct models.

    This is the core of the Heterogeneous Latent Handoff. It takes a set of matching semantic anchor
    tensors from both models and calculates the optimal transformation matrix `Q`.

    Args:
        sender_hidden_states: Tensors from Agent A (e.g., Qwen-2B).
        receiver_hidden_states: Corresponding ground-truth tensors from Agent B (e.g., Qwen-0.8B).
        strategy: "orthogonal" (Procrustes - preserves geometry) or "ridge" (Linear regression).
        center: Whether to center the vectors at the origin before alignment.
        adaptive_projection_strength: How aggressively to clip standard deviation on the receiver end to prevent OOM drift.

    Returns:
        An AlignmentState dictionary containing the transformation matrix `mapping_matrix` (Q) and optional bias parameters.
    """
    normalized_strategy = str(strategy).strip().lower()
    if normalized_strategy not in {"orthogonal", "ridge", "hybrid_affine"}:
        raise ValueError(
            f"Unsupported alignment strategy {strategy!r}. "
            "Expected 'orthogonal', 'ridge', or 'hybrid_affine'."
        )

    paired_layers, normalized_weights, sender_anchor_mean, receiver_anchor_mean = (
        _prepare_alignment_layers(
            sender_hidden_states,
            receiver_hidden_states,
            layer_weights=layer_weights,
            center=center,
        )
    )
    sender_rows, receiver_rows = _stack_weighted_alignment_rows(
        paired_layers,
        normalized_weights,
    )
    sender_rows = sender_rows.float()
    receiver_rows = receiver_rows.float().to(sender_rows.device)

    cross_covariance = _compute_cross_covariance_from_pairs(
        paired_layers,
        normalized_weights,
    )
    u, singular_values, vh = _safe_svd(cross_covariance)
    orthogonal_q = u @ vh

    residual_matrix = torch.zeros(
        (sender_rows.shape[-1], receiver_rows.shape[-1]),
        device=sender_rows.device,
        dtype=torch.float32,
    )
    mapping_matrix = orthogonal_q

    if normalized_strategy == "ridge":
        mapping_matrix = compute_ridge_mapping(
            sender_rows,
            receiver_rows,
            regularization=regularization,
        )
        orthogonal_q = mapping_matrix
    elif normalized_strategy == "hybrid_affine":
        residual_target = receiver_rows - (sender_rows @ orthogonal_q)
        residual_matrix = compute_ridge_mapping(
            sender_rows,
            residual_target,
            regularization=regularization,
        )
        scaled_residual = residual_matrix * float(residual_alpha)
        orthogonal_norm = float(torch.linalg.matrix_norm(orthogonal_q).item())
        residual_norm = float(torch.linalg.matrix_norm(scaled_residual).item())
        if (
            residual_max_norm_ratio is not None
            and residual_max_norm_ratio > 0.0
            and orthogonal_norm > 0.0
            and residual_norm > orthogonal_norm * float(residual_max_norm_ratio)
        ):
            cap = orthogonal_norm * float(residual_max_norm_ratio)
            scaled_residual = scaled_residual * (cap / max(residual_norm, 1e-8))
        residual_matrix = scaled_residual
        mapping_matrix = orthogonal_q + residual_matrix

    bias_vector = torch.zeros(
        (1, receiver_rows.shape[-1]),
        device=sender_rows.device,
        dtype=torch.float32,
    )
    if use_bias:
        sender_anchor_mean = sender_anchor_mean.to(sender_rows.device, dtype=torch.float32)
        receiver_anchor_mean = receiver_anchor_mean.to(
            sender_rows.device, dtype=torch.float32
        )
        bias_vector = receiver_anchor_mean - (sender_anchor_mean @ mapping_matrix)

    sender_unweighted_rows, receiver_unweighted_rows = _stack_weighted_alignment_rows(
        _prepare_alignment_layers(
            sender_hidden_states,
            receiver_hidden_states,
            layer_weights=layer_weights,
            center=False,
        )[0],
        normalized_weights,
    )
    sender_unweighted_rows = sender_unweighted_rows.float()
    receiver_unweighted_rows = receiver_unweighted_rows.float().to(sender_unweighted_rows.device)
    mapped_sender_unweighted = (sender_unweighted_rows @ mapping_matrix) + bias_vector

    pre_projection_state = build_adaptive_projection_state(
        sender_unweighted_rows,
        sender_unweighted_rows,
        strength=adaptive_projection_strength,
        clip_std_multiplier=adaptive_projection_clip_std_multiplier,
    )
    post_projection_state = build_adaptive_projection_state(
        mapped_sender_unweighted,
        receiver_unweighted_rows,
        strength=adaptive_projection_strength,
        clip_std_multiplier=adaptive_projection_clip_std_multiplier,
    )
    adapted_sender_unweighted = apply_adaptive_projection(
        mapped_sender_unweighted.to(dtype=torch.float32),
        post_projection_state,
    )

    orthogonal_norm = float(torch.linalg.matrix_norm(orthogonal_q).item())
    residual_norm = float(torch.linalg.matrix_norm(residual_matrix).item())
    residual_norm_ratio = 0.0 if orthogonal_norm <= 0.0 else residual_norm / orthogonal_norm
    bias_norm = float(torch.linalg.vector_norm(bias_vector.reshape(-1)).item())

    return {
        "alignment_strategy": normalized_strategy,
        "mapping_matrix": mapping_matrix.detach().cpu(),
        "mapping_bias": bias_vector.detach().cpu(),
        "orthogonal_q": orthogonal_q.detach().cpu(),
        "residual_matrix": residual_matrix.detach().cpu(),
        "center_anchors": bool(center),
        "use_bias": bool(use_bias),
        "regularization": float(regularization),
        "residual_alpha": float(residual_alpha),
        "residual_max_norm_ratio": None
        if residual_max_norm_ratio is None
        else float(residual_max_norm_ratio),
        "sender_anchor_mean": sender_anchor_mean.detach().cpu(),
        "receiver_anchor_mean": receiver_anchor_mean.detach().cpu(),
        "alignment_singular_values": singular_values.detach().cpu(),
        "pre_projection_state": pre_projection_state,
        "post_projection_state": post_projection_state,
        "anchor_reconstruction_mse": float(
            torch.mean((adapted_sender_unweighted - receiver_unweighted_rows) ** 2).item()
        ),
        "anchor_pairwise_distance_distortion": _pairwise_distance_distortion(
            adapted_sender_unweighted,
            receiver_unweighted_rows,
        ),
        "anchor_cosine_structure_error": _cosine_structure_error(
            adapted_sender_unweighted,
            receiver_unweighted_rows,
        ),
        "residual_norm_ratio": residual_norm_ratio,
        "bias_norm": bias_norm,
    }


def score_anchor_stability(
    sender_hidden_states: torch.Tensor,
    receiver_hidden_states: torch.Tensor,
    *,
    strategy: str = "hybrid_affine",
    regularization: float = 1e-3,
    residual_alpha: float = 1.0,
    residual_max_norm_ratio: Optional[float] = 0.25,
    center: bool = True,
    use_bias: bool = True,
    bootstrap_count: int = 3,
    bootstrap_ratio: float = 0.8,
    bootstrap_weight: float = 0.5,
    seed: int = 0,
) -> AlignmentState:
    sender = _flatten_hidden_states(sender_hidden_states).float()
    receiver = _flatten_hidden_states(receiver_hidden_states).float().to(sender.device)
    if sender.shape[0] != receiver.shape[0]:
        raise ValueError("sender and receiver must provide the same number of anchor rows")
    if sender.shape[0] < 2:
        raise ValueError("Need at least two anchors to score anchor stability")

    base_state = compute_alignment_state(
        sender,
        receiver,
        strategy=strategy,
        center=center,
        use_bias=use_bias,
        regularization=regularization,
        residual_alpha=residual_alpha,
        residual_max_norm_ratio=residual_max_norm_ratio,
    )
    base_mapped = apply_alignment(sender, base_state).float()
    base_errors = torch.linalg.vector_norm(base_mapped - receiver, dim=-1) / math.sqrt(receiver.shape[-1])
    base_errors_cpu = base_errors.detach().cpu()

    oob_error_sum = torch.zeros_like(base_errors_cpu)
    oob_error_sq_sum = torch.zeros_like(base_errors_cpu)
    oob_count = torch.zeros_like(base_errors_cpu)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    subset_size = min(sender.shape[0] - 1, max(2, int(round(sender.shape[0] * bootstrap_ratio))))

    for _ in range(max(0, bootstrap_count)):
        permutation = torch.randperm(sender.shape[0], generator=generator)
        train_indices = permutation[:subset_size]
        oob_indices = permutation[subset_size:]
        if oob_indices.numel() == 0:
            continue
        bootstrap_state = compute_alignment_state(
            sender.index_select(0, train_indices.to(sender.device)),
            receiver.index_select(0, train_indices.to(receiver.device)),
            strategy=strategy,
            center=center,
            use_bias=use_bias,
            regularization=regularization,
            residual_alpha=residual_alpha,
            residual_max_norm_ratio=residual_max_norm_ratio,
        )
        bootstrap_mapped = apply_alignment(
            sender.index_select(0, oob_indices.to(sender.device)),
            bootstrap_state,
        ).float()
        bootstrap_errors = torch.linalg.vector_norm(
            bootstrap_mapped - receiver.index_select(0, oob_indices.to(receiver.device)),
            dim=-1,
        ) / math.sqrt(receiver.shape[-1])
        bootstrap_errors_cpu = bootstrap_errors.detach().cpu()
        oob_error_sum[oob_indices] += bootstrap_errors_cpu
        oob_error_sq_sum[oob_indices] += bootstrap_errors_cpu**2
        oob_count[oob_indices] += 1

    fallback_counts = oob_count.clamp_min(1.0)
    oob_mean = torch.where(
        oob_count > 0,
        oob_error_sum / fallback_counts,
        base_errors_cpu,
    )
    oob_variance = torch.where(
        oob_count > 0,
        (oob_error_sq_sum / fallback_counts) - (oob_mean**2),
        torch.zeros_like(base_errors_cpu),
    ).clamp_min(0.0)
    oob_std = torch.sqrt(oob_variance)
    combined_score = base_errors_cpu + (float(bootstrap_weight) * oob_mean) + (0.5 * oob_std)

    return {
        "base_reconstruction_error": base_errors_cpu,
        "oob_reconstruction_error": oob_mean.detach().cpu(),
        "oob_reconstruction_std": oob_std.detach().cpu(),
        "combined_score": combined_score.detach().cpu(),
        "bootstrap_count": int(bootstrap_count),
        "bootstrap_ratio": float(bootstrap_ratio),
        "bootstrap_weight": float(bootstrap_weight),
    }


def apply_linear_mapping(hidden_states: torch.Tensor, mapping_matrix: torch.Tensor) -> torch.Tensor:
    return hidden_states @ _device_resident(
        mapping_matrix,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )


def apply_orthogonal_mapping(hidden_states: torch.Tensor, mapping_q: torch.Tensor) -> torch.Tensor:
    return apply_linear_mapping(hidden_states, mapping_q)


def apply_alignment(
    hidden_states: torch.Tensor,
    alignment: torch.Tensor | AlignmentState,
) -> torch.Tensor:
    if isinstance(alignment, torch.Tensor):
        return apply_linear_mapping(hidden_states, alignment)

    per_step_states = alignment.get("per_step_states")
    if (
        per_step_states
        and hidden_states.dim() == 3
        and int(hidden_states.shape[1]) == len(per_step_states)
    ):
        # Position-wise alignment: each sequence step has its own fitted state.
        # Inputs whose step count does not match fall through to the shared
        # mapping below, so mismatched sequences degrade gracefully.
        return torch.stack(
            [
                apply_alignment(hidden_states[:, step_index, :], per_step_states[step_index])
                for step_index in range(len(per_step_states))
            ],
            dim=1,
        )

    pre_projection_state = alignment.get("pre_projection_state")
    post_projection_state = alignment.get("post_projection_state")
    mapping_matrix = alignment.get("mapping_matrix", alignment.get("global_alignment_q"))
    if mapping_matrix is None:
        raise ValueError("alignment state is missing a mapping_matrix")
    mapped = apply_adaptive_projection(hidden_states, pre_projection_state)
    mapped = apply_linear_mapping(mapped, mapping_matrix)
    mapping_bias = alignment.get("mapping_bias", alignment.get("global_alignment_bias"))
    if mapping_bias is not None:
        mapped = mapped + _device_resident(
            mapping_bias,
            device=mapped.device,
            dtype=mapped.dtype,
        )
    mapped = apply_adaptive_projection(mapped, post_projection_state)
    return mapped
