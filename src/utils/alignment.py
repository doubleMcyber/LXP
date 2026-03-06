from __future__ import annotations

from collections.abc import Sequence
import re

import torch


HiddenStateInput = torch.Tensor | Sequence[torch.Tensor]
LayerWeightInput = Sequence[float] | torch.Tensor | None
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


def compute_cross_covariance(
    sender_hidden_states: HiddenStateInput,
    receiver_hidden_states: HiddenStateInput,
    *,
    layer_weights: LayerWeightInput = None,
) -> torch.Tensor:
    sender_layers = _normalize_hidden_state_layers(
        sender_hidden_states, name="sender_hidden_states"
    )
    receiver_layers = _normalize_hidden_state_layers(
        receiver_hidden_states, name="receiver_hidden_states"
    )
    if len(sender_layers) != len(receiver_layers):
        raise ValueError("sender and receiver must provide the same number of layers")

    normalized_weights = _normalize_layer_weights(len(sender_layers), layer_weights)
    mean_cross_covariance: torch.Tensor | None = None
    for layer_index, (sender_layer, receiver_layer) in enumerate(
        zip(sender_layers, receiver_layers)
    ):
        sender, receiver = _pair_anchor_states(sender_layer, receiver_layer)
        layer_cross_covariance = sender.transpose(0, 1) @ receiver

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
    u, _, vh = torch.linalg.svd(cross_covariance, full_matrices=False)
    return u @ vh


def compute_ridge_mapping(
    source_hidden_states: torch.Tensor,
    target_hidden_states: torch.Tensor,
    *,
    regularization: float = 1e-4,
) -> torch.Tensor:
    if regularization < 0:
        raise ValueError("regularization must be non-negative")

    source = _flatten_hidden_states(source_hidden_states).to(torch.float32)
    target = _flatten_hidden_states(target_hidden_states).to(torch.float32).to(source.device)
    if source.shape[0] != target.shape[0]:
        raise ValueError(
            "source and target must provide the same number of anchor rows for ridge mapping"
        )

    gram = source.transpose(0, 1) @ source
    rhs = source.transpose(0, 1) @ target
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return torch.linalg.solve(gram + (regularization * identity), rhs)


def apply_linear_mapping(hidden_states: torch.Tensor, mapping_matrix: torch.Tensor) -> torch.Tensor:
    return hidden_states @ mapping_matrix.to(
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )


def apply_orthogonal_mapping(hidden_states: torch.Tensor, mapping_q: torch.Tensor) -> torch.Tensor:
    return apply_linear_mapping(hidden_states, mapping_q)
