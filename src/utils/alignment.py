from __future__ import annotations

from collections.abc import Sequence

import torch


HiddenStateInput = torch.Tensor | Sequence[torch.Tensor]
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
        for anchor in sorted(shared_token_ids):
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
    sender_hidden_states: HiddenStateInput, receiver_hidden_states: HiddenStateInput
) -> torch.Tensor:
    sender_layers = _normalize_hidden_state_layers(
        sender_hidden_states, name="sender_hidden_states"
    )
    receiver_layers = _normalize_hidden_state_layers(
        receiver_hidden_states, name="receiver_hidden_states"
    )
    if len(sender_layers) != len(receiver_layers):
        raise ValueError("sender and receiver must provide the same number of layers")

    mean_cross_covariance: torch.Tensor | None = None
    for sender_layer, receiver_layer in zip(sender_layers, receiver_layers):
        sender, receiver = _pair_anchor_states(sender_layer, receiver_layer)
        layer_cross_covariance = sender.transpose(0, 1) @ receiver

        if mean_cross_covariance is None:
            mean_cross_covariance = layer_cross_covariance
            continue
        if layer_cross_covariance.shape != mean_cross_covariance.shape:
            raise ValueError("all layer pairs must share the same hidden dimensions")

        mean_cross_covariance = mean_cross_covariance + layer_cross_covariance.to(
            mean_cross_covariance.device
        )

    if mean_cross_covariance is None:
        raise ValueError("sender and receiver hidden states are empty")

    return mean_cross_covariance / len(sender_layers)


def compute_orthogonal_mapping(
    sender_hidden_states: HiddenStateInput, receiver_hidden_states: HiddenStateInput
) -> torch.Tensor:
    cross_covariance = compute_cross_covariance(sender_hidden_states, receiver_hidden_states)
    u, _, vh = torch.linalg.svd(cross_covariance, full_matrices=False)
    return u @ vh


def apply_orthogonal_mapping(hidden_states: torch.Tensor, mapping_q: torch.Tensor) -> torch.Tensor:
    return hidden_states @ mapping_q.to(device=hidden_states.device, dtype=hidden_states.dtype)
