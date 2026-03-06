from __future__ import annotations

import torch

from src.utils.alignment import (
    compute_cross_covariance,
    compute_orthogonal_mapping,
    get_preferred_semantic_anchors,
    resolve_shared_semantic_anchor_ids,
)


def test_compute_orthogonal_mapping_keeps_single_layer_behavior() -> None:
    sender = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    receiver = sender.clone()

    mapping = compute_orthogonal_mapping(sender, receiver)

    assert mapping.shape == (2, 2)
    assert torch.allclose(mapping, torch.eye(2), atol=1e-6)


def test_mean_cross_covariance_alignment_increases_rank_with_multiple_layers() -> None:
    sender_layers = [
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[[0.0, 1.0]]]),
    ]
    receiver_layers = [
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[[0.0, 1.0]]]),
    ]

    single_layer_covariance = compute_cross_covariance(
        sender_layers[0], receiver_layers[0]
    )
    multi_layer_covariance = compute_cross_covariance(sender_layers, receiver_layers)
    mapping = compute_orthogonal_mapping(sender_layers, receiver_layers)

    assert torch.linalg.matrix_rank(single_layer_covariance).item() == 1
    assert torch.linalg.matrix_rank(multi_layer_covariance).item() == 2
    assert torch.allclose(mapping.transpose(0, 1) @ mapping, torch.eye(2), atol=1e-6)


class _FakeTokenizer:
    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens
        self._ids = {token: index for index, token in enumerate(tokens)}
        self.all_special_ids: tuple[int, ...] = ()

    def get_vocab(self) -> dict[str, int]:
        return dict(self._ids)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        token_id = self._ids.get(text)
        return [] if token_id is None else [token_id]

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(self._tokens[token_id] for token_id in token_ids)


def test_resolve_shared_semantic_anchor_ids_prefers_curated_anchor_list() -> None:
    preferred = list(get_preferred_semantic_anchors())
    tokenizer_a = _FakeTokenizer(preferred + ["extra"])
    tokenizer_b = _FakeTokenizer(preferred + ["fallback"])

    anchor_strings, anchor_ids_a, anchor_ids_b = resolve_shared_semantic_anchor_ids(
        tokenizer_a,
        tokenizer_b,
        anchor_count=100,
    )

    assert len(anchor_strings) == 100
    assert anchor_strings == tuple(preferred)
    assert anchor_ids_a.shape == (100,)
    assert anchor_ids_b.shape == (100,)
