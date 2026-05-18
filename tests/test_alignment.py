from __future__ import annotations

import torch

from src.utils.alignment import (
    apply_alignment,
    apply_linear_mapping,
    compute_alignment_state,
    compute_cross_covariance,
    compute_orthogonal_mapping,
    compute_ridge_mapping,
    get_preferred_semantic_anchors,
    resolve_shared_semantic_anchor_ids,
    score_anchor_stability,
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


def test_compute_cross_covariance_uniform_layer_weights_matches_default() -> None:
    sender_layers = [
        torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
        torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]),
    ]
    receiver_layers = [
        torch.tensor([[[2.0, 1.0], [4.0, 3.0]]]),
        torch.tensor([[[6.0, 5.0], [8.0, 7.0]]]),
    ]

    default_covariance = compute_cross_covariance(sender_layers, receiver_layers)
    weighted_covariance = compute_cross_covariance(
        sender_layers,
        receiver_layers,
        layer_weights=[0.5, 0.5],
    )

    assert torch.allclose(default_covariance, weighted_covariance)


def test_compute_ridge_mapping_matches_identity_for_identical_anchor_spaces() -> None:
    source = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    mapping = compute_ridge_mapping(source, source, regularization=1e-6)
    transformed = apply_linear_mapping(source, mapping)

    assert mapping.shape == (2, 2)
    assert torch.allclose(transformed, source, atol=1e-4)


def test_compute_alignment_state_matches_orthogonal_solver_when_hybrid_features_disabled() -> None:
    sender = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    receiver = sender.clone()

    orthogonal = compute_orthogonal_mapping(sender, receiver)
    alignment_state = compute_alignment_state(
        sender,
        receiver,
        strategy="orthogonal",
        center=False,
        use_bias=False,
        adaptive_projection_strength=0.0,
    )

    assert torch.allclose(alignment_state["mapping_matrix"], orthogonal, atol=1e-6)
    transformed = apply_alignment(sender, alignment_state)
    assert torch.allclose(transformed, receiver, atol=1e-6)


def test_hybrid_affine_alignment_reduces_mean_shift_error() -> None:
    sender = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    receiver = (sender * 1.2) + torch.tensor([0.5, -0.25])

    orthogonal_state = compute_alignment_state(
        sender,
        receiver,
        strategy="orthogonal",
        center=False,
        use_bias=False,
        adaptive_projection_strength=0.0,
    )
    hybrid_state = compute_alignment_state(
        sender,
        receiver,
        strategy="hybrid_affine",
        center=True,
        use_bias=True,
        regularization=1e-4,
        residual_alpha=1.0,
        residual_max_norm_ratio=0.5,
        adaptive_projection_strength=0.0,
    )

    orthogonal_error = torch.mean((apply_alignment(sender, orthogonal_state) - receiver) ** 2)
    hybrid_error = torch.mean((apply_alignment(sender, hybrid_state) - receiver) ** 2)

    assert hybrid_error < orthogonal_error


def test_hybrid_affine_residual_norm_cap_is_enforced() -> None:
    sender = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    receiver = torch.tensor(
        [
            [0.0, 0.0],
            [3.0, 0.0],
            [0.0, 0.25],
            [3.0, 0.25],
        ]
    )
    alignment_state = compute_alignment_state(
        sender,
        receiver,
        strategy="hybrid_affine",
        center=True,
        use_bias=True,
        residual_alpha=4.0,
        residual_max_norm_ratio=0.1,
        adaptive_projection_strength=0.0,
    )

    assert alignment_state["residual_norm_ratio"] <= 0.1001


def test_score_anchor_stability_returns_cpu_rank_statistics() -> None:
    sender = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    receiver = sender + torch.tensor([0.2, -0.1])

    stability = score_anchor_stability(
        sender,
        receiver,
        strategy="hybrid_affine",
        bootstrap_count=2,
        seed=7,
    )

    assert stability["combined_score"].device.type == "cpu"
    assert stability["oob_reconstruction_error"].device.type == "cpu"
    assert stability["combined_score"].shape == (4,)


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


def test_resolve_shared_semantic_anchor_ids_scales_beyond_curated_seed_bank() -> None:
    preferred = list(get_preferred_semantic_anchors())
    extras = [f"tok{index:03d}" for index in range(600)]
    tokenizer_a = _FakeTokenizer(preferred + extras)
    tokenizer_b = _FakeTokenizer(preferred + extras)

    anchor_strings, anchor_ids_a, anchor_ids_b = resolve_shared_semantic_anchor_ids(
        tokenizer_a,
        tokenizer_b,
        anchor_count=500,
    )

    assert len(anchor_strings) == 500
    assert anchor_strings[:100] == tuple(preferred)
    assert anchor_ids_a.shape == (500,)
    assert anchor_ids_b.shape == (500,)
