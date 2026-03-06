from __future__ import annotations

import torch
from omegaconf import OmegaConf

from analyze_distance_accuracy import _assign_distance_deciles, _breaking_point_decile
from latent_pipeline import (
    _aggregate_hidden_layers,
    _build_alignment_cache_key,
    _resolve_reasoning_layer_indices_from_counts,
    _resolve_reasoning_layer_weights,
)


def _make_cfg():
    return OmegaConf.create(
        {
            "alignment": {
                "semantic_anchor_count": 250,
                "cache_dir": ".cache/alignment",
                "reasoning_layer_weights": [0.2, 0.3, 0.5],
            }
        }
    )


def test_aggregate_hidden_layers_uniform_weights_matches_mean() -> None:
    hidden_layers = [
        torch.tensor([[[1.0, 3.0]]]),
        torch.tensor([[[5.0, 7.0]]]),
        torch.tensor([[[9.0, 11.0]]]),
    ]

    aggregated = _aggregate_hidden_layers(hidden_layers, [1.0, 1.0, 1.0])
    expected = torch.stack(hidden_layers, dim=0).mean(dim=0)

    assert torch.allclose(aggregated, expected)


def test_resolve_reasoning_layer_weights_renormalizes_surviving_preferred_layers() -> None:
    cfg = _make_cfg()

    layer_weights = _resolve_reasoning_layer_weights(cfg, (12, 16))

    assert layer_weights == (0.4, 0.6)


def test_resolve_reasoning_layer_indices_falls_back_to_deepest_layers_when_needed() -> None:
    layer_indices = _resolve_reasoning_layer_indices_from_counts(10, 10)

    assert layer_indices == (8, 9, 10)


def test_build_alignment_cache_key_changes_with_anchor_count_and_layer_weights() -> None:
    first_key = _build_alignment_cache_key(
        agent_a_model="a",
        agent_b_model="b",
        torch_dtype="bfloat16",
        semantic_anchor_count=100,
        reasoning_layer_indices=(12, 16, 20),
        reasoning_layer_weights=(0.2, 0.3, 0.5),
    )
    second_key = _build_alignment_cache_key(
        agent_a_model="a",
        agent_b_model="b",
        torch_dtype="bfloat16",
        semantic_anchor_count=250,
        reasoning_layer_indices=(12, 16, 20),
        reasoning_layer_weights=(0.2, 0.3, 0.5),
    )
    third_key = _build_alignment_cache_key(
        agent_a_model="a",
        agent_b_model="b",
        torch_dtype="bfloat16",
        semantic_anchor_count=250,
        reasoning_layer_indices=(12, 16, 20),
        reasoning_layer_weights=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    )

    assert first_key != second_key
    assert second_key != third_key


def test_distance_decile_breaking_point_tracks_accuracy_drop() -> None:
    rows = [
        {"post_alignment_l2_distance": 0.10, "correct": True, "latency_seconds": 1.0, "error": ""},
        {"post_alignment_l2_distance": 0.20, "correct": True, "latency_seconds": 1.0, "error": ""},
        {"post_alignment_l2_distance": 0.30, "correct": True, "latency_seconds": 1.0, "error": ""},
        {"post_alignment_l2_distance": 0.40, "correct": False, "latency_seconds": 1.0, "error": ""},
        {"post_alignment_l2_distance": 0.50, "correct": False, "latency_seconds": 1.0, "error": ""},
    ]

    _assign_distance_deciles(rows)
    decile_rows = [
        {"distance_decile": 1, "accuracy_percentage": 100.0},
        {"distance_decile": 2, "accuracy_percentage": 100.0},
        {"distance_decile": 3, "accuracy_percentage": 100.0},
        {"distance_decile": 4, "accuracy_percentage": 0.0},
        {"distance_decile": 5, "accuracy_percentage": 0.0},
    ]

    assert rows[0]["distance_decile"] == 1
    assert rows[-1]["distance_decile"] == 5
    assert _breaking_point_decile(decile_rows) == 4
