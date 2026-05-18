from __future__ import annotations

import torch
from omegaconf import OmegaConf

from analyze_distance_accuracy import (
    _build_q_generalization_report,
    _solver_alignment_strategy_name,
)


def test_build_q_generalization_report_passes_with_stable_bootstraps() -> None:
    summary_rows = [
        {
            "row_type": "overall",
            "config_id": "anchors_250_uniform",
            "control_name": "full_anchor",
            "selected_default": True,
            "accuracy_percentage": 72.0,
            "answer_perplexity": 1.8,
            "mean_anchor_eval_post_alignment_l2_distance": 0.20,
        },
        {
            "row_type": "overall",
            "config_id": "anchors_250_uniform",
            "control_name": "heldout_anchor_generalization",
            "selected_default": False,
            "accuracy_percentage": 70.5,
            "answer_perplexity": 1.9,
            "mean_anchor_eval_post_alignment_l2_distance": 0.21,
        },
        {
            "row_type": "overall",
            "config_id": "anchors_250_uniform",
            "control_name": "shuffled_anchor_control",
            "selected_default": False,
            "accuracy_percentage": 41.0,
            "answer_perplexity": 3.5,
            "mean_anchor_eval_post_alignment_l2_distance": 0.45,
        },
        {
            "row_type": "overall",
            "config_id": "anchors_250_uniform",
            "control_name": "anchor_subset_stability",
            "bootstrap_index": "0",
            "selected_default": False,
            "accuracy_percentage": 71.5,
            "mean_anchor_eval_post_alignment_l2_distance": 0.22,
        },
        {
            "row_type": "overall",
            "config_id": "anchors_250_uniform",
            "control_name": "anchor_subset_stability",
            "bootstrap_index": "1",
            "selected_default": False,
            "accuracy_percentage": 70.8,
            "mean_anchor_eval_post_alignment_l2_distance": 0.21,
        },
    ]

    report = _build_q_generalization_report(
        summary_rows,
        bootstrap_qs={
            "anchors_250_uniform": [
                torch.eye(4),
                torch.eye(4) * 0.99,
            ]
        },
        bootstrap_singular_values={
            "anchors_250_uniform": [
                torch.tensor([4.0, 3.0, 2.0, 1.0]),
                torch.tensor([3.9, 3.0, 2.0, 1.1]),
            ]
        },
    )

    assert report["passed"] is True
    assert report["shuffled_anchor_is_worse"] is True
    assert report["heldout_anchor_accuracy_drop"] == 1.5


def test_solver_alignment_strategy_name_uses_configured_solver_strategy() -> None:
    cfg = OmegaConf.create(
        {
            "alignment": {
                "strategy": "orthogonal",
                "prompt_calibration": {"enabled": False},
            }
        }
    )

    assert _solver_alignment_strategy_name(cfg, "heldout_anchor_generalization") == "orthogonal"


def test_solver_alignment_strategy_name_normalizes_calibration_variant() -> None:
    cfg = OmegaConf.create(
        {
            "alignment": {
                "strategy": "hybrid_affine",
                "prompt_calibration": {"enabled": True},
            }
        }
    )

    assert _solver_alignment_strategy_name(cfg, "hybrid_affine_plus_calibration") == "hybrid_affine"
