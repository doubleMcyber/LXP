from __future__ import annotations

from omegaconf import OmegaConf

from benchmark_all import _methods_for_suite
from src.utils.benchmarking import (
    aggregate_standard_rows,
    build_distance_calibration_report,
    build_phase1_gate_report,
    build_phase3_gate_report,
    build_standard_row_base,
    build_training_phase2_report,
)


def _make_cfg():
    return OmegaConf.create(
        {
            "agent_a_model": "reasoner-a",
            "agent_b_model": "actor-b",
            "torch_dtype": "bfloat16",
            "latent_steps": 10,
            "alignment": {
                "semantic_anchor_count": 250,
                "reasoning_layer_weights": [0.2, 0.3, 0.5],
            },
            "training": {
                "compressed_steps": 16,
            },
        }
    )


def test_build_standard_row_base_uses_cfg_metadata() -> None:
    row = build_standard_row_base(
        _make_cfg(),
        evaluation_surface="benchmark_all",
        suite="standard",
        method="hybrid_hl_mas",
        dataset="gsm8k",
        repetition=2,
        alignment_mode="semantic_anchor_global",
    )

    assert row["evaluation_surface"] == "benchmark_all"
    assert row["suite"] == "standard"
    assert row["method"] == "hybrid_hl_mas"
    assert row["dataset"] == "gsm8k"
    assert row["repetition"] == 2
    assert row["model_pair"] == "reasoner-a -> actor-b"
    assert row["compression_steps"] == 16
    assert row["semantic_anchor_count"] == 250
    assert row["reasoning_layer_weights"] == "0.200000,0.300000,0.500000"


def test_aggregate_standard_rows_computes_rates_and_means() -> None:
    row_base = build_standard_row_base(
        _make_cfg(),
        evaluation_surface="benchmark_all",
        suite="standard",
        method="global_anchor_latent",
        dataset="gsm8k",
        repetition=0,
        compression_steps=10,
        alignment_mode="semantic_anchor_global",
    )
    rows = [
        {
            **row_base,
            "sample_index": 0,
            "kv_cache_transferred": True,
            "prompt": "q1",
            "target_answer": "1",
            "predicted_answer": "1",
            "decoded_text": "1",
            "generated_tokens": 1,
            "correct": True,
            "latency_seconds": 1.0,
            "pre_alignment_l2_distance": 0.5,
            "pre_alignment_cosine_distance": 0.3,
            "post_alignment_l2_distance": 0.2,
            "post_alignment_cosine_distance": 0.1,
            "raw_handoff_entropy": 5.0,
            "handoff_uncertainty": 4.0,
            "confidence_gate_triggered": False,
            "fallback_discrete_reasoning_steps": 0,
            "latent_trajectory_steps": 10,
            "total_reasoning_steps": 10,
            "continuous_integration_seconds": 1.5,
            "error": "",
        },
        {
            **row_base,
            "sample_index": 1,
            "kv_cache_transferred": True,
            "prompt": "q2",
            "target_answer": "2",
            "predicted_answer": "0",
            "decoded_text": "0",
            "generated_tokens": 1,
            "correct": False,
            "latency_seconds": 3.0,
            "pre_alignment_l2_distance": 0.7,
            "pre_alignment_cosine_distance": 0.5,
            "post_alignment_l2_distance": 0.4,
            "post_alignment_cosine_distance": 0.2,
            "raw_handoff_entropy": 7.0,
            "handoff_uncertainty": 6.0,
            "confidence_gate_triggered": True,
            "fallback_discrete_reasoning_steps": 2,
            "latent_trajectory_steps": 12,
            "total_reasoning_steps": 14,
            "continuous_integration_seconds": 2.5,
            "error": "",
        },
    ]

    [summary] = aggregate_standard_rows(rows)

    assert summary["sample_count"] == 2
    assert summary["accuracy_percentage"] == 50.0
    assert summary["average_latency_seconds"] == 2.0
    assert summary["tokens_per_second"] == 0.5
    assert summary["cache_transfer_rate_percentage"] == 100.0
    assert summary["confidence_gate_trigger_rate_percentage"] == 50.0
    assert summary["mean_post_alignment_l2_distance"] == 0.30000000000000004


def test_build_phase1_gate_report_passes_when_thresholds_are_met() -> None:
    summary_rows = [
        {
            "method": "homogeneous_ridge_latent",
            "repetition_count": 3,
            "failure_rate_percentage": 0.0,
            "error_count": 0,
            "sample_count": 5,
            "cache_transfer_rate_percentage": 100.0,
            "non_empty_decoded_rate_percentage": 100.0,
        },
        {
            "method": "homogeneous_orthogonal_latent",
            "repetition_count": 3,
            "failure_rate_percentage": 0.0,
            "error_count": 0,
            "sample_count": 5,
            "cache_transfer_rate_percentage": 100.0,
            "non_empty_decoded_rate_percentage": 100.0,
        },
    ]

    report = build_phase1_gate_report(
        summary_rows,
        required_repetitions=3,
        max_error_rate_percentage=0.0,
        min_cache_transfer_rate_percentage=100.0,
        min_non_empty_decoded_rate_percentage=100.0,
    )

    assert report["passed"] is True
    assert report["missing_requirements"] == []


def test_build_phase3_gate_report_requires_q_global_to_beat_prompt_local() -> None:
    summary_rows = [
        {"method": "prompt_local_latent", "accuracy_percentage": 40.0},
        {"method": "global_anchor_latent", "accuracy_percentage": 50.0},
        {"method": "hybrid_hl_mas", "accuracy_percentage": 55.0},
    ]

    report = build_phase3_gate_report(
        summary_rows,
        require_q_global_beats_prompt_local=True,
    )

    assert report["passed"] is True
    assert report["q_global_beats_prompt_local"] is True


def test_build_distance_calibration_report_detects_useful_separation() -> None:
    rows = [
        {"correct": True, "post_alignment_l2_distance": 0.2},
        {"correct": True, "post_alignment_l2_distance": 0.3},
        {"correct": False, "post_alignment_l2_distance": 0.6},
        {"correct": False, "post_alignment_l2_distance": 0.7},
    ]
    summary_rows = [
        {
            "row_type": "overall",
            "config_id": "anchors_250_linear_deep_bias",
            "selected_default": True,
            "breaking_point_decile": 4,
        }
    ]

    report = build_distance_calibration_report(rows, summary_rows)

    assert report["passed"] is True
    assert report["correct_mean_post_alignment_l2_distance"] < report["incorrect_mean_post_alignment_l2_distance"]


def test_build_training_phase2_report_flags_missing_real_mode_requirements() -> None:
    history = [
        {"epoch": 0.0, "step": 2.0, "heldout_exact_match_accuracy": 60.0, "heldout_eval_samples": 16.0}
    ]
    cfg = _make_cfg()
    alignment_context = {
        "alignment_mode": "semantic_anchor_global",
        "semantic_anchor_count": 250,
        "reasoning_layer_weights": (0.2, 0.3, 0.5),
    }

    report = build_training_phase2_report(
        history=history,
        cfg=cfg,
        alignment_context=alignment_context,
        dataset_name="gsm8k",
        training_mode="smoke",
        seed_count=1,
        required_seed_count=3,
        min_accuracy_retention_ratio=0.85,
        baseline_accuracy_percentage=None,
    )

    assert report["passed"] is False
    assert report["final_heldout_exact_match_accuracy"] == 60.0
    assert "Training mode is not 'real'." in report["missing_requirements"]


def test_methods_for_suite_exposes_phase1_homogeneous_entrypoint() -> None:
    phase1_methods = [name for name, _ in _methods_for_suite("phase1_homogeneous")]
    standard_methods = [name for name, _ in _methods_for_suite("standard")]

    assert phase1_methods == [
        "pure_text_cot",
        "homogeneous_ridge_latent",
        "homogeneous_orthogonal_latent",
    ]
    assert "global_anchor_latent" in standard_methods
    assert "hybrid_hl_mas" in standard_methods
