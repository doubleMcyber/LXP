from __future__ import annotations

from omegaconf import OmegaConf
import pytest

from benchmark_all import (
    DEFAULT_HETERO_SMOKE_AGENT_A_MODEL,
    DEFAULT_HETERO_SMOKE_AGENT_B_MODEL,
    _apply_model_profile_defaults,
    _decoded_answer_is_obvious_degenerate,
    _methods_for_suite,
    _predicted_answer,
)
from src.utils.benchmarking import (
    REPORT_SCHEMA_VERSION,
    aggregate_standard_rows,
    build_distance_calibration_report,
    build_phase1_gate_report,
    build_phase3_gate_report,
    build_runtime_smoke_report,
    build_semantic_smoke_report,
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
                "strategy": "hybrid_affine",
                "semantic_anchor_count": 250,
                "reasoning_layer_weights": [0.2, 0.3, 0.5],
                "prompt_calibration": {
                    "enabled": True,
                },
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
        dataset_split="validation",
        repetition=2,
        alignment_mode="semantic_anchor_global",
        alignment_strategy="hybrid_affine",
    )

    assert row["evaluation_surface"] == "benchmark_all"
    assert row["suite"] == "standard"
    assert row["method"] == "hybrid_hl_mas"
    assert row["dataset"] == "gsm8k"
    assert row["dataset_split"] == "validation"
    assert row["repetition"] == 2
    assert row["model_pair"] == "reasoner-a -> actor-b"
    assert row["compression_steps"] == 16
    assert row["semantic_anchor_count"] == 250
    assert row["reasoning_layer_weights"] == "0.200000,0.300000,0.500000"
    assert row["alignment_strategy"] == "hybrid_affine"
    assert row["report_schema_version"] == REPORT_SCHEMA_VERSION
    assert row["model_pair_compatibility_status"] == ""


def test_aggregate_standard_rows_computes_rates_and_means() -> None:
    row_base = build_standard_row_base(
        _make_cfg(),
        evaluation_surface="benchmark_all",
        suite="standard",
        method="global_anchor_hybrid_affine",
        dataset="gsm8k",
        dataset_split="validation",
        repetition=0,
        compression_steps=10,
        alignment_mode="global_anchor_hybrid_affine",
        alignment_strategy="hybrid_affine",
    )
    rows = [
        {
            **row_base,
            "sample_index": 0,
            "kv_cache_transferred": True,
            "kv_cache_status": "transferred",
            "kv_cache_reason": "compatible",
            "handoff_status": "ok",
            "handoff_surface": "input_embedding",
            "decode_status": "decoded",
            "prompt": "q1",
            "target_answer": "1",
            "predicted_answer": "1",
            "decoded_text": "1",
            "generated_tokens": 1,
            "answer_token_count": 1,
            "answer_nll": 0.5,
            "answer_perplexity": 1.6487,
            "correct": True,
            "latency_seconds": 1.0,
            "pre_alignment_l2_distance": 0.5,
            "pre_alignment_cosine_distance": 0.3,
            "post_alignment_l2_distance": 0.2,
            "post_alignment_cosine_distance": 0.1,
            "alignment_residual_norm_ratio": 0.05,
            "alignment_bias_norm": 0.02,
            "prompt_calibration_enabled": True,
            "prompt_calibration_bias_norm": 0.01,
            "raw_handoff_entropy": 5.0,
            "handoff_uncertainty": 4.0,
            "confidence_gate_triggered": False,
            "fallback_discrete_reasoning_steps": 0,
            "latent_trajectory_steps": 10,
            "total_reasoning_steps": 10,
            "continuous_integration_seconds": 1.5,
            "global_alignment_cache_hit": True,
            "error": "",
        },
        {
            **row_base,
            "sample_index": 1,
            "kv_cache_transferred": True,
            "kv_cache_status": "transferred",
            "kv_cache_reason": "compatible",
            "handoff_status": "ok",
            "handoff_surface": "input_embedding",
            "decode_status": "decoded",
            "prompt": "q2",
            "target_answer": "2",
            "predicted_answer": "0",
            "decoded_text": "0",
            "generated_tokens": 1,
            "answer_token_count": 1,
            "answer_nll": 1.0,
            "answer_perplexity": 2.7183,
            "correct": False,
            "latency_seconds": 3.0,
            "pre_alignment_l2_distance": 0.7,
            "pre_alignment_cosine_distance": 0.5,
            "post_alignment_l2_distance": 0.4,
            "post_alignment_cosine_distance": 0.2,
            "alignment_residual_norm_ratio": 0.07,
            "alignment_bias_norm": 0.03,
            "prompt_calibration_enabled": True,
            "prompt_calibration_bias_norm": 0.02,
            "raw_handoff_entropy": 7.0,
            "handoff_uncertainty": 6.0,
            "confidence_gate_triggered": True,
            "fallback_discrete_reasoning_steps": 2,
            "latent_trajectory_steps": 12,
            "total_reasoning_steps": 14,
            "continuous_integration_seconds": 2.5,
            "global_alignment_cache_hit": False,
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
    assert summary["prompt_calibration_rate_percentage"] == 100.0
    assert summary["explicit_status_rate_percentage"] == 100.0
    assert summary["handoff_ok_rate_percentage"] == 100.0
    assert summary["answer_perplexity"] == summary["answer_perplexity"]


def test_runtime_smoke_report_checks_errors_and_statuses() -> None:
    report = build_runtime_smoke_report(
        [
            {
                "error": "",
                "handoff_status": "ok",
                "kv_cache_status": "unsupported_architecture_mismatch",
                "decode_status": "empty_decode",
            }
        ],
        max_error_count=0,
        require_explicit_statuses=True,
    )

    assert report["passed"] is True
    assert report["phase"] == "runtime_smoke"


def test_semantic_smoke_report_checks_baseline_decode_cache_and_perplexity() -> None:
    rows = [
        {
            "method": "pure_text_cot",
            "predicted_answer": "42",
            "target_answer": "42",
            "correct": True,
            "decoded_text": "Final answer: 42",
            "answer_perplexity": 2.0,
        },
        {
            "method": "hybrid_hl_mas",
            "predicted_answer": "42",
            "target_answer": "42",
            "correct": True,
            "decoded_text": "Final answer: 42",
            "kv_cache_transferred": True,
            "answer_perplexity": 3.0,
        },
    ]

    report = build_semantic_smoke_report(
        rows,
        baseline_methods=("pure_text_cot",),
        latent_methods=("hybrid_hl_mas",),
        model_pair_compatibility={
            "kv_cache_compatible": True,
            "status": "predicted_compatible",
            "reason": "matching_cache_topology",
        },
    )

    assert report["passed"] is True
    assert report["baseline_answer_extraction_rate_percentage"] == 100.0
    assert report["baseline_accuracy_percentage"] == 100.0
    assert report["latent_accuracy_percentage"] == 100.0
    assert report["method_accuracy_percentage"]["pure_text_cot"] == 100.0
    assert report["method_accuracy_percentage"]["hybrid_hl_mas"] == 100.0
    assert report["latent_non_empty_decoded_rate_percentage"] == 100.0
    assert report["compatible_cache_transfer_rate_percentage"] == 100.0
    assert report["max_answer_perplexity"] == 3.0
    assert report["degenerate_decode_count"] == 0


def test_semantic_smoke_report_flags_degenerate_decode_and_high_perplexity() -> None:
    report = build_semantic_smoke_report(
        [
            {
                "method": "pure_text_cot",
                "predicted_answer": "2",
                "target_answer": "2",
                "correct": True,
                "decoded_text": "Final answer: 2",
                "answer_perplexity": 2.0,
            },
            {
                "method": "hybrid_hl_mas",
                "predicted_answer": "2",
                "target_answer": "1",
                "correct": False,
                "decoded_text": " ".join(["2"] * 32),
                "kv_cache_transferred": True,
                "answer_perplexity": 20000.0,
            },
        ],
        baseline_methods=("pure_text_cot",),
        latent_methods=("hybrid_hl_mas",),
        model_pair_compatibility={"kv_cache_compatible": True},
        max_answer_perplexity=10000.0,
    )

    assert report["passed"] is False
    assert report["degenerate_decode_count"] == 1
    assert report["wrong_answer_count"] == 1
    assert report["worst_answer_perplexity_rows"][0]["method"] == "hybrid_hl_mas"
    assert any("perplexity" in item for item in report["missing_requirements"])
    assert any("degenerate" in item for item in report["missing_requirements"])


def test_semantic_smoke_report_can_require_accuracy_and_final_answer_markers() -> None:
    report = build_semantic_smoke_report(
        [
            {
                "method": "pure_text_cot",
                "predicted_answer": "2",
                "target_answer": "2",
                "correct": True,
                "decoded_text": "Final answer: 2",
                "answer_perplexity": 2.0,
            },
            {
                "method": "hybrid_hl_mas",
                "predicted_answer": "13",
                "target_answer": "2",
                "correct": False,
                "decoded_text": "Reasoning mentions 10 + 3 = 13.",
                "kv_cache_transferred": True,
                "answer_perplexity": 10.0,
            },
        ],
        baseline_methods=("pure_text_cot",),
        latent_methods=("hybrid_hl_mas",),
        model_pair_compatibility={"kv_cache_compatible": True},
        min_baseline_accuracy_percentage=1.0,
        min_latent_accuracy_percentage=1.0,
        min_method_accuracy_percentage=1.0,
        require_final_answer_marker_methods=("pure_text_cot", "hybrid_hl_mas"),
    )

    assert report["passed"] is False
    assert report["baseline_accuracy_percentage"] == 100.0
    assert report["latent_accuracy_percentage"] == 0.0
    assert report["method_accuracy_percentage"]["hybrid_hl_mas"] == 0.0
    assert report["required_final_answer_marker_rate_percentage"] == 50.0
    assert any("Latent accuracy" in item for item in report["missing_requirements"])
    assert any("Method hybrid_hl_mas accuracy" in item for item in report["missing_requirements"])
    assert any("final-answer marker" in item for item in report["missing_requirements"])


def test_gsm8k_prediction_prefers_final_answer_marker() -> None:
    decoded = "Reasoning mentions 100 and 200. Final answer: 42. Then ignores 9000."

    assert _predicted_answer("gsm8k", decoded) == "42"


def test_guarded_latent_methods_are_available_for_standard_suite() -> None:
    standard_methods = [name for name, _ in _methods_for_suite("standard")]

    assert "guarded_latent_transfer" in standard_methods
    assert "same_family_guarded_latent" in standard_methods


@pytest.mark.parametrize(
    "decoded_text",
    [
        "",
        "Final answer: 000",
        "Final answer: 0.00",
        "Final answer: -0",
    ],
)
def test_decoded_answer_degenerate_guard_catches_zero_forms(decoded_text: str) -> None:
    assert _decoded_answer_is_obvious_degenerate(decoded_text) is True


def test_decoded_answer_degenerate_guard_allows_nonzero_answer() -> None:
    assert _decoded_answer_is_obvious_degenerate("Final answer: 28") is False


def test_hetero_smoke_uses_cross_family_default_models() -> None:
    cfg = _make_cfg()

    _apply_model_profile_defaults(
        cfg,
        agent_a_model=None,
        agent_b_model=None,
        hetero_smoke=True,
    )

    assert cfg.agent_a_model == DEFAULT_HETERO_SMOKE_AGENT_A_MODEL
    assert cfg.agent_b_model == DEFAULT_HETERO_SMOKE_AGENT_B_MODEL


def test_hetero_smoke_model_overrides_win() -> None:
    cfg = _make_cfg()

    _apply_model_profile_defaults(
        cfg,
        agent_a_model="custom-a",
        agent_b_model="custom-b",
        hetero_smoke=True,
    )

    assert cfg.agent_a_model == "custom-a"
    assert cfg.agent_b_model == "custom-b"


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
        {"method": "global_anchor_orthogonal", "accuracy_percentage": 50.0, "answer_perplexity": 2.2},
        {
            "method": "global_anchor_hybrid_affine_plus_calibration",
            "accuracy_percentage": 52.0,
            "answer_perplexity": 2.0,
        },
        {
            "method": "text_text_hybrid",
            "accuracy_percentage": 50.0,
            "answer_perplexity": 2.0,
        },
        {
            "method": "hybrid_hl_mas",
            "accuracy_percentage": 55.0,
            "answer_perplexity": 1.6,
        },
    ]

    report = build_phase3_gate_report(
        summary_rows,
        require_q_global_beats_prompt_local=True,
    )

    assert report["passed"] is True
    assert report["q_global_beats_prompt_local"] is True
    assert report["hybrid_affine_beats_orthogonal"] is True
    assert report["hybrid_beats_text_text"] is True


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
        "text_text_hybrid",
        "homogeneous_ridge_latent",
        "homogeneous_orthogonal_latent",
    ]
    assert "text_text_hybrid" in standard_methods
    assert "global_anchor_orthogonal" in standard_methods
    assert "global_anchor_ridge" in standard_methods
    assert "global_anchor_hybrid_affine" in standard_methods
    assert "global_anchor_hybrid_affine_plus_calibration" in standard_methods
    assert "guarded_latent_transfer" in standard_methods
    assert "same_family_guarded_latent" in standard_methods
    assert "hybrid_hl_mas" in standard_methods
    assert "homogeneous_orthogonal_latent" in standard_methods


def test_methods_for_suite_filters_requested_methods_in_order() -> None:
    selected_methods = [
        name
        for name, _ in _methods_for_suite(
            "standard",
            ["hybrid_hl_mas", "pure_text_cot"],
        )
    ]

    assert selected_methods == ["hybrid_hl_mas", "pure_text_cot"]
    with pytest.raises(ValueError, match="Unknown methods"):
        _methods_for_suite("standard", ["does_not_exist"])
