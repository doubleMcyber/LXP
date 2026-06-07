from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf
import pytest
import torch

from benchmark_all import (
    DEFAULT_HETERO_SMOKE_AGENT_A_MODEL,
    DEFAULT_HETERO_SMOKE_AGENT_B_MODEL,
    DEFAULT_HETERO_SMOKE_GENERATED_ADAPTER_TRAIN_LIMIT,
    DEFAULT_HETERO_SMOKE_LATENT_METHODS,
    DEFAULT_HETERO_SMOKE_METHODS,
    DEFAULT_HETERO_SMOKE_REASONER_MAX_NEW_TOKENS,
    FINAL_ANSWER_COMPLETE_REGEX,
    _apply_eval_manifest_to_args,
    _apply_model_profile_defaults,
    _apply_generated_adapter_local_residual,
    _build_generated_adapter_local_residual_state,
    _build_eval_manifest,
    _answers_match,
    _cache_key_digest,
    _cache_key_metadata,
    _final_answer_tail_needs_scalar_verification,
    _format_sender_answer_text_handoff_prompt,
    _format_token_context_handoff_prompt,
    _format_verified_token_context_handoff_prompt,
    _format_verified_final_answer_text,
    _generated_trajectory_adapter_train_on_missing,
    _generated_trajectory_adapter_input_space,
    _generated_trajectory_adapter_target_alignment,
    _generated_trajectory_adapter_target_text,
    _generated_trajectory_adapter_trace_cache_path,
    _generated_trajectory_adapter_training_rows_cache_key,
    _generated_trajectory_trace_cache_key,
    _generated_adapter_include_prompt_values,
    _handoff_decode_prompt,
    _load_generated_trajectory_training_rows_from_disk,
    _load_generated_trajectory_trace_from_disk,
    _load_eval_manifest,
    _load_generated_trajectory_adapter_from_disk,
    _methods_for_suite,
    _predicted_answer,
    _sample_fingerprints,
    _resolve_sender_trace_reasoning_metadata_from_layer_counts,
    _require_requested_device_available,
    _reasoner_metadata_for_text_hybrid,
    _select_generated_adapter_memory_rows,
    _sender_generation_cache_fingerprint,
    _validate_eval_manifest_sample_lock,
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
    build_heterogeneous_transfer_report,
    build_transfer_comparison_report,
    build_training_phase2_report,
    build_training_smoke_report,
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


def test_handoff_decode_prompt_respects_receiver_context_mode() -> None:
    cfg = OmegaConf.create({"handoff": {"receiver_context": {"mode": "auto"}}})
    assert _handoff_decode_prompt("question", cfg) == "question"

    cfg.handoff.receiver_context.mode = "prompt_prefix"
    assert _handoff_decode_prompt("question", cfg) == "question"

    cfg.handoff.receiver_context.mode = "none"
    assert _handoff_decode_prompt("question", cfg) is None


def test_benchmark_guard_fails_fast_when_mps_unavailable() -> None:
    cfg = OmegaConf.create({"device_map": "mps"})

    with patch("benchmark_all.torch.backends.mps.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="MPS is not available"):
            _require_requested_device_available(cfg)


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
            "sender_reasoning_text": "Final answer: 1",
            "sender_reasoning_status": "complete",
            "sender_trace_cache_hit": True,
            "sender_trace_cache_path": ".cache/trace-a.pt",
            "sender_revision_enabled": True,
            "sender_revision_applied": True,
            "sender_initial_predicted_answer": "0",
            "sender_revision_predicted_answer": "1",
            "sender_revision_decision_applied": True,
            "sender_revision_decision_predicted_answer": "1",
            "sender_final_answer_marker": True,
            "sender_predicted_answer": "1",
            "sender_answer_matches_target": True,
            "predicted_answer": "1",
            "decoded_text": "1",
            "generated_tokens": 1,
            "receiver_input_token_count": 120,
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
            "sender_reasoning_text": "Final answer: 2",
            "sender_reasoning_status": "complete",
            "sender_trace_cache_hit": False,
            "sender_trace_cache_path": ".cache/trace-b.pt",
            "sender_revision_enabled": True,
            "sender_revision_applied": False,
            "sender_initial_predicted_answer": "2",
            "sender_revision_predicted_answer": None,
            "sender_revision_decision_applied": False,
            "sender_revision_decision_predicted_answer": None,
            "sender_final_answer_marker": True,
            "sender_predicted_answer": "2",
            "sender_answer_matches_target": True,
            "predicted_answer": "0",
            "decoded_text": "0",
            "generated_tokens": 1,
            "receiver_input_token_count": 180,
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
    assert summary["sender_final_answer_marker_rate_percentage"] == 100.0
    assert summary["sender_trace_cache_hit_rate_percentage"] == 50.0
    assert summary["sender_revision_applied_rate_percentage"] == 50.0
    assert summary["sender_revision_decision_applied_rate_percentage"] == 50.0
    assert summary["sender_accuracy_percentage"] == 100.0
    assert summary["sender_correct_sample_count"] == 2
    assert summary["accuracy_when_sender_correct_percentage"] == 50.0
    assert summary["average_latency_seconds"] == 2.0
    assert summary["tokens_per_second"] == 0.5
    assert summary["total_receiver_input_tokens"] == 300
    assert summary["mean_receiver_input_token_count"] == 150.0
    assert summary["max_receiver_input_token_count"] == 180
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


def test_semantic_smoke_report_does_not_require_kv_for_soft_prefix_rows() -> None:
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
            "kv_cache_transferred": False,
            "kv_cache_status": "not_provided",
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
    assert report["cache_transfer_required"] is False
    assert report["compatible_cache_transfer_rate_percentage"] is None


def test_semantic_smoke_report_supports_latent_only_focused_runs() -> None:
    report = build_semantic_smoke_report(
        [
            {
                "method": "generated_latent_handoff",
                "predicted_answer": "42",
                "target_answer": "42",
                "correct": True,
                "decoded_text": "Final answer: 42",
                "sender_reasoning_text": "Final answer: 42",
                "sender_reasoning_status": "complete",
                "sender_final_answer_marker": True,
                "sender_predicted_answer": "42",
                "sender_answer_matches_target": True,
                "kv_cache_status": "not_provided",
                "answer_perplexity": 2.0,
            },
        ],
        baseline_methods=(),
        latent_methods=("generated_latent_handoff",),
        min_baseline_accuracy_percentage=1.0,
        min_latent_accuracy_percentage=1.0,
        min_latent_accuracy_when_sender_correct_percentage=100.0,
        min_sender_final_answer_marker_rate_percentage=100.0,
        min_method_accuracy_percentage=1.0,
        require_baseline_final_answer_marker=True,
        require_final_answer_marker_methods=("generated_latent_handoff",),
    )

    assert report["passed"] is True
    assert report["baseline_sample_count"] == 0
    assert report["sender_final_answer_marker_rate_percentage"] == 100.0
    assert report["sender_accuracy_percentage"] == 100.0
    assert report["latent_accuracy_when_sender_correct_percentage"] == 100.0
    assert report["latent_accuracy_percentage"] == 100.0


def test_semantic_smoke_report_flags_sender_correct_latent_regressions() -> None:
    report = build_semantic_smoke_report(
        [
            {
                "method": "generated_latent_handoff",
                "predicted_answer": "41",
                "target_answer": "42",
                "correct": False,
                "decoded_text": "Final answer: 41",
                "sender_reasoning_text": "Final answer: 42",
                "sender_reasoning_status": "complete",
                "sender_final_answer_marker": True,
                "sender_predicted_answer": "42",
                "sender_answer_matches_target": True,
                "kv_cache_status": "not_provided",
                "answer_perplexity": 2.0,
            },
            {
                "method": "generated_latent_handoff",
                "predicted_answer": "10",
                "target_answer": "99",
                "correct": False,
                "decoded_text": "Final answer: 10",
                "sender_reasoning_text": "Final answer: 10",
                "sender_reasoning_status": "complete",
                "sender_final_answer_marker": True,
                "sender_predicted_answer": "10",
                "sender_answer_matches_target": False,
                "kv_cache_status": "not_provided",
                "answer_perplexity": 2.0,
            },
        ],
        baseline_methods=(),
        latent_methods=("generated_latent_handoff",),
        min_latent_accuracy_percentage=None,
        min_latent_accuracy_when_sender_correct_percentage=100.0,
        min_method_accuracy_percentage=None,
        require_final_answer_marker_methods=("generated_latent_handoff",),
    )

    assert report["passed"] is False
    assert report["sender_correct_latent_sample_count"] == 1
    assert report["latent_accuracy_when_sender_correct_percentage"] == 0.0
    assert any(
        "Latent accuracy when sender is correct" in item
        for item in report["missing_requirements"]
    )


def test_semantic_smoke_report_flags_sender_accuracy_regressions_separately() -> None:
    report = build_semantic_smoke_report(
        [
            {
                "method": "generated_latent_handoff",
                "predicted_answer": "42",
                "target_answer": "42",
                "correct": True,
                "decoded_text": "Final answer: 42",
                "sender_reasoning_text": "Final answer: 42",
                "sender_reasoning_status": "complete",
                "sender_final_answer_marker": True,
                "sender_predicted_answer": "42",
                "sender_answer_matches_target": True,
                "kv_cache_status": "not_provided",
                "answer_perplexity": 2.0,
            },
            {
                "method": "generated_latent_handoff",
                "predicted_answer": "108",
                "target_answer": "107",
                "correct": False,
                "decoded_text": "Final answer: 108",
                "sender_reasoning_text": "Final answer: 108",
                "sender_reasoning_status": "complete",
                "sender_final_answer_marker": True,
                "sender_predicted_answer": "108",
                "sender_answer_matches_target": False,
                "kv_cache_status": "not_provided",
                "answer_perplexity": 2.0,
            },
        ],
        baseline_methods=(),
        latent_methods=("generated_latent_handoff",),
        min_latent_accuracy_percentage=None,
        min_latent_accuracy_when_sender_correct_percentage=100.0,
        min_sender_accuracy_percentage=100.0,
        min_method_accuracy_percentage=None,
        require_final_answer_marker_methods=("generated_latent_handoff",),
    )

    assert report["passed"] is False
    assert report["sender_accuracy_percentage"] == 50.0
    assert report["latent_accuracy_when_sender_correct_percentage"] == 100.0
    assert any("Sender accuracy" in item for item in report["missing_requirements"])
    assert not any(
        "Latent accuracy when sender is correct" in item
        for item in report["missing_requirements"]
    )


def test_semantic_smoke_report_flags_incomplete_sender_reasoning() -> None:
    report = build_semantic_smoke_report(
        [
            {
                "method": "generated_latent_handoff",
                "predicted_answer": "28",
                "target_answer": "252",
                "correct": False,
                "decoded_text": "Final answer: 28",
                "sender_reasoning_text": "10% of 280 is 28",
                "sender_reasoning_status": "max_tokens_without_final_answer",
                "sender_final_answer_marker": False,
                "sender_predicted_answer": "28",
                "sender_answer_matches_target": False,
                "kv_cache_status": "not_provided",
                "answer_perplexity": 2.0,
            },
        ],
        baseline_methods=(),
        latent_methods=("generated_latent_handoff",),
        min_sender_final_answer_marker_rate_percentage=100.0,
        min_method_accuracy_percentage=None,
        require_final_answer_marker_methods=("generated_latent_handoff",),
    )

    assert report["passed"] is False
    assert report["sender_final_answer_marker_rate_percentage"] == 0.0
    assert any(
        "Sender final-answer marker rate" in item
        for item in report["missing_requirements"]
    )


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


def test_semantic_smoke_answer_perplexity_gate_scopes_to_latent_rows() -> None:
    report = build_semantic_smoke_report(
        [
            {
                "method": "pure_text_cot",
                "predicted_answer": "13",
                "target_answer": "2",
                "correct": False,
                "decoded_text": "Final answer: 13",
                "answer_perplexity": 20000.0,
            },
            {
                "method": "hybrid_hl_mas",
                "predicted_answer": "2",
                "target_answer": "2",
                "correct": True,
                "decoded_text": "Final answer: 2",
                "kv_cache_status": "not_provided",
                "answer_perplexity": 20.0,
            },
        ],
        baseline_methods=("pure_text_cot",),
        latent_methods=("hybrid_hl_mas",),
        max_answer_perplexity=10000.0,
    )

    assert report["max_answer_perplexity"] == 20.0
    assert report["max_all_answer_perplexity"] == 20000.0
    assert not any("perplexity" in item for item in report["missing_requirements"])


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


def test_gsm8k_prediction_uses_latest_final_answer_marker() -> None:
    decoded = (
        "Draft solution says Final answer: 108.\n\n"
        "Verification finds the copied relationship was off.\n"
        "Final answer: 107."
    )

    assert _predicted_answer("gsm8k", decoded) == "107"


def test_gsm8k_prediction_prefers_complete_boxed_final_answer() -> None:
    decoded = (
        "Final answer: \\boxed{28}\n\n"
        "```\nFinal answer: \\boxed{28}\n```\n\n"
        "```\nFinal answer: \\boxed{2"
    )

    assert _predicted_answer("gsm8k", decoded) == "28"


def test_final_answer_tail_flags_non_scalar_sender_answers() -> None:
    assert _final_answer_tail_needs_scalar_verification("Final answer: 10 boys") is True
    assert _final_answer_tail_needs_scalar_verification("Final answer: 17.") is False


def test_verified_final_answer_payload_stays_compact() -> None:
    assert _format_verified_final_answer_text("300") == (
        "\n\nVerification decision:\nFinal answer: 300.\n"
    )


def test_sender_answer_text_handoff_prompt_is_copy_only() -> None:
    prompt = _format_sender_answer_text_handoff_prompt("300")

    assert "Verified upstream final answer" in prompt
    assert "300" in prompt
    assert prompt.rstrip().endswith("Final answer:")
    assert "original problem" not in prompt.casefold()


def test_token_context_handoff_prompt_uses_sender_reasoning() -> None:
    prompt = _format_token_context_handoff_prompt(
        "What is 2 + 2?",
        "Final answer: 4.",
    )

    assert "What is 2 + 2?" in prompt
    assert "Transferred token context from Agent A" in prompt
    assert "Final answer: 4." in prompt


def test_verified_token_context_handoff_prompt_prioritizes_verified_answer() -> None:
    prompt = _format_verified_token_context_handoff_prompt(
        "4",
        "Reasoning was noisy. Final answer: 5.",
    )

    assert "Verified upstream final answer" in prompt
    assert "4" in prompt
    assert "Transferred token context from Agent A" in prompt
    assert "Final answer: 5." in prompt
    assert "authoritative" in prompt
    assert prompt.rstrip().endswith("Final answer:")


def test_eval_manifest_is_locked_by_digest(tmp_path) -> None:
    sample_fingerprints = [
        {
            "sample_index": 9,
            "prompt_sha256": "prompt-a",
            "target_sha256": "target-a",
            "prompt_char_count": 100,
            "target_char_count": 12,
        },
        {
            "sample_index": 11,
            "prompt_sha256": "prompt-b",
            "target_sha256": "target-b",
            "prompt_char_count": 90,
            "target_char_count": 8,
        },
    ]
    manifest = _build_eval_manifest(
        suite_name="standard",
        dataset_name="gsm8k",
        dataset_split="validation",
        limit=3,
        sample_indices=[9, 11, 19],
        methods=("sender_answer_text_handoff", "generated_context_latent_handoff"),
        agent_a_model="agent-a",
        agent_b_model="agent-b",
        seed=7,
        semantic_smoke=False,
        mvp_smoke=False,
        hetero_smoke=True,
        sample_fingerprints=sample_fingerprints,
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = _load_eval_manifest(path)

    assert loaded["manifest_digest"] == manifest["manifest_digest"]
    assert loaded["sample_indices"] == [9, 11, 19]
    assert loaded["sample_fingerprints"] == sample_fingerprints
    assert loaded["sample_content_digest"]

    tampered = dict(manifest)
    tampered["sample_indices"] = [9, 11, 18]
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        _load_eval_manifest(path)


def test_eval_manifest_locks_generated_adapter_identity(tmp_path) -> None:
    generated_identity = {
        "enabled": True,
        "train_on_missing": False,
        "train_limit": 8,
        "dataset_name": "long_context_handoff",
        "train_split": "test",
        "source_mode": "final_answer_tail",
        "source_tail_tokens": 12,
        "input_space": "raw",
        "target_mode": "final_answer_line",
        "target_alignment": "linear",
        "local_residual": {
            "enabled": True,
            "top_k": 8,
            "temperature": 0.05,
            "blend": 1.0,
            "max_memory_rows": 4096,
        },
    }
    handoff_identity = {
        "latent_pooling": "last_token",
        "latent_prefix_mode": "sequence",
        "receiver_context_mode": "prompt_prefix",
        "receiver_context_latent_position": "after_context",
        "embedding_manifold": {"enabled": True, "top_k": 4, "blend": 1.0},
    }
    manifest = _build_eval_manifest(
        suite_name="standard",
        dataset_name="long_context_handoff",
        dataset_split="test",
        limit=3,
        sample_indices=[0, 1, 2],
        methods=("token_context_handoff", "generated_latent_handoff"),
        agent_a_model="agent-a",
        agent_b_model="agent-b",
        seed=0,
        semantic_smoke=False,
        mvp_smoke=False,
        hetero_smoke=True,
        max_new_tokens=32,
        reasoner_max_new_tokens=64,
        torch_dtype="float32",
        device_map="mps",
        generated_trajectory_adapter_identity=generated_identity,
        handoff_identity=handoff_identity,
        sample_fingerprints=[],
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = _load_eval_manifest(path)

    assert loaded["manifest_schema_version"] == 2
    assert loaded["generated_trajectory_adapter"] == generated_identity
    assert loaded["handoff"] == handoff_identity

    tampered = dict(manifest)
    tampered["generated_trajectory_adapter"] = dict(generated_identity, train_limit=32)
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        _load_eval_manifest(path)


def test_apply_eval_manifest_restores_generated_adapter_identity() -> None:
    manifest = _build_eval_manifest(
        suite_name="standard",
        dataset_name="long_context_handoff",
        dataset_split="test",
        limit=3,
        sample_indices=[0, 1, 2],
        methods=("generated_latent_handoff",),
        agent_a_model="agent-a",
        agent_b_model="agent-b",
        seed=0,
        semantic_smoke=False,
        mvp_smoke=False,
        hetero_smoke=True,
        generated_trajectory_adapter_identity={
            "enabled": True,
            "train_on_missing": False,
            "train_limit": 8,
            "train_split": "test",
            "source_mode": "final_answer_tail",
            "source_tail_tokens": 12,
            "input_space": "raw",
            "target_mode": "final_answer_line",
            "target_alignment": "linear",
            "local_residual": {
                "enabled": True,
                "top_k": 8,
                "temperature": 0.05,
                "blend": 1.0,
                "max_memory_rows": 4096,
            },
        },
        handoff_identity={
            "latent_pooling": "last_token",
            "receiver_context_mode": "prompt_prefix",
            "receiver_context_latent_position": "after_context",
            "embedding_manifold": {"enabled": True, "top_k": 4, "blend": 1.0},
        },
        sample_fingerprints=[],
    )
    args = Namespace(
        enable_generated_trajectory_adapter=False,
        disable_generated_trajectory_adapter=False,
        generated_trajectory_adapter_train_on_missing=False,
        generated_trajectory_adapter_no_train_on_missing=False,
        generated_trajectory_adapter_train_limit=None,
        generated_trajectory_adapter_train_split=None,
        generated_trajectory_adapter_input_space=None,
        generated_trajectory_adapter_source_mode=None,
        generated_trajectory_adapter_source_tail_tokens=None,
        generated_trajectory_adapter_target_mode=None,
        generated_trajectory_adapter_target_alignment=None,
        enable_generated_trajectory_local_residual=False,
        disable_generated_trajectory_local_residual=False,
        generated_trajectory_local_residual_top_k=None,
        generated_trajectory_local_residual_temperature=None,
        generated_trajectory_local_residual_blend=None,
        generated_trajectory_local_residual_max_memory_rows=None,
        latent_pooling=None,
        receiver_context_mode=None,
        receiver_context_latent_position=None,
        enable_embedding_manifold=False,
        disable_embedding_manifold=False,
        embedding_manifold_top_k=None,
        embedding_manifold_blend=None,
    )

    _apply_eval_manifest_to_args(args, manifest)

    assert args.enable_generated_trajectory_adapter is True
    assert args.generated_trajectory_adapter_no_train_on_missing is True
    assert args.generated_trajectory_adapter_train_limit == 8
    assert args.generated_trajectory_adapter_train_split == "test"
    assert args.generated_trajectory_adapter_input_space == "raw"
    assert args.generated_trajectory_adapter_source_mode == "final_answer_tail"
    assert args.generated_trajectory_adapter_source_tail_tokens == 12
    assert args.generated_trajectory_adapter_target_mode == "final_answer_line"
    assert args.generated_trajectory_adapter_target_alignment == "linear"
    assert args.enable_generated_trajectory_local_residual is True
    assert args.generated_trajectory_local_residual_top_k == 8
    assert args.generated_trajectory_local_residual_temperature == 0.05
    assert args.generated_trajectory_local_residual_max_memory_rows == 4096
    assert args.receiver_context_mode == "prompt_prefix"
    assert args.enable_embedding_manifold is True
    assert args.embedding_manifold_top_k == 4


def test_sample_fingerprints_lock_prompt_and_target_content() -> None:
    rows = [
        {"question": "How many?", "answer": "#### 3"},
        {"problem": "Find x", "solution": "\\boxed{4}"},
    ]

    fingerprints = _sample_fingerprints(rows, limit=2, sample_indices=[5, 7])

    assert [row["sample_index"] for row in fingerprints] == [5, 7]
    assert fingerprints[0]["prompt_sha256"] != fingerprints[1]["prompt_sha256"]
    assert fingerprints[0]["target_sha256"] != fingerprints[1]["target_sha256"]
    assert fingerprints[0]["prompt_char_count"] == len("How many?")


def test_eval_manifest_sample_lock_flags_changed_content() -> None:
    locked_manifest = _build_eval_manifest(
        suite_name="standard",
        dataset_name="gsm8k",
        dataset_split="validation",
        limit=1,
        sample_indices=[0],
        methods=("generated_context_latent_handoff",),
        agent_a_model="agent-a",
        agent_b_model="agent-b",
        seed=0,
        semantic_smoke=False,
        mvp_smoke=False,
        hetero_smoke=True,
        sample_fingerprints=[
            {
                "sample_index": 0,
                "prompt_sha256": "old-prompt",
                "target_sha256": "old-target",
                "prompt_char_count": 10,
                "target_char_count": 2,
            }
        ],
    )
    resolved_manifest = _build_eval_manifest(
        suite_name="standard",
        dataset_name="gsm8k",
        dataset_split="validation",
        limit=1,
        sample_indices=[0],
        methods=("generated_context_latent_handoff",),
        agent_a_model="agent-a",
        agent_b_model="agent-b",
        seed=0,
        semantic_smoke=False,
        mvp_smoke=False,
        hetero_smoke=True,
        sample_fingerprints=[
            {
                "sample_index": 0,
                "prompt_sha256": "new-prompt",
                "target_sha256": "old-target",
                "prompt_char_count": 10,
                "target_char_count": 2,
            }
        ],
    )

    with pytest.raises(ValueError, match="sample content digest mismatch"):
        _validate_eval_manifest_sample_lock(resolved_manifest, locked_manifest)


def test_transfer_comparison_report_tracks_accuracy_latency_and_retention() -> None:
    summary_rows = [
        {
            "method": "verified_token_context_handoff",
            "sample_count": 10,
            "accuracy_percentage": 80.0,
            "average_latency_seconds": 2.0,
            "answer_perplexity": 1.5,
            "mean_receiver_input_token_count": 1000.0,
        },
        {
            "method": "generated_context_latent_handoff",
            "sample_count": 10,
            "accuracy_percentage": 72.0,
            "average_latency_seconds": 3.0,
            "answer_perplexity": 1.8,
            "mean_receiver_input_token_count": 250.0,
            "cache_transfer_rate_percentage": None,
            "handoff_ok_rate_percentage": 100.0,
            "non_empty_decoded_rate_percentage": 100.0,
        },
    ]

    report = build_transfer_comparison_report(
        summary_rows,
        baseline_methods=("verified_token_context_handoff",),
        latent_methods=("generated_context_latent_handoff",),
        primary_baseline_method="verified_token_context_handoff",
        min_accuracy_retention_ratio=0.9,
        max_latency_ratio=2.0,
    )

    assert report["passed"] is True
    [comparison] = report["comparisons"]
    assert comparison["accuracy_delta_percentage"] == -8.0
    assert comparison["accuracy_retention_ratio"] == 0.9
    assert comparison["latency_ratio"] == 1.5
    assert comparison["answer_perplexity_delta"] == pytest.approx(0.3)
    assert comparison["receiver_input_token_ratio"] == 0.25
    assert comparison["receiver_input_token_savings_percentage"] == 75.0


def test_transfer_comparison_report_flags_retention_failures() -> None:
    report = build_transfer_comparison_report(
        [
            {"method": "token_context_handoff", "accuracy_percentage": 90.0},
            {"method": "generated_context_latent_handoff", "accuracy_percentage": 45.0},
        ],
        baseline_methods=("token_context_handoff",),
        latent_methods=("generated_context_latent_handoff",),
        min_accuracy_retention_ratio=0.9,
    )

    assert report["passed"] is False
    assert any("retained" in item for item in report["missing_requirements"])


def test_heterogeneous_transfer_report_requires_adapter_and_context_for_incompatible_pairs() -> None:
    report = build_heterogeneous_transfer_report(
        [
            {
                "method": "generated_context_latent_handoff",
                "kv_cache_status": "not_provided",
                "handoff_adapter_status": "generated_trajectory_loaded_raw",
                "handoff_adapter_applied": True,
                "receiver_context_status": "used_prompt_prefix",
            },
        ],
        latent_methods=("generated_context_latent_handoff",),
        generated_methods=("generated_context_latent_handoff",),
        context_generated_methods=("generated_context_latent_handoff",),
        model_pair_compatibility={"kv_cache_compatible": False},
    )

    assert report["passed"] is True
    assert report["generated_adapter_applied_row_count"] == 1
    assert report["receiver_context_used_row_count"] == 1


def test_heterogeneous_transfer_report_flags_direct_kv_and_missing_adapter() -> None:
    report = build_heterogeneous_transfer_report(
        [
            {
                "method": "generated_context_latent_handoff",
                "kv_cache_status": "unsupported_architecture_mismatch",
                "handoff_adapter_status": "generated_trajectory_missing_raw",
                "handoff_adapter_applied": False,
                "receiver_context_status": "not_used",
            },
        ],
        latent_methods=("generated_context_latent_handoff",),
        generated_methods=("generated_context_latent_handoff",),
        context_generated_methods=("generated_context_latent_handoff",),
        model_pair_compatibility={"kv_cache_compatible": False},
    )

    assert report["passed"] is False
    assert report["direct_cache_attempt_row_count"] == 1
    assert report["missing_generated_adapter_row_count"] == 1
    assert any("direct KV-cache" in item for item in report["missing_requirements"])


def test_gsm8k_answer_matching_accepts_integer_valued_decimals() -> None:
    assert _answers_match("gsm8k", "252.00", "252")
    assert _answers_match("gsm8k", "9,800.0", "9800")
    assert not _answers_match("gsm8k", "252.01", "252")


def test_final_answer_stop_regex_waits_for_numeric_delimiter() -> None:
    assert FINAL_ANSWER_COMPLETE_REGEX.search("Final answer: 9") is None
    assert FINAL_ANSWER_COMPLETE_REGEX.search("Final answer: 9800 ") is not None
    assert FINAL_ANSWER_COMPLETE_REGEX.search("Final answer: 2.") is not None
    assert FINAL_ANSWER_COMPLETE_REGEX.search("Final answer: **252**") is not None
    assert FINAL_ANSWER_COMPLETE_REGEX.search("Final answer: \\boxed{28}") is not None
    assert FINAL_ANSWER_COMPLETE_REGEX.search("Final answer: \\boxed{2") is None


def test_generated_trajectory_adapter_input_space_is_validated() -> None:
    cfg = OmegaConf.create(
        {
            "handoff": {
                "generated_trajectory_adapter": {
                    "input_space": "raw",
                    "target_alignment": "character",
                },
            },
        }
    )

    assert _generated_trajectory_adapter_input_space(cfg) == "raw"
    assert _generated_trajectory_adapter_target_alignment(cfg) == "character"
    cfg.handoff.generated_trajectory_adapter.input_space = "invalid"
    with pytest.raises(ValueError, match="input_space"):
        _generated_trajectory_adapter_input_space(cfg)
    cfg.handoff.generated_trajectory_adapter.input_space = "aligned"
    cfg.handoff.generated_trajectory_adapter.target_alignment = "invalid"
    with pytest.raises(ValueError, match="target_alignment"):
        _generated_trajectory_adapter_target_alignment(cfg)
    cfg.handoff.generated_trajectory_adapter.target_mode = "final_answer_line"
    cfg.handoff.generated_trajectory_adapter.target_alignment = "character"
    with pytest.raises(ValueError, match="generated_text"):
        _generated_trajectory_adapter_target_alignment(cfg)


def test_generated_trajectory_adapter_requires_explicit_train_on_missing() -> None:
    repo_config = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs" / "main.yaml")

    assert repo_config.handoff.adapter.train_on_missing is False
    assert repo_config.handoff.generated_trajectory_adapter.train_on_missing is False
    assert _generated_trajectory_adapter_train_on_missing(OmegaConf.create({})) is False

    cfg = OmegaConf.create(
        {
            "handoff": {
                "generated_trajectory_adapter": {
                    "train_on_missing": True,
                },
            },
        }
    )

    assert _generated_trajectory_adapter_train_on_missing(cfg) is True


def test_generated_trajectory_final_answer_target_uses_latest_marker() -> None:
    cfg = OmegaConf.create(
        {
            "handoff": {
                "generated_trajectory_adapter": {
                    "target_mode": "final_answer_line",
                },
            },
        }
    )
    generated_text = (
        "Initial attempt.\nFinal answer: 175.\n\n"
        "Verification decision:\nFinal answer: 300.\n"
    )

    assert _generated_trajectory_adapter_target_text(cfg, generated_text) == (
        "Final answer: 300"
    )


def test_text_hybrid_reuses_generated_sender_trace_cache(tmp_path) -> None:
    cfg = OmegaConf.create(
        {
            "agent_a_model": "agent-a",
            "agent_b_model": "agent-b",
            "torch_dtype": "bfloat16",
            "max_new_tokens": 64,
            "benchmark": {
                "answer_only_final": True,
                "text_hybrid_reasoning_max_new_tokens": 64,
                "sender_revision": {
                    "enabled": True,
                    "max_new_tokens": 512,
                    "disagreement_verifier_enabled": True,
                    "disagreement_verifier_max_new_tokens": 512,
                },
            },
            "handoff": {
                "generated_trajectory_adapter": {
                    "trace_cache_enabled": True,
                    "trace_cache_dir": str(tmp_path),
                },
            },
        }
    )
    state = {
        "global_reasoning_layer_indices": (1, 2),
        "global_reasoning_layer_weights": (0.4, 0.6),
    }
    prompt = "What is 40 + 2?"
    cache_key = _generated_trajectory_trace_cache_key(
        cfg,
        state,
        prompt,
        include_prompt=False,
    )
    cache_path = _generated_trajectory_adapter_trace_cache_path(cfg, cache_key)
    torch.save(
        {
            "trace_cache_format_version": 2,
            "cache_key": _cache_key_metadata(cache_key),
            "consensus_hidden_states": torch.zeros(1, 1, 2),
            "generated_token_ids": [4, 2],
            "generated_reasoning_text": "Final answer: 42.",
            "generated_reasoning_token_count": 2,
            "generated_reasoning_status": "complete",
            "generated_reasoning_final_answer_marker": True,
            "generated_latent_includes_prompt": False,
            "sender_revision_enabled": True,
            "sender_revision_applied": True,
            "sender_initial_predicted_answer": "41",
            "sender_revision_predicted_answer": "42",
            "sender_revision_decision_applied": True,
            "sender_revision_decision_predicted_answer": "42",
        },
        cache_path,
    )

    metadata = _reasoner_metadata_for_text_hybrid(prompt, cfg, state)

    assert metadata["token_ids"] == [4, 2]
    assert metadata["reasoning_text"] == "Final answer: 42."
    assert metadata["trace_cache_hit"] is True
    assert metadata["trace_cache_path"] == str(cache_path)
    assert metadata["sender_revision_decision_predicted_answer"] == "42"


def test_generated_trajectory_training_rows_cache_key_scopes_source_rows_only() -> None:
    cfg = OmegaConf.create(
        {
            "agent_a_model": "a",
            "agent_b_model": "b",
            "torch_dtype": "bfloat16",
            "max_new_tokens": 64,
            "benchmark": {
                "answer_only_final": True,
                "text_hybrid_reasoning_max_new_tokens": 128,
            },
            "handoff": {
                "generated_trajectory_adapter": {
                    "dataset_name": "gsm8k",
                    "train_split": "train",
                    "train_limit": 8,
                    "input_space": "raw",
                    "source_mode": "generated_text",
                    "source_tail_tokens": 32,
                    "target_mode": "generated_text",
                    "target_alignment": "character",
                    "strategy": "hybrid_affine",
                    "local_residual": {
                        "enabled": False,
                        "top_k": 8,
                        "temperature": 0.05,
                        "blend": 1.0,
                        "max_memory_rows": 4096,
                    },
                }
            },
        }
    )
    state = {"global_alignment_cache_key": ("alignment", 1)}

    first_key = _generated_trajectory_adapter_training_rows_cache_key(
        cfg,
        state,
        include_prompt=False,
    )
    cfg.benchmark.sender_revision = {
        "enabled": True,
        "max_new_tokens": 256,
    }
    assert _generated_trajectory_adapter_training_rows_cache_key(
        cfg,
        state,
        include_prompt=False,
    ) != first_key
    revision_key = _generated_trajectory_adapter_training_rows_cache_key(
        cfg,
        state,
        include_prompt=False,
    )
    cfg.benchmark.sender_revision.disagreement_verifier_enabled = False
    assert _generated_trajectory_adapter_training_rows_cache_key(
        cfg,
        state,
        include_prompt=False,
    ) != revision_key
    cfg.benchmark.sender_revision.enabled = False
    cfg.benchmark.sender_revision.disagreement_verifier_enabled = True

    cfg.handoff.generated_trajectory_adapter.strategy = "ridge"
    cfg.handoff.generated_trajectory_adapter.local_residual.enabled = True
    cfg.handoff.generated_trajectory_adapter.local_residual.top_k = 16
    assert _generated_trajectory_adapter_training_rows_cache_key(
        cfg,
        state,
        include_prompt=False,
    ) == first_key

    cfg.handoff.generated_trajectory_adapter.source_mode = "final_answer_tail"
    assert _generated_trajectory_adapter_training_rows_cache_key(
        cfg,
        state,
        include_prompt=False,
    ) != first_key


def test_load_generated_trajectory_training_rows_validates_disk_cache(tmp_path) -> None:
    cache_path = tmp_path / "rows.pt"
    expected_key = ("rows", "expected")
    torch.save(
        {
            "source_matrix": torch.zeros(2, 3),
            "target_matrix": torch.ones(2, 4),
            "training_prompt_count": 1,
            "training_token_count": 2,
            "training_rows_cache_key_digest": _cache_key_digest(expected_key),
        },
        cache_path,
    )

    loaded = _load_generated_trajectory_training_rows_from_disk(
        cache_path,
        expected_cache_key=expected_key,
    )

    assert loaded is not None
    assert loaded["source_matrix"].shape == (2, 3)

    torch.save(
        {
            "source_matrix": torch.zeros(2, 3),
            "target_matrix": torch.ones(2, 4),
            "training_rows_cache_key_digest": _cache_key_digest(("rows", "other")),
        },
        cache_path,
    )
    assert (
        _load_generated_trajectory_training_rows_from_disk(
            cache_path,
            expected_cache_key=expected_key,
        )
        is None
    )

    torch.save({"source_matrix": torch.zeros(2, 3), "target_matrix": torch.ones(3, 4)}, cache_path)
    assert _load_generated_trajectory_training_rows_from_disk(cache_path) is None


def test_load_generated_trajectory_adapter_rejects_digest_mismatch(tmp_path) -> None:
    cache_path = tmp_path / "adapter.pt"
    expected_key = ("adapter", "expected")
    torch.save(
        {
            "mapping_matrix": torch.eye(2),
            "adapter_cache_key_digest": _cache_key_digest(("adapter", "other")),
        },
        cache_path,
    )

    assert (
        _load_generated_trajectory_adapter_from_disk(
            cache_path,
            expected_cache_key=expected_key,
        )
        is None
    )

    torch.save(
        {
            "mapping_matrix": torch.eye(2),
            "adapter_cache_key_digest": _cache_key_digest(expected_key),
        },
        cache_path,
    )
    assert (
        _load_generated_trajectory_adapter_from_disk(
            cache_path,
            expected_cache_key=expected_key,
        )
        is not None
    )


def test_generated_trajectory_trace_cache_key_tracks_prompt_and_sender_setup() -> None:
    cfg = OmegaConf.create(
        {
            "agent_a_model": "a",
            "agent_b_model": "b",
            "torch_dtype": "bfloat16",
            "benchmark": {
                "answer_only_final": True,
                "text_hybrid_reasoning_max_new_tokens": 128,
            },
            "handoff": {
                "latent_pooling": "mean",
                "generated_trajectory_adapter": {
                    "strategy": "hybrid_affine",
                    "local_residual": {"enabled": False},
                },
            },
        }
    )
    state = {
        "global_reasoning_layer_indices": (1, 2),
        "global_reasoning_layer_weights": (0.25, 0.75),
    }

    first_key = _generated_trajectory_trace_cache_key(
        cfg,
        state,
        "How many?",
        include_prompt=False,
    )
    first_generation_fingerprint = _sender_generation_cache_fingerprint(cfg)
    cfg.benchmark.sender_revision = {
        "enabled": True,
        "max_new_tokens": 256,
    }
    assert _sender_generation_cache_fingerprint(cfg) != first_generation_fingerprint
    assert (
        _generated_trajectory_trace_cache_key(
            cfg,
            state,
            "How many?",
            include_prompt=False,
        )
        != first_key
    )
    cfg.benchmark.sender_revision.enabled = False

    cfg.handoff.generated_trajectory_adapter.strategy = "ridge"
    cfg.handoff.generated_trajectory_adapter.local_residual.enabled = True
    assert (
        _generated_trajectory_trace_cache_key(
            cfg,
            state,
            "How many?",
            include_prompt=False,
        )
        == first_key
    )

    assert (
        _generated_trajectory_trace_cache_key(
            cfg,
            state,
            "How many?",
            include_prompt=True,
        )
        != first_key
    )
    assert (
        _generated_trajectory_trace_cache_key(
            cfg,
            state,
            "A different prompt",
            include_prompt=False,
        )
        != first_key
    )
    state["global_reasoning_layer_weights"] = (0.5, 0.5)
    assert (
        _generated_trajectory_trace_cache_key(
            cfg,
            state,
            "How many?",
            include_prompt=False,
        )
        != first_key
    )


def test_sender_trace_reasoning_metadata_uses_receiver_layer_bound() -> None:
    cfg = OmegaConf.create({"alignment": {"reasoning_layer_weights": [0.2, 0.3, 0.5]}})

    indices, weights = _resolve_sender_trace_reasoning_metadata_from_layer_counts(
        cfg,
        sender_layer_count=30,
        receiver_layer_count=18,
    )

    assert indices == (12, 16)
    assert weights == pytest.approx((0.4, 0.6))


def test_load_generated_trajectory_trace_validates_disk_cache(tmp_path) -> None:
    cache_path = tmp_path / "trace.pt"
    expected_key = ("generated_trajectory_trace_v1", "model", ["weights"])
    torch.save(
        {
            "cache_key": _cache_key_metadata(expected_key),
            "consensus_hidden_states": torch.zeros(1, 2, 3),
            "generated_token_ids": [1, 2],
            "generated_reasoning_text": "Final answer: 4",
        },
        cache_path,
    )

    loaded = _load_generated_trajectory_trace_from_disk(
        cache_path,
        expected_cache_key=expected_key,
    )

    assert loaded is not None
    assert loaded["generated_token_ids"] == [1, 2]
    assert (
        _load_generated_trajectory_trace_from_disk(
            cache_path,
            expected_cache_key=("generated_trajectory_trace_v1", "other", ["weights"]),
        )
        is None
    )

    torch.save(
        {
            "cache_key": _cache_key_metadata(expected_key),
            "consensus_hidden_states": torch.zeros(2, 3),
            "generated_token_ids": [1, 2],
            "generated_reasoning_text": "Final answer: 4",
        },
        cache_path,
    )
    assert _load_generated_trajectory_trace_from_disk(cache_path) is None


def test_generated_adapter_include_prompt_values_follow_selected_methods() -> None:
    assert _generated_adapter_include_prompt_values(None) == (False,)
    assert _generated_adapter_include_prompt_values(["text_text_hybrid"]) == (False,)
    assert _generated_adapter_include_prompt_values(
        [
            "generated_latent_handoff",
            "prompt_generated_latent_handoff",
            "generated_context_latent_handoff",
        ]
    ) == (False, True)


def test_generated_trajectory_residual_memory_keeps_hard_rows_and_coverage() -> None:
    source = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    residual = torch.zeros(10, 2)
    residual[5] = torch.tensor([100.0, 0.0])

    source_memory, residual_memory = _select_generated_adapter_memory_rows(
        source,
        residual,
        max_rows=4,
    )

    assert int(source_memory.shape[0]) == 4
    assert any(torch.allclose(row, source[5]) for row in source_memory)
    assert torch.linalg.vector_norm(residual_memory, dim=-1).max().item() == 100.0


def test_generated_trajectory_local_residual_corrects_nearest_training_error() -> None:
    cfg = OmegaConf.create(
        {
            "handoff": {
                "generated_trajectory_adapter": {
                    "local_residual": {
                        "enabled": True,
                        "top_k": 1,
                        "temperature": 0.05,
                        "blend": 1.0,
                        "max_memory_rows": 8,
                        "chunk_size": 2,
                    }
                }
            }
        }
    )
    source = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    fitted = torch.tensor([[0.5, 0.0], [0.0, 0.5]])
    target = torch.tensor([[1.5, 0.0], [0.0, 1.5]])

    residual_state = _build_generated_adapter_local_residual_state(
        cfg,
        source,
        target,
        fitted,
    )
    adapter_state = {"local_residual_state": residual_state}
    corrected, metrics = _apply_generated_adapter_local_residual(
        source.reshape(1, 2, 2),
        fitted.reshape(1, 2, 2),
        adapter_state,
    )

    assert metrics["generated_adapter_local_residual_applied"] is True
    assert metrics["generated_adapter_local_residual_memory_rows"] == 2
    assert torch.allclose(corrected.reshape(2, 2), target)


def test_raw_latent_methods_are_available_for_standard_suite() -> None:
    standard_methods = [name for name, _ in _methods_for_suite("standard")]

    assert "prompt_local_latent" in standard_methods
    assert "global_anchor_hybrid_affine_plus_calibration" in standard_methods
    assert "hybrid_hl_mas" in standard_methods


def test_hetero_smoke_defaults_to_generated_trajectory_mvp() -> None:
    assert DEFAULT_HETERO_SMOKE_METHODS == (
        "text_text_hybrid",
        "generated_latent_handoff",
    )
    assert DEFAULT_HETERO_SMOKE_LATENT_METHODS == ("generated_latent_handoff",)
    assert DEFAULT_HETERO_SMOKE_REASONER_MAX_NEW_TOKENS == 640
    assert DEFAULT_HETERO_SMOKE_GENERATED_ADAPTER_TRAIN_LIMIT == 32


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
        runtime_metadata={
            "effective_device": "mps",
            "effective_torch_dtype": "float32",
        },
    )

    assert report["passed"] is False
    assert report["final_heldout_exact_match_accuracy"] == 60.0
    assert report["effective_device"] == "mps"
    assert report["effective_torch_dtype"] == "float32"
    assert "Training mode is not 'real'." in report["missing_requirements"]


def test_build_training_smoke_report_passes_structural_smoke_without_accuracy() -> None:
    report = build_training_smoke_report(
        [
            {"epoch": 0.0, "step": 0.0, "loss": 4.0},
            {
                "epoch": 0.0,
                "step": 1.0,
                "heldout_exact_match_accuracy": 0.0,
                "heldout_answer_extraction_rate_percentage": 100.0,
                "heldout_decode_answer_extraction_rate_percentage": 33.333,
                "heldout_candidate_fallback_rate_percentage": 66.667,
                "heldout_latent_sequence_decoder_token_accuracy": 100.0,
                "heldout_latent_sequence_decoder_sequence_accuracy": 100.0,
                "heldout_latent_sequence_decoder_length_accuracy": 100.0,
                "heldout_latent_sequence_decoder_unique_predicted_answer_count": 3.0,
                "heldout_latent_generation_smoke_ready": True,
                "heldout_latent_generation_smoke_skipped_count": 0.0,
                "heldout_latent_generation_sequence_accuracy_threshold": 95.0,
                "heldout_extraction_failure_count": 0.0,
                "heldout_eval_diagnostics": "target=13 | predicted=13 | source=candidate_nll",
                "heldout_answer_perplexity": 250.0,
                "heldout_eval_samples": 3.0,
            },
        ]
    )

    assert report["passed"] is True
    assert report["final_heldout_exact_match_accuracy"] == 0.0
    assert report["final_heldout_answer_extraction_rate_percentage"] == 100.0
    assert report["final_heldout_decode_answer_extraction_rate_percentage"] == 33.333
    assert report["final_heldout_candidate_fallback_rate_percentage"] == 66.667
    assert report["final_heldout_latent_sequence_decoder_sequence_accuracy"] == 100.0
    assert report["latent_sequence_decoder_ready"] is True
    assert report["final_heldout_extraction_failure_count"] == 0.0
    assert "source=candidate_nll" in report["heldout_eval_diagnostics"]


def test_build_training_smoke_report_flags_nonfinite_loss() -> None:
    report = build_training_smoke_report(
        [
            {"epoch": 0.0, "step": 0.0, "loss": float("nan")},
            {
                "epoch": 0.0,
                "step": 1.0,
                "heldout_answer_extraction_rate_percentage": 100.0,
                "heldout_answer_perplexity": 1.0,
                "heldout_eval_samples": 1.0,
            },
        ]
    )

    assert report["passed"] is False
    assert any("non-finite" in item for item in report["missing_requirements"])


def test_build_training_smoke_report_flags_low_extraction_rate() -> None:
    report = build_training_smoke_report(
        [
            {"epoch": 0.0, "step": 0.0, "loss": 4.0},
            {
                "epoch": 0.0,
                "step": 1.0,
                "heldout_answer_extraction_rate_percentage": 33.333,
                "heldout_answer_perplexity": 1.0,
                "heldout_eval_samples": 3.0,
            },
        ]
    )

    assert report["passed"] is False
    assert any("Answer extraction rate" in item for item in report["missing_requirements"])


def test_build_training_smoke_report_flags_degenerate_predictions() -> None:
    report = build_training_smoke_report(
        [
            {"epoch": 0.0, "step": 0.0, "loss": 4.0},
            {
                "epoch": 0.0,
                "step": 1.0,
                "heldout_exact_match_accuracy": 0.0,
                "heldout_answer_extraction_rate_percentage": 100.0,
                "heldout_unique_predicted_answer_count": 1.0,
                "heldout_answer_perplexity": 1.0,
                "heldout_eval_samples": 3.0,
            },
        ]
    )

    assert report["passed"] is False
    assert report["final_heldout_degenerate_prediction"] is True
    assert any("degenerate" in item for item in report["missing_requirements"])


def test_build_training_smoke_report_detects_degenerate_predictions_from_diagnostics() -> None:
    report = build_training_smoke_report(
        [
            {"epoch": 0.0, "step": 0.0, "loss": 4.0},
            {
                "epoch": 0.0,
                "step": 1.0,
                "heldout_exact_match_accuracy": 0.0,
                "heldout_answer_extraction_rate_percentage": 100.0,
                "heldout_answer_perplexity": 1.0,
                "heldout_eval_samples": 3.0,
                "heldout_eval_diagnostics": "\n".join(
                    [
                        "target=13 | predicted=100000000000000 | source=decode",
                        "target=42 | predicted=100000000000000 | source=decode",
                        "target=3x^2 | predicted=100000000000000 | source=decode",
                    ]
                ),
            },
        ]
    )

    assert report["passed"] is False
    assert report["final_heldout_unique_predicted_answer_count"] == 1
    assert report["final_heldout_degenerate_prediction"] is True


def test_build_training_smoke_report_flags_degenerate_actor_text_baseline() -> None:
    report = build_training_smoke_report(
        [
            {"epoch": 0.0, "step": 0.0, "loss": 4.0},
            {
                "epoch": 0.0,
                "step": 1.0,
                "heldout_exact_match_accuracy": 100.0,
                "heldout_answer_extraction_rate_percentage": 100.0,
                "heldout_unique_predicted_answer_count": 3.0,
                "heldout_actor_text_baseline_accuracy": 0.0,
                "heldout_actor_text_baseline_unique_predicted_answer_count": 1.0,
                "heldout_answer_perplexity": 1.0,
                "heldout_eval_samples": 3.0,
            },
        ]
    )

    assert report["passed"] is False
    assert report["latent_training_ready"] is False
    assert report["final_heldout_actor_text_baseline_degenerate_prediction"] is True
    assert any("Actor text baseline is degenerate" in item for item in report["missing_requirements"])


def test_build_training_smoke_report_gates_enabled_token_decoder() -> None:
    report = build_training_smoke_report(
        [
            {"epoch": 0.0, "step": 0.0, "loss": 4.0},
            {
                "epoch": 0.0,
                "step": 1.0,
                "heldout_exact_match_accuracy": 66.667,
                "heldout_answer_extraction_rate_percentage": 100.0,
                "heldout_unique_predicted_answer_count": 2.0,
                "heldout_latent_token_decode_enabled": True,
                "heldout_latent_token_decode_require_ready": True,
                "heldout_latent_token_decode_accuracy": 66.667,
                "heldout_latent_token_decode_answer_extraction_rate_percentage": 100.0,
                "heldout_latent_token_decode_unique_predicted_answer_count": 2.0,
                "heldout_answer_perplexity": 1.0,
                "heldout_eval_samples": 3.0,
            },
        ]
    )

    assert report["passed"] is False
    assert report["latent_token_decoder_ready"] is False
    assert any("Latent token decoder" in item for item in report["missing_requirements"])


def test_build_training_smoke_report_gates_enabled_actor_bridge_decoder() -> None:
    report = build_training_smoke_report(
        [
            {"epoch": 0.0, "step": 0.0, "loss": 4.0},
            {
                "epoch": 0.0,
                "step": 1.0,
                "heldout_exact_match_accuracy": 66.667,
                "heldout_answer_extraction_rate_percentage": 100.0,
                "heldout_unique_predicted_answer_count": 2.0,
                "heldout_actor_semantic_bridge_decode_enabled": True,
                "heldout_actor_semantic_bridge_decode_accuracy": 66.667,
                "heldout_actor_semantic_bridge_decode_answer_extraction_rate_percentage": 100.0,
                "heldout_actor_semantic_bridge_decode_unique_predicted_answer_count": 2.0,
                "heldout_answer_perplexity": 1.0,
                "heldout_eval_samples": 3.0,
            },
        ]
    )

    assert report["passed"] is False
    assert report["actor_semantic_bridge_decoder_ready"] is False
    assert any("Actor semantic bridge decode" in item for item in report["missing_requirements"])


def test_build_training_smoke_report_gates_required_raw_actor_decode() -> None:
    report = build_training_smoke_report(
        [
            {"epoch": 0.0, "step": 0.0, "loss": 4.0},
            {
                "epoch": 0.0,
                "step": 1.0,
                "heldout_exact_match_accuracy": 66.667,
                "heldout_answer_extraction_rate_percentage": 100.0,
                "heldout_unique_predicted_answer_count": 2.0,
                "heldout_raw_decode_require_ready": True,
                "heldout_raw_decode_exact_match_accuracy": 66.667,
                "heldout_raw_decode_answer_extraction_rate_percentage": 100.0,
                "heldout_raw_decode_unique_predicted_answer_count": 2.0,
                "heldout_answer_perplexity": 1.0,
                "heldout_eval_samples": 3.0,
            },
        ]
    )

    assert report["passed"] is False
    assert report["raw_actor_free_decoder_ready"] is False
    assert any("Raw actor free decode" in item for item in report["missing_requirements"])


def test_methods_for_suite_exposes_phase1_homogeneous_entrypoint() -> None:
    phase1_methods = [name for name, _ in _methods_for_suite("phase1_homogeneous")]
    standard_methods = [name for name, _ in _methods_for_suite("standard")]

    assert phase1_methods == [
        "pure_text_cot",
        "text_text_hybrid",
        "token_context_handoff",
        "verified_token_context_handoff",
        "sender_answer_text_handoff",
        "homogeneous_ridge_latent",
        "homogeneous_orthogonal_latent",
    ]
    assert "text_text_hybrid" in standard_methods
    assert "token_context_handoff" in standard_methods
    assert "verified_token_context_handoff" in standard_methods
    assert "sender_answer_text_handoff" in standard_methods
    assert "global_anchor_orthogonal" in standard_methods
    assert "global_anchor_ridge" in standard_methods
    assert "global_anchor_hybrid_affine" in standard_methods
    assert "global_anchor_hybrid_affine_plus_calibration" in standard_methods
    assert "prompt_local_latent" in standard_methods
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
