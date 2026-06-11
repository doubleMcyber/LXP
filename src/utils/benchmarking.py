from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

REPORT_SCHEMA_VERSION = 22

STANDARD_SAMPLE_FIELDS: list[str] = [
    "report_schema_version",
    "evaluation_surface",
    "suite",
    "method",
    "dataset",
    "dataset_split",
    "repetition",
    "seed",
    "sample_index",
    "agent_a_model",
    "agent_b_model",
    "model_pair",
    "model_pair_kv_cache_compatible",
    "model_pair_compatibility_status",
    "model_pair_compatibility_reason",
    "torch_dtype",
    "compression_steps",
    "semantic_anchor_count",
    "reasoning_layer_weights",
    "alignment_mode",
    "alignment_strategy",
    "handoff_status",
    "handoff_surface",
    "kv_cache_transferred",
    "kv_cache_status",
    "kv_cache_reason",
    "active_kv_cache_transferred",
    "active_kv_cache_status",
    "active_kv_cache_reason",
    "active_kv_cache_source",
    "receiver_context_status",
    "receiver_context_reason",
    "receiver_context_token_count",
    "receiver_context_latent_position",
    "receiver_input_token_count",
    "decode_status",
    "prompt",
    "target_answer",
    "sender_reasoning_text",
    "sender_reasoning_token_count",
    "sender_reasoning_status",
    "sender_trace_cache_hit",
    "sender_trace_cache_path",
    "sender_revision_enabled",
    "sender_revision_applied",
    "sender_initial_predicted_answer",
    "sender_revision_predicted_answer",
    "sender_revision_decision_applied",
    "sender_revision_decision_predicted_answer",
    "sender_final_answer_marker",
    "sender_predicted_answer",
    "sender_answer_matches_target",
    "predicted_answer",
    "decoded_text",
    "generated_tokens",
    "answer_token_count",
    "answer_nll",
    "answer_perplexity",
    "correct",
    "latency_seconds",
    "pre_alignment_l2_distance",
    "pre_alignment_cosine_distance",
    "post_alignment_l2_distance",
    "post_alignment_cosine_distance",
    "alignment_residual_norm_ratio",
    "alignment_bias_norm",
    "prompt_calibration_enabled",
    "prompt_calibration_bias_norm",
    "handoff_adapter_enabled",
    "handoff_adapter_status",
    "handoff_adapter_applied",
    "handoff_adapter_delta_norm",
    "handoff_adapter_cache_hit",
    "handoff_adapter_cache_path",
    "handoff_adapter_cache_key_digest",
    "handoff_adapter_training_prompt_count",
    "handoff_adapter_training_token_count",
    "handoff_adapter_training_row_cache_hit",
    "handoff_adapter_training_row_cache_path",
    "handoff_adapter_training_rows_cache_key_digest",
    "handoff_adapter_training_trace_cache_hit_count",
    "handoff_adapter_training_trace_cache_miss_count",
    "handoff_adapter_training_trace_cache_hit_rate_percentage",
    "handoff_adapter_training_reconstruction_mse",
    "handoff_adapter_training_mean_cosine_similarity",
    "generated_adapter_local_residual_applied",
    "generated_adapter_local_residual_delta_norm",
    "generated_adapter_local_residual_mean_top_similarity",
    "generated_adapter_local_residual_memory_rows",
    "generated_adapter_semantic_memory_applied",
    "generated_adapter_semantic_memory_similarity",
    "generated_adapter_semantic_memory_entry_count",
    "generated_adapter_semantic_memory_target_text",
    "embedding_manifold_enabled",
    "embedding_manifold_applied",
    "embedding_manifold_delta_norm",
    "embedding_manifold_mean_top_similarity",
    "embedding_manifold_unique_token_count",
    "raw_handoff_entropy",
    "handoff_uncertainty",
    "confidence_gate_triggered",
    "fallback_discrete_reasoning_steps",
    "latent_trajectory_steps",
    "total_reasoning_steps",
    "continuous_integration_seconds",
    "global_alignment_cache_hit",
    "error",
]
STANDARD_SUMMARY_FIELDS: list[str] = [
    "report_schema_version",
    "evaluation_surface",
    "suite",
    "method",
    "dataset",
    "dataset_split",
    "agent_a_model",
    "agent_b_model",
    "model_pair",
    "model_pair_kv_cache_compatible",
    "model_pair_compatibility_status",
    "model_pair_compatibility_reason",
    "torch_dtype",
    "seed",
    "compression_steps",
    "semantic_anchor_count",
    "reasoning_layer_weights",
    "alignment_mode",
    "alignment_strategy",
    "handoff_status",
    "handoff_surface",
    "kv_cache_status",
    "kv_cache_reason",
    "active_kv_cache_status",
    "active_kv_cache_reason",
    "active_kv_cache_source",
    "receiver_context_status",
    "receiver_context_reason",
    "receiver_context_latent_position",
    "decode_status",
    "repetition_count",
    "sample_count",
    "accuracy_percentage",
    "sender_answer_extraction_rate_percentage",
    "sender_final_answer_marker_rate_percentage",
    "sender_trace_cache_hit_rate_percentage",
    "sender_revision_applied_rate_percentage",
    "sender_revision_decision_applied_rate_percentage",
    "sender_accuracy_percentage",
    "sender_correct_sample_count",
    "accuracy_when_sender_correct_percentage",
    "total_latency_seconds",
    "average_latency_seconds",
    "total_generated_tokens",
    "total_receiver_input_tokens",
    "mean_receiver_input_token_count",
    "max_receiver_input_token_count",
    "tokens_per_second",
    "total_answer_tokens",
    "mean_answer_nll",
    "answer_perplexity",
    "mean_pre_alignment_l2_distance",
    "mean_pre_alignment_cosine_distance",
    "mean_post_alignment_l2_distance",
    "mean_post_alignment_cosine_distance",
    "mean_alignment_residual_norm_ratio",
    "mean_alignment_bias_norm",
    "prompt_calibration_rate_percentage",
    "mean_prompt_calibration_bias_norm",
    "handoff_adapter_rate_percentage",
    "handoff_adapter_cache_hit_rate_percentage",
    "handoff_adapter_training_row_cache_hit_rate_percentage",
    "mean_handoff_adapter_training_trace_cache_hit_count",
    "mean_handoff_adapter_training_trace_cache_miss_count",
    "mean_handoff_adapter_training_trace_cache_hit_rate_percentage",
    "mean_handoff_adapter_delta_norm",
    "mean_handoff_adapter_training_reconstruction_mse",
    "mean_handoff_adapter_training_mean_cosine_similarity",
    "generated_adapter_local_residual_rate_percentage",
    "mean_generated_adapter_local_residual_delta_norm",
    "mean_generated_adapter_local_residual_top_similarity",
    "mean_generated_adapter_local_residual_memory_rows",
    "generated_adapter_semantic_memory_rate_percentage",
    "mean_generated_adapter_semantic_memory_similarity",
    "mean_generated_adapter_semantic_memory_entry_count",
    "embedding_manifold_rate_percentage",
    "mean_embedding_manifold_delta_norm",
    "mean_embedding_manifold_top_similarity",
    "mean_embedding_manifold_unique_token_count",
    "mean_raw_handoff_entropy",
    "mean_handoff_uncertainty",
    "confidence_gate_trigger_rate_percentage",
    "cache_transfer_rate_percentage",
    "global_alignment_cache_hit_rate_percentage",
    "explicit_status_rate_percentage",
    "handoff_ok_rate_percentage",
    "empty_decode_rate_percentage",
    "non_empty_decoded_rate_percentage",
    "failure_rate_percentage",
    "error_count",
]


def _cfg_value(cfg: Any, path: str, default: Any = None) -> Any:
    current = cfg
    for part in path.split("."):
        if current is None:
            return default
        current = getattr(current, part, None)
    return default if current is None else current


def serialize_reasoning_layer_weights(weights: Any) -> str:
    if weights is None:
        return ""
    if isinstance(weights, str):
        return weights
    return ",".join(f"{float(weight):.6f}" for weight in weights)


def build_standard_row_base(
    cfg: Any,
    *,
    evaluation_surface: str,
    suite: str,
    method: str,
    dataset: str,
    dataset_split: str,
    repetition: int,
    compression_steps: Optional[int] = None,
    alignment_mode: Optional[str] = None,
    alignment_strategy: Optional[str] = None,
    semantic_anchor_count: Optional[int] = None,
    reasoning_layer_weights: Optional[Sequence[float] | str] = None,
    seed: Optional[int] = None,
    model_pair_kv_cache_compatible: Optional[bool] = None,
    model_pair_compatibility_status: str = "",
    model_pair_compatibility_reason: str = "",
) -> dict[str, Any]:
    if compression_steps is None:
        compression_steps = int(
            _cfg_value(cfg, "training.compressed_steps", _cfg_value(cfg, "latent_steps", 0))
        )
    if semantic_anchor_count is None:
        semantic_anchor_count = int(_cfg_value(cfg, "alignment.semantic_anchor_count", 0))
    if reasoning_layer_weights is None:
        reasoning_layer_weights = _cfg_value(cfg, "alignment.reasoning_layer_weights", ())
    if seed is None:
        seed = int(_cfg_value(cfg, "seed", 0))

    agent_a_model = str(_cfg_value(cfg, "agent_a_model", ""))
    agent_b_model = str(_cfg_value(cfg, "agent_b_model", ""))
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "evaluation_surface": evaluation_surface,
        "suite": suite,
        "method": method,
        "dataset": dataset,
        "dataset_split": dataset_split,
        "repetition": int(repetition),
        "seed": int(seed),
        "agent_a_model": agent_a_model,
        "agent_b_model": agent_b_model,
        "model_pair": f"{agent_a_model} -> {agent_b_model}",
        "model_pair_kv_cache_compatible": model_pair_kv_cache_compatible,
        "model_pair_compatibility_status": str(model_pair_compatibility_status),
        "model_pair_compatibility_reason": str(model_pair_compatibility_reason),
        "torch_dtype": str(_cfg_value(cfg, "torch_dtype", "")),
        "compression_steps": int(compression_steps),
        "semantic_anchor_count": int(semantic_anchor_count),
        "reasoning_layer_weights": serialize_reasoning_layer_weights(reasoning_layer_weights),
        "alignment_mode": "" if alignment_mode is None else str(alignment_mode),
        "alignment_strategy": "" if alignment_strategy is None else str(alignment_strategy),
    }


def _mean_or_none(rows: Sequence[dict[str, Any]], field: str) -> Optional[float]:
    values = [
        float(row[field])
        for row in rows
        if row.get(field) is not None and row.get(field) != ""
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _unique_join(rows: Sequence[dict[str, Any]], field: str) -> str:
    values = sorted(
        {
            str(row.get(field))
            for row in rows
            if row.get(field) is not None and row.get(field) != ""
        }
    )
    return ",".join(values)


def aggregate_standard_rows(
    rows: Sequence[dict[str, Any]],
    *,
    group_fields: Optional[Sequence[str]] = None,
) -> list[dict[str, Any]]:
    if group_fields is None:
        group_fields = (
            "report_schema_version",
            "evaluation_surface",
            "suite",
            "method",
            "dataset",
            "dataset_split",
            "agent_a_model",
            "agent_b_model",
            "model_pair",
            "model_pair_kv_cache_compatible",
            "model_pair_compatibility_status",
            "model_pair_compatibility_reason",
            "torch_dtype",
            "seed",
            "compression_steps",
            "semantic_anchor_count",
            "reasoning_layer_weights",
            "alignment_mode",
            "alignment_strategy",
        )

    grouped_rows: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped_rows.setdefault(tuple(row.get(field) for field in group_fields), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, group_rows in grouped_rows.items():
        row_template = {field: value for field, value in zip(group_fields, key)}
        sample_count = len(group_rows)
        correct_values = [
            value for value in (_row_correct_value(row) for row in group_rows) if value is not None
        ]
        correct_count = sum(correct_values)
        error_count = sum(1 for row in group_rows if bool(row.get("error")))
        sender_reasoning_rows = [
            row for row in group_rows if str(row.get("sender_reasoning_text") or "").strip()
        ]
        sender_answer_rows = [
            row for row in sender_reasoning_rows if str(row.get("sender_predicted_answer") or "").strip()
        ]
        sender_final_answer_marker_rows = [
            row
            for row in sender_reasoning_rows
            if _optional_bool_value(row.get("sender_final_answer_marker")) is True
            or str(row.get("sender_reasoning_status") or "") == "complete"
        ]
        sender_trace_cache_rows = [
            _optional_bool_value(row.get("sender_trace_cache_hit"))
            for row in group_rows
            if row.get("sender_trace_cache_hit") is not None
            and row.get("sender_trace_cache_hit") != ""
        ]
        sender_trace_cache_values = [
            value for value in sender_trace_cache_rows if value is not None
        ]
        sender_revision_rows = [
            _optional_bool_value(row.get("sender_revision_applied"))
            for row in group_rows
            if row.get("sender_revision_applied") is not None
            and row.get("sender_revision_applied") != ""
        ]
        sender_revision_values = [
            value for value in sender_revision_rows if value is not None
        ]
        sender_revision_decision_rows = [
            _optional_bool_value(row.get("sender_revision_decision_applied"))
            for row in group_rows
            if row.get("sender_revision_decision_applied") is not None
            and row.get("sender_revision_decision_applied") != ""
        ]
        sender_revision_decision_values = [
            value for value in sender_revision_decision_rows if value is not None
        ]
        sender_correct_values = [
            value
            for value in (
                _optional_bool_value(row.get("sender_answer_matches_target"))
                for row in sender_reasoning_rows
            )
            if value is not None
        ]
        sender_correct_rows = [
            row
            for row in group_rows
            if _optional_bool_value(row.get("sender_answer_matches_target")) is True
        ]
        sender_correct_row_values = [
            value for value in (_row_correct_value(row) for row in sender_correct_rows) if value is not None
        ]
        total_latency_seconds = sum(float(row.get("latency_seconds", 0.0)) for row in group_rows)
        total_generated_tokens = sum(int(row.get("generated_tokens", 0) or 0) for row in group_rows)
        receiver_input_token_counts = [
            int(row.get("receiver_input_token_count", 0) or 0)
            for row in group_rows
            if row.get("receiver_input_token_count") not in (None, "")
        ]
        total_receiver_input_tokens = sum(receiver_input_token_counts)
        total_answer_tokens = sum(int(row.get("answer_token_count", 0) or 0) for row in group_rows)
        total_answer_nll = sum(
            float(row["answer_nll"]) * int(row.get("answer_token_count", 0) or 0)
            for row in group_rows
            if row.get("answer_nll") is not None and row.get("answer_nll") != ""
        )
        kv_transfer_rows = [
            bool(row["kv_cache_transferred"])
            for row in group_rows
            if row.get("kv_cache_transferred") is not None and row.get("kv_cache_transferred") != ""
        ]
        global_alignment_cache_rows = [
            bool(row["global_alignment_cache_hit"])
            for row in group_rows
            if row.get("global_alignment_cache_hit") is not None and row.get("global_alignment_cache_hit") != ""
        ]
        confidence_gate_rows = [
            bool(row["confidence_gate_triggered"])
            for row in group_rows
            if row.get("confidence_gate_triggered") is not None and row.get("confidence_gate_triggered") != ""
        ]
        prompt_calibration_rows = [
            bool(row["prompt_calibration_enabled"])
            for row in group_rows
            if row.get("prompt_calibration_enabled") is not None and row.get("prompt_calibration_enabled") != ""
        ]
        handoff_adapter_rows = [
            bool(row["handoff_adapter_applied"])
            for row in group_rows
            if row.get("handoff_adapter_applied") is not None and row.get("handoff_adapter_applied") != ""
        ]
        handoff_adapter_cache_rows = [
            bool(row["handoff_adapter_cache_hit"])
            for row in group_rows
            if row.get("handoff_adapter_cache_hit") is not None and row.get("handoff_adapter_cache_hit") != ""
        ]
        handoff_adapter_training_row_cache_rows = [
            bool(row["handoff_adapter_training_row_cache_hit"])
            for row in group_rows
            if row.get("handoff_adapter_training_row_cache_hit") is not None
            and row.get("handoff_adapter_training_row_cache_hit") != ""
        ]
        generated_adapter_local_residual_rows = [
            bool(row["generated_adapter_local_residual_applied"])
            for row in group_rows
            if row.get("generated_adapter_local_residual_applied") is not None
            and row.get("generated_adapter_local_residual_applied") != ""
        ]
        generated_adapter_semantic_memory_rows = [
            bool(row["generated_adapter_semantic_memory_applied"])
            for row in group_rows
            if row.get("generated_adapter_semantic_memory_applied") is not None
            and row.get("generated_adapter_semantic_memory_applied") != ""
        ]
        embedding_manifold_rows = [
            bool(row["embedding_manifold_applied"])
            for row in group_rows
            if row.get("embedding_manifold_applied") is not None
            and row.get("embedding_manifold_applied") != ""
        ]
        explicit_status_rows = [
            row for row in group_rows
            if row.get("handoff_status") not in (None, "")
            and row.get("kv_cache_status") not in (None, "")
            and row.get("decode_status") not in (None, "")
        ]
        handoff_status_rows = [
            str(row.get("handoff_status"))
            for row in group_rows
            if row.get("handoff_status") not in (None, "")
        ]
        empty_decode_rows = [
            row for row in group_rows if str(row.get("decode_status", "")) == "empty_decode"
        ]
        non_empty_rows = [
            1 for row in group_rows if str(row.get("decoded_text", "")).strip()
        ]
        repetitions = {int(row.get("repetition", 0)) for row in group_rows}

        summary_rows.append(
            {
                **row_template,
                "handoff_status": _unique_join(group_rows, "handoff_status"),
                "handoff_surface": _unique_join(group_rows, "handoff_surface"),
                "kv_cache_status": _unique_join(group_rows, "kv_cache_status"),
                "kv_cache_reason": _unique_join(group_rows, "kv_cache_reason"),
                "active_kv_cache_status": _unique_join(group_rows, "active_kv_cache_status"),
                "active_kv_cache_reason": _unique_join(group_rows, "active_kv_cache_reason"),
                "active_kv_cache_source": _unique_join(group_rows, "active_kv_cache_source"),
                "receiver_context_status": _unique_join(group_rows, "receiver_context_status"),
                "receiver_context_reason": _unique_join(group_rows, "receiver_context_reason"),
                "receiver_context_latent_position": _unique_join(group_rows, "receiver_context_latent_position"),
                "decode_status": _unique_join(group_rows, "decode_status"),
                "repetition_count": len(repetitions),
                "sample_count": sample_count,
                "accuracy_percentage": (
                    100.0 * correct_count / len(correct_values) if correct_values else None
                ),
                "sender_answer_extraction_rate_percentage": _percentage(
                    len(sender_answer_rows),
                    len(sender_reasoning_rows),
                ),
                "sender_final_answer_marker_rate_percentage": _percentage(
                    len(sender_final_answer_marker_rows),
                    len(sender_reasoning_rows),
                ),
                "sender_trace_cache_hit_rate_percentage": _percentage(
                    sum(sender_trace_cache_values),
                    len(sender_trace_cache_values),
                ),
                "sender_revision_applied_rate_percentage": _percentage(
                    sum(sender_revision_values),
                    len(sender_revision_values),
                ),
                "sender_revision_decision_applied_rate_percentage": _percentage(
                    sum(sender_revision_decision_values),
                    len(sender_revision_decision_values),
                ),
                "sender_accuracy_percentage": _percentage(
                    sum(sender_correct_values),
                    len(sender_correct_values),
                ),
                "sender_correct_sample_count": len(sender_correct_rows),
                "accuracy_when_sender_correct_percentage": _percentage(
                    sum(sender_correct_row_values),
                    len(sender_correct_row_values),
                ),
                "total_latency_seconds": total_latency_seconds,
                "average_latency_seconds": (total_latency_seconds / sample_count) if sample_count else 0.0,
                "total_generated_tokens": total_generated_tokens,
                "total_receiver_input_tokens": total_receiver_input_tokens,
                "mean_receiver_input_token_count": (
                    total_receiver_input_tokens / len(receiver_input_token_counts)
                    if receiver_input_token_counts
                    else None
                ),
                "max_receiver_input_token_count": (
                    max(receiver_input_token_counts)
                    if receiver_input_token_counts
                    else None
                ),
                "tokens_per_second": (
                    total_generated_tokens / total_latency_seconds if total_latency_seconds > 0 else 0.0
                ),
                "total_answer_tokens": total_answer_tokens,
                "mean_answer_nll": (
                    total_answer_nll / total_answer_tokens if total_answer_tokens > 0 else None
                ),
                "answer_perplexity": (
                    math.exp(total_answer_nll / total_answer_tokens)
                    if total_answer_tokens > 0
                    else None
                ),
                "mean_pre_alignment_l2_distance": _mean_or_none(group_rows, "pre_alignment_l2_distance"),
                "mean_pre_alignment_cosine_distance": _mean_or_none(group_rows, "pre_alignment_cosine_distance"),
                "mean_post_alignment_l2_distance": _mean_or_none(group_rows, "post_alignment_l2_distance"),
                "mean_post_alignment_cosine_distance": _mean_or_none(group_rows, "post_alignment_cosine_distance"),
                "mean_alignment_residual_norm_ratio": _mean_or_none(group_rows, "alignment_residual_norm_ratio"),
                "mean_alignment_bias_norm": _mean_or_none(group_rows, "alignment_bias_norm"),
                "prompt_calibration_rate_percentage": (
                    100.0 * sum(1 for item in prompt_calibration_rows if item) / len(prompt_calibration_rows)
                    if prompt_calibration_rows
                    else None
                ),
                "mean_prompt_calibration_bias_norm": _mean_or_none(group_rows, "prompt_calibration_bias_norm"),
                "handoff_adapter_rate_percentage": (
                    100.0 * sum(1 for item in handoff_adapter_rows if item) / len(handoff_adapter_rows)
                    if handoff_adapter_rows
                    else None
                ),
                "handoff_adapter_cache_hit_rate_percentage": (
                    100.0 * sum(1 for item in handoff_adapter_cache_rows if item)
                    / len(handoff_adapter_cache_rows)
                    if handoff_adapter_cache_rows
                    else None
                ),
                "handoff_adapter_training_row_cache_hit_rate_percentage": (
                    100.0
                    * sum(1 for item in handoff_adapter_training_row_cache_rows if item)
                    / len(handoff_adapter_training_row_cache_rows)
                    if handoff_adapter_training_row_cache_rows
                    else None
                ),
                "mean_handoff_adapter_training_trace_cache_hit_count": _mean_or_none(
                    group_rows,
                    "handoff_adapter_training_trace_cache_hit_count",
                ),
                "mean_handoff_adapter_training_trace_cache_miss_count": _mean_or_none(
                    group_rows,
                    "handoff_adapter_training_trace_cache_miss_count",
                ),
                "mean_handoff_adapter_training_trace_cache_hit_rate_percentage": _mean_or_none(
                    group_rows,
                    "handoff_adapter_training_trace_cache_hit_rate_percentage",
                ),
                "mean_handoff_adapter_delta_norm": _mean_or_none(group_rows, "handoff_adapter_delta_norm"),
                "mean_handoff_adapter_training_reconstruction_mse": _mean_or_none(
                    group_rows,
                    "handoff_adapter_training_reconstruction_mse",
                ),
                "mean_handoff_adapter_training_mean_cosine_similarity": _mean_or_none(
                    group_rows,
                    "handoff_adapter_training_mean_cosine_similarity",
                ),
                "generated_adapter_local_residual_rate_percentage": (
                    100.0
                    * sum(1 for item in generated_adapter_local_residual_rows if item)
                    / len(generated_adapter_local_residual_rows)
                    if generated_adapter_local_residual_rows
                    else None
                ),
                "mean_generated_adapter_local_residual_delta_norm": _mean_or_none(
                    group_rows,
                    "generated_adapter_local_residual_delta_norm",
                ),
                "mean_generated_adapter_local_residual_top_similarity": _mean_or_none(
                    group_rows,
                    "generated_adapter_local_residual_mean_top_similarity",
                ),
                "mean_generated_adapter_local_residual_memory_rows": _mean_or_none(
                    group_rows,
                    "generated_adapter_local_residual_memory_rows",
                ),
                "generated_adapter_semantic_memory_rate_percentage": (
                    100.0
                    * sum(1 for item in generated_adapter_semantic_memory_rows if item)
                    / len(generated_adapter_semantic_memory_rows)
                    if generated_adapter_semantic_memory_rows
                    else None
                ),
                "mean_generated_adapter_semantic_memory_similarity": _mean_or_none(
                    group_rows,
                    "generated_adapter_semantic_memory_similarity",
                ),
                "mean_generated_adapter_semantic_memory_entry_count": _mean_or_none(
                    group_rows,
                    "generated_adapter_semantic_memory_entry_count",
                ),
                "embedding_manifold_rate_percentage": (
                    100.0 * sum(1 for item in embedding_manifold_rows if item)
                    / len(embedding_manifold_rows)
                    if embedding_manifold_rows
                    else None
                ),
                "mean_embedding_manifold_delta_norm": _mean_or_none(
                    group_rows,
                    "embedding_manifold_delta_norm",
                ),
                "mean_embedding_manifold_top_similarity": _mean_or_none(
                    group_rows,
                    "embedding_manifold_mean_top_similarity",
                ),
                "mean_embedding_manifold_unique_token_count": _mean_or_none(
                    group_rows,
                    "embedding_manifold_unique_token_count",
                ),
                "mean_raw_handoff_entropy": _mean_or_none(group_rows, "raw_handoff_entropy"),
                "mean_handoff_uncertainty": _mean_or_none(group_rows, "handoff_uncertainty"),
                "confidence_gate_trigger_rate_percentage": (
                    100.0 * sum(1 for item in confidence_gate_rows if item) / len(confidence_gate_rows)
                    if confidence_gate_rows
                    else None
                ),
                "cache_transfer_rate_percentage": (
                    100.0 * sum(1 for item in kv_transfer_rows if item) / len(kv_transfer_rows)
                    if kv_transfer_rows
                    else None
                ),
                "global_alignment_cache_hit_rate_percentage": (
                    100.0 * sum(1 for item in global_alignment_cache_rows if item) / len(global_alignment_cache_rows)
                    if global_alignment_cache_rows
                    else None
                ),
                "explicit_status_rate_percentage": (
                    100.0 * len(explicit_status_rows) / sample_count if sample_count else 0.0
                ),
                "handoff_ok_rate_percentage": (
                    100.0 * sum(1 for item in handoff_status_rows if item == "ok") / len(handoff_status_rows)
                    if handoff_status_rows
                    else None
                ),
                "empty_decode_rate_percentage": (
                    100.0 * len(empty_decode_rows) / sample_count if sample_count else 0.0
                ),
                "non_empty_decoded_rate_percentage": (
                    100.0 * len(non_empty_rows) / sample_count if sample_count else 0.0
                ),
                "failure_rate_percentage": (
                    100.0 * (len(correct_values) - correct_count) / len(correct_values)
                    if correct_values
                    else None
                ),
                "error_count": error_count,
            }
        )

    return summary_rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def build_runtime_smoke_report(
    rows: Sequence[dict[str, Any]],
    *,
    max_error_count: int = 0,
    require_explicit_statuses: bool = True,
) -> dict[str, Any]:
    error_count = sum(1 for row in rows if bool(row.get("error")))
    missing_status_count = sum(
        1
        for row in rows
        if row.get("handoff_status") in (None, "")
        or row.get("kv_cache_status") in (None, "")
        or row.get("decode_status") in (None, "")
    )
    missing_requirements: list[str] = []
    if error_count > max_error_count:
        missing_requirements.append(
            f"Observed error count {error_count} exceeds allowed {max_error_count}."
        )
    if require_explicit_statuses and missing_status_count > 0:
        missing_requirements.append(
            f"{missing_status_count} rows are missing explicit runtime status fields."
        )

    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "phase": "runtime_smoke",
        "passed": not missing_requirements,
        "sample_count": len(rows),
        "max_error_count": max_error_count,
        "observed_error_count": error_count,
        "missing_status_count": missing_status_count,
        "require_explicit_statuses": require_explicit_statuses,
        "missing_requirements": missing_requirements,
    }


def _percentage(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return 100.0 * numerator / denominator


def _optional_bool_value(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip():
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def build_semantic_smoke_report(
    rows: Sequence[dict[str, Any]],
    *,
    baseline_methods: Sequence[str],
    latent_methods: Sequence[str],
    model_pair_compatibility: Optional[dict[str, Any]] = None,
    min_baseline_accuracy_percentage: Optional[float] = None,
    min_latent_accuracy_percentage: Optional[float] = None,
    min_latent_accuracy_when_sender_correct_percentage: Optional[float] = None,
    min_sender_accuracy_percentage: Optional[float] = None,
    min_sender_final_answer_marker_rate_percentage: Optional[float] = None,
    min_method_accuracy_percentage: Optional[float] = None,
    min_latent_non_empty_decoded_rate_percentage: float = 100.0,
    min_compatible_cache_transfer_rate_percentage: float = 100.0,
    max_answer_perplexity: Optional[float] = None,
    max_degenerate_decode_count: int = 0,
    require_baseline_final_answer_marker: bool = False,
    require_final_answer_marker_methods: Optional[Sequence[str]] = None,
    max_diagnostic_rows: int = 5,
) -> dict[str, Any]:
    baseline_method_set = set(baseline_methods)
    latent_method_set = set(latent_methods)
    baseline_rows = [row for row in rows if row.get("method") in baseline_method_set]
    latent_rows = [row for row in rows if row.get("method") in latent_method_set]
    marker_method_set = (
        set(require_final_answer_marker_methods)
        if require_final_answer_marker_methods is not None
        else baseline_method_set
    )
    marker_scope_rows = [row for row in rows if row.get("method") in marker_method_set]

    baseline_answer_rows = [
        row for row in baseline_rows if str(row.get("predicted_answer") or "").strip()
    ]
    latent_non_empty_rows = [
        row for row in latent_rows if str(row.get("decoded_text") or "").strip()
    ]
    latent_kv_rows = [
        row
        for row in latent_rows
        if row.get("kv_cache_transferred") is not None and row.get("kv_cache_transferred") != ""
        and str(row.get("kv_cache_status") or "") != "not_provided"
    ]
    compatible_cache_rows = [
        row for row in latent_kv_rows if bool(row.get("kv_cache_transferred"))
    ]
    perplexity_rows = [
        row
        for row in rows
        if row.get("answer_perplexity") is not None and row.get("answer_perplexity") != ""
    ]
    finite_perplexity_rows = [
        row
        for row in perplexity_rows
        if math.isfinite(float(row["answer_perplexity"]))
    ]
    latent_perplexity_rows = [
        row
        for row in latent_rows
        if row.get("answer_perplexity") is not None and row.get("answer_perplexity") != ""
    ]
    finite_latent_perplexity_rows = [
        row
        for row in latent_perplexity_rows
        if math.isfinite(float(row["answer_perplexity"]))
    ]
    baseline_final_answer_marker_rows = [
        row for row in baseline_rows if "final answer" in str(row.get("decoded_text") or "").casefold()
    ]
    marker_rows = [
        row for row in marker_scope_rows if "final answer" in str(row.get("decoded_text") or "").casefold()
    ]
    degenerate_rows = [
        row for row in rows if _is_degenerate_decode(str(row.get("decoded_text") or ""))
    ]
    baseline_correct_values = [
        value for value in (_row_correct_value(row) for row in baseline_rows) if value is not None
    ]
    latent_correct_values = [
        value for value in (_row_correct_value(row) for row in latent_rows) if value is not None
    ]
    sender_reasoning_rows = [
        row for row in rows if str(row.get("sender_reasoning_text") or "").strip()
    ]
    sender_answer_rows = [
        row for row in sender_reasoning_rows if str(row.get("sender_predicted_answer") or "").strip()
    ]
    sender_final_answer_marker_rows = [
        row
        for row in sender_reasoning_rows
        if _optional_bool_value(row.get("sender_final_answer_marker")) is True
        or str(row.get("sender_reasoning_status") or "") == "complete"
    ]
    sender_correct_values = [
        value
        for value in (
            _optional_bool_value(row.get("sender_answer_matches_target"))
            for row in sender_reasoning_rows
        )
        if value is not None
    ]
    sender_correct_latent_rows = [
        row
        for row in latent_rows
        if _optional_bool_value(row.get("sender_answer_matches_target")) is True
    ]
    sender_correct_latent_values = [
        value
        for value in (_row_correct_value(row) for row in sender_correct_latent_rows)
        if value is not None
    ]
    wrong_answer_rows = [
        row for row in rows if _row_correct_value(row) is False
    ]
    semantic_methods = list(dict.fromkeys([*baseline_methods, *latent_methods]))
    method_accuracy = {
        method: _accuracy_rate([row for row in rows if row.get("method") == method])
        for method in semantic_methods
    }
    diagnostic_limit = max(0, int(max_diagnostic_rows))
    worst_perplexity_rows = sorted(
        finite_perplexity_rows,
        key=lambda row: float(row["answer_perplexity"]),
        reverse=True,
    )[:diagnostic_limit]

    baseline_answer_rate = _percentage(len(baseline_answer_rows), len(baseline_rows))
    baseline_accuracy_rate = _percentage(sum(baseline_correct_values), len(baseline_correct_values))
    latent_accuracy_rate = _percentage(sum(latent_correct_values), len(latent_correct_values))
    sender_answer_rate = _percentage(len(sender_answer_rows), len(sender_reasoning_rows))
    sender_final_answer_marker_rate = _percentage(
        len(sender_final_answer_marker_rows),
        len(sender_reasoning_rows),
    )
    sender_accuracy_rate = _percentage(sum(sender_correct_values), len(sender_correct_values))
    latent_accuracy_when_sender_correct = _percentage(
        sum(sender_correct_latent_values),
        len(sender_correct_latent_values),
    )
    latent_non_empty_rate = _percentage(len(latent_non_empty_rows), len(latent_rows))
    compatible_cache_transfer_rate = _percentage(len(compatible_cache_rows), len(latent_kv_rows))
    max_all_observed_perplexity = (
        max(float(row["answer_perplexity"]) for row in finite_perplexity_rows)
        if finite_perplexity_rows
        else None
    )
    max_observed_perplexity = (
        max(float(row["answer_perplexity"]) for row in finite_latent_perplexity_rows)
        if finite_latent_perplexity_rows
        else None
    )
    compatibility = model_pair_compatibility or {}
    cache_transfer_required = bool(compatibility.get("kv_cache_compatible", False)) and bool(latent_kv_rows)

    missing_requirements: list[str] = []
    baseline_required = bool(baseline_methods)
    if baseline_required and not baseline_rows:
        missing_requirements.append("No baseline rows were provided.")
    elif len(baseline_answer_rows) < len(baseline_rows):
        missing_requirements.append("At least one baseline row did not extract a predicted answer.")
    if baseline_required and min_baseline_accuracy_percentage is not None:
        if baseline_accuracy_rate is None:
            missing_requirements.append("No baseline correctness rows were provided.")
        elif baseline_accuracy_rate < float(min_baseline_accuracy_percentage):
            missing_requirements.append(
                "Baseline accuracy "
                f"{baseline_accuracy_rate:.2f}% is below required "
                f"{float(min_baseline_accuracy_percentage):.2f}%."
            )

    if not latent_rows:
        missing_requirements.append("No latent rows were provided.")
    elif (
        latent_non_empty_rate is not None
        and latent_non_empty_rate < min_latent_non_empty_decoded_rate_percentage
    ):
        missing_requirements.append(
            "Latent non-empty decode rate "
            f"{latent_non_empty_rate:.2f}% is below required "
                f"{min_latent_non_empty_decoded_rate_percentage:.2f}%."
        )
    if min_latent_accuracy_percentage is not None:
        if latent_accuracy_rate is None:
            missing_requirements.append("No latent correctness rows were provided.")
        elif latent_accuracy_rate < float(min_latent_accuracy_percentage):
            missing_requirements.append(
                "Latent accuracy "
                f"{latent_accuracy_rate:.2f}% is below required "
                f"{float(min_latent_accuracy_percentage):.2f}%."
            )
    if min_latent_accuracy_when_sender_correct_percentage is not None:
        if latent_accuracy_when_sender_correct is None:
            missing_requirements.append("No sender-correct latent correctness rows were provided.")
        elif latent_accuracy_when_sender_correct < float(
            min_latent_accuracy_when_sender_correct_percentage
        ):
            missing_requirements.append(
                "Latent accuracy when sender is correct "
                f"{latent_accuracy_when_sender_correct:.2f}% is below required "
                f"{float(min_latent_accuracy_when_sender_correct_percentage):.2f}%."
            )
    if min_sender_accuracy_percentage is not None:
        if sender_accuracy_rate is None:
            missing_requirements.append("No sender correctness rows were provided.")
        elif sender_accuracy_rate < float(min_sender_accuracy_percentage):
            missing_requirements.append(
                "Sender accuracy "
                f"{sender_accuracy_rate:.2f}% is below required "
                f"{float(min_sender_accuracy_percentage):.2f}%."
            )
    if min_sender_final_answer_marker_rate_percentage is not None:
        if sender_final_answer_marker_rate is None:
            missing_requirements.append("No sender reasoning rows were provided.")
        elif sender_final_answer_marker_rate < float(
            min_sender_final_answer_marker_rate_percentage
        ):
            missing_requirements.append(
                "Sender final-answer marker rate "
                f"{sender_final_answer_marker_rate:.2f}% is below required "
                f"{float(min_sender_final_answer_marker_rate_percentage):.2f}%."
            )
    if min_method_accuracy_percentage is not None:
        for method, accuracy_rate in method_accuracy.items():
            if accuracy_rate is None:
                missing_requirements.append(f"No correctness rows were provided for method {method}.")
            elif accuracy_rate < float(min_method_accuracy_percentage):
                missing_requirements.append(
                    f"Method {method} accuracy {accuracy_rate:.2f}% is below required "
                    f"{float(min_method_accuracy_percentage):.2f}%."
                )

    if cache_transfer_required:
        if compatible_cache_transfer_rate is None:
            missing_requirements.append("No latent KV cache transfer rows were provided.")
        elif compatible_cache_transfer_rate < min_compatible_cache_transfer_rate_percentage:
            missing_requirements.append(
                "Compatible-pair cache transfer rate "
                f"{compatible_cache_transfer_rate:.2f}% is below required "
                f"{min_compatible_cache_transfer_rate_percentage:.2f}%."
            )

    if len(perplexity_rows) < len(rows):
        missing_requirements.append("At least one row did not report answer perplexity.")
    if len(finite_perplexity_rows) < len(perplexity_rows):
        missing_requirements.append("At least one answer perplexity was not finite.")
    if max_answer_perplexity is not None and max_observed_perplexity is not None:
        if max_observed_perplexity > float(max_answer_perplexity):
            missing_requirements.append(
                f"Observed max latent answer perplexity {max_observed_perplexity:.4f} "
                f"exceeds allowed {float(max_answer_perplexity):.4f}."
            )
    if len(degenerate_rows) > int(max_degenerate_decode_count):
        missing_requirements.append(
            f"Observed {len(degenerate_rows)} degenerate decode rows, "
            f"exceeds allowed {int(max_degenerate_decode_count)}."
        )
    if (
        require_baseline_final_answer_marker
        and baseline_required
        and len(baseline_final_answer_marker_rows) < len(baseline_rows)
    ):
        missing_requirements.append(
            "At least one baseline row did not include a final-answer marker."
        )
    if require_final_answer_marker_methods is not None:
        if not marker_scope_rows:
            missing_requirements.append("No rows were provided for required final-answer marker methods.")
        elif len(marker_rows) < len(marker_scope_rows):
            missing_requirements.append(
                "At least one required-method row did not include a final-answer marker."
            )

    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "phase": "semantic_smoke",
        "passed": not missing_requirements,
        "baseline_methods": list(baseline_methods),
        "latent_methods": list(latent_methods),
        "baseline_sample_count": len(baseline_rows),
        "latent_sample_count": len(latent_rows),
        "baseline_answer_extraction_rate_percentage": baseline_answer_rate,
        "baseline_accuracy_percentage": baseline_accuracy_rate,
        "min_baseline_accuracy_percentage": min_baseline_accuracy_percentage,
        "sender_reasoning_sample_count": len(sender_reasoning_rows),
        "sender_answer_extraction_rate_percentage": sender_answer_rate,
        "sender_final_answer_marker_rate_percentage": sender_final_answer_marker_rate,
        "min_sender_final_answer_marker_rate_percentage": (
            min_sender_final_answer_marker_rate_percentage
        ),
        "sender_accuracy_percentage": sender_accuracy_rate,
        "min_sender_accuracy_percentage": min_sender_accuracy_percentage,
        "sender_correct_latent_sample_count": len(sender_correct_latent_rows),
        "latent_accuracy_when_sender_correct_percentage": latent_accuracy_when_sender_correct,
        "min_latent_accuracy_when_sender_correct_percentage": (
            min_latent_accuracy_when_sender_correct_percentage
        ),
        "latent_accuracy_percentage": latent_accuracy_rate,
        "min_latent_accuracy_percentage": min_latent_accuracy_percentage,
        "method_accuracy_percentage": method_accuracy,
        "min_method_accuracy_percentage": min_method_accuracy_percentage,
        "latent_non_empty_decoded_rate_percentage": latent_non_empty_rate,
        "compatible_cache_transfer_rate_percentage": compatible_cache_transfer_rate,
        "cache_transfer_required": cache_transfer_required,
        "model_pair_compatibility_status": compatibility.get("status"),
        "model_pair_compatibility_reason": compatibility.get("reason"),
        "finite_answer_perplexity_count": len(finite_perplexity_rows),
        "answer_perplexity_count": len(perplexity_rows),
        "max_answer_perplexity": max_observed_perplexity,
        "max_all_answer_perplexity": max_all_observed_perplexity,
        "max_allowed_answer_perplexity": max_answer_perplexity,
        "degenerate_decode_count": len(degenerate_rows),
        "max_allowed_degenerate_decode_count": int(max_degenerate_decode_count),
        "baseline_final_answer_marker_rate_percentage": _percentage(
            len(baseline_final_answer_marker_rows),
            len(baseline_rows),
        ),
        "require_baseline_final_answer_marker": bool(require_baseline_final_answer_marker),
        "required_final_answer_marker_methods": (
            list(require_final_answer_marker_methods)
            if require_final_answer_marker_methods is not None
            else None
        ),
        "required_final_answer_marker_sample_count": len(marker_scope_rows),
        "required_final_answer_marker_rate_percentage": _percentage(
            len(marker_rows),
            len(marker_scope_rows),
        ),
        "worst_answer_perplexity_rows": [
            _semantic_row_diagnostic(row) for row in worst_perplexity_rows
        ],
        "wrong_answer_rows": [
            _semantic_row_diagnostic(row) for row in wrong_answer_rows[:diagnostic_limit]
        ],
        "wrong_answer_count": len(wrong_answer_rows),
        "missing_requirements": missing_requirements,
    }


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def _summary_row_by_method(
    rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_method: dict[str, dict[str, Any]] = {}
    for row in rows:
        method = str(row.get("method") or "")
        if method:
            by_method[method] = row
    return by_method


def build_transfer_comparison_report(
    summary_rows: Sequence[dict[str, Any]],
    *,
    baseline_methods: Sequence[str],
    latent_methods: Sequence[str],
    primary_baseline_method: Optional[str] = None,
    min_accuracy_retention_ratio: Optional[float] = None,
    max_latency_ratio: Optional[float] = None,
    require_latent_accuracy_gain: bool = False,
) -> dict[str, Any]:
    """Compare latent transfer rows against token/text handoff baselines."""
    by_method = _summary_row_by_method(summary_rows)
    available_baselines = [
        method for method in baseline_methods if method in by_method
    ]
    available_latents = [
        method for method in latent_methods if method in by_method
    ]
    selected_baseline = (
        primary_baseline_method
        if primary_baseline_method in by_method
        else available_baselines[0]
        if available_baselines
        else None
    )
    baseline_row = by_method.get(selected_baseline or "")
    baseline_accuracy = _safe_float(
        None if baseline_row is None else baseline_row.get("accuracy_percentage")
    )
    baseline_latency = _safe_float(
        None if baseline_row is None else baseline_row.get("average_latency_seconds")
    )
    baseline_perplexity = _safe_float(
        None if baseline_row is None else baseline_row.get("answer_perplexity")
    )
    baseline_receiver_tokens = _safe_float(
        None if baseline_row is None else baseline_row.get("mean_receiver_input_token_count")
    )

    comparisons: list[dict[str, Any]] = []
    missing_requirements: list[str] = []
    if not available_baselines:
        missing_requirements.append("No configured baseline methods were present in summary rows.")
    if not available_latents:
        missing_requirements.append("No configured latent methods were present in summary rows.")

    for method in available_latents:
        latent_row = by_method[method]
        latent_accuracy = _safe_float(latent_row.get("accuracy_percentage"))
        latent_latency = _safe_float(latent_row.get("average_latency_seconds"))
        latent_perplexity = _safe_float(latent_row.get("answer_perplexity"))
        latent_receiver_tokens = _safe_float(latent_row.get("mean_receiver_input_token_count"))
        accuracy_delta = (
            None
            if latent_accuracy is None or baseline_accuracy is None
            else latent_accuracy - baseline_accuracy
        )
        latency_ratio = _safe_ratio(latent_latency, baseline_latency)
        retention_ratio = _safe_ratio(latent_accuracy, baseline_accuracy)
        receiver_token_ratio = _safe_ratio(latent_receiver_tokens, baseline_receiver_tokens)
        perplexity_delta = (
            None
            if latent_perplexity is None or baseline_perplexity is None
            else latent_perplexity - baseline_perplexity
        )
        comparison = {
            "method": method,
            "baseline_method": selected_baseline,
            "sample_count": latent_row.get("sample_count"),
            "baseline_accuracy_percentage": baseline_accuracy,
            "latent_accuracy_percentage": latent_accuracy,
            "accuracy_delta_percentage": accuracy_delta,
            "accuracy_retention_ratio": retention_ratio,
            "baseline_average_latency_seconds": baseline_latency,
            "latent_average_latency_seconds": latent_latency,
            "latency_ratio": latency_ratio,
            "baseline_mean_receiver_input_token_count": baseline_receiver_tokens,
            "latent_mean_receiver_input_token_count": latent_receiver_tokens,
            "receiver_input_token_ratio": receiver_token_ratio,
            "receiver_input_token_savings_percentage": (
                None
                if receiver_token_ratio is None
                else 100.0 * (1.0 - receiver_token_ratio)
            ),
            "baseline_answer_perplexity": baseline_perplexity,
            "latent_answer_perplexity": latent_perplexity,
            "answer_perplexity_delta": perplexity_delta,
            "latent_cache_transfer_rate_percentage": latent_row.get(
                "cache_transfer_rate_percentage"
            ),
            "latent_handoff_ok_rate_percentage": latent_row.get(
                "handoff_ok_rate_percentage"
            ),
            "latent_non_empty_decoded_rate_percentage": latent_row.get(
                "non_empty_decoded_rate_percentage"
            ),
        }
        comparisons.append(comparison)

        if min_accuracy_retention_ratio is not None:
            if retention_ratio is None:
                missing_requirements.append(
                    f"Method {method} did not provide enough accuracy data for retention."
                )
            elif retention_ratio < float(min_accuracy_retention_ratio):
                missing_requirements.append(
                    f"Method {method} retained {retention_ratio:.4f} of baseline accuracy, "
                    f"below required {float(min_accuracy_retention_ratio):.4f}."
                )
        if max_latency_ratio is not None:
            if latency_ratio is None:
                missing_requirements.append(
                    f"Method {method} did not provide enough latency data for comparison."
                )
            elif latency_ratio > float(max_latency_ratio):
                missing_requirements.append(
                    f"Method {method} latency ratio {latency_ratio:.4f} exceeds allowed "
                    f"{float(max_latency_ratio):.4f}."
                )
        if require_latent_accuracy_gain and accuracy_delta is not None and accuracy_delta <= 0.0:
            missing_requirements.append(
                f"Method {method} did not beat baseline accuracy "
                f"({accuracy_delta:.2f} percentage point delta)."
            )

    best_latent = None
    if comparisons:
        best_latent = max(
            comparisons,
            key=lambda row: (
                -1.0
                if row["latent_accuracy_percentage"] is None
                else float(row["latent_accuracy_percentage"]),
                -float(row["latency_ratio"])
                if row["latency_ratio"] is not None
                else float("-inf"),
            ),
        )

    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "phase": "transfer_comparison",
        "passed": not missing_requirements,
        "baseline_methods": list(baseline_methods),
        "latent_methods": list(latent_methods),
        "available_baseline_methods": available_baselines,
        "available_latent_methods": available_latents,
        "primary_baseline_method": selected_baseline,
        "min_accuracy_retention_ratio": min_accuracy_retention_ratio,
        "max_latency_ratio": max_latency_ratio,
        "require_latent_accuracy_gain": bool(require_latent_accuracy_gain),
        "comparisons": comparisons,
        "best_latent_method": None if best_latent is None else best_latent["method"],
        "best_latent_accuracy_percentage": (
            None if best_latent is None else best_latent["latent_accuracy_percentage"]
        ),
        "missing_requirements": missing_requirements,
    }


def build_heterogeneous_transfer_report(
    rows: Sequence[dict[str, Any]],
    *,
    latent_methods: Sequence[str],
    model_pair_compatibility: Optional[dict[str, Any]] = None,
    generated_methods: Sequence[str] = (),
    context_generated_methods: Sequence[str] = (),
    require_generated_adapter_for_incompatible_pair: bool = True,
    require_context_for_context_methods: bool = True,
) -> dict[str, Any]:
    """Gate cross-family latent rows on robust, non-KV transfer surfaces."""
    compatibility = model_pair_compatibility or {}
    pair_kv_compatible = bool(compatibility.get("kv_cache_compatible", False))
    latent_method_set = set(latent_methods)
    generated_method_set = set(generated_methods)
    context_method_set = set(context_generated_methods)
    latent_rows = [row for row in rows if row.get("method") in latent_method_set]
    generated_rows = [row for row in latent_rows if row.get("method") in generated_method_set]
    context_rows = [row for row in latent_rows if row.get("method") in context_method_set]

    direct_cache_statuses = {
        "transferred",
        "unsupported",
        "unsupported_architecture_mismatch",
        "layer_count_mismatch",
    }
    direct_cache_attempt_rows = [
        row
        for row in latent_rows
        if str(row.get("kv_cache_status") or "") in direct_cache_statuses
        or _optional_bool_value(row.get("kv_cache_transferred")) is True
    ]
    generated_adapter_rows = [
        row
        for row in generated_rows
        if "generated_trajectory_" in str(row.get("handoff_adapter_status") or "")
    ]
    generated_adapter_applied_rows = [
        row
        for row in generated_adapter_rows
        if _optional_bool_value(row.get("handoff_adapter_applied")) is True
    ]
    missing_generated_adapter_rows = [
        row
        for row in generated_rows
        if "generated_trajectory_missing" in str(row.get("handoff_adapter_status") or "")
        or "generated_trajectory_disabled" in str(row.get("handoff_adapter_status") or "")
        or "generated_trajectory_" not in str(row.get("handoff_adapter_status") or "")
    ]
    context_used_rows = [
        row
        for row in context_rows
        if str(row.get("receiver_context_status") or "").startswith("used_")
        or str(row.get("active_kv_cache_source") or "") == "receiver_context"
    ]

    missing_requirements: list[str] = []
    if not latent_rows:
        missing_requirements.append("No latent rows were available for heterogeneous reporting.")
    if not pair_kv_compatible and direct_cache_attempt_rows:
        missing_requirements.append(
            "Incompatible model pair produced rows on direct KV-cache transfer surfaces; "
            "use generated trajectory/context handoff for production hetero evaluation."
        )
    if (
        require_generated_adapter_for_incompatible_pair
        and not pair_kv_compatible
        and generated_rows
        and len(generated_adapter_applied_rows) < len(generated_rows)
    ):
        missing_requirements.append(
            "At least one generated latent row did not apply a generated-trajectory adapter "
            "on an incompatible model pair."
        )
    if (
        require_context_for_context_methods
        and context_rows
        and len(context_used_rows) < len(context_rows)
    ):
        missing_requirements.append(
            "At least one generated context latent row did not use receiver prompt context."
        )

    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "phase": "heterogeneous_transfer",
        "passed": not missing_requirements,
        "model_pair_kv_cache_compatible": pair_kv_compatible,
        "model_pair_compatibility_status": compatibility.get("status"),
        "model_pair_compatibility_reason": compatibility.get("reason"),
        "latent_row_count": len(latent_rows),
        "generated_latent_row_count": len(generated_rows),
        "direct_cache_attempt_row_count": len(direct_cache_attempt_rows),
        "generated_adapter_row_count": len(generated_adapter_rows),
        "generated_adapter_applied_row_count": len(generated_adapter_applied_rows),
        "missing_generated_adapter_row_count": len(missing_generated_adapter_rows),
        "receiver_context_row_count": len(context_rows),
        "receiver_context_used_row_count": len(context_used_rows),
        "missing_requirements": missing_requirements,
    }


def _row_correct_value(row: dict[str, Any]) -> Optional[bool]:
    value = row.get("correct")
    bool_value = _optional_bool_value(value)
    if bool_value is not None:
        return bool_value

    predicted = row.get("predicted_answer")
    target = row.get("target_answer")
    if predicted is not None and target is not None:
        return str(predicted).strip() == str(target).strip()
    return None


def _accuracy_rate(rows: Sequence[dict[str, Any]]) -> Optional[float]:
    correct_values = [
        value for value in (_row_correct_value(row) for row in rows) if value is not None
    ]
    return _percentage(sum(correct_values), len(correct_values))


def _semantic_row_diagnostic(row: dict[str, Any]) -> dict[str, Any]:
    decoded_text = " ".join(str(row.get("decoded_text") or "").split())
    return {
        "method": row.get("method"),
        "sample_index": row.get("sample_index"),
        "target_answer": row.get("target_answer"),
        "sender_reasoning_status": row.get("sender_reasoning_status"),
        "sender_final_answer_marker": _optional_bool_value(
            row.get("sender_final_answer_marker")
        ),
        "sender_predicted_answer": row.get("sender_predicted_answer"),
        "sender_answer_matches_target": _optional_bool_value(
            row.get("sender_answer_matches_target")
        ),
        "predicted_answer": row.get("predicted_answer"),
        "correct": _row_correct_value(row),
        "answer_perplexity": row.get("answer_perplexity"),
        "generated_tokens": row.get("generated_tokens"),
        "kv_cache_status": row.get("kv_cache_status"),
        "active_kv_cache_status": row.get("active_kv_cache_status"),
        "active_kv_cache_source": row.get("active_kv_cache_source"),
        "receiver_context_status": row.get("receiver_context_status"),
        "receiver_context_reason": row.get("receiver_context_reason"),
        "receiver_context_latent_position": row.get("receiver_context_latent_position"),
        "decoded_preview": decoded_text[:240],
    }


def _is_degenerate_decode(decoded_text: str) -> bool:
    tokens = decoded_text.split()
    if len(tokens) < 24:
        return False
    max_run = 1
    current_run = 1
    previous_token = tokens[0]
    for token in tokens[1:]:
        if token == previous_token:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
            previous_token = token
    unique_ratio = len(set(tokens)) / len(tokens)
    return max_run >= 16 or unique_ratio < 0.08


def build_phase1_gate_report(
    summary_rows: Sequence[dict[str, Any]],
    *,
    required_repetitions: int,
    max_error_rate_percentage: float,
    min_cache_transfer_rate_percentage: float,
    min_non_empty_decoded_rate_percentage: float,
) -> dict[str, Any]:
    relevant_rows = [
        row for row in summary_rows if row.get("method") in {"homogeneous_ridge_latent", "homogeneous_orthogonal_latent"}
    ]
    if not relevant_rows:
        return {
            "phase": "phase_1",
            "passed": False,
            "missing_requirements": ["No homogeneous latent baseline rows were provided."],
        }

    max_failure_rate = max(float(row["failure_rate_percentage"]) for row in relevant_rows)
    max_error_rate = max(
        (100.0 * float(row["error_count"]) / float(row["sample_count"])) if float(row["sample_count"]) else 100.0
        for row in relevant_rows
    )
    min_cache_rate = min(
        float(row["cache_transfer_rate_percentage"])
        for row in relevant_rows
        if row.get("cache_transfer_rate_percentage") is not None
    )
    min_non_empty_rate = min(float(row["non_empty_decoded_rate_percentage"]) for row in relevant_rows)
    min_repetition_count = min(int(row["repetition_count"]) for row in relevant_rows)
    missing_requirements: list[str] = []
    if max_error_rate > max_error_rate_percentage:
        missing_requirements.append(
            f"Observed error rate {max_error_rate:.2f}% exceeds allowed {max_error_rate_percentage:.2f}%."
        )
    if min_cache_rate < min_cache_transfer_rate_percentage:
        missing_requirements.append(
            f"Observed cache transfer rate {min_cache_rate:.2f}% is below required {min_cache_transfer_rate_percentage:.2f}%."
        )
    if min_non_empty_rate < min_non_empty_decoded_rate_percentage:
        missing_requirements.append(
            f"Observed non-empty decode rate {min_non_empty_rate:.2f}% is below required {min_non_empty_decoded_rate_percentage:.2f}%."
        )
    if min_repetition_count < required_repetitions:
        missing_requirements.append(
            f"Observed repetition count {min_repetition_count} is below required {required_repetitions}."
        )

    return {
        "phase": "phase_1",
        "passed": not missing_requirements,
        "required_repetitions": required_repetitions,
        "max_error_rate_percentage": max_error_rate_percentage,
        "min_cache_transfer_rate_percentage": min_cache_transfer_rate_percentage,
        "min_non_empty_decoded_rate_percentage": min_non_empty_decoded_rate_percentage,
        "observed_max_failure_rate_percentage": max_failure_rate,
        "observed_max_error_rate_percentage": max_error_rate,
        "observed_min_cache_transfer_rate_percentage": min_cache_rate,
        "observed_min_non_empty_decoded_rate_percentage": min_non_empty_rate,
        "observed_min_repetition_count": min_repetition_count,
        "methods": [row["method"] for row in relevant_rows],
        "missing_requirements": missing_requirements,
    }


def build_phase3_gate_report(
    summary_rows: Sequence[dict[str, Any]],
    *,
    require_q_global_beats_prompt_local: bool,
    min_hybrid_accuracy_gain_over_text_text: float = 3.0,
    max_accuracy_gap_for_perplexity_tie: float = 1.0,
    min_perplexity_improvement_ratio: float = 0.10,
) -> dict[str, Any]:
    def _best_row(method_name: str) -> Optional[dict[str, Any]]:
        candidates = [row for row in summary_rows if str(row.get("method")) == method_name]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda row: (
                -float(row["accuracy_percentage"]),
                float(row.get("average_latency_seconds", 0.0)),
                float(row.get("answer_perplexity", float("inf")) or float("inf")),
            ),
        )

    prompt_local = _best_row("prompt_local_latent")
    global_anchor_orthogonal = _best_row("global_anchor_orthogonal")
    global_anchor_hybrid = _best_row("global_anchor_hybrid_affine_plus_calibration")
    if global_anchor_hybrid is None:
        global_anchor_hybrid = _best_row("global_anchor_hybrid_affine")
    hybrid = _best_row("hybrid_hl_mas")
    text_text_hybrid = _best_row("text_text_hybrid")
    missing_requirements: list[str] = []

    if prompt_local is None or global_anchor_orthogonal is None:
        missing_requirements.append(
            "Prompt-local and global-anchor orthogonal baselines are both required for phase 3 comparison."
        )

    q_global_beats_prompt_local: Optional[bool] = None
    if prompt_local is not None and global_anchor_orthogonal is not None:
        q_global_beats_prompt_local = float(global_anchor_orthogonal["accuracy_percentage"]) >= float(
            prompt_local["accuracy_percentage"]
        )
        if require_q_global_beats_prompt_local and not q_global_beats_prompt_local:
            missing_requirements.append("Global semantic-anchor alignment does not beat prompt-local alignment.")

    hybrid_affine_beats_orthogonal: Optional[bool] = None
    if global_anchor_hybrid is None or global_anchor_orthogonal is None:
        missing_requirements.append(
            "Orthogonal and hybrid-affine global-anchor baselines are both required for phase 3 comparison."
        )
    else:
        hybrid_affine_accuracy = float(global_anchor_hybrid["accuracy_percentage"])
        orthogonal_accuracy = float(global_anchor_orthogonal["accuracy_percentage"])
        hybrid_affine_perplexity = global_anchor_hybrid.get("answer_perplexity")
        orthogonal_perplexity = global_anchor_orthogonal.get("answer_perplexity")
        hybrid_affine_beats_orthogonal = (
            hybrid_affine_accuracy >= orthogonal_accuracy
            and (
                hybrid_affine_perplexity is None
                or orthogonal_perplexity is None
                or float(hybrid_affine_perplexity) <= float(orthogonal_perplexity)
            )
        )
        if not hybrid_affine_beats_orthogonal:
            missing_requirements.append(
                "Hybrid-affine global alignment does not beat or match the orthogonal global baseline."
            )

    hybrid_available = hybrid is not None
    if not hybrid_available:
        missing_requirements.append("Hybrid ODE benchmark row is missing.")

    hybrid_beats_text_text: Optional[bool] = None
    if hybrid is None or text_text_hybrid is None:
        missing_requirements.append("Text-text hybrid and latent ODE rows are both required for phase 3 comparison.")
    else:
        hybrid_accuracy = float(hybrid["accuracy_percentage"])
        text_hybrid_accuracy = float(text_text_hybrid["accuracy_percentage"])
        hybrid_perplexity = hybrid.get("answer_perplexity")
        text_hybrid_perplexity = text_text_hybrid.get("answer_perplexity")
        accuracy_gain = hybrid_accuracy - text_hybrid_accuracy
        perplexity_improvement_ratio = None
        if (
            hybrid_perplexity is not None
            and text_hybrid_perplexity is not None
            and float(text_hybrid_perplexity) > 0.0
        ):
            perplexity_improvement_ratio = (
                float(text_hybrid_perplexity) - float(hybrid_perplexity)
            ) / float(text_hybrid_perplexity)
        hybrid_beats_text_text = (
            accuracy_gain >= min_hybrid_accuracy_gain_over_text_text
            or (
                accuracy_gain >= -max_accuracy_gap_for_perplexity_tie
                and perplexity_improvement_ratio is not None
                and perplexity_improvement_ratio >= min_perplexity_improvement_ratio
            )
        )
        if not hybrid_beats_text_text:
            missing_requirements.append(
                "Hybrid latent transfer does not clear the required text-text hybrid comparison thresholds."
            )

    return {
        "phase": "phase_3",
        "passed": not missing_requirements,
        "require_q_global_beats_prompt_local": require_q_global_beats_prompt_local,
        "min_hybrid_accuracy_gain_over_text_text": min_hybrid_accuracy_gain_over_text_text,
        "max_accuracy_gap_for_perplexity_tie": max_accuracy_gap_for_perplexity_tie,
        "min_perplexity_improvement_ratio": min_perplexity_improvement_ratio,
        "q_global_beats_prompt_local": q_global_beats_prompt_local,
        "hybrid_affine_beats_orthogonal": hybrid_affine_beats_orthogonal,
        "hybrid_beats_text_text": hybrid_beats_text_text,
        "prompt_local_accuracy_percentage": None if prompt_local is None else float(prompt_local["accuracy_percentage"]),
        "global_anchor_orthogonal_accuracy_percentage": (
            None if global_anchor_orthogonal is None else float(global_anchor_orthogonal["accuracy_percentage"])
        ),
        "global_anchor_hybrid_affine_accuracy_percentage": (
            None if global_anchor_hybrid is None else float(global_anchor_hybrid["accuracy_percentage"])
        ),
        "text_text_hybrid_accuracy_percentage": (
            None if text_text_hybrid is None else float(text_text_hybrid["accuracy_percentage"])
        ),
        "text_text_hybrid_answer_perplexity": (
            None if text_text_hybrid is None else text_text_hybrid.get("answer_perplexity")
        ),
        "hybrid_accuracy_percentage": None if hybrid is None else float(hybrid["accuracy_percentage"]),
        "hybrid_answer_perplexity": None if hybrid is None else hybrid.get("answer_perplexity"),
        "missing_requirements": missing_requirements,
    }


def build_distance_calibration_report(
    rows: Sequence[dict[str, Any]],
    summary_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    correct_rows = [
        row for row in rows if bool(row.get("correct")) and row.get("post_alignment_l2_distance") is not None
    ]
    incorrect_rows = [
        row for row in rows if (not bool(row.get("correct"))) and row.get("post_alignment_l2_distance") is not None
    ]
    correct_mean = None if not correct_rows else sum(float(row["post_alignment_l2_distance"]) for row in correct_rows) / len(correct_rows)
    incorrect_mean = None if not incorrect_rows else sum(float(row["post_alignment_l2_distance"]) for row in incorrect_rows) / len(incorrect_rows)

    overall_rows = [row for row in summary_rows if row.get("row_type") == "overall"]
    selected_default = next((row for row in overall_rows if bool(row.get("selected_default"))), None)
    selected_breaking_point = None if selected_default is None else selected_default.get("breaking_point_decile")
    missing_requirements: list[str] = []
    if correct_mean is None or incorrect_mean is None:
        missing_requirements.append("Need both correct and incorrect samples to evaluate distance separation.")
    elif not correct_mean < incorrect_mean:
        missing_requirements.append("Correct samples are not closer than incorrect samples in actor space.")
    if selected_breaking_point in (None, ""):
        missing_requirements.append("No breaking-point decile was detected for the selected default configuration.")

    return {
        "phase": "phase_3_calibration",
        "passed": not missing_requirements,
        "correct_mean_post_alignment_l2_distance": correct_mean,
        "incorrect_mean_post_alignment_l2_distance": incorrect_mean,
        "selected_default_config_id": None if selected_default is None else selected_default.get("config_id"),
        "selected_default_breaking_point_decile": selected_breaking_point,
        "missing_requirements": missing_requirements,
    }


def build_training_phase2_report(
    *,
    history: Sequence[dict[str, Any]],
    cfg: Any,
    alignment_context: dict[str, Any],
    dataset_name: str,
    training_mode: str,
    seed_count: int,
    required_seed_count: int,
    min_accuracy_retention_ratio: float,
    baseline_accuracy_percentage: Optional[float] = None,
    runtime_metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    heldout_entries = [entry for entry in history if "heldout_exact_match_accuracy" in entry]
    final_heldout_entry = heldout_entries[-1] if heldout_entries else {}
    final_heldout_accuracy = None if not heldout_entries else float(heldout_entries[-1]["heldout_exact_match_accuracy"])
    final_heldout_answer_perplexity = (
        None if not heldout_entries else heldout_entries[-1].get("heldout_answer_perplexity")
    )
    accuracy_retention_ratio = None
    if baseline_accuracy_percentage is not None and baseline_accuracy_percentage > 0 and final_heldout_accuracy is not None:
        accuracy_retention_ratio = final_heldout_accuracy / float(baseline_accuracy_percentage)

    missing_requirements: list[str] = []
    if training_mode != "real":
        missing_requirements.append("Training mode is not 'real'.")
    if seed_count < required_seed_count:
        missing_requirements.append(
            f"Observed seed count {seed_count} is below required {required_seed_count}."
        )
    if baseline_accuracy_percentage is None:
        missing_requirements.append("Baseline accuracy is not available, so retention cannot be computed.")
    elif accuracy_retention_ratio is None or accuracy_retention_ratio < min_accuracy_retention_ratio:
        missing_requirements.append(
            f"Observed accuracy retention {0.0 if accuracy_retention_ratio is None else accuracy_retention_ratio:.4f} "
            f"is below required {min_accuracy_retention_ratio:.4f}."
        )

    report = {
        "evaluation_surface": "run_training",
        "method": "stage2_compression_training",
        "phase": "phase_2",
        "passed": not missing_requirements,
        "dataset": dataset_name,
        "training_mode": training_mode,
        "agent_a_model": str(_cfg_value(cfg, "agent_a_model", "")),
        "agent_b_model": str(_cfg_value(cfg, "agent_b_model", "")),
        "torch_dtype": str(_cfg_value(cfg, "torch_dtype", "")),
        "seed": int(_cfg_value(cfg, "seed", 0)),
        "compression_steps": int(_cfg_value(cfg, "training.compressed_steps", 0)),
        "seed_count": seed_count,
        "required_seed_count": required_seed_count,
        "min_accuracy_retention_ratio": min_accuracy_retention_ratio,
        "baseline_accuracy_percentage": baseline_accuracy_percentage,
        "final_heldout_exact_match_accuracy": final_heldout_accuracy,
        "final_heldout_answer_perplexity": final_heldout_answer_perplexity,
        "final_heldout_answer_perplexity_surface": final_heldout_entry.get(
            "heldout_answer_perplexity_surface"
        ),
        "final_heldout_raw_decode_exact_match_accuracy": final_heldout_entry.get(
            "heldout_raw_decode_exact_match_accuracy"
        ),
        "final_heldout_actor_semantic_bridge_decode_accuracy": final_heldout_entry.get(
            "heldout_actor_semantic_bridge_decode_accuracy"
        ),
        "final_heldout_actor_semantic_bridge_decode_answer_extraction_rate_percentage": final_heldout_entry.get(
            "heldout_actor_semantic_bridge_decode_answer_extraction_rate_percentage"
        ),
        "final_heldout_actor_semantic_bridge_decode_unique_predicted_answer_count": final_heldout_entry.get(
            "heldout_actor_semantic_bridge_decode_unique_predicted_answer_count"
        ),
        "final_heldout_latent_token_decode_accuracy": final_heldout_entry.get(
            "heldout_latent_token_decode_accuracy"
        ),
        "final_heldout_latent_token_decode_enabled": final_heldout_entry.get(
            "heldout_latent_token_decode_enabled"
        ),
        "final_heldout_latent_token_decode_require_ready": final_heldout_entry.get(
            "heldout_latent_token_decode_require_ready"
        ),
        "final_heldout_latent_token_decode_surface_rate_percentage": final_heldout_entry.get(
            "heldout_latent_token_decode_surface_rate_percentage"
        ),
        "final_heldout_latent_token_decode_answer_extraction_rate_percentage": final_heldout_entry.get(
            "heldout_latent_token_decode_answer_extraction_rate_percentage"
        ),
        "final_heldout_latent_token_decode_unique_predicted_answer_count": final_heldout_entry.get(
            "heldout_latent_token_decode_unique_predicted_answer_count"
        ),
        "final_heldout_latent_semantic_readout_accuracy": final_heldout_entry.get(
            "heldout_latent_semantic_readout_accuracy"
        ),
        "final_heldout_latent_semantic_readout_rate_percentage": final_heldout_entry.get(
            "heldout_latent_semantic_readout_rate_percentage"
        ),
        "final_heldout_latent_semantic_readout_unique_predicted_answer_count": final_heldout_entry.get(
            "heldout_latent_semantic_readout_unique_predicted_answer_count"
        ),
        "final_heldout_latent_candidate_accuracy": final_heldout_entry.get(
            "heldout_latent_candidate_accuracy"
        ),
        "final_heldout_latent_probe_accuracy": final_heldout_entry.get(
            "heldout_latent_probe_accuracy"
        ),
        "final_heldout_latent_sequence_decoder_token_accuracy": final_heldout_entry.get(
            "heldout_latent_sequence_decoder_token_accuracy"
        ),
        "final_heldout_latent_sequence_decoder_sequence_accuracy": final_heldout_entry.get(
            "heldout_latent_sequence_decoder_sequence_accuracy"
        ),
        "final_heldout_latent_sequence_decoder_length_accuracy": final_heldout_entry.get(
            "heldout_latent_sequence_decoder_length_accuracy"
        ),
        "final_heldout_latent_generation_smoke_ready": final_heldout_entry.get(
            "heldout_latent_generation_smoke_ready"
        ),
        "final_heldout_latent_generation_smoke_skipped_count": final_heldout_entry.get(
            "heldout_latent_generation_smoke_skipped_count"
        ),
        "final_heldout_latent_first_token_accuracy": final_heldout_entry.get(
            "heldout_latent_first_token_accuracy"
        ),
        "final_heldout_latent_first_token_rank_mean": final_heldout_entry.get(
            "heldout_latent_first_token_rank_mean"
        ),
        "final_heldout_actor_text_baseline_evaluated": final_heldout_entry.get(
            "heldout_actor_text_baseline_evaluated"
        ),
        "accuracy_retention_ratio": accuracy_retention_ratio,
        "alignment_mode": alignment_context.get("alignment_mode"),
        "semantic_anchor_count": alignment_context.get("semantic_anchor_count"),
        "reasoning_layer_weights": serialize_reasoning_layer_weights(
            alignment_context.get("reasoning_layer_weights")
        ),
        "model_pair": f"{_cfg_value(cfg, 'agent_a_model', '')} -> {_cfg_value(cfg, 'agent_b_model', '')}",
        "missing_requirements": missing_requirements,
    }
    if runtime_metadata is not None:
        report["runtime"] = dict(runtime_metadata)
        if runtime_metadata.get("effective_torch_dtype"):
            report["effective_torch_dtype"] = str(runtime_metadata["effective_torch_dtype"])
        if runtime_metadata.get("effective_device"):
            report["effective_device"] = str(runtime_metadata["effective_device"])
    return report


_TRAINING_DIAGNOSTIC_PREDICTED_REGEX = re.compile(r"(?:^|\|\s*)predicted=([^|]+)")


def _unique_training_prediction_count(
    final_eval: Mapping[str, Any],
    *,
    eval_samples: int,
) -> int | None:
    unique_prediction_raw = final_eval.get("heldout_unique_predicted_answer_count")
    if unique_prediction_raw is not None and unique_prediction_raw != "":
        return int(float(unique_prediction_raw))

    diagnostics = str(final_eval.get("heldout_eval_diagnostics") or "")
    predicted_answers: list[str] = []
    for line in diagnostics.splitlines():
        match = _TRAINING_DIAGNOSTIC_PREDICTED_REGEX.search(line)
        if match is None:
            continue
        predicted = match.group(1).strip().casefold().replace(" ", "")
        if predicted and predicted != "none":
            predicted_answers.append(predicted)
    if eval_samples > 0 and len(predicted_answers) >= eval_samples:
        return len(set(predicted_answers))
    return None


def build_training_smoke_report(
    history: Sequence[dict[str, Any]],
    *,
    min_eval_samples: int = 1,
    max_loss: float = 1000.0,
    max_answer_perplexity: float = 10000.0,
    min_answer_extraction_rate_percentage: float = 100.0,
) -> dict[str, Any]:
    loss_entries = [entry for entry in history if "loss" in entry]
    heldout_entries = [entry for entry in history if "heldout_eval_samples" in entry]
    initial_eval = heldout_entries[0] if len(heldout_entries) > 1 else {}
    final_eval = heldout_entries[-1] if heldout_entries else {}
    losses = [float(entry["loss"]) for entry in loss_entries if entry.get("loss") is not None]
    finite_losses = [value for value in losses if math.isfinite(value)]
    final_loss = finite_losses[-1] if finite_losses else None
    final_perplexity_raw = final_eval.get("heldout_answer_perplexity")
    final_perplexity = (
        None if final_perplexity_raw is None else float(final_perplexity_raw)
    )
    final_extraction_rate_raw = final_eval.get("heldout_answer_extraction_rate_percentage")
    final_extraction_rate = (
        None if final_extraction_rate_raw is None else float(final_extraction_rate_raw)
    )
    final_accuracy_raw = final_eval.get("heldout_exact_match_accuracy")
    final_accuracy = None if final_accuracy_raw is None else float(final_accuracy_raw)
    eval_samples = int(float(final_eval.get("heldout_eval_samples", 0) or 0))
    unique_prediction_count = _unique_training_prediction_count(
        final_eval,
        eval_samples=eval_samples,
    )
    actor_baseline_accuracy_raw = final_eval.get("heldout_actor_text_baseline_accuracy")
    actor_baseline_accuracy = (
        None if actor_baseline_accuracy_raw is None else float(actor_baseline_accuracy_raw)
    )
    actor_baseline_evaluated = bool(final_eval.get("heldout_actor_text_baseline_evaluated", True))
    actor_baseline_unique_raw = final_eval.get(
        "heldout_actor_text_baseline_unique_predicted_answer_count"
    )
    actor_baseline_unique_count = (
        None if actor_baseline_unique_raw is None else int(float(actor_baseline_unique_raw))
    )
    actor_bridge_accuracy_raw = final_eval.get("heldout_actor_semantic_bridge_decode_accuracy")
    actor_bridge_accuracy = (
        None if actor_bridge_accuracy_raw is None else float(actor_bridge_accuracy_raw)
    )
    actor_bridge_enabled = bool(
        final_eval.get("heldout_actor_semantic_bridge_decode_enabled", False)
    )
    actor_bridge_rate_raw = final_eval.get(
        "heldout_actor_semantic_bridge_decode_answer_extraction_rate_percentage"
    )
    actor_bridge_rate = None if actor_bridge_rate_raw is None else float(actor_bridge_rate_raw)
    actor_bridge_unique_raw = final_eval.get(
        "heldout_actor_semantic_bridge_decode_unique_predicted_answer_count"
    )
    actor_bridge_unique_count = (
        None if actor_bridge_unique_raw is None else int(float(actor_bridge_unique_raw))
    )
    latent_probe_accuracy_raw = final_eval.get("heldout_latent_probe_accuracy")
    latent_probe_accuracy = (
        None if latent_probe_accuracy_raw is None else float(latent_probe_accuracy_raw)
    )
    latent_probe_unique_raw = final_eval.get("heldout_latent_probe_unique_predicted_answer_count")
    latent_probe_unique_count = (
        None if latent_probe_unique_raw is None else int(float(latent_probe_unique_raw))
    )
    latent_candidate_accuracy_raw = final_eval.get("heldout_latent_candidate_accuracy")
    latent_candidate_accuracy = (
        None if latent_candidate_accuracy_raw is None else float(latent_candidate_accuracy_raw)
    )
    latent_candidate_unique_raw = final_eval.get(
        "heldout_latent_candidate_unique_predicted_answer_count"
    )
    latent_candidate_unique_count = (
        None if latent_candidate_unique_raw is None else int(float(latent_candidate_unique_raw))
    )
    latent_token_decode_accuracy_raw = final_eval.get("heldout_latent_token_decode_accuracy")
    latent_token_decode_accuracy = (
        None if latent_token_decode_accuracy_raw is None else float(latent_token_decode_accuracy_raw)
    )
    latent_token_decode_rate_raw = final_eval.get(
        "heldout_latent_token_decode_answer_extraction_rate_percentage"
    )
    latent_token_decode_rate = (
        None if latent_token_decode_rate_raw is None else float(latent_token_decode_rate_raw)
    )
    latent_token_decode_unique_raw = final_eval.get(
        "heldout_latent_token_decode_unique_predicted_answer_count"
    )
    latent_token_decode_unique_count = (
        None if latent_token_decode_unique_raw is None else int(float(latent_token_decode_unique_raw))
    )
    latent_token_decode_enabled = bool(final_eval.get("heldout_latent_token_decode_enabled", False))
    latent_token_decode_require_ready = bool(
        final_eval.get("heldout_latent_token_decode_require_ready", False)
    )
    raw_decode_accuracy_raw = final_eval.get("heldout_raw_decode_exact_match_accuracy")
    raw_decode_accuracy = (
        None if raw_decode_accuracy_raw is None else float(raw_decode_accuracy_raw)
    )
    raw_decode_rate_raw = final_eval.get("heldout_raw_decode_answer_extraction_rate_percentage")
    raw_decode_rate = None if raw_decode_rate_raw is None else float(raw_decode_rate_raw)
    raw_decode_unique_raw = final_eval.get("heldout_raw_decode_unique_predicted_answer_count")
    raw_decode_unique_count = (
        None if raw_decode_unique_raw is None else int(float(raw_decode_unique_raw))
    )
    raw_decode_require_ready = bool(
        final_eval.get("heldout_raw_decode_require_ready", False)
    )
    latent_sequence_accuracy_raw = final_eval.get(
        "heldout_latent_sequence_decoder_sequence_accuracy"
    )
    latent_sequence_accuracy = (
        None if latent_sequence_accuracy_raw is None else float(latent_sequence_accuracy_raw)
    )
    latent_sequence_unique_raw = final_eval.get(
        "heldout_latent_sequence_decoder_unique_predicted_answer_count"
    )
    latent_sequence_unique_count = (
        None if latent_sequence_unique_raw is None else int(float(latent_sequence_unique_raw))
    )

    missing_requirements: list[str] = []
    if not loss_entries:
        missing_requirements.append("No training loss entries were logged.")
    if len(finite_losses) != len(losses):
        missing_requirements.append("At least one logged training loss is non-finite.")
    if final_loss is not None and final_loss > max_loss:
        missing_requirements.append(
            f"Final training loss {final_loss:.4f} exceeds smoke limit {max_loss:.4f}."
        )
    if not heldout_entries:
        missing_requirements.append("No heldout evaluation entry was logged.")
    if eval_samples < min_eval_samples:
        missing_requirements.append(
            f"Heldout eval samples {eval_samples} is below required {min_eval_samples}."
        )
    if final_perplexity is None or not math.isfinite(final_perplexity):
        missing_requirements.append("Final heldout answer perplexity is missing or non-finite.")
    elif final_perplexity > max_answer_perplexity:
        missing_requirements.append(
            f"Final heldout answer perplexity {final_perplexity:.4f} exceeds smoke limit "
            f"{max_answer_perplexity:.4f}."
        )
    if final_extraction_rate is None or not math.isfinite(final_extraction_rate):
        missing_requirements.append("Final heldout answer extraction rate is missing or non-finite.")
    elif final_extraction_rate < min_answer_extraction_rate_percentage:
        missing_requirements.append(
            f"Answer extraction rate {final_extraction_rate:.2f}% is below required "
            f"{min_answer_extraction_rate_percentage:.2f}%."
        )
    degenerate_prediction = (
        eval_samples > 1
        and unique_prediction_count is not None
        and unique_prediction_count <= 1
        and final_accuracy is not None
        and final_accuracy < 100.0
    )
    if degenerate_prediction:
        missing_requirements.append(
            "Heldout predictions are degenerate: one unique predicted answer was emitted "
            f"across {eval_samples} samples while exact-match accuracy was {final_accuracy:.2f}%."
        )
    actor_text_baseline_degenerate = (
        actor_baseline_evaluated
        and eval_samples > 1
        and actor_baseline_unique_count is not None
        and actor_baseline_unique_count <= 1
        and actor_baseline_accuracy is not None
        and actor_baseline_accuracy < 100.0
    )
    if actor_text_baseline_degenerate:
        missing_requirements.append(
            "Actor text baseline is degenerate: one unique predicted answer was emitted "
            f"across {eval_samples} samples while baseline accuracy was "
            f"{actor_baseline_accuracy:.2f}%."
        )
    if actor_bridge_enabled:
        if actor_bridge_rate is None or actor_bridge_rate < 100.0:
            missing_requirements.append(
                "Actor semantic bridge decode is enabled but did not extract an answer "
                f"for every sample ({0.0 if actor_bridge_rate is None else actor_bridge_rate:.2f}%)."
            )
        if actor_bridge_accuracy is None or actor_bridge_accuracy < 100.0:
            missing_requirements.append(
                "Actor semantic bridge decode is enabled but exact-match accuracy is below "
                f"100% ({0.0 if actor_bridge_accuracy is None else actor_bridge_accuracy:.2f}%)."
            )
        if eval_samples > 1 and (
            actor_bridge_unique_count is None or actor_bridge_unique_count <= 1
        ):
            missing_requirements.append(
                "Actor semantic bridge decode predictions are degenerate: fewer than two "
                f"unique answers were decoded across {eval_samples} samples."
            )
    if latent_token_decode_enabled and latent_token_decode_require_ready:
        if latent_token_decode_rate is None or latent_token_decode_rate < 100.0:
            missing_requirements.append(
                "Latent token decoder is enabled but did not extract an answer for every "
                f"sample ({0.0 if latent_token_decode_rate is None else latent_token_decode_rate:.2f}%)."
            )
        if latent_token_decode_accuracy is None or latent_token_decode_accuracy < 100.0:
            missing_requirements.append(
                "Latent token decoder is enabled but exact-match accuracy is below 100% "
                f"({0.0 if latent_token_decode_accuracy is None else latent_token_decode_accuracy:.2f}%)."
            )
        if eval_samples > 1 and (
            latent_token_decode_unique_count is None or latent_token_decode_unique_count <= 1
        ):
            missing_requirements.append(
                "Latent token decoder predictions are degenerate: fewer than two unique "
                f"answers were decoded across {eval_samples} samples."
            )
    if raw_decode_require_ready:
        if raw_decode_rate is None or raw_decode_rate < 100.0:
            missing_requirements.append(
                "Raw actor free decode is required but did not extract an answer for every "
                f"sample ({0.0 if raw_decode_rate is None else raw_decode_rate:.2f}%)."
            )
        if raw_decode_accuracy is None or raw_decode_accuracy < 100.0:
            missing_requirements.append(
                "Raw actor free decode is required but exact-match accuracy is below 100% "
                f"({0.0 if raw_decode_accuracy is None else raw_decode_accuracy:.2f}%)."
            )
        if eval_samples > 1 and (
            raw_decode_unique_count is None or raw_decode_unique_count <= 1
        ):
            missing_requirements.append(
                "Raw actor free decode predictions are degenerate: fewer than two unique "
                f"answers were decoded across {eval_samples} samples."
            )

    return {
        "phase": "training_smoke",
        "passed": not missing_requirements,
        "loss_entry_count": len(loss_entries),
        "max_loss": None if not finite_losses else max(finite_losses),
        "final_loss": final_loss,
        "initial_heldout_exact_match_accuracy": initial_eval.get(
            "heldout_exact_match_accuracy"
        ),
        "initial_heldout_unique_predicted_answer_count": initial_eval.get(
            "heldout_unique_predicted_answer_count"
        ),
        "initial_heldout_eval_diagnostics": initial_eval.get("heldout_eval_diagnostics"),
        "final_heldout_exact_match_accuracy": final_accuracy,
        "final_heldout_answer_extraction_rate_percentage": final_extraction_rate,
        "final_heldout_decode_answer_extraction_rate_percentage": final_eval.get(
            "heldout_decode_answer_extraction_rate_percentage"
        ),
        "final_heldout_raw_decode_exact_match_accuracy": final_eval.get(
            "heldout_raw_decode_exact_match_accuracy"
        ),
        "final_heldout_raw_decode_require_ready": raw_decode_require_ready,
        "final_heldout_raw_decode_answer_extraction_rate_percentage": final_eval.get(
            "heldout_raw_decode_answer_extraction_rate_percentage"
        ),
        "final_heldout_raw_decode_unique_predicted_answer_count": final_eval.get(
            "heldout_raw_decode_unique_predicted_answer_count"
        ),
        "final_heldout_actor_semantic_bridge_decode_accuracy": actor_bridge_accuracy,
        "final_heldout_actor_semantic_bridge_decode_enabled": actor_bridge_enabled,
        "final_heldout_actor_semantic_bridge_decode_surface_rate_percentage": final_eval.get(
            "heldout_actor_semantic_bridge_decode_surface_rate_percentage"
        ),
        "final_heldout_actor_semantic_bridge_decode_answer_extraction_rate_percentage": (
            actor_bridge_rate
        ),
        "final_heldout_actor_semantic_bridge_decode_unique_predicted_answer_count": (
            actor_bridge_unique_count
        ),
        "actor_semantic_bridge_decoder_ready": (
            actor_bridge_enabled
            and actor_bridge_accuracy is not None
            and actor_bridge_accuracy >= 100.0
            and actor_bridge_rate is not None
            and actor_bridge_rate >= 100.0
            and (
                eval_samples <= 1
                or (
                    actor_bridge_unique_count is not None
                    and actor_bridge_unique_count > 1
                )
            )
        ),
        "raw_actor_free_decoder_ready": (
            raw_decode_require_ready
            and raw_decode_accuracy is not None
            and raw_decode_accuracy >= 100.0
            and raw_decode_rate is not None
            and raw_decode_rate >= 100.0
            and (
                eval_samples <= 1
                or (
                    raw_decode_unique_count is not None
                    and raw_decode_unique_count > 1
                )
            )
        ),
        "final_heldout_latent_token_decode_accuracy": latent_token_decode_accuracy,
        "final_heldout_latent_token_decode_enabled": latent_token_decode_enabled,
        "final_heldout_latent_token_decode_require_ready": latent_token_decode_require_ready,
        "final_heldout_latent_token_decode_surface_rate_percentage": final_eval.get(
            "heldout_latent_token_decode_surface_rate_percentage"
        ),
        "final_heldout_latent_token_decode_answer_extraction_rate_percentage": (
            latent_token_decode_rate
        ),
        "final_heldout_latent_token_decode_unique_predicted_answer_count": (
            latent_token_decode_unique_count
        ),
        "latent_token_decoder_ready": (
            latent_token_decode_enabled
            and latent_token_decode_accuracy is not None
            and latent_token_decode_accuracy >= 100.0
            and latent_token_decode_rate is not None
            and latent_token_decode_rate >= 100.0
            and (
                eval_samples <= 1
                or (
                    latent_token_decode_unique_count is not None
                    and latent_token_decode_unique_count > 1
                )
            )
        ),
        "final_heldout_latent_semantic_readout_accuracy": final_eval.get(
            "heldout_latent_semantic_readout_accuracy"
        ),
        "final_heldout_latent_semantic_readout_rate_percentage": final_eval.get(
            "heldout_latent_semantic_readout_rate_percentage"
        ),
        "final_heldout_latent_semantic_readout_unique_predicted_answer_count": final_eval.get(
            "heldout_latent_semantic_readout_unique_predicted_answer_count"
        ),
        "final_heldout_candidate_fallback_rate_percentage": final_eval.get(
            "heldout_candidate_fallback_rate_percentage"
        ),
        "final_heldout_latent_candidate_accuracy": final_eval.get(
            "heldout_latent_candidate_accuracy"
        ),
        "final_heldout_latent_candidate_unique_predicted_answer_count": final_eval.get(
            "heldout_latent_candidate_unique_predicted_answer_count"
        ),
        "latent_candidate_fallback_ready": (
            latent_candidate_accuracy is not None
            and latent_candidate_accuracy >= 100.0
            and (
                eval_samples <= 1
                or (
                    latent_candidate_unique_count is not None
                    and latent_candidate_unique_count > 1
                )
            )
        ),
        "final_heldout_latent_probe_accuracy": latent_probe_accuracy,
        "final_heldout_latent_probe_unique_predicted_answer_count": latent_probe_unique_count,
        "final_heldout_latent_sequence_decoder_token_accuracy": final_eval.get(
            "heldout_latent_sequence_decoder_token_accuracy"
        ),
        "final_heldout_latent_sequence_decoder_sequence_accuracy": latent_sequence_accuracy,
        "final_heldout_latent_sequence_decoder_length_accuracy": final_eval.get(
            "heldout_latent_sequence_decoder_length_accuracy"
        ),
        "final_heldout_latent_sequence_decoder_unique_predicted_answer_count": (
            latent_sequence_unique_count
        ),
        "final_heldout_latent_generation_smoke_ready": final_eval.get(
            "heldout_latent_generation_smoke_ready"
        ),
        "final_heldout_latent_generation_smoke_skipped_count": final_eval.get(
            "heldout_latent_generation_smoke_skipped_count"
        ),
        "final_heldout_latent_generation_sequence_accuracy_threshold": final_eval.get(
            "heldout_latent_generation_sequence_accuracy_threshold"
        ),
        "final_heldout_latent_first_token_accuracy": final_eval.get(
            "heldout_latent_first_token_accuracy"
        ),
        "final_heldout_latent_first_token_rank_mean": final_eval.get(
            "heldout_latent_first_token_rank_mean"
        ),
        "latent_probe_ready": (
            latent_probe_accuracy is not None
            and latent_probe_accuracy >= 100.0
            and latent_probe_unique_count is not None
            and latent_probe_unique_count > 1
        ),
        "latent_sequence_decoder_ready": (
            latent_sequence_accuracy is not None
            and latent_sequence_accuracy >= float(
                final_eval.get("heldout_latent_generation_sequence_accuracy_threshold", 95.0) or 95.0
            )
            and latent_sequence_unique_count is not None
            and latent_sequence_unique_count > 1
        ),
        "final_heldout_extraction_failure_count": final_eval.get(
            "heldout_extraction_failure_count"
        ),
        "final_heldout_unique_predicted_answer_count": unique_prediction_count,
        "final_heldout_degenerate_prediction": degenerate_prediction,
        "final_heldout_actor_text_baseline_accuracy": actor_baseline_accuracy,
        "final_heldout_actor_text_baseline_evaluated": actor_baseline_evaluated,
        "final_heldout_actor_text_baseline_answer_extraction_rate_percentage": final_eval.get(
            "heldout_actor_text_baseline_answer_extraction_rate_percentage"
        ),
        "final_heldout_actor_text_baseline_unique_predicted_answer_count": actor_baseline_unique_count,
        "final_heldout_actor_text_baseline_degenerate_prediction": actor_text_baseline_degenerate,
        "final_heldout_actor_text_baseline_candidate_accuracy": final_eval.get(
            "heldout_actor_text_baseline_candidate_accuracy"
        ),
        "final_heldout_actor_text_baseline_candidate_unique_predicted_answer_count": final_eval.get(
            "heldout_actor_text_baseline_candidate_unique_predicted_answer_count"
        ),
        "latent_training_ready": not missing_requirements,
        "final_heldout_answer_perplexity": final_perplexity,
        "final_heldout_answer_perplexity_surface": final_eval.get(
            "heldout_answer_perplexity_surface"
        ),
        "heldout_eval_diagnostics": final_eval.get("heldout_eval_diagnostics"),
        "heldout_eval_samples": eval_samples,
        "missing_requirements": missing_requirements,
    }


def build_ode_scaling_report(
    summary_rows: Sequence[dict[str, Any]],
    *,
    method: str = "hybrid_hl_mas",
    accuracy_tolerance: float = 1.0,
) -> dict[str, Any]:
    relevant_rows = [
        row for row in summary_rows if str(row.get("method")) == method
    ]
    if len(relevant_rows) < 2:
        return {
            "method": method,
            "available": False,
            "missing_requirements": ["Need at least two latent-step settings to build an ODE scaling report."],
        }

    ordered_rows = sorted(relevant_rows, key=lambda row: int(row["compression_steps"]))
    x_values = [math.log2(max(1, int(row["compression_steps"]))) for row in ordered_rows]
    y_values = [float(row["accuracy_percentage"]) for row in ordered_rows]
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    variance = sum((x - x_mean) ** 2 for x in x_values)
    slope = 0.0 if variance == 0 else sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values)) / variance
    best_accuracy = max(y_values)
    best_row = max(ordered_rows, key=lambda row: float(row["accuracy_percentage"]))
    near_optimal_row = min(
        (
            row for row in ordered_rows
            if best_accuracy - float(row["accuracy_percentage"]) <= accuracy_tolerance
        ),
        key=lambda row: int(row["compression_steps"]),
    )
    return {
        "method": method,
        "available": True,
        "slope_accuracy_vs_log2_steps": slope,
        "best_accuracy_percentage": float(best_row["accuracy_percentage"]),
        "best_answer_perplexity": best_row.get("answer_perplexity"),
        "best_step_count": int(best_row["compression_steps"]),
        "near_optimal_step_count": int(near_optimal_row["compression_steps"]),
        "near_optimal_answer_perplexity": near_optimal_row.get("answer_perplexity"),
        "rows": [
            {
                "compression_steps": int(row["compression_steps"]),
                "accuracy_percentage": float(row["accuracy_percentage"]),
                "answer_perplexity": row.get("answer_perplexity"),
                "average_latency_seconds": float(row["average_latency_seconds"]),
            }
            for row in ordered_rows
        ],
    }
