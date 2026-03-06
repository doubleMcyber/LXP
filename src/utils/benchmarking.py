from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional, Sequence


STANDARD_SAMPLE_FIELDS: list[str] = [
    "evaluation_surface",
    "suite",
    "method",
    "dataset",
    "repetition",
    "sample_index",
    "agent_a_model",
    "agent_b_model",
    "model_pair",
    "torch_dtype",
    "compression_steps",
    "semantic_anchor_count",
    "reasoning_layer_weights",
    "alignment_mode",
    "kv_cache_transferred",
    "prompt",
    "target_answer",
    "predicted_answer",
    "decoded_text",
    "generated_tokens",
    "correct",
    "latency_seconds",
    "pre_alignment_l2_distance",
    "pre_alignment_cosine_distance",
    "post_alignment_l2_distance",
    "post_alignment_cosine_distance",
    "raw_handoff_entropy",
    "handoff_uncertainty",
    "confidence_gate_triggered",
    "fallback_discrete_reasoning_steps",
    "latent_trajectory_steps",
    "total_reasoning_steps",
    "continuous_integration_seconds",
    "error",
]
STANDARD_SUMMARY_FIELDS: list[str] = [
    "evaluation_surface",
    "suite",
    "method",
    "dataset",
    "agent_a_model",
    "agent_b_model",
    "model_pair",
    "torch_dtype",
    "compression_steps",
    "semantic_anchor_count",
    "reasoning_layer_weights",
    "alignment_mode",
    "repetition_count",
    "sample_count",
    "accuracy_percentage",
    "total_latency_seconds",
    "average_latency_seconds",
    "total_generated_tokens",
    "tokens_per_second",
    "mean_pre_alignment_l2_distance",
    "mean_pre_alignment_cosine_distance",
    "mean_post_alignment_l2_distance",
    "mean_post_alignment_cosine_distance",
    "mean_raw_handoff_entropy",
    "mean_handoff_uncertainty",
    "confidence_gate_trigger_rate_percentage",
    "cache_transfer_rate_percentage",
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
    repetition: int,
    compression_steps: Optional[int] = None,
    alignment_mode: Optional[str] = None,
    semantic_anchor_count: Optional[int] = None,
    reasoning_layer_weights: Optional[Sequence[float] | str] = None,
) -> dict[str, Any]:
    if compression_steps is None:
        compression_steps = int(
            _cfg_value(cfg, "training.compressed_steps", _cfg_value(cfg, "latent_steps", 0))
        )
    if semantic_anchor_count is None:
        semantic_anchor_count = int(_cfg_value(cfg, "alignment.semantic_anchor_count", 0))
    if reasoning_layer_weights is None:
        reasoning_layer_weights = _cfg_value(cfg, "alignment.reasoning_layer_weights", ())

    agent_a_model = str(_cfg_value(cfg, "agent_a_model", ""))
    agent_b_model = str(_cfg_value(cfg, "agent_b_model", ""))
    return {
        "evaluation_surface": evaluation_surface,
        "suite": suite,
        "method": method,
        "dataset": dataset,
        "repetition": int(repetition),
        "agent_a_model": agent_a_model,
        "agent_b_model": agent_b_model,
        "model_pair": f"{agent_a_model} -> {agent_b_model}",
        "torch_dtype": str(_cfg_value(cfg, "torch_dtype", "")),
        "compression_steps": int(compression_steps),
        "semantic_anchor_count": int(semantic_anchor_count),
        "reasoning_layer_weights": serialize_reasoning_layer_weights(reasoning_layer_weights),
        "alignment_mode": "" if alignment_mode is None else str(alignment_mode),
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


def aggregate_standard_rows(
    rows: Sequence[dict[str, Any]],
    *,
    group_fields: Optional[Sequence[str]] = None,
) -> list[dict[str, Any]]:
    if group_fields is None:
        group_fields = (
            "evaluation_surface",
            "suite",
            "method",
            "dataset",
            "agent_a_model",
            "agent_b_model",
            "model_pair",
            "torch_dtype",
            "compression_steps",
            "semantic_anchor_count",
            "reasoning_layer_weights",
            "alignment_mode",
        )

    grouped_rows: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped_rows.setdefault(tuple(row.get(field) for field in group_fields), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, group_rows in grouped_rows.items():
        row_template = {field: value for field, value in zip(group_fields, key)}
        sample_count = len(group_rows)
        correct_count = sum(1 for row in group_rows if bool(row.get("correct")))
        error_count = sum(1 for row in group_rows if bool(row.get("error")))
        total_latency_seconds = sum(float(row.get("latency_seconds", 0.0)) for row in group_rows)
        total_generated_tokens = sum(int(row.get("generated_tokens", 0) or 0) for row in group_rows)
        kv_transfer_rows = [
            bool(row["kv_cache_transferred"])
            for row in group_rows
            if row.get("kv_cache_transferred") is not None and row.get("kv_cache_transferred") != ""
        ]
        confidence_gate_rows = [
            bool(row["confidence_gate_triggered"])
            for row in group_rows
            if row.get("confidence_gate_triggered") is not None and row.get("confidence_gate_triggered") != ""
        ]
        non_empty_rows = [
            1 for row in group_rows if str(row.get("decoded_text", "")).strip()
        ]
        repetitions = {int(row.get("repetition", 0)) for row in group_rows}

        summary_rows.append(
            {
                **row_template,
                "repetition_count": len(repetitions),
                "sample_count": sample_count,
                "accuracy_percentage": (100.0 * correct_count / sample_count) if sample_count else 0.0,
                "total_latency_seconds": total_latency_seconds,
                "average_latency_seconds": (total_latency_seconds / sample_count) if sample_count else 0.0,
                "total_generated_tokens": total_generated_tokens,
                "tokens_per_second": (
                    total_generated_tokens / total_latency_seconds if total_latency_seconds > 0 else 0.0
                ),
                "mean_pre_alignment_l2_distance": _mean_or_none(group_rows, "pre_alignment_l2_distance"),
                "mean_pre_alignment_cosine_distance": _mean_or_none(group_rows, "pre_alignment_cosine_distance"),
                "mean_post_alignment_l2_distance": _mean_or_none(group_rows, "post_alignment_l2_distance"),
                "mean_post_alignment_cosine_distance": _mean_or_none(group_rows, "post_alignment_cosine_distance"),
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
                "non_empty_decoded_rate_percentage": (
                    100.0 * len(non_empty_rows) / sample_count if sample_count else 0.0
                ),
                "failure_rate_percentage": (
                    100.0 * (sample_count - correct_count) / sample_count if sample_count else 0.0
                ),
                "error_count": error_count,
            }
        )

    return summary_rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2))


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
) -> dict[str, Any]:
    summary_by_method = {str(row["method"]): row for row in summary_rows}
    prompt_local = summary_by_method.get("prompt_local_latent")
    global_anchor = summary_by_method.get("global_anchor_latent")
    hybrid = summary_by_method.get("hybrid_hl_mas")
    missing_requirements: list[str] = []

    if prompt_local is None or global_anchor is None:
        missing_requirements.append(
            "Prompt-local and global-anchor latent baselines are both required for phase 3 comparison."
        )

    q_global_beats_prompt_local: Optional[bool] = None
    if prompt_local is not None and global_anchor is not None:
        q_global_beats_prompt_local = float(global_anchor["accuracy_percentage"]) >= float(
            prompt_local["accuracy_percentage"]
        )
        if require_q_global_beats_prompt_local and not q_global_beats_prompt_local:
            missing_requirements.append("Global semantic-anchor alignment does not beat prompt-local alignment.")

    hybrid_available = hybrid is not None
    if not hybrid_available:
        missing_requirements.append("Hybrid ODE benchmark row is missing.")

    return {
        "phase": "phase_3",
        "passed": not missing_requirements,
        "require_q_global_beats_prompt_local": require_q_global_beats_prompt_local,
        "q_global_beats_prompt_local": q_global_beats_prompt_local,
        "prompt_local_accuracy_percentage": None if prompt_local is None else float(prompt_local["accuracy_percentage"]),
        "global_anchor_accuracy_percentage": None if global_anchor is None else float(global_anchor["accuracy_percentage"]),
        "hybrid_accuracy_percentage": None if hybrid is None else float(hybrid["accuracy_percentage"]),
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
) -> dict[str, Any]:
    heldout_entries = [entry for entry in history if "heldout_exact_match_accuracy" in entry]
    final_heldout_accuracy = None if not heldout_entries else float(heldout_entries[-1]["heldout_exact_match_accuracy"])
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

    return {
        "evaluation_surface": "run_training",
        "method": "stage2_compression_training",
        "phase": "phase_2",
        "passed": not missing_requirements,
        "dataset": dataset_name,
        "training_mode": training_mode,
        "agent_a_model": str(_cfg_value(cfg, "agent_a_model", "")),
        "agent_b_model": str(_cfg_value(cfg, "agent_b_model", "")),
        "torch_dtype": str(_cfg_value(cfg, "torch_dtype", "")),
        "compression_steps": int(_cfg_value(cfg, "training.compressed_steps", 0)),
        "seed_count": seed_count,
        "required_seed_count": required_seed_count,
        "min_accuracy_retention_ratio": min_accuracy_retention_ratio,
        "baseline_accuracy_percentage": baseline_accuracy_percentage,
        "final_heldout_exact_match_accuracy": final_heldout_accuracy,
        "accuracy_retention_ratio": accuracy_retention_ratio,
        "alignment_mode": alignment_context.get("alignment_mode"),
        "semantic_anchor_count": alignment_context.get("semantic_anchor_count"),
        "reasoning_layer_weights": serialize_reasoning_layer_weights(
            alignment_context.get("reasoning_layer_weights")
        ),
        "model_pair": f"{_cfg_value(cfg, 'agent_a_model', '')} -> {_cfg_value(cfg, 'agent_b_model', '')}",
        "missing_requirements": missing_requirements,
    }
