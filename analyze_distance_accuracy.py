from __future__ import annotations

import argparse
import csv
import math
import re
import time
from pathlib import Path
from typing import Any, Optional

import torch
from omegaconf import OmegaConf

from latent_pipeline import (
    _aggregate_hidden_layers,
    _collect_single_token_hidden_states,
    _cosine_distance,
    _get_pipeline_state,
    _normalized_l2_distance,
    _select_hidden_layers,
    compute_semantic_alignment_from_token_ids,
    initialize_hybrid_pipeline,
    run_hybrid_pipeline,
)
from src.data.loader import get_dataloader, pick_field
from src.utils.alignment import apply_alignment
from src.utils.benchmarking import (
    build_distance_calibration_report,
    build_standard_row_base,
    write_json,
)
from src.utils.metrics import extract_boxed_text, normalize_answer

DEFAULT_DATASET = "gsm8k"
DEFAULT_LIMIT = 100
DEFAULT_SPLIT = None
DEFAULT_ANCHOR_COUNTS = (100, 250, 500)
DEFAULT_WEIGHT_SCHEDULES = (
    "uniform",
    "linear_deep_bias",
    "strong_deep_bias",
)
DEFAULT_ALIGNMENT_STRATEGIES = (
    "orthogonal",
    "ridge",
    "hybrid_affine",
    "hybrid_affine_plus_calibration",
)
DEFAULT_CONTROLS = (
    "full_anchor",
    "heldout_anchor_generalization",
    "anchor_subset_stability",
    "shuffled_anchor_control",
)
DEFAULT_SAMPLES_OUTPUT = Path("outputs/distance_accuracy_samples.csv")
DEFAULT_SUMMARY_OUTPUT = Path("outputs/distance_accuracy_summary.csv")
DEFAULT_REPORT_OUTPUT = Path("outputs/distance_accuracy_report.json")
GSM8K_FINAL_ANSWER_REGEX = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")
NUMERIC_ANSWER_REGEX = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_WEIGHT_SCHEDULES: dict[str, tuple[float, float, float]] = {
    "uniform": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    "linear_deep_bias": (1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0),
    "strong_deep_bias": (0.125, 0.25, 0.625),
}


def _load_cfg() -> Any:
    return OmegaConf.load(Path(__file__).resolve().parent / "configs" / "main.yaml")


def _clone_cfg(base_cfg: Any) -> Any:
    return OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))


def _dataset_cfg(cfg: Any, dataset_name: str) -> Any:
    return getattr(getattr(cfg, "datasets", None), dataset_name, None)


def _validation_size(cfg: Any, dataset_name: str) -> int | None:
    dataset_cfg = _dataset_cfg(cfg, dataset_name)
    if dataset_cfg is None:
        return None
    raw_value = getattr(dataset_cfg, "validation_size", None)
    return None if raw_value is None else int(raw_value)


def _default_split_for_dataset(dataset_name: str) -> str:
    if dataset_name == "gsm8k":
        return "validation"
    return "test"


def _benchmark_cfg(cfg: Any) -> Any:
    return getattr(cfg, "benchmark", None)


def _q_generalization_cfg(cfg: Any) -> Any:
    return getattr(_benchmark_cfg(cfg), "q_generalization", None)


def _heldout_anchor_ratio(cfg: Any) -> float:
    q_cfg = _q_generalization_cfg(cfg)
    return float(getattr(q_cfg, "heldout_anchor_ratio", 0.2))


def _bootstrap_count(cfg: Any) -> int:
    q_cfg = _q_generalization_cfg(cfg)
    return int(getattr(q_cfg, "bootstrap_count", 5))


def _solver_alignment_strategy_name(cfg: Any, requested_name: str) -> str:
    configured_alignment = getattr(cfg, "alignment", None)
    configured_strategy = str(
        getattr(configured_alignment, "strategy", "hybrid_affine")
    ).strip()
    if configured_strategy in {"orthogonal", "ridge", "hybrid_affine"}:
        return configured_strategy
    if requested_name == "hybrid_affine_plus_calibration":
        return "hybrid_affine"
    if requested_name in {"orthogonal", "ridge", "hybrid_affine"}:
        return requested_name
    raise ValueError(
        f"Unsupported solver alignment strategy {requested_name!r}. "
        "Expected a benchmark variant that resolves to 'orthogonal', 'ridge', or 'hybrid_affine'."
    )


def _parse_int_list(raw_values: str) -> list[int]:
    values: list[int] = []
    for part in raw_values.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        values.append(int(stripped))
    if not values:
        raise ValueError("At least one integer value is required")
    return values


def _parse_name_list(raw_values: str) -> list[str]:
    values = [part.strip() for part in raw_values.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one schedule name is required")
    return values


def _load_samples(dataset_name: str, split: str, limit: int, validation_size: Optional[int]) -> Any:
    return get_dataloader(
        dataset_name,
        split=split,
        limit=limit,
        validation_size=validation_size,
    )


def _extract_gsm8k_target_answer(text: str) -> Optional[str]:
    match = GSM8K_FINAL_ANSWER_REGEX.search(text)
    return None if match is None else match.group(1)


def _extract_gsm8k_predicted_answer(text: str) -> Optional[str]:
    matches = NUMERIC_ANSWER_REGEX.findall(text)
    if not matches:
        return None
    return matches[-1]


def _normalize_numeric_answer(answer: Optional[str]) -> Optional[str]:
    normalized = normalize_answer(answer)
    if normalized is None:
        return None
    return normalized.replace(",", "")


def _target_answer(dataset_name: str, row: Any) -> Optional[str]:
    if dataset_name == "gsm8k":
        return _extract_gsm8k_target_answer(pick_field(row, ("answer", "solution")))
    return extract_boxed_text(pick_field(row, ("solution", "answer")))


def _predicted_answer(dataset_name: str, decoded_text: str) -> Optional[str]:
    if dataset_name == "gsm8k":
        return _extract_gsm8k_predicted_answer(decoded_text)
    return extract_boxed_text(decoded_text)


def _answers_match(
    dataset_name: str,
    predicted_answer: Optional[str],
    target_answer: Optional[str],
) -> bool:
    if dataset_name == "gsm8k":
        return _normalize_numeric_answer(predicted_answer) == _normalize_numeric_answer(target_answer)
    return normalize_answer(predicted_answer) == normalize_answer(target_answer)


def _reasoning_layer_weights_for_schedule(schedule_name: str) -> tuple[float, float, float]:
    schedule = _WEIGHT_SCHEDULES.get(schedule_name)
    if schedule is None:
        raise ValueError(
            f"Unsupported weight schedule {schedule_name!r}. "
            f"Supported: {', '.join(sorted(_WEIGHT_SCHEDULES))}"
        )
    return schedule


def _config_id(anchor_count: int, weight_schedule_name: str, alignment_strategy: str) -> str:
    return f"anchors_{anchor_count}_{weight_schedule_name}_{alignment_strategy}"


def _assign_distance_deciles(rows: list[dict[str, Any]]) -> None:
    valid_rows = [
        row for row in rows if row.get("post_alignment_l2_distance") is not None
    ]
    if not valid_rows:
        for row in rows:
            row["distance_decile"] = ""
        return

    sorted_rows = sorted(valid_rows, key=lambda row: float(row["post_alignment_l2_distance"]))
    bucket_count = min(10, len(sorted_rows))
    for index, row in enumerate(sorted_rows):
        bucket = min(bucket_count - 1, index * bucket_count // len(sorted_rows))
        row["distance_decile"] = bucket + 1

    for row in rows:
        row.setdefault("distance_decile", "")


def _breaking_point_decile(decile_rows: list[dict[str, Any]]) -> Optional[int]:
    if not decile_rows:
        return None
    best_accuracy = max(float(row["accuracy_percentage"]) for row in decile_rows)
    for row in sorted(decile_rows, key=lambda item: int(item["distance_decile"])):
        if float(row["accuracy_percentage"]) <= best_accuracy - 10.0:
            return int(row["distance_decile"])
    return None


def _index_tensor(values: torch.Tensor, indices: torch.LongTensor) -> torch.Tensor:
    return values.index_select(0, indices.to(device=values.device, dtype=torch.long))


def _split_anchor_indices(
    anchor_count: int,
    *,
    heldout_ratio: float,
    seed: int,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    if anchor_count < 2:
        raise ValueError("Need at least two anchors for held-out generalization")
    heldout_count = max(1, int(round(anchor_count * heldout_ratio)))
    heldout_count = min(anchor_count - 1, heldout_count)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    permutation = torch.randperm(anchor_count, generator=generator)
    heldout_indices = permutation[:heldout_count]
    train_indices = permutation[heldout_count:]
    return train_indices, heldout_indices


def _normalized_frobenius_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat = left.float().reshape(-1)
    right_flat = right.float().reshape(-1)
    denominator = max(float(torch.linalg.vector_norm(right_flat).item()), 1e-8)
    return float(torch.linalg.vector_norm(left_flat - right_flat).item() / denominator)


def _pairwise_mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _standard_deviation(values: list[float]) -> Optional[float]:
    if len(values) < 2:
        return 0.0 if values else None
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _anchor_eval_metrics(
    *,
    state: dict[str, Any],
    alignment_state: dict[str, Any] | torch.Tensor,
    sender_anchor_ids: torch.LongTensor,
    receiver_anchor_ids: torch.LongTensor,
) -> dict[str, float]:
    reasoning_layer_indices = tuple(state["global_reasoning_layer_indices"])
    reasoning_layer_weights = tuple(state["global_reasoning_layer_weights"])
    agent_a = state["agent_a"]
    agent_b = state["agent_b"]
    sender_hidden_states = _collect_single_token_hidden_states(
        agent_a,
        sender_anchor_ids,
        next(agent_a.parameters()).device,
    )
    receiver_hidden_states = _collect_single_token_hidden_states(
        agent_b,
        receiver_anchor_ids,
        next(agent_b.parameters()).device,
    )
    sender_consensus = _aggregate_hidden_layers(
        _select_hidden_layers(sender_hidden_states, reasoning_layer_indices),
        reasoning_layer_weights,
    )
    receiver_consensus = _aggregate_hidden_layers(
        _select_hidden_layers(receiver_hidden_states, reasoning_layer_indices),
        reasoning_layer_weights,
    ).to(device=sender_consensus.device, dtype=sender_consensus.dtype)
    mapped_sender = apply_alignment(sender_consensus, alignment_state)
    return {
        "anchor_eval_post_alignment_l2_distance": float(
            _normalized_l2_distance(mapped_sender, receiver_consensus).mean().detach().cpu().item()
        ),
        "anchor_eval_post_alignment_cosine_distance": float(
            _cosine_distance(mapped_sender, receiver_consensus).mean().detach().cpu().item()
        ),
    }


def _aggregate_group_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sample_count = len(rows)
    correct_count = sum(1 for row in rows if row["correct"])
    total_latency = sum(float(row["latency_seconds"]) for row in rows)
    total_answer_tokens = sum(int(row.get("answer_token_count", 0) or 0) for row in rows)
    total_answer_nll = sum(
        float(row["answer_nll"]) * int(row.get("answer_token_count", 0) or 0)
        for row in rows
        if row.get("answer_nll") is not None
    )

    def _mean(field: str) -> Optional[float]:
        values = [float(row[field]) for row in rows if row.get(field) is not None and row.get(field) != ""]
        if not values:
            return None
        return sum(values) / len(values)

    def _unique(field: str) -> str:
        values = sorted(
            {
                str(row.get(field))
                for row in rows
                if row.get(field) is not None and row.get(field) != ""
            }
        )
        return ",".join(values)

    return {
        "sample_count": sample_count,
        "handoff_status": _unique("handoff_status"),
        "handoff_surface": _unique("handoff_surface"),
        "kv_cache_status": _unique("kv_cache_status"),
        "kv_cache_reason": _unique("kv_cache_reason"),
        "decode_status": _unique("decode_status"),
        "accuracy_percentage": (100.0 * correct_count / sample_count) if sample_count else 0.0,
        "average_latency_seconds": total_latency / sample_count if sample_count else 0.0,
        "mean_answer_nll": (total_answer_nll / total_answer_tokens) if total_answer_tokens > 0 else None,
        "answer_perplexity": (
            math.exp(total_answer_nll / total_answer_tokens) if total_answer_tokens > 0 else None
        ),
        "mean_pre_alignment_l2_distance": _mean("pre_alignment_l2_distance"),
        "mean_post_alignment_l2_distance": _mean("post_alignment_l2_distance"),
        "mean_post_alignment_cosine_distance": _mean("post_alignment_cosine_distance"),
        "mean_alignment_residual_norm_ratio": _mean("alignment_residual_norm_ratio"),
        "mean_alignment_bias_norm": _mean("alignment_bias_norm"),
        "prompt_calibration_rate_percentage": (
            100.0 * sum(1 for row in rows if bool(row.get("prompt_calibration_enabled"))) / sample_count
            if sample_count
            else 0.0
        ),
        "mean_prompt_calibration_bias_norm": _mean("prompt_calibration_bias_norm"),
        "mean_anchor_eval_post_alignment_l2_distance": _mean("anchor_eval_post_alignment_l2_distance"),
        "mean_anchor_eval_post_alignment_cosine_distance": _mean("anchor_eval_post_alignment_cosine_distance"),
        "mean_raw_handoff_entropy": _mean("raw_handoff_entropy"),
        "confidence_gate_trigger_rate_percentage": (
            100.0 * sum(1 for row in rows if bool(row["confidence_gate_triggered"])) / sample_count
            if sample_count else 0.0
        ),
        "failure_rate_percentage": (100.0 * (sample_count - correct_count) / sample_count)
        if sample_count
        else 0.0,
        "error_count": sum(1 for row in rows if row["error"]),
        "global_alignment_cache_hit_rate_percentage": (
            100.0 * sum(1 for row in rows if bool(row["global_alignment_cache_hit"])) / sample_count
            if sample_count else 0.0
        ),
        "explicit_status_rate_percentage": (
            100.0
            * sum(
                1
                for row in rows
                if row.get("handoff_status") not in (None, "")
                and row.get("kv_cache_status") not in (None, "")
                and row.get("decode_status") not in (None, "")
            )
            / sample_count
            if sample_count
            else 0.0
        ),
    }


def _summarize_rows(
    rows: list[dict[str, Any]],
    *,
    bootstrap_frobenius: dict[str, float],
) -> list[dict[str, Any]]:
    grouped_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped_rows.setdefault(
            (
                str(row["config_id"]),
                str(row["control_name"]),
                str(row["bootstrap_index"]),
            ),
            [],
        ).append(row)

    summary_rows: list[dict[str, Any]] = []
    full_anchor_overall_rows: list[dict[str, Any]] = []
    for (config_id, control_name, bootstrap_index), group_rows in grouped_rows.items():
        base_row = group_rows[0]
        aggregate = _aggregate_group_rows(group_rows)
        summary_row = {
            "row_type": "overall",
            "report_schema_version": base_row.get("report_schema_version", 2),
            "evaluation_surface": base_row["evaluation_surface"],
            "suite": base_row["suite"],
            "method": base_row["method"],
            "config_id": config_id,
            "control_name": control_name,
            "bootstrap_index": bootstrap_index,
            "dataset": base_row["dataset"],
            "dataset_split": base_row["dataset_split"],
            "agent_a_model": base_row["agent_a_model"],
            "agent_b_model": base_row["agent_b_model"],
            "model_pair": base_row["model_pair"],
            "torch_dtype": base_row["torch_dtype"],
            "seed": base_row["seed"],
            "compression_steps": base_row["compression_steps"],
            "semantic_anchor_count": base_row["semantic_anchor_count"],
            "weight_schedule_name": base_row["weight_schedule_name"],
            "reasoning_layer_weights": base_row["reasoning_layer_weights"],
            "alignment_mode": base_row["alignment_mode"],
            "alignment_strategy": base_row.get("alignment_strategy"),
            "q_frobenius_distance_to_full": bootstrap_frobenius.get(
                f"{config_id}:{control_name}:{bootstrap_index}"
            ),
            **aggregate,
            "selected_default": False,
        }
        summary_rows.append(summary_row)
        if control_name == "full_anchor":
            full_anchor_overall_rows.append(summary_row)

    if full_anchor_overall_rows:
        selected_default = min(
            full_anchor_overall_rows,
            key=lambda row: (-float(row["accuracy_percentage"]), float(row["average_latency_seconds"])),
        )
        selected_default["selected_default"] = True

    return summary_rows


def _build_q_generalization_report(
    summary_rows: list[dict[str, Any]],
    *,
    bootstrap_qs: dict[str, list[torch.Tensor]],
    bootstrap_singular_values: dict[str, list[torch.Tensor]],
) -> dict[str, Any]:
    overall_rows = [row for row in summary_rows if row["row_type"] == "overall"]
    selected_full_anchor = next(
        (
            row for row in overall_rows
            if row["control_name"] == "full_anchor" and bool(row["selected_default"])
        ),
        None,
    )
    missing_requirements: list[str] = []
    if selected_full_anchor is None:
        return {
            "phase": "q_generalization",
            "passed": False,
            "missing_requirements": ["No selected full-anchor configuration is available."],
        }

    config_id = str(selected_full_anchor["config_id"])
    heldout_row = next(
        (row for row in overall_rows if row["config_id"] == config_id and row["control_name"] == "heldout_anchor_generalization"),
        None,
    )
    shuffled_row = next(
        (row for row in overall_rows if row["config_id"] == config_id and row["control_name"] == "shuffled_anchor_control"),
        None,
    )
    bootstrap_rows = [
        row for row in overall_rows
        if row["config_id"] == config_id and row["control_name"] == "anchor_subset_stability"
    ]

    if heldout_row is None:
        missing_requirements.append("Held-out anchor generalization row is missing.")
    if shuffled_row is None:
        missing_requirements.append("Shuffled-anchor control row is missing.")
    if not bootstrap_rows:
        missing_requirements.append("Bootstrap stability rows are missing.")

    heldout_accuracy_drop = None
    heldout_distance_degradation = None
    shuffled_is_worse = None
    bootstrap_accuracy_std = None
    bootstrap_frobenius_mean = None
    bootstrap_singular_value_std_mean = None

    if heldout_row is not None:
        heldout_accuracy_drop = float(selected_full_anchor["accuracy_percentage"]) - float(heldout_row["accuracy_percentage"])
        full_distance = selected_full_anchor.get("mean_anchor_eval_post_alignment_l2_distance")
        heldout_distance = heldout_row.get("mean_anchor_eval_post_alignment_l2_distance")
        if full_distance not in (None, 0.0) and heldout_distance is not None:
            heldout_distance_degradation = (float(heldout_distance) - float(full_distance)) / float(full_distance)
        if heldout_accuracy_drop is None or heldout_accuracy_drop > 2.0:
            missing_requirements.append("Held-out anchor accuracy drop exceeds 2 absolute points.")
        if heldout_distance_degradation is None or heldout_distance_degradation > 0.10:
            missing_requirements.append("Held-out anchor distance degradation exceeds 10%.")

    if shuffled_row is not None:
        shuffled_is_worse = (
            float(shuffled_row["accuracy_percentage"]) < float(selected_full_anchor["accuracy_percentage"])
            and float(shuffled_row["mean_anchor_eval_post_alignment_l2_distance"])
            > float(selected_full_anchor["mean_anchor_eval_post_alignment_l2_distance"])
        )
        if not shuffled_is_worse:
            missing_requirements.append("Shuffled-anchor control is not clearly worse than full-anchor alignment.")

    if bootstrap_rows:
        bootstrap_accuracy_std = _standard_deviation(
            [float(row["accuracy_percentage"]) for row in bootstrap_rows]
        )
        if bootstrap_accuracy_std is None or bootstrap_accuracy_std > 1.5:
            missing_requirements.append("Bootstrap accuracy variance exceeds 1.5 points.")

    bootstrap_q_values = bootstrap_qs.get(config_id, [])
    if len(bootstrap_q_values) >= 2:
        pairwise_frobenius: list[float] = []
        for left_index in range(len(bootstrap_q_values)):
            for right_index in range(left_index + 1, len(bootstrap_q_values)):
                pairwise_frobenius.append(
                    _normalized_frobenius_distance(
                        bootstrap_q_values[left_index],
                        bootstrap_q_values[right_index],
                    )
                )
        bootstrap_frobenius_mean = _pairwise_mean(pairwise_frobenius)

    singular_value_sets = bootstrap_singular_values.get(config_id, [])
    if singular_value_sets:
        min_width = min(int(values.numel()) for values in singular_value_sets)
        if min_width > 0:
            stacked = torch.stack([values[:min_width].float() for values in singular_value_sets], dim=0)
            bootstrap_singular_value_std_mean = float(stacked.std(dim=0, unbiased=False).mean().item())

    return {
        "phase": "q_generalization",
        "passed": not missing_requirements,
        "selected_default_config_id": config_id,
        "full_anchor_accuracy_percentage": float(selected_full_anchor["accuracy_percentage"]),
        "full_anchor_answer_perplexity": selected_full_anchor.get("answer_perplexity"),
        "heldout_anchor_accuracy_drop": heldout_accuracy_drop,
        "heldout_anchor_distance_degradation": heldout_distance_degradation,
        "bootstrap_accuracy_std": bootstrap_accuracy_std,
        "bootstrap_q_frobenius_mean": bootstrap_frobenius_mean,
        "bootstrap_singular_value_std_mean": bootstrap_singular_value_std_mean,
        "shuffled_anchor_is_worse": shuffled_is_worse,
        "missing_requirements": missing_requirements,
    }


def run_analysis(
    *,
    dataset_name: str,
    dataset_split: Optional[str],
    limit: int,
    anchor_counts: list[int],
    weight_schedule_names: list[str],
    alignment_strategy_names: list[str],
    controls: list[str],
    samples_output_path: Path,
    summary_output_path: Path,
    report_output_path: Path,
    max_new_tokens: Optional[int] = None,
    agent_a_model: Optional[str] = None,
    agent_b_model: Optional[str] = None,
    seed: Optional[int] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    base_cfg = _load_cfg()
    if agent_a_model is not None:
        base_cfg.agent_a_model = str(agent_a_model)
    if agent_b_model is not None:
        base_cfg.agent_b_model = str(agent_b_model)
    if seed is not None:
        base_cfg.seed = int(seed)
    if max_new_tokens is not None:
        base_cfg.max_new_tokens = int(max_new_tokens)

    effective_split = dataset_split or _default_split_for_dataset(dataset_name)
    samples = _load_samples(
        dataset_name,
        effective_split,
        limit,
        _validation_size(base_cfg, dataset_name),
    )
    rows: list[dict[str, Any]] = []
    bootstrap_frobenius: dict[str, float] = {}
    bootstrap_qs: dict[str, list[torch.Tensor]] = {}
    bootstrap_singular_values: dict[str, list[torch.Tensor]] = {}

    for anchor_count in anchor_counts:
        for weight_schedule_name in weight_schedule_names:
            for alignment_strategy_name in alignment_strategy_names:
                schedule_weights = _reasoning_layer_weights_for_schedule(weight_schedule_name)
                step_cfg = _clone_cfg(base_cfg)
                step_cfg.alignment.semantic_anchor_count = int(anchor_count)
                step_cfg.alignment.reasoning_layer_weights = list(schedule_weights)
                if alignment_strategy_name == "hybrid_affine_plus_calibration":
                    step_cfg.alignment.strategy = "hybrid_affine"
                    step_cfg.alignment.prompt_calibration.enabled = True
                else:
                    step_cfg.alignment.strategy = str(alignment_strategy_name)
                    step_cfg.alignment.prompt_calibration.enabled = False
                solver_strategy_name = _solver_alignment_strategy_name(
                    step_cfg,
                    alignment_strategy_name,
                )
                config_id = _config_id(anchor_count, weight_schedule_name, alignment_strategy_name)

                initialize_hybrid_pipeline(step_cfg)
                state = _get_pipeline_state(step_cfg)
                tokenizer_a = state["tokenizer_a"]
                tokenizer_b = state["tokenizer_b"]
                reasoning_layer_indices = tuple(state["global_reasoning_layer_indices"])
                reasoning_layer_weights = tuple(state["global_reasoning_layer_weights"])
                semantic_anchor_strings = tuple(state["semantic_anchor_strings"])
                anchor_ids_a = torch.tensor(
                    state["semantic_anchor_ids_a"],
                    dtype=torch.long,
                )
                anchor_ids_b = torch.tensor(
                    state["semantic_anchor_ids_b"],
                    dtype=torch.long,
                )
                full_alignment = compute_semantic_alignment_from_token_ids(
                    cfg=step_cfg,
                    tokenizer_a=tokenizer_a,
                    tokenizer_b=tokenizer_b,
                    agent_a=state["agent_a"],
                    agent_b=state["agent_b"],
                    sender_anchor_ids=anchor_ids_a,
                    receiver_anchor_ids=anchor_ids_b,
                    reasoning_layer_indices=reasoning_layer_indices,
                    reasoning_layer_weights=reasoning_layer_weights,
                    alignment_mode=solver_strategy_name,
                    semantic_anchor_strings=semantic_anchor_strings,
                )
                full_q = full_alignment["global_alignment_q"]
                primary_train_indices, primary_heldout_indices = _split_anchor_indices(
                    anchor_count,
                    heldout_ratio=_heldout_anchor_ratio(step_cfg),
                    seed=int(getattr(step_cfg, "seed", 0)),
                )
                full_eval_sender_ids = _index_tensor(anchor_ids_a, primary_heldout_indices)
                full_eval_receiver_ids = _index_tensor(anchor_ids_b, primary_heldout_indices)

                control_variants: list[dict[str, Any]] = []
                if "full_anchor" in controls:
                    control_variants.append(
                        {
                            "control_name": "full_anchor",
                            "bootstrap_index": "",
                            "alignment_state": full_alignment,
                            "eval_sender_ids": full_eval_sender_ids,
                            "eval_receiver_ids": full_eval_receiver_ids,
                        }
                    )

                if "heldout_anchor_generalization" in controls:
                    heldout_alignment = compute_semantic_alignment_from_token_ids(
                        cfg=step_cfg,
                        tokenizer_a=tokenizer_a,
                        tokenizer_b=tokenizer_b,
                        agent_a=state["agent_a"],
                        agent_b=state["agent_b"],
                        sender_anchor_ids=_index_tensor(anchor_ids_a, primary_train_indices),
                        receiver_anchor_ids=_index_tensor(anchor_ids_b, primary_train_indices),
                        reasoning_layer_indices=reasoning_layer_indices,
                        reasoning_layer_weights=reasoning_layer_weights,
                        alignment_mode=solver_strategy_name,
                        semantic_anchor_strings=[semantic_anchor_strings[index] for index in primary_train_indices.tolist()],
                    )
                    control_variants.append(
                        {
                            "control_name": "heldout_anchor_generalization",
                            "bootstrap_index": "",
                            "alignment_state": heldout_alignment,
                            "eval_sender_ids": full_eval_sender_ids,
                            "eval_receiver_ids": full_eval_receiver_ids,
                        }
                    )

                if "shuffled_anchor_control" in controls:
                    generator = torch.Generator()
                    generator.manual_seed(int(getattr(step_cfg, "seed", 0)) + 97)
                    shuffled_indices = torch.randperm(anchor_count, generator=generator)
                    shuffled_alignment = compute_semantic_alignment_from_token_ids(
                        cfg=step_cfg,
                        tokenizer_a=tokenizer_a,
                        tokenizer_b=tokenizer_b,
                        agent_a=state["agent_a"],
                        agent_b=state["agent_b"],
                        sender_anchor_ids=anchor_ids_a,
                        receiver_anchor_ids=_index_tensor(anchor_ids_b, shuffled_indices),
                        reasoning_layer_indices=reasoning_layer_indices,
                        reasoning_layer_weights=reasoning_layer_weights,
                        alignment_mode=solver_strategy_name,
                        semantic_anchor_strings=semantic_anchor_strings,
                    )
                    control_variants.append(
                        {
                            "control_name": "shuffled_anchor_control",
                            "bootstrap_index": "",
                            "alignment_state": shuffled_alignment,
                            "eval_sender_ids": full_eval_sender_ids,
                            "eval_receiver_ids": full_eval_receiver_ids,
                        }
                    )

                if "anchor_subset_stability" in controls:
                    bootstrap_qs[config_id] = []
                    bootstrap_singular_values[config_id] = []
                    for bootstrap_index in range(_bootstrap_count(step_cfg)):
                        train_indices, heldout_indices = _split_anchor_indices(
                            anchor_count,
                            heldout_ratio=_heldout_anchor_ratio(step_cfg),
                            seed=int(getattr(step_cfg, "seed", 0)) + bootstrap_index + 1,
                        )
                        bootstrap_alignment = compute_semantic_alignment_from_token_ids(
                            cfg=step_cfg,
                            tokenizer_a=tokenizer_a,
                            tokenizer_b=tokenizer_b,
                            agent_a=state["agent_a"],
                            agent_b=state["agent_b"],
                            sender_anchor_ids=_index_tensor(anchor_ids_a, train_indices),
                            receiver_anchor_ids=_index_tensor(anchor_ids_b, train_indices),
                            reasoning_layer_indices=reasoning_layer_indices,
                            reasoning_layer_weights=reasoning_layer_weights,
                            alignment_mode=solver_strategy_name,
                            semantic_anchor_strings=[semantic_anchor_strings[index] for index in train_indices.tolist()],
                        )
                        bootstrap_q = bootstrap_alignment["global_alignment_q"]
                        bootstrap_qs[config_id].append(bootstrap_q.float())
                        bootstrap_singular_values[config_id].append(
                            bootstrap_alignment["alignment_singular_values"].float()
                        )
                        bootstrap_frobenius[
                            f"{config_id}:anchor_subset_stability:{bootstrap_index}"
                        ] = _normalized_frobenius_distance(bootstrap_q.float(), full_q.float())
                        control_variants.append(
                            {
                                "control_name": "anchor_subset_stability",
                                "bootstrap_index": str(bootstrap_index),
                                "alignment_state": bootstrap_alignment,
                                "eval_sender_ids": _index_tensor(anchor_ids_a, heldout_indices),
                                "eval_receiver_ids": _index_tensor(anchor_ids_b, heldout_indices),
                            }
                        )

                for variant in control_variants:
                    alignment_state = variant["alignment_state"]
                    anchor_metrics = _anchor_eval_metrics(
                        state=state,
                        alignment_state=alignment_state,
                        sender_anchor_ids=variant["eval_sender_ids"],
                        receiver_anchor_ids=variant["eval_receiver_ids"],
                    )
                    print(
                        f"Running {dataset_name}/{effective_split} analysis for {config_id} "
                        f"control={variant['control_name']} bootstrap={variant['bootstrap_index'] or 'na'} "
                        f"on {len(samples)} samples..."
                    )

                    for index, row in enumerate(samples):
                        prompt = pick_field(row, ("question", "problem"))
                        target_answer = _target_answer(dataset_name, row)
                        decoded_text = ""
                        predicted_answer: Optional[str] = None
                        error: Optional[str] = None
                        pipeline_output: dict[str, Any] = {}

                        start = time.perf_counter()
                        try:
                            pipeline_output = run_hybrid_pipeline(
                                step_cfg,
                                prompt=prompt,
                                collect_alignment_metrics=True,
                                target_answer_text=target_answer,
                                alignment_q_override=alignment_state,
                                alignment_mode_override=f"{variant['control_name']}:{alignment_strategy_name}",
                            )
                            decoded_text = str(pipeline_output["decoded_text"])
                            predicted_answer = _predicted_answer(dataset_name, decoded_text)
                        except Exception as exc:  # noqa: BLE001
                            error = str(exc)
                        latency = time.perf_counter() - start

                        row_base = build_standard_row_base(
                            step_cfg,
                            evaluation_surface="analyze_distance_accuracy",
                            suite="phase3_alignment_generalization",
                            method="hybrid_hl_mas",
                            dataset=dataset_name,
                            dataset_split=effective_split,
                            repetition=0,
                            seed=int(getattr(step_cfg, "seed", 0)),
                            compression_steps=int(getattr(step_cfg, "latent_steps", 0)),
                            alignment_mode=str(variant["control_name"]),
                            alignment_strategy=str(alignment_strategy_name),
                            semantic_anchor_count=anchor_count,
                            reasoning_layer_weights=schedule_weights,
                        )
                        rows.append(
                            {
                                **row_base,
                                "config_id": config_id,
                                "control_name": variant["control_name"],
                                "bootstrap_index": variant["bootstrap_index"],
                                "sample_index": index,
                                "weight_schedule_name": weight_schedule_name,
                                "prompt": prompt,
                                "target_answer": target_answer,
                                "predicted_answer": predicted_answer,
                                "decoded_text": decoded_text,
                                "decode_status": pipeline_output.get(
                                    "decode_status",
                                    "decoded" if decoded_text.strip() else "empty_decode",
                                ),
                                "handoff_status": pipeline_output.get("handoff_status"),
                                "handoff_surface": pipeline_output.get("handoff_surface"),
                                "generated_tokens": int(pipeline_output.get("generated_tokens", 0) or 0),
                                "answer_token_count": int(pipeline_output.get("answer_token_count", 0) or 0),
                                "answer_nll": pipeline_output.get("answer_nll"),
                                "answer_perplexity": pipeline_output.get("answer_perplexity"),
                                "correct": _answers_match(dataset_name, predicted_answer, target_answer),
                                "latency_seconds": latency,
                                "pre_alignment_l2_distance": pipeline_output.get("pre_alignment_l2_distance"),
                                "pre_alignment_cosine_distance": pipeline_output.get("pre_alignment_cosine_distance"),
                                "post_alignment_l2_distance": pipeline_output.get("post_alignment_l2_distance"),
                                "post_alignment_cosine_distance": pipeline_output.get("post_alignment_cosine_distance"),
                                "alignment_residual_norm_ratio": pipeline_output.get("alignment_residual_norm_ratio"),
                                "alignment_bias_norm": pipeline_output.get("alignment_bias_norm"),
                                "prompt_calibration_enabled": pipeline_output.get("prompt_calibration_enabled"),
                                "prompt_calibration_bias_norm": pipeline_output.get("prompt_calibration_bias_norm"),
                                "anchor_eval_post_alignment_l2_distance": anchor_metrics["anchor_eval_post_alignment_l2_distance"],
                                "anchor_eval_post_alignment_cosine_distance": anchor_metrics["anchor_eval_post_alignment_cosine_distance"],
                                "raw_handoff_entropy": pipeline_output.get("raw_handoff_entropy"),
                                "handoff_uncertainty": pipeline_output.get("handoff_uncertainty"),
                                "confidence_gate_triggered": pipeline_output.get("confidence_gate_triggered"),
                                "kv_cache_transferred": pipeline_output.get("kv_cache_transferred"),
                                "kv_cache_status": pipeline_output.get("kv_cache_status"),
                                "kv_cache_reason": pipeline_output.get("kv_cache_reason"),
                                "fallback_discrete_reasoning_steps": pipeline_output.get("fallback_discrete_reasoning_steps"),
                                "latent_trajectory_steps": pipeline_output.get("latent_trajectory_steps"),
                                "total_reasoning_steps": pipeline_output.get("total_reasoning_steps"),
                                "continuous_integration_seconds": pipeline_output.get("continuous_integration_seconds"),
                                "global_alignment_cache_hit": pipeline_output.get("global_alignment_cache_hit"),
                                "error": error,
                            }
                        )

    summary_rows = _summarize_rows(rows, bootstrap_frobenius=bootstrap_frobenius)
    full_anchor_rows = [row for row in rows if row["control_name"] == "full_anchor"]
    full_anchor_summary_rows = [row for row in summary_rows if row["control_name"] == "full_anchor"]
    report_payload = {
        "report_schema_version": 2,
        "evaluation_surface": "analyze_distance_accuracy",
        "dataset": dataset_name,
        "dataset_split": effective_split,
        "limit": limit,
        "distance_calibration_report": build_distance_calibration_report(
            full_anchor_rows,
            full_anchor_summary_rows,
        ),
        "q_generalization_report": _build_q_generalization_report(
            summary_rows,
            bootstrap_qs=bootstrap_qs,
            bootstrap_singular_values=bootstrap_singular_values,
        ),
    }

    sample_fieldnames = [
        "report_schema_version",
        "evaluation_surface",
        "suite",
        "method",
        "config_id",
        "control_name",
        "bootstrap_index",
        "dataset",
        "dataset_split",
        "agent_a_model",
        "agent_b_model",
        "model_pair",
        "torch_dtype",
        "seed",
        "compression_steps",
        "repetition",
        "sample_index",
        "semantic_anchor_count",
        "weight_schedule_name",
        "reasoning_layer_weights",
        "alignment_mode",
        "alignment_strategy",
        "prompt",
        "target_answer",
        "predicted_answer",
        "decoded_text",
        "decode_status",
        "handoff_status",
        "handoff_surface",
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
        "anchor_eval_post_alignment_l2_distance",
        "anchor_eval_post_alignment_cosine_distance",
        "raw_handoff_entropy",
        "handoff_uncertainty",
        "confidence_gate_triggered",
        "kv_cache_transferred",
        "kv_cache_status",
        "kv_cache_reason",
        "fallback_discrete_reasoning_steps",
        "latent_trajectory_steps",
        "total_reasoning_steps",
        "continuous_integration_seconds",
        "global_alignment_cache_hit",
        "error",
    ]
    summary_fieldnames = [
        "row_type",
        "report_schema_version",
        "evaluation_surface",
        "suite",
        "method",
        "config_id",
        "control_name",
        "bootstrap_index",
        "dataset",
        "dataset_split",
        "agent_a_model",
        "agent_b_model",
        "model_pair",
        "torch_dtype",
        "seed",
        "compression_steps",
        "semantic_anchor_count",
        "weight_schedule_name",
        "reasoning_layer_weights",
        "alignment_mode",
        "alignment_strategy",
        "handoff_status",
        "handoff_surface",
        "kv_cache_status",
        "kv_cache_reason",
        "decode_status",
        "sample_count",
        "accuracy_percentage",
        "average_latency_seconds",
        "mean_answer_nll",
        "answer_perplexity",
        "mean_pre_alignment_l2_distance",
        "mean_post_alignment_l2_distance",
        "mean_post_alignment_cosine_distance",
        "mean_alignment_residual_norm_ratio",
        "mean_alignment_bias_norm",
        "prompt_calibration_rate_percentage",
        "mean_prompt_calibration_bias_norm",
        "mean_anchor_eval_post_alignment_l2_distance",
        "mean_anchor_eval_post_alignment_cosine_distance",
        "mean_raw_handoff_entropy",
        "confidence_gate_trigger_rate_percentage",
        "global_alignment_cache_hit_rate_percentage",
        "explicit_status_rate_percentage",
        "failure_rate_percentage",
        "error_count",
        "q_frobenius_distance_to_full",
        "selected_default",
    ]
    samples_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    with samples_output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=sample_fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with summary_output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    write_json(report_output_path, report_payload)
    return rows, summary_rows, report_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Q-matrix distance calibration and generalization controls "
            "including held-out anchors, bootstrap subsets, and shuffled anchors."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("gsm8k", "math"),
        default=DEFAULT_DATASET,
        help=f"Dataset to evaluate (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default=DEFAULT_SPLIT,
        help="Dataset split to evaluate. Defaults to validation for GSM8K and test for MATH.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Number of samples to evaluate (default: {DEFAULT_LIMIT}).",
    )
    parser.add_argument(
        "--anchor-counts",
        default=",".join(str(value) for value in DEFAULT_ANCHOR_COUNTS),
        help="Comma-separated semantic anchor counts to evaluate.",
    )
    parser.add_argument(
        "--weight-schedules",
        default=",".join(DEFAULT_WEIGHT_SCHEDULES),
        help="Comma-separated layer weight schedules to evaluate.",
    )
    parser.add_argument(
        "--alignment-strategies",
        default=",".join(DEFAULT_ALIGNMENT_STRATEGIES),
        help="Comma-separated alignment strategies to evaluate.",
    )
    parser.add_argument(
        "--controls",
        default=",".join(DEFAULT_CONTROLS),
        help="Comma-separated control names to evaluate.",
    )
    parser.add_argument(
        "--samples-output",
        type=Path,
        default=DEFAULT_SAMPLES_OUTPUT,
        help=f"Per-sample CSV output path (default: {DEFAULT_SAMPLES_OUTPUT}).",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
        help=f"Summary CSV output path (default: {DEFAULT_SUMMARY_OUTPUT}).",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=DEFAULT_REPORT_OUTPUT,
        help=f"JSON report output path (default: {DEFAULT_REPORT_OUTPUT}).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional override for cfg.max_new_tokens to support faster smoke tests.",
    )
    parser.add_argument(
        "--agent-a-model",
        default=None,
        help="Optional override for Agent A model.",
    )
    parser.add_argument(
        "--agent-b-model",
        default=None,
        help="Optional override for Agent B model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional deterministic seed override.",
    )
    args = parser.parse_args()

    _, summary_rows, report_payload = run_analysis(
        dataset_name=args.dataset,
        dataset_split=args.split,
        limit=args.limit,
        anchor_counts=_parse_int_list(args.anchor_counts),
        weight_schedule_names=_parse_name_list(args.weight_schedules),
        alignment_strategy_names=_parse_name_list(args.alignment_strategies),
        controls=_parse_name_list(args.controls),
        samples_output_path=args.samples_output,
        summary_output_path=args.summary_output,
        report_output_path=args.report_output,
        max_new_tokens=args.max_new_tokens,
        agent_a_model=args.agent_a_model,
        agent_b_model=args.agent_b_model,
        seed=args.seed,
    )

    print(f"Wrote per-sample diagnostics to {args.samples_output}")
    print(f"Wrote summary diagnostics to {args.summary_output}")
    print(f"Wrote calibration report to {args.report_output}")
    q_report = report_payload["q_generalization_report"]
    distance_report = report_payload["distance_calibration_report"]
    print(f"Distance calibration passed: {distance_report['passed']}")
    print(f"Q generalization passed: {q_report['passed']}")
    for row in summary_rows:
        perplexity = row["answer_perplexity"]
        perplexity_label = "n/a" if perplexity is None else f"{float(perplexity):.2f}"
        anchor_l2 = row["mean_anchor_eval_post_alignment_l2_distance"]
        anchor_l2_label = "n/a" if anchor_l2 is None else f"{float(anchor_l2):.4f}"
        print(
            f"{row['config_id']} | {row['control_name']} | bootstrap={row['bootstrap_index'] or 'na'} | "
            f"accuracy={float(row['accuracy_percentage']):.2f}% | "
            f"answer_ppl={perplexity_label} | "
            f"anchor_l2={anchor_l2_label}"
        )


if __name__ == "__main__":
    main()
