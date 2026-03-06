from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf

from latent_pipeline import _get_pipeline_state, initialize_hybrid_pipeline, run_hybrid_pipeline
from src.data.loader import load_gsm8k, load_math_level5, pick_field
from src.utils.benchmarking import build_distance_calibration_report, build_standard_row_base, write_json
from src.utils.metrics import extract_boxed_text, normalize_answer

DEFAULT_DATASET = "gsm8k"
DEFAULT_LIMIT = 100
DEFAULT_ANCHOR_COUNTS = (100, 250, 500)
DEFAULT_WEIGHT_SCHEDULES = (
    "uniform",
    "linear_deep_bias",
    "strong_deep_bias",
)
DEFAULT_SAMPLES_OUTPUT = Path("distance_accuracy_samples.csv")
DEFAULT_SUMMARY_OUTPUT = Path("distance_accuracy_summary.csv")
DEFAULT_REPORT_OUTPUT = Path("distance_accuracy_report.json")
GSM8K_FINAL_ANSWER_REGEX = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")
NUMERIC_ANSWER_REGEX = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_WEIGHT_SCHEDULES: dict[str, tuple[float, float, float]] = {
    "uniform": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    "linear_deep_bias": (1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0),
    "strong_deep_bias": (0.125, 0.25, 0.625),
}


def _load_cfg() -> Any:
    return OmegaConf.load(Path(__file__).resolve().parent / "configs" / "main.yaml")


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


def _load_samples(dataset_name: str, limit: int) -> Any:
    if dataset_name == "gsm8k":
        return load_gsm8k(limit=limit)
    if dataset_name == "math":
        return load_math_level5(limit=limit)
    raise ValueError(f"Unsupported dataset {dataset_name!r}")


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


def _clone_cfg(base_cfg: Any) -> Any:
    return OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))


def _reasoning_layer_weights_for_schedule(schedule_name: str) -> tuple[float, float, float]:
    schedule = _WEIGHT_SCHEDULES.get(schedule_name)
    if schedule is None:
        raise ValueError(
            f"Unsupported weight schedule {schedule_name!r}. "
            f"Supported: {', '.join(sorted(_WEIGHT_SCHEDULES))}"
        )
    return schedule


def _config_id(anchor_count: int, weight_schedule_name: str) -> str:
    return f"anchors_{anchor_count}_{weight_schedule_name}"


def _assign_distance_deciles(rows: list[dict[str, Any]]) -> None:
    valid_rows = [
        row for row in rows if row["post_alignment_l2_distance"] is not None
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


def _aggregate_group_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sample_count = len(rows)
    correct_count = sum(1 for row in rows if row["correct"])
    total_latency = sum(float(row["latency_seconds"]) for row in rows)
    valid_post_l2 = [
        float(row["post_alignment_l2_distance"])
        for row in rows
        if row["post_alignment_l2_distance"] is not None
    ]
    valid_post_cosine = [
        float(row["post_alignment_cosine_distance"])
        for row in rows
        if row["post_alignment_cosine_distance"] is not None
    ]
    valid_handoff_entropy = [
        float(row["raw_handoff_entropy"])
        for row in rows
        if row["raw_handoff_entropy"] is not None
    ]
    confidence_gate_trigger_count = sum(
        1 for row in rows if bool(row["confidence_gate_triggered"])
    )
    error_count = sum(1 for row in rows if row["error"])
    return {
        "sample_count": sample_count,
        "accuracy_percentage": (100.0 * correct_count / sample_count) if sample_count else 0.0,
        "average_latency_seconds": total_latency / sample_count if sample_count else 0.0,
        "mean_post_alignment_l2_distance": (
            sum(valid_post_l2) / len(valid_post_l2) if valid_post_l2 else None
        ),
        "mean_post_alignment_cosine_distance": (
            sum(valid_post_cosine) / len(valid_post_cosine) if valid_post_cosine else None
        ),
        "failure_rate_percentage": (100.0 * (sample_count - correct_count) / sample_count)
        if sample_count
        else 0.0,
        "mean_raw_handoff_entropy": (
            sum(valid_handoff_entropy) / len(valid_handoff_entropy) if valid_handoff_entropy else None
        ),
        "confidence_gate_trigger_rate_percentage": (
            100.0 * confidence_gate_trigger_count / sample_count if sample_count else 0.0
        ),
        "error_count": error_count,
    }


def _breaking_point_decile(decile_rows: list[dict[str, Any]]) -> Optional[int]:
    if not decile_rows:
        return None
    best_accuracy = max(float(row["accuracy_percentage"]) for row in decile_rows)
    for row in sorted(decile_rows, key=lambda item: int(item["distance_decile"])):
        if float(row["accuracy_percentage"]) <= best_accuracy - 10.0:
            return int(row["distance_decile"])
    return None


def _summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped_configs: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped_configs.setdefault(str(row["config_id"]), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []

    for config_id, config_rows in grouped_configs.items():
        base_row = config_rows[0]
        decile_groups: dict[int, list[dict[str, Any]]] = {}
        for row in config_rows:
            distance_decile = row["distance_decile"]
            if distance_decile == "":
                continue
            decile_groups.setdefault(int(distance_decile), []).append(row)

        decile_summary_rows: list[dict[str, Any]] = []
        for distance_decile, decile_group_rows in sorted(decile_groups.items()):
            aggregate = _aggregate_group_rows(decile_group_rows)
            decile_summary_rows.append(
                {
                    "row_type": "distance_decile",
                    "evaluation_surface": base_row["evaluation_surface"],
                    "suite": base_row["suite"],
                    "method": base_row["method"],
                    "config_id": config_id,
                    "dataset": base_row["dataset"],
                    "agent_a_model": base_row["agent_a_model"],
                    "agent_b_model": base_row["agent_b_model"],
                    "model_pair": base_row["model_pair"],
                    "torch_dtype": base_row["torch_dtype"],
                    "compression_steps": base_row["compression_steps"],
                    "semantic_anchor_count": base_row["semantic_anchor_count"],
                    "weight_schedule_name": base_row["weight_schedule_name"],
                    "reasoning_layer_weights": base_row["reasoning_layer_weights"],
                    "alignment_mode": base_row["alignment_mode"],
                    "distance_decile": distance_decile,
                    **aggregate,
                    "breaking_point_decile": "",
                    "selected_default": False,
                }
            )

        breaking_point = _breaking_point_decile(decile_summary_rows)
        aggregate = _aggregate_group_rows(config_rows)
        overall_row = {
            "row_type": "overall",
            "evaluation_surface": base_row["evaluation_surface"],
            "suite": base_row["suite"],
            "method": base_row["method"],
            "config_id": config_id,
            "dataset": base_row["dataset"],
            "agent_a_model": base_row["agent_a_model"],
            "agent_b_model": base_row["agent_b_model"],
            "model_pair": base_row["model_pair"],
            "torch_dtype": base_row["torch_dtype"],
            "compression_steps": base_row["compression_steps"],
            "semantic_anchor_count": base_row["semantic_anchor_count"],
            "weight_schedule_name": base_row["weight_schedule_name"],
            "reasoning_layer_weights": base_row["reasoning_layer_weights"],
            "alignment_mode": base_row["alignment_mode"],
            "distance_decile": "",
            **aggregate,
            "breaking_point_decile": "" if breaking_point is None else breaking_point,
            "selected_default": False,
        }
        overall_rows.append(overall_row)
        summary_rows.append(overall_row)
        summary_rows.extend(decile_summary_rows)

    if overall_rows:
        selected_row = min(
            overall_rows,
            key=lambda row: (-float(row["accuracy_percentage"]), float(row["average_latency_seconds"])),
        )
        selected_row["selected_default"] = True

    return summary_rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(
    *,
    dataset_name: str,
    limit: int,
    anchor_counts: list[int],
    weight_schedule_names: list[str],
    samples_output_path: Path,
    summary_output_path: Path,
    report_output_path: Path,
    max_new_tokens: Optional[int] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    base_cfg = _load_cfg()
    if max_new_tokens is not None:
        base_cfg.max_new_tokens = int(max_new_tokens)

    samples = _load_samples(dataset_name, limit)
    rows: list[dict[str, Any]] = []

    for anchor_count in anchor_counts:
        for weight_schedule_name in weight_schedule_names:
            schedule_weights = _reasoning_layer_weights_for_schedule(weight_schedule_name)
            step_cfg = _clone_cfg(base_cfg)
            step_cfg.alignment.semantic_anchor_count = int(anchor_count)
            step_cfg.alignment.reasoning_layer_weights = list(schedule_weights)
            config_id = _config_id(anchor_count, weight_schedule_name)

            initialize_hybrid_pipeline(step_cfg)
            state = _get_pipeline_state(step_cfg)
            tokenizer_b = state["tokenizer_b"]
            print(
                f"Running {dataset_name} analysis for "
                f"{config_id} on {len(samples)} samples..."
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
                    )
                    decoded_text = str(pipeline_output["decoded_text"])
                    predicted_answer = _predicted_answer(dataset_name, decoded_text)
                except Exception as exc:  # noqa: BLE001
                    error = str(exc)
                latency = time.perf_counter() - start

                correct = _answers_match(dataset_name, predicted_answer, target_answer)
                row_base = build_standard_row_base(
                    step_cfg,
                    evaluation_surface="analyze_distance_accuracy",
                    suite="phase3_alignment_sweep",
                    method="hybrid_hl_mas",
                    dataset=dataset_name,
                    repetition=0,
                    compression_steps=int(getattr(step_cfg, "latent_steps", 0)),
                    alignment_mode=str(pipeline_output.get("alignment_mode", "semantic_anchor_global")),
                    semantic_anchor_count=anchor_count,
                    reasoning_layer_weights=schedule_weights,
                )
                rows.append(
                    {
                        **row_base,
                        "config_id": config_id,
                        "sample_index": index,
                        "weight_schedule_name": weight_schedule_name,
                        "prompt": prompt,
                        "target_answer": target_answer,
                        "predicted_answer": predicted_answer,
                        "decoded_text": decoded_text,
                        "generated_tokens": len(
                            tokenizer_b.encode(decoded_text, add_special_tokens=False)
                        ) if decoded_text else 0,
                        "correct": correct,
                        "latency_seconds": latency,
                        "pre_alignment_l2_distance": pipeline_output.get("pre_alignment_l2_distance"),
                        "pre_alignment_cosine_distance": pipeline_output.get(
                            "pre_alignment_cosine_distance"
                        ),
                        "post_alignment_l2_distance": pipeline_output.get(
                            "post_alignment_l2_distance"
                        ),
                        "post_alignment_cosine_distance": pipeline_output.get(
                            "post_alignment_cosine_distance"
                        ),
                        "raw_handoff_entropy": pipeline_output.get("raw_handoff_entropy"),
                        "handoff_uncertainty": pipeline_output.get("handoff_uncertainty"),
                        "confidence_gate_triggered": pipeline_output.get(
                            "confidence_gate_triggered"
                        ),
                        "kv_cache_transferred": pipeline_output.get("kv_cache_transferred"),
                        "fallback_discrete_reasoning_steps": pipeline_output.get(
                            "fallback_discrete_reasoning_steps"
                        ),
                        "latent_trajectory_steps": pipeline_output.get("latent_trajectory_steps"),
                        "total_reasoning_steps": pipeline_output.get("total_reasoning_steps"),
                        "continuous_integration_seconds": pipeline_output.get(
                            "continuous_integration_seconds"
                        ),
                        "error": error,
                    }
                )

    for config_id in sorted({str(row["config_id"]) for row in rows}):
        _assign_distance_deciles([row for row in rows if row["config_id"] == config_id])

    summary_rows = _summarize_rows(rows)
    report_payload = {
        "evaluation_surface": "analyze_distance_accuracy",
        "dataset": dataset_name,
        "limit": limit,
        "report": build_distance_calibration_report(rows, summary_rows),
    }

    sample_fieldnames = [
        "evaluation_surface",
        "suite",
        "method",
        "config_id",
        "dataset",
        "agent_a_model",
        "agent_b_model",
        "model_pair",
        "torch_dtype",
        "compression_steps",
        "repetition",
        "sample_index",
        "semantic_anchor_count",
        "weight_schedule_name",
        "reasoning_layer_weights",
        "alignment_mode",
        "distance_decile",
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
        "kv_cache_transferred",
        "fallback_discrete_reasoning_steps",
        "latent_trajectory_steps",
        "total_reasoning_steps",
        "continuous_integration_seconds",
        "error",
    ]
    summary_fieldnames = [
        "row_type",
        "evaluation_surface",
        "suite",
        "method",
        "config_id",
        "dataset",
        "agent_a_model",
        "agent_b_model",
        "model_pair",
        "torch_dtype",
        "compression_steps",
        "semantic_anchor_count",
        "weight_schedule_name",
        "reasoning_layer_weights",
        "alignment_mode",
        "distance_decile",
        "sample_count",
        "accuracy_percentage",
        "average_latency_seconds",
        "mean_post_alignment_l2_distance",
        "mean_post_alignment_cosine_distance",
        "mean_raw_handoff_entropy",
        "confidence_gate_trigger_rate_percentage",
        "failure_rate_percentage",
        "error_count",
        "breaking_point_decile",
        "selected_default",
    ]
    _write_csv(samples_output_path, rows, sample_fieldnames)
    _write_csv(summary_output_path, summary_rows, summary_fieldnames)
    write_json(report_output_path, report_payload)
    return rows, summary_rows, report_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep semantic-anchor counts and layer weighting schedules, then "
            "measure post-alignment distance versus exact-match accuracy."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("gsm8k", "math"),
        default=DEFAULT_DATASET,
        help=f"Dataset to evaluate (default: {DEFAULT_DATASET}).",
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
    args = parser.parse_args()

    rows, summary_rows, report_payload = run_analysis(
        dataset_name=args.dataset,
        limit=args.limit,
        anchor_counts=_parse_int_list(args.anchor_counts),
        weight_schedule_names=_parse_name_list(args.weight_schedules),
        samples_output_path=args.samples_output,
        summary_output_path=args.summary_output,
        report_output_path=args.report_output,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"Wrote per-sample diagnostics to {args.samples_output}")
    print(f"Wrote summary diagnostics to {args.summary_output}")
    print(f"Wrote calibration report to {args.report_output}")
    for row in summary_rows:
        if row["row_type"] != "overall":
            continue
        print(
            f"{row['config_id']}: "
            f"accuracy={row['accuracy_percentage']:.2f}% "
            f"avg_latency={row['average_latency_seconds']:.4f}s "
            f"breaking_point_decile={row['breaking_point_decile'] or 'none'} "
            f"selected_default={row['selected_default']}"
        )
    print(f"Collected {len(rows)} per-sample rows.")
    print(f"Phase 3 calibration passed: {report_payload['report']['passed']}")


if __name__ == "__main__":
    main()
