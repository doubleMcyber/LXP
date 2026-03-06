from __future__ import annotations

import argparse
import csv
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf

from latent_pipeline import _get_pipeline_state, initialize_hybrid_pipeline, run_hybrid_pipeline
from src.data.loader import load_gsm8k, pick_field
from src.utils.metrics import EvalSampleResult, calculate_latency_stats, normalize_answer

DEFAULT_LATENT_STEPS = [1, 2, 4, 8, 16, 32, 64]
DEFAULT_LIMIT = 10
DEFAULT_OUTPUT_PATH = Path("scaling_results.csv")
GSM8K_FINAL_ANSWER_REGEX = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")
NUMERIC_ANSWER_REGEX = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _load_cfg() -> Any:
    return OmegaConf.load(Path(__file__).resolve().parent / "configs" / "main.yaml")


def _parse_steps(raw_steps: str) -> list[int]:
    values = []
    for part in raw_steps.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("At least one latent step value is required")
    return values


def _extract_gsm8k_target_answer(text: str) -> Optional[str]:
    match = GSM8K_FINAL_ANSWER_REGEX.search(text)
    if match is None:
        return None
    return match.group(1)


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


def evaluate_gsm8k(cfg: Any, samples: Any, tokenizer: Any) -> dict[str, Any]:
    per_sample: list[EvalSampleResult] = []
    total_generated_tokens = 0
    total_correct = 0

    for idx, row in enumerate(samples):
        problem = pick_field(row, ("question", "problem"))
        target_answer = _extract_gsm8k_target_answer(pick_field(row, ("answer", "solution")))

        error: Optional[str] = None
        decoded_text = ""
        predicted_answer: Optional[str] = None
        generated_tokens = 0

        start = time.perf_counter()
        try:
            pipeline_output = run_hybrid_pipeline(cfg, prompt=problem)
            decoded_text = str(pipeline_output["decoded_text"])
            predicted_answer = _extract_gsm8k_predicted_answer(decoded_text)
            generated_tokens = len(tokenizer.encode(decoded_text, add_special_tokens=False))
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        latency = time.perf_counter() - start

        correct = _normalize_numeric_answer(predicted_answer) == _normalize_numeric_answer(
            target_answer
        )
        total_correct += int(correct)
        total_generated_tokens += generated_tokens

        per_sample.append(
            EvalSampleResult(
                index=idx,
                latency_seconds=latency,
                generated_tokens=generated_tokens,
                predicted_boxed=predicted_answer,
                target_boxed=target_answer,
                correct=correct,
                error=error,
            )
        )

    latency_stats = calculate_latency_stats(per_sample)
    sample_count = len(per_sample)
    error_count = sum(1 for item in per_sample if item.error is not None)
    accuracy_percentage = (100.0 * total_correct / sample_count) if sample_count else 0.0

    return {
        "benchmark": "GSM8K",
        "sample_count": sample_count,
        "total_latency_seconds": latency_stats["total_latency_seconds"],
        "average_latency_seconds": latency_stats["average_latency_seconds"],
        "tokens_per_second": latency_stats["tokens_per_second"],
        "total_text_tokens_generated": total_generated_tokens,
        "accuracy_percentage": accuracy_percentage,
        "error_count": error_count,
        "results": [asdict(item) for item in per_sample],
    }


def run_sweep(
    *,
    latent_steps_values: list[int],
    limit: int,
    output_path: Path,
    max_new_tokens: Optional[int] = None,
) -> list[dict[str, Any]]:
    base_cfg = _load_cfg()
    if max_new_tokens is not None:
        base_cfg.max_new_tokens = int(max_new_tokens)

    samples = load_gsm8k(limit=limit)
    initialize_hybrid_pipeline(base_cfg)
    tokenizer = _get_pipeline_state(base_cfg)["tokenizer_b"]

    baseline_steps = max(latent_steps_values)
    rows: list[dict[str, Any]] = []

    for latent_steps in latent_steps_values:
        step_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
        step_cfg.latent_steps = int(latent_steps)

        print(f"Running GSM8K sweep for latent_steps={latent_steps} on {limit} samples...")
        metrics = evaluate_gsm8k(cfg=step_cfg, samples=samples, tokenizer=tokenizer)
        row = {
            "latent_steps": latent_steps,
            "compression_ratio": baseline_steps / latent_steps,
            "accuracy_percentage": metrics["accuracy_percentage"],
            "total_latency_seconds": metrics["total_latency_seconds"],
            "average_latency_seconds": metrics["average_latency_seconds"],
            "tokens_per_second": metrics["tokens_per_second"],
            "total_text_tokens_generated": metrics["total_text_tokens_generated"],
            "error_count": metrics["error_count"],
            "sample_count": metrics["sample_count"],
        }
        rows.append(row)

    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep latent step counts to identify the optimal compression ratio on GSM8K."
    )
    parser.add_argument(
        "--steps",
        default=",".join(str(step) for step in DEFAULT_LATENT_STEPS),
        help="Comma-separated latent step values to evaluate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Number of GSM8K samples to evaluate (default: {DEFAULT_LIMIT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"CSV output path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional override for cfg.max_new_tokens to support faster smoke tests.",
    )
    args = parser.parse_args()

    rows = run_sweep(
        latent_steps_values=_parse_steps(args.steps),
        limit=args.limit,
        output_path=args.output,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"Wrote {args.output}")
    for row in rows:
        print(
            f"latent_steps={row['latent_steps']:<2} "
            f"compression_ratio={row['compression_ratio']:.2f} "
            f"accuracy={row['accuracy_percentage']:.2f}% "
            f"total_latency={row['total_latency_seconds']:.4f}s"
        )


if __name__ == "__main__":
    main()
