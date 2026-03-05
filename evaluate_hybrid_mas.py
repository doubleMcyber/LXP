from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from latent_pipeline import (
    initialize_hybrid_pipeline,
    run_hybrid_pipeline,
)
from src.data.loader import load_math_level5, pick_field
from src.utils.metrics import (
    EvalSampleResult,
    calculate_latency_stats,
    extract_boxed_text,
    normalize_answer,
)

RESULTS_PATH = Path("results.json")


def _load_cfg():
    return OmegaConf.load(Path(__file__).resolve().parent / "configs" / "main.yaml")


def evaluate(limit: int = 100, cfg=None) -> dict:
    if cfg is None:
        cfg = _load_cfg()
    samples = load_math_level5(limit=limit)
    initialize_hybrid_pipeline(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.agent_b_model)

    per_sample: list[EvalSampleResult] = []
    total_latency = 0.0
    total_generated_tokens = 0
    total_correct = 0

    for idx, row in enumerate(samples):
        problem = pick_field(row, ("problem", "question"))
        target_boxed = extract_boxed_text(pick_field(row, ("solution", "answer")))

        error: Optional[str] = None
        decoded_text = ""
        predicted_boxed: Optional[str] = None
        generated_tokens = 0
        latency = 0.0

        start = time.perf_counter()
        try:
            pipeline_output = run_hybrid_pipeline(cfg, prompt=problem)
            decoded_text = str(pipeline_output["decoded_text"])
            predicted_boxed = extract_boxed_text(decoded_text)
            generated_tokens = len(tokenizer.encode(decoded_text, add_special_tokens=False))
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        latency = time.perf_counter() - start

        correct = normalize_answer(predicted_boxed) == normalize_answer(target_boxed)

        total_latency += latency
        total_generated_tokens += generated_tokens
        total_correct += int(correct)

        per_sample.append(
            EvalSampleResult(
                index=idx,
                latency_seconds=latency,
                generated_tokens=generated_tokens,
                predicted_boxed=predicted_boxed,
                target_boxed=target_boxed,
                correct=correct,
                error=error,
            )
        )

    sample_count = len(per_sample)
    accuracy_percentage = (100.0 * total_correct / sample_count) if sample_count else 0.0
    latency_stats = calculate_latency_stats(per_sample)

    return {
        "benchmark": "MATH Level 5",
        "sample_count": sample_count,
        "total_latency_seconds": latency_stats["total_latency_seconds"],
        "average_latency_seconds": latency_stats["average_latency_seconds"],
        "tokens_per_second": latency_stats["tokens_per_second"],
        "total_text_tokens_generated": total_generated_tokens,
        "accuracy_percentage": accuracy_percentage,
        "results": [asdict(item) for item in per_sample],
    }


def main() -> None:
    results = evaluate(limit=100)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"Wrote evaluation results to {RESULTS_PATH}")
    print(f"Accuracy: {results['accuracy_percentage']:.2f}%")
    print(f"Total latency (s): {results['total_latency_seconds']:.4f}")
    print(f"Total generated tokens: {results['total_text_tokens_generated']}")


if __name__ == "__main__":
    main()
