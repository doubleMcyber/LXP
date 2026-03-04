from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from transformers import AutoTokenizer

from latent_pipeline import (
    AGENT_B_MODEL_NAME,
    initialize_hybrid_pipeline,
    run_hybrid_pipeline,
)

BOXED_REGEX = re.compile(r"\\boxed\s*\{([^{}]+)\}")
RESULTS_PATH = Path("results.json")


@dataclass
class EvalSampleResult:
    index: int
    latency_seconds: float
    generated_tokens: int
    predicted_boxed: Optional[str]
    target_boxed: Optional[str]
    correct: bool
    error: Optional[str] = None


def extract_boxed_text(text: str) -> Optional[str]:
    matches = BOXED_REGEX.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    cleaned = answer.replace("$", "").strip()
    return re.sub(r"\s+", "", cleaned)


def load_math_level5(limit: int = 100):
    dataset_candidates = [
        ("hendrycks/competition_math", "test"),
        ("competition_math", "test"),
    ]
    last_error: Optional[Exception] = None

    for dataset_name, split in dataset_candidates:
        try:
            dataset = load_dataset(dataset_name, split=split)
            if "level" in dataset.column_names:
                level5 = dataset.filter(lambda row: "Level 5" in str(row.get("level", "")))
            else:
                level5 = dataset
            return level5.select(range(min(limit, len(level5))))
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise RuntimeError("Failed to load a MATH dataset candidate") from last_error


def _pick_field(row: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return str(value)
    return ""


def evaluate(limit: int = 100) -> dict:
    samples = load_math_level5(limit=limit)
    initialize_hybrid_pipeline()
    tokenizer = AutoTokenizer.from_pretrained(AGENT_B_MODEL_NAME)

    per_sample: list[EvalSampleResult] = []
    total_latency = 0.0
    total_generated_tokens = 0
    total_correct = 0

    for idx, row in enumerate(samples):
        problem = _pick_field(row, ("problem", "question"))
        target_boxed = extract_boxed_text(_pick_field(row, ("solution", "answer")))

        error: Optional[str] = None
        decoded_text = ""
        predicted_boxed: Optional[str] = None
        generated_tokens = 0
        latency = 0.0

        start = time.perf_counter()
        try:
            pipeline_output = run_hybrid_pipeline(prompt=problem)
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
    avg_latency = (total_latency / sample_count) if sample_count else 0.0

    return {
        "benchmark": "MATH Level 5",
        "sample_count": sample_count,
        "total_latency_seconds": total_latency,
        "average_latency_seconds": avg_latency,
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
