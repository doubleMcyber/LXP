"""Evaluation metrics: answer extraction, normalization, and latency stats."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

BOXED_REGEX = re.compile(r"\\boxed\s*\{([^{}]+)\}")


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
    """Return the last \\boxed{...} content in *text*, or None."""
    matches = BOXED_REGEX.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def normalize_answer(answer: Optional[str]) -> Optional[str]:
    """Strip dollars, whitespace-collapse, and lowercase for comparison."""
    if answer is None:
        return None
    cleaned = answer.replace("$", "").strip()
    return re.sub(r"\s+", "", cleaned)


def calculate_latency_stats(
    results: Sequence[EvalSampleResult],
) -> dict[str, float]:
    """Compute aggregate latency and throughput from a list of sample results."""
    if not results:
        return {
            "total_latency_seconds": 0.0,
            "average_latency_seconds": 0.0,
            "tokens_per_second": 0.0,
        }

    total_latency = sum(r.latency_seconds for r in results)
    total_tokens = sum(r.generated_tokens for r in results)
    sample_count = len(results)

    return {
        "total_latency_seconds": total_latency,
        "average_latency_seconds": total_latency / sample_count,
        "tokens_per_second": total_tokens / total_latency if total_latency > 0 else 0.0,
    }
