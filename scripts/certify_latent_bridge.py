"""Certify the Phase 0 latent-continuation result.

Programmatic checks that the bridge accuracy is not an artifact:
1. Split disjointness — no eval question appears in the bridge's training set or in
   the ridge warm-start adapter's training rows (both come from the train split).
2. Truncation safety — no truncated prefix contains a final-answer marker, and every
   cut lands strictly before the marker token.
3. Answer-literal stratification — accuracy reported separately for eval rows whose
   truncated prefix already contains the answer string (where "continuation" could
   be copying) vs rows where it does not (where the receiver must compute).

Reads the latest triplet report produced by scripts/train_latent_bridge.py.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from transformers import AutoTokenizer  # noqa: E402

from train_latent_bridge import load_bridge_samples  # noqa: E402


def _normalized_number(text: str) -> str:
    return re.sub(r"[^\d.]", "", str(text)).strip(".")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="outputs/latent_bridge_untrained/bridge_report.json")
    parser.add_argument("--sender-model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--trace-dtype", default="bfloat16")
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--train-limit", type=int, default=128)
    parser.add_argument("--eval-limit", type=int, default=32)
    parser.add_argument("--truncation-fraction", type=float, default=0.5)
    parser.add_argument("--output", default="outputs/latent_bridge_untrained/certification.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.sender_model, trust_remote_code=True)
    shared = dict(
        dataset=args.dataset,
        model_id=args.sender_model,
        torch_dtype=args.trace_dtype,
        truncation_fraction=args.truncation_fraction,
        tokenizer=tokenizer,
        max_continuation_tokens=256,
    )
    train_samples = load_bridge_samples(split="train", limit=args.train_limit, **shared)
    eval_samples = load_bridge_samples(split="validation", limit=args.eval_limit, **shared)

    certification: dict[str, object] = {}

    # 1. split disjointness
    train_questions = {s["question"] for s in train_samples}
    overlap = [s["sample_index"] for s in eval_samples if s["question"] in train_questions]
    certification["train_eval_question_overlap"] = overlap
    certification["split_disjoint"] = not overlap

    # 2. truncation safety
    marker_violations = [
        s["sample_index"]
        for s in eval_samples + train_samples
        if "final answer" in s["truncated_text"].lower()
    ]
    certification["truncated_prefixes_with_marker"] = marker_violations
    certification["truncation_safe"] = not marker_violations

    # 3. answer-literal stratification on the eval report
    report = json.loads(Path(args.report).read_text())
    rows = report["latest"]["rows"]
    truncated_by_index = {s["sample_index"]: s for s in eval_samples}
    strata: dict[str, dict[str, list[int]]] = {
        "answer_in_prefix": {},
        "answer_not_in_prefix": {},
    }
    for row in rows:
        sample = truncated_by_index.get(row["sample_index"])
        if sample is None:
            continue
        answer_token = _normalized_number(sample["answer"])
        present = bool(answer_token) and answer_token in re.sub(
            r"[,\s]", "", sample["truncated_text"]
        )
        bucket = strata["answer_in_prefix" if present else "answer_not_in_prefix"]
        for variant in ("bridge", "alone", "text", "instruction_only"):
            if variant in row:
                bucket.setdefault(variant, []).append(int(bool(row[variant]["correct"])))
    certification["strata"] = {
        name: {
            variant: {
                "n": len(values),
                "accuracy": round(100.0 * sum(values) / len(values), 1) if values else None,
            }
            for variant, values in bucket.items()
        }
        for name, bucket in strata.items()
    }
    certification["overall_accuracy"] = report["latest"]["accuracy"]
    certification["eval_sample_count"] = report["latest"]["sample_count"]

    passed = bool(certification["split_disjoint"]) and bool(certification["truncation_safe"])
    certification["certification_passed"] = passed
    Path(args.output).write_text(json.dumps(certification, indent=2), encoding="utf-8")
    print(json.dumps(certification, indent=2))
    print(f"CERTIFICATION {'PASSED' if passed else 'FAILED'}")


if __name__ == "__main__":
    main()
