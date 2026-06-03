from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Sequence

import torch


def build_command(args: argparse.Namespace) -> list[str]:
    output_dir = Path(args.output_dir)
    return [
        args.python,
        "run_training.py",
        "runtime.device=mps",
        f"runtime.mps.fallback_to_cpu={str(args.allow_cpu_fallback).lower()}",
        "runtime.mps.torch_dtype=float32",
        "device_map=none",
        f"agent_a_model={args.agent_a_model}",
        f"agent_b_model={args.agent_b_model}",
        "training.data.mode=smoke",
        f"training.data.batch_size={args.batch_size}",
        f"training.data.smoke_num_samples={args.smoke_samples}",
        f"training.reasoner_max_length={args.max_length}",
        f"training.actor_max_length={args.max_length}",
        f"training.compressed_steps={args.compressed_steps}",
        f"training.num_epochs={args.epochs}",
        "training.checkpointing.enabled=false",
        f"reporting.training.history_output={output_dir / 'mac_mps_training_history.csv'}",
        f"reporting.training.report_output={output_dir / 'mac_mps_training_report.json'}",
    ]


def _print_command(command: Sequence[str]) -> None:
    print(" ".join(command), flush=True)


def _print_report_summary(report_path: Path) -> None:
    if not report_path.is_file():
        return
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    smoke_report = report.get("training_smoke_report") or {}
    print("\nMac MPS smoke summary", flush=True)
    print(json.dumps(
        {
            "training_smoke_passed": smoke_report.get("passed"),
            "phase2_gate_passed": report.get("passed"),
            "effective_device": report.get("effective_device"),
            "effective_torch_dtype": report.get("effective_torch_dtype"),
            "final_heldout_exact_match_accuracy": report.get(
                "final_heldout_exact_match_accuracy"
            ),
            "final_heldout_answer_perplexity": report.get(
                "final_heldout_answer_perplexity"
            ),
            "final_answer_extraction_rate": smoke_report.get(
                "final_heldout_answer_extraction_rate_percentage"
            ),
            "decode_answer_extraction_rate": smoke_report.get(
                "final_heldout_decode_answer_extraction_rate_percentage"
            ),
            "candidate_fallback_rate": smoke_report.get(
                "final_heldout_candidate_fallback_rate_percentage"
            ),
            "unique_predicted_answer_count": smoke_report.get(
                "final_heldout_unique_predicted_answer_count"
            ),
            "degenerate_prediction": smoke_report.get(
                "final_heldout_degenerate_prediction"
            ),
            "actor_text_baseline_accuracy": smoke_report.get(
                "final_heldout_actor_text_baseline_accuracy"
            ),
            "actor_text_baseline_unique_predicted_answer_count": smoke_report.get(
                "final_heldout_actor_text_baseline_unique_predicted_answer_count"
            ),
            "eval_diagnostics": smoke_report.get("heldout_eval_diagnostics"),
            "initial_eval_diagnostics": smoke_report.get("initial_heldout_eval_diagnostics"),
            "missing_phase2_requirements": report.get("missing_requirements"),
            "missing_smoke_requirements": smoke_report.get("missing_requirements"),
        },
        indent=2,
        sort_keys=True,
    ), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small Stage-II training smoke configured for Apple Silicon MPS. "
            "By default this prints the command only; pass --execute to run it."
        )
    )
    parser.add_argument("--python", default="venv/bin/python")
    parser.add_argument("--output-dir", default="outputs/mac_mps")
    parser.add_argument("--agent-a-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--agent-b-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--smoke-samples", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--compressed-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    command = build_command(args)
    print(f"MPS available: {torch.backends.mps.is_available()}", flush=True)
    _print_command(command)
    if not args.execute:
        print("\nDry run only. Re-run with --execute on the Mac.", flush=True)
        return 0
    if not torch.backends.mps.is_available() and not args.allow_cpu_fallback:
        raise SystemExit("MPS is unavailable. Re-run with --allow-cpu-fallback to test CPU only.")
    subprocess.run(command, check=True)
    _print_report_summary(Path(args.output_dir) / "mac_mps_training_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
