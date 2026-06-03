from __future__ import annotations

import argparse
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
