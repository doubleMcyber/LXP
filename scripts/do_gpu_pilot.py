from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence


DEFAULT_METHODS = (
    "token_context_handoff",
    "verified_token_context_handoff",
    "sender_answer_text_handoff",
    "generated_context_latent_handoff",
)


def _indices_csv(limit: int) -> str:
    return ",".join(str(index) for index in range(max(0, int(limit))))


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    output_dir = Path(args.output_dir)
    eval_indices = _indices_csv(args.eval_limit)
    methods = ",".join(DEFAULT_METHODS)
    common_generated_args = [
        "--hetero-smoke",
        "--generated-trajectory-adapter-input-space",
        "raw",
        "--generated-trajectory-adapter-train-limit",
        str(args.train_limit),
        "--enable-sender-revision",
    ]
    return [
        [args.python, "-m", "pytest", "-q"],
        [
            args.python,
            "benchmark_all.py",
            *common_generated_args,
            "--sample-indices",
            eval_indices,
            "--limit",
            str(args.eval_limit),
            "--methods",
            "generated_context_latent_handoff",
            "--prepare-generated-trajectory-eval-traces",
            "--report-output",
            str(output_dir / f"do_pilot_eval_trace_warm_{args.eval_limit}.json"),
        ],
        [
            args.python,
            "benchmark_all.py",
            *common_generated_args,
            "--methods",
            "generated_context_latent_handoff",
            "--prepare-generated-trajectory-adapter",
            "--report-output",
            str(output_dir / f"do_pilot_adapter_raw_{args.train_limit}_prepare.json"),
        ],
        [
            args.python,
            "benchmark_all.py",
            *common_generated_args,
            "--sample-indices",
            eval_indices,
            "--limit",
            str(args.eval_limit),
            "--methods",
            methods,
            "--generated-trajectory-adapter-no-train-on-missing",
            "--report-output",
            str(output_dir / f"do_pilot_context_vs_latent_{args.eval_limit}_report.json"),
            "--samples-output",
            str(output_dir / f"do_pilot_context_vs_latent_{args.eval_limit}_samples.csv"),
            "--summary-output",
            str(output_dir / f"do_pilot_context_vs_latent_{args.eval_limit}_summary.csv"),
            "--write-eval-manifest",
            str(output_dir / f"do_pilot_eval_manifest_{args.eval_limit}.json"),
        ],
    ]


def _print_command(command: Sequence[str]) -> None:
    print(" ".join(command), flush=True)


def _print_report_summary(report_path: Path) -> None:
    if not report_path.is_file():
        return
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    semantic = report.get("semantic_smoke_report") or {}
    provenance = report.get("latent_provenance_report") or {}
    print("\nPilot summary", flush=True)
    print(json.dumps(
        {
            "semantic_passed": semantic.get("passed"),
            "method_accuracy_percentage": semantic.get("method_accuracy_percentage"),
            "missing_requirements": semantic.get("missing_requirements"),
            "latent_failure_counts": provenance.get("failure_counts_by_class"),
            "eval_manifest_digest": (
                (report.get("eval_manifest") or {}).get("manifest_digest")
            ),
        },
        indent=2,
        sort_keys=True,
    ), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the bounded DigitalOcean GPU pilot sequence. By default this "
            "prints commands only; pass --execute on the GPU host to run them."
        )
    )
    parser.add_argument("--python", default="venv/bin/python")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--eval-limit", type=int, default=20)
    parser.add_argument("--train-limit", type=int, default=128)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    commands = build_commands(args)
    for command in commands:
        _print_command(command)
        if args.execute:
            subprocess.run(command, check=True)

    if args.execute:
        _print_report_summary(
            Path(args.output_dir)
            / f"do_pilot_context_vs_latent_{args.eval_limit}_report.json"
        )
    else:
        print("\nDry run only. Re-run with --execute on the GPU host.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
