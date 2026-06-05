from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Sequence


DEFAULT_METHODS = (
    "token_context_handoff",
    "verified_token_context_handoff",
    "sender_answer_text_handoff",
    "generated_context_latent_handoff",
)


PROFILE_DEFAULTS = {
    "local": {"eval_limit": 5, "train_limit": 32},
    "gpu": {"eval_limit": 20, "train_limit": 128},
    "scale": {"eval_limit": 64, "train_limit": 256},
}


def _indices_csv(limit: int) -> str:
    return ",".join(str(index) for index in range(max(0, int(limit))))


def _with_output_dir(args: argparse.Namespace, filename: str) -> str:
    return str(Path(args.output_dir) / filename)


def _common_hetero_args(args: argparse.Namespace) -> list[str]:
    return [
        "--hetero-smoke",
        "--sample-indices",
        _indices_csv(args.eval_limit),
        "--limit",
        str(args.eval_limit),
        "--generated-trajectory-adapter-input-space",
        args.generated_trajectory_adapter_input_space,
        "--generated-trajectory-adapter-train-limit",
        str(args.train_limit),
        "--enable-sender-revision",
    ]


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    methods_csv = ",".join(DEFAULT_METHODS)
    manifest_path = _with_output_dir(
        args,
        f"production_context_vs_latent_{args.eval_limit}_manifest.json",
    )
    commands: list[list[str]] = []
    if args.include_tests:
        commands.append([args.python, "-m", "pytest", "-q"])

    if args.prepare:
        commands.append(
            [
                args.python,
                "benchmark_all.py",
                *_common_hetero_args(args),
                "--methods",
                "generated_context_latent_handoff",
                "--prepare-generated-trajectory-eval-traces",
                "--report-output",
                _with_output_dir(
                    args,
                    f"production_eval_trace_warm_{args.eval_limit}.json",
                ),
            ]
        )
        commands.append(
            [
                args.python,
                "benchmark_all.py",
                *_common_hetero_args(args),
                "--methods",
                "generated_context_latent_handoff",
                "--prepare-generated-trajectory-adapter",
                "--report-output",
                _with_output_dir(
                    args,
                    f"production_adapter_{args.generated_trajectory_adapter_input_space}_{args.train_limit}.json",
                ),
            ]
        )

    commands.append(
        [
            args.python,
            "benchmark_all.py",
            *_common_hetero_args(args),
            "--methods",
            methods_csv,
            "--generated-trajectory-adapter-no-train-on-missing",
            "--report-output",
            _with_output_dir(args, f"production_context_vs_latent_{args.eval_limit}_report.json"),
            "--samples-output",
            _with_output_dir(args, f"production_context_vs_latent_{args.eval_limit}_samples.csv"),
            "--summary-output",
            _with_output_dir(args, f"production_context_vs_latent_{args.eval_limit}_summary.csv"),
            "--write-eval-manifest",
            manifest_path,
        ]
    )
    if args.replay:
        commands.append(
            [
                args.python,
                "benchmark_all.py",
                "--eval-manifest",
                manifest_path,
                "--generated-trajectory-adapter-input-space",
                args.generated_trajectory_adapter_input_space,
                "--enable-sender-revision",
                "--generated-trajectory-adapter-no-train-on-missing",
                "--report-output",
                _with_output_dir(
                    args,
                    f"production_context_vs_latent_{args.eval_limit}_replay_report.json",
                ),
                "--samples-output",
                _with_output_dir(
                    args,
                    f"production_context_vs_latent_{args.eval_limit}_replay_samples.csv",
                ),
                "--summary-output",
                _with_output_dir(
                    args,
                    f"production_context_vs_latent_{args.eval_limit}_replay_summary.csv",
                ),
            ]
        )
    return commands


def _print_command(command: Sequence[str]) -> None:
    print(" ".join(command), flush=True)


def _print_report_summary(report_path: Path) -> None:
    if not report_path.is_file():
        return
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    semantic = report.get("semantic_smoke_report") or {}
    comparison = report.get("transfer_comparison_report") or {}
    hetero = report.get("heterogeneous_transfer_report") or {}
    manifest = report.get("eval_manifest") or {}
    print("\nProduction validation summary", flush=True)
    print(
        json.dumps(
            {
                "semantic_passed": semantic.get("passed"),
                "method_accuracy_percentage": semantic.get("method_accuracy_percentage"),
                "comparison_passed": comparison.get("passed"),
                "best_latent_method": comparison.get("best_latent_method"),
                "heterogeneous_passed": hetero.get("passed"),
                "heterogeneous_missing_requirements": hetero.get("missing_requirements"),
                "sample_content_digest": manifest.get("sample_content_digest"),
                "eval_manifest_digest": manifest.get("manifest_digest"),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run or print the staged production validation ladder: tests, generated "
            "trace warm-up, adapter preparation, locked context-vs-latent comparison, "
            "and optional manifest replay."
        )
    )
    parser.add_argument("--python", default="venv/bin/python")
    parser.add_argument("--output-dir", default="outputs/production_validation")
    parser.add_argument("--profile", choices=tuple(PROFILE_DEFAULTS), default="local")
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument(
        "--generated-trajectory-adapter-input-space",
        choices=("raw", "aligned"),
        default="raw",
    )
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    defaults = PROFILE_DEFAULTS[args.profile]
    args.eval_limit = int(args.eval_limit or defaults["eval_limit"])
    args.train_limit = int(args.train_limit or defaults["train_limit"])
    args.include_tests = not args.skip_tests
    args.prepare = not args.skip_prepare

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    commands = build_commands(args)
    for command in commands:
        _print_command(command)
        if args.execute:
            subprocess.run(command, check=True)

    if args.execute:
        _print_report_summary(
            output_dir / f"production_context_vs_latent_{args.eval_limit}_report.json"
        )
    else:
        print("\nDry run only. Re-run with --execute to run the ladder.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
