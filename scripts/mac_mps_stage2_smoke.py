from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.render_stage2_report import render_stage2_report


def build_command(args: argparse.Namespace) -> list[str]:
    output_dir = Path(args.output_dir)
    full_decode_eval = bool(getattr(args, "full_decode_eval", False))
    auxiliary_decode_surfaces = not full_decode_eval
    raw_decode_output_steps = int(getattr(args, "raw_decode_output_steps", 2))
    raw_answer_loss_weight = float(getattr(args, "raw_answer_loss_weight", 12.0))
    raw_answer_first_token_loss_weight = float(
        getattr(args, "raw_answer_first_token_loss_weight", 4.0)
    )
    raw_logit_steering_weight = float(getattr(args, "raw_logit_steering_weight", 160.0))
    raw_logit_steering_lr_multiplier = float(
        getattr(args, "raw_logit_steering_lr_multiplier", 20.0)
    )
    raw_logit_steering_generation_scale = float(
        getattr(args, "raw_logit_steering_generation_scale", 1.0)
    )
    raw_logit_steering_answer_token_weight = float(
        getattr(args, "raw_logit_steering_answer_token_weight", 1.0)
    )
    raw_logit_steering_later_answer_token_weight = float(
        getattr(args, "raw_logit_steering_later_answer_token_weight", 1.0)
    )
    raw_logit_steering_eos_weight = float(
        getattr(args, "raw_logit_steering_eos_weight", 1.0)
    )
    raw_smoke_max_loss = float(getattr(args, "raw_smoke_max_loss", 5000.0))
    command = [
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
        f"training.evaluation.baseline_few_shot_examples={args.baseline_few_shot_examples}",
        "training.checkpointing.enabled=false",
        f"reporting.training.history_output={output_dir / 'mac_mps_training_history.csv'}",
        f"reporting.training.report_output={output_dir / 'mac_mps_training_report.json'}",
    ]
    if args.eval_on_train:
        command.extend(
            [
                "training.evaluation.smoke_eval_set=train_overfit",
                f"training.evaluation.semantic_readout_only={str(not full_decode_eval).lower()}",
                f"training.evaluation.semantic_bridge_actor_decode={str(auxiliary_decode_surfaces).lower()}",
                "training.evaluation.semantic_bridge_selected_answer_bias=100.0",
                "training.evaluation.latent_token_decoder_probe_prior_weight=8.0",
                f"training.evaluation.require_raw_decode_ready={str(full_decode_eval).lower()}",
                "training.evaluation.raw_decode_stop_after_steering=true",
                f"training.evaluation.raw_decode_stop_by_semantic_readout_length={str(full_decode_eval).lower()}",
                f"training.evaluation.early_stop_raw_decode_ready={str(full_decode_eval).lower()}",
                f"training.evaluation.smoke_max_loss={raw_smoke_max_loss if full_decode_eval else 1000.0}",
                "training.curriculum.enabled=false",
                "training.adaptive_loss.enabled=false",
                "training.learning_rate=3.0e-4",
                f"training.max_grad_norm={10.0 if full_decode_eval else 5.0}",
                f"training.lambda_answer={raw_answer_loss_weight if full_decode_eval else 0.0}",
                "training.lambda_answer_first_token="
                f"{raw_answer_first_token_loss_weight if full_decode_eval else 0.0}",
                f"training.lambda_logit_steering={raw_logit_steering_weight if full_decode_eval else 0.0}",
                f"training.lambda_latent_token_decoder={0.0 if full_decode_eval else 160.0}",
                "training.lambda_answer_contrast=0.0",
                "training.lambda_answer_probe=20.0",
                "training.answer_contrast_temperature=0.5",
                "training.answer_first_token_weight=8.0",
                "training.answer_first_token_margin=4.0",
                "training.logit_steering_margin=8.0",
                "training.lambda_task=0.0",
                "training.lambda_pref=0.0",
                "training.lambda_geom=0.0",
                "training.lambda_plan=0.0",
                "training.lambda_contrast=0.0",
                "training.train_reasoner=false",
                "training.latent_handoff_adapter.enabled=true",
                "training.latent_answer_probe.enabled=true",
                f"training.latent_logit_steering.enabled={str(full_decode_eval).lower()}",
                "training.latent_logit_steering.rank=128",
                "training.latent_logit_steering.vocabulary_mode=low_rank",
                f"training.latent_logit_steering.lr_multiplier={raw_logit_steering_lr_multiplier}",
                f"training.latent_logit_steering.output_steps={raw_decode_output_steps}",
                f"training.latent_logit_steering.generation_scale={raw_logit_steering_generation_scale}",
                f"training.latent_logit_steering.answer_token_weight={raw_logit_steering_answer_token_weight}",
                "training.latent_logit_steering.later_answer_token_weight="
                f"{raw_logit_steering_later_answer_token_weight}",
                f"training.latent_logit_steering.eos_weight={raw_logit_steering_eos_weight}",
                "training.latent_logit_steering.pooling=mean_last",
                f"training.latent_token_decoder.enabled={str(auxiliary_decode_surfaces).lower()}",
                "training.latent_token_decoder.rank=128",
                "training.latent_token_decoder.vocabulary_mode=low_rank",
                "training.latent_token_decoder.lr_multiplier=10.0",
                "training.latent_token_decoder.output_steps=8",
                "training.latent_token_decoder.candidate_token_mask=true",
                f"training.latent_token_decoder.require_ready={str(auxiliary_decode_surfaces).lower()}",
                "training.latent_token_decoder.eos_weight=2.0",
                "training.latent_token_decoder.margin=4.0",
                "training.latent_soft_prompt_decoder.enabled=false",
                "training.latent_soft_prompt_decoder.output_steps=0",
            ]
        )
    return command


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
            "raw_decode_exact_match_accuracy": smoke_report.get(
                "final_heldout_raw_decode_exact_match_accuracy"
            ),
            "raw_decode_answer_extraction_rate": smoke_report.get(
                "final_heldout_raw_decode_answer_extraction_rate_percentage"
            ),
            "raw_decode_require_ready": smoke_report.get(
                "final_heldout_raw_decode_require_ready"
            ),
            "raw_actor_free_decoder_ready": smoke_report.get(
                "raw_actor_free_decoder_ready"
            ),
            "actor_semantic_bridge_decode_accuracy": smoke_report.get(
                "final_heldout_actor_semantic_bridge_decode_accuracy"
            ),
            "actor_semantic_bridge_decode_extraction_rate": smoke_report.get(
                "final_heldout_actor_semantic_bridge_decode_answer_extraction_rate_percentage"
            ),
            "actor_semantic_bridge_decoder_ready": smoke_report.get(
                "actor_semantic_bridge_decoder_ready"
            ),
            "latent_token_decode_accuracy": smoke_report.get(
                "final_heldout_latent_token_decode_accuracy"
            ),
            "latent_token_decode_enabled": smoke_report.get(
                "final_heldout_latent_token_decode_enabled"
            ),
            "latent_token_decode_require_ready": smoke_report.get(
                "final_heldout_latent_token_decode_require_ready"
            ),
            "latent_token_decode_extraction_rate": smoke_report.get(
                "final_heldout_latent_token_decode_answer_extraction_rate_percentage"
            ),
            "latent_token_decoder_ready": smoke_report.get(
                "latent_token_decoder_ready"
            ),
            "latent_semantic_readout_accuracy": smoke_report.get(
                "final_heldout_latent_semantic_readout_accuracy"
            ),
            "latent_semantic_readout_rate": smoke_report.get(
                "final_heldout_latent_semantic_readout_rate_percentage"
            ),
            "candidate_fallback_rate": smoke_report.get(
                "final_heldout_candidate_fallback_rate_percentage"
            ),
            "latent_candidate_accuracy": smoke_report.get(
                "final_heldout_latent_candidate_accuracy"
            ),
            "latent_candidate_fallback_ready": smoke_report.get(
                "latent_candidate_fallback_ready"
            ),
            "latent_probe_accuracy": smoke_report.get(
                "final_heldout_latent_probe_accuracy"
            ),
            "latent_first_token_accuracy": smoke_report.get(
                "final_heldout_latent_first_token_accuracy"
            ),
            "latent_first_token_rank_mean": smoke_report.get(
                "final_heldout_latent_first_token_rank_mean"
            ),
            "latent_probe_ready": smoke_report.get("latent_probe_ready"),
            "unique_predicted_answer_count": smoke_report.get(
                "final_heldout_unique_predicted_answer_count"
            ),
            "degenerate_prediction": smoke_report.get(
                "final_heldout_degenerate_prediction"
            ),
            "actor_text_baseline_accuracy": smoke_report.get(
                "final_heldout_actor_text_baseline_accuracy"
            ),
            "actor_text_baseline_degenerate_prediction": smoke_report.get(
                "final_heldout_actor_text_baseline_degenerate_prediction"
            ),
            "actor_text_baseline_unique_predicted_answer_count": smoke_report.get(
                "final_heldout_actor_text_baseline_unique_predicted_answer_count"
            ),
            "actor_text_baseline_candidate_accuracy": smoke_report.get(
                "final_heldout_actor_text_baseline_candidate_accuracy"
            ),
            "latent_training_ready": smoke_report.get("latent_training_ready"),
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
    parser.add_argument("--baseline-few-shot-examples", type=int, default=6)
    parser.add_argument("--eval-on-train", action="store_true")
    parser.add_argument("--full-decode-eval", action="store_true")
    parser.add_argument("--raw-decode-output-steps", type=int, default=2)
    parser.add_argument("--raw-answer-loss-weight", type=float, default=12.0)
    parser.add_argument("--raw-answer-first-token-loss-weight", type=float, default=4.0)
    parser.add_argument("--raw-logit-steering-weight", type=float, default=160.0)
    parser.add_argument("--raw-logit-steering-lr-multiplier", type=float, default=20.0)
    parser.add_argument("--raw-logit-steering-generation-scale", type=float, default=1.0)
    parser.add_argument("--raw-logit-steering-answer-token-weight", type=float, default=1.0)
    parser.add_argument("--raw-logit-steering-later-answer-token-weight", type=float, default=1.0)
    parser.add_argument("--raw-logit-steering-eos-weight", type=float, default=1.0)
    parser.add_argument("--raw-smoke-max-loss", type=float, default=5000.0)
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
    output_dir = Path(args.output_dir)
    report_path = output_dir / "mac_mps_training_report.json"
    history_path = output_dir / "mac_mps_training_history.csv"
    html_path = output_dir / "mac_mps_training_report.html"
    render_stage2_report(report_path, history_path, html_path)
    print(f"Wrote visual training report to {html_path}", flush=True)
    _print_report_summary(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
