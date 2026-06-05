from __future__ import annotations

import argparse
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from run_training import (
    _format_actor_answer_prompt,
    _predicted_answer_for_target,
    _resolve_training_device,
    _resolve_training_torch_dtype,
    _select_candidate_answer_by_token_decoder,
    _select_latent_candidate_fallback,
    _unique_candidate_answers,
    _training_device_map,
)
from scripts.mac_mps_stage2_smoke import build_command


def _cfg(device: str = "auto", *, fallback_to_cpu: bool = True):
    return OmegaConf.create(
        {
            "torch_dtype": "bfloat16",
            "device_map": "auto",
            "runtime": {
                "device": device,
                "mps": {
                    "enabled": True,
                    "fallback_to_cpu": fallback_to_cpu,
                    "torch_dtype": "float32",
                },
            },
        }
    )


def test_resolve_training_device_prefers_mps_in_auto_mode() -> None:
    with patch("torch.backends.mps.is_available", return_value=True):
        assert _resolve_training_device(_cfg("auto")) == torch.device("mps")


def test_resolve_training_device_can_fallback_from_requested_mps_to_cpu() -> None:
    with patch("torch.backends.mps.is_available", return_value=False):
        assert _resolve_training_device(_cfg("mps")) == torch.device("cpu")


def test_resolve_training_device_can_require_mps() -> None:
    with patch("torch.backends.mps.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="mps"):
            _resolve_training_device(_cfg("mps", fallback_to_cpu=False))


def test_mps_dtype_uses_runtime_override_and_device_map_none() -> None:
    cfg = _cfg("mps")
    assert _resolve_training_torch_dtype(cfg, torch.device("mps")) == torch.float32
    assert _training_device_map(cfg, torch.device("mps")) is None


def test_smoke_answer_extraction_is_target_aware() -> None:
    assert _predicted_answer_for_target("smoke", "Final answer: 13.", "13") == "13"
    assert _predicted_answer_for_target("smoke", "The answer is 42.", "42") == "42"
    assert _predicted_answer_for_target("smoke", "3x^2", "3x^2") == "3x^2"
    assert (
        _predicted_answer_for_target(
            "smoke",
            "Final answer: 42 Final answer: 4",
            "4",
        )
        == "42"
    )


def test_smoke_candidate_answers_are_unique_and_ordered() -> None:
    candidates = _unique_candidate_answers(
        [
            {"prompt": "a", "answer": "13"},
            {"prompt": "b", "answer": "42"},
            {"prompt": "c", "answer": "13"},
            {"prompt": "d", "answer": None},
        ]
    )

    assert candidates == ("13", "42")


def test_actor_answer_prompt_uses_few_shot_examples_without_target_leak() -> None:
    prompt = _format_actor_answer_prompt(
        "What is 8 + 5?",
        baseline_examples=[
            {"prompt": "What is 2 + 2?", "answer": "4"},
            {"prompt": "What is 8 + 5?", "answer": "13"},
            {"prompt": "What is 6 * 7?", "answer": "42"},
        ],
    )

    assert "Question: What is 2 + 2?\nFinal answer: 4" in prompt
    assert "Question: What is 6 * 7?\nFinal answer: 42" in prompt
    assert "Final answer: 13" not in prompt
    assert prompt.rstrip().endswith("Question: What is 8 + 5?")


class _CandidateTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return {
            "4": [4],
            "42": [42],
            "5": [5],
        }.get(str(text).strip(), [])


def test_candidate_token_decoder_scores_whole_candidate_sequences() -> None:
    logits = torch.zeros(1, 3, 64)
    logits[0, 0, 42] = 8.0
    logits[0, 1, 0] = 8.0
    logits[0, 0, 4] = 2.0
    logits[0, 0, 5] = 1.0

    answer, nll = _select_candidate_answer_by_token_decoder(
        token_logits=logits,
        tokenizer=_CandidateTokenizer(),
        candidate_answers=("4", "42", "5"),
        max_answer_length=3,
    )

    assert answer == "42"
    assert nll is not None


def test_candidate_token_decoder_can_use_probe_prior_for_ambiguous_logits() -> None:
    logits = torch.zeros(1, 2, 64)
    logits[0, 0, 4] = 4.0
    logits[0, 1, 0] = 4.0
    logits[0, 0, 42] = 3.9

    answer, _ = _select_candidate_answer_by_token_decoder(
        token_logits=logits,
        tokenizer=_CandidateTokenizer(),
        candidate_answers=("4", "42"),
        max_answer_length=2,
        candidate_prior_nll={"4": 5.0, "42": 0.0},
        candidate_prior_weight=1.0,
    )

    assert answer == "42"


def test_latent_candidate_fallback_prefers_trained_latent_readouts() -> None:
    answer, source = _select_latent_candidate_fallback(
        candidate_answers=("4", "42", "5"),
        latent_probe_answer="42",
        latent_semantic_readout_answer="4",
        latent_token_decoder_answer="5",
        actor_nll_answer="4",
    )

    assert answer == "42"
    assert source == "latent_probe"


def test_mac_mps_stage2_smoke_command_is_small_and_explicit() -> None:
    args = argparse.Namespace(
        python="venv/bin/python",
        output_dir="outputs/mac_mps",
        agent_a_model="Qwen/Qwen3.5-0.8B",
        agent_b_model="Qwen/Qwen3.5-0.8B",
        batch_size=1,
        smoke_samples=4,
        max_length=64,
        compressed_steps=8,
        epochs=1,
        baseline_few_shot_examples=6,
        eval_on_train=False,
        allow_cpu_fallback=False,
    )

    command = build_command(args)

    assert command[:2] == ["venv/bin/python", "run_training.py"]
    assert "runtime.device=mps" in command
    assert "device_map=none" in command
    assert "runtime.mps.torch_dtype=float32" in command
    assert "training.data.batch_size=1" in command
    assert "training.data.smoke_num_samples=4" in command
    assert "training.evaluation.baseline_few_shot_examples=6" in command
    assert "agent_a_model=Qwen/Qwen3.5-0.8B" in command
    assert "agent_b_model=Qwen/Qwen3.5-0.8B" in command


def test_mac_mps_stage2_smoke_command_can_eval_on_train() -> None:
    args = argparse.Namespace(
        python="venv/bin/python",
        output_dir="outputs/mac_mps",
        agent_a_model="Qwen/Qwen3.5-0.8B",
        agent_b_model="Qwen/Qwen3.5-0.8B",
        batch_size=1,
        smoke_samples=4,
        max_length=64,
        compressed_steps=8,
        epochs=3,
        baseline_few_shot_examples=4,
        eval_on_train=True,
        allow_cpu_fallback=False,
    )

    command = build_command(args)

    assert "training.evaluation.smoke_eval_set=train_overfit" in command
    assert "training.evaluation.semantic_readout_only=true" in command
    assert "training.evaluation.semantic_bridge_actor_decode=true" in command
    assert "training.evaluation.semantic_bridge_selected_answer_bias=100.0" in command
    assert "training.evaluation.latent_token_decoder_probe_prior_weight=8.0" in command
    assert "training.evaluation.baseline_few_shot_examples=4" in command
    assert "training.train_reasoner=false" in command
    assert "training.learning_rate=3.0e-4" in command
    assert "training.max_grad_norm=5.0" in command
    assert "training.lambda_answer=0.0" in command
    assert "training.lambda_answer_first_token=0.0" in command
    assert "training.lambda_logit_steering=0.0" in command
    assert "training.lambda_latent_token_decoder=160.0" in command
    assert "training.lambda_answer_contrast=0.0" in command
    assert "training.lambda_answer_probe=20.0" in command
    assert "training.answer_contrast_temperature=0.5" in command
    assert "training.answer_first_token_weight=8.0" in command
    assert "training.answer_first_token_margin=4.0" in command
    assert "training.logit_steering_margin=8.0" in command
    assert "training.adaptive_loss.enabled=false" in command
    assert "training.latent_answer_probe.enabled=true" in command
    assert "training.latent_logit_steering.enabled=false" in command
    assert "training.latent_token_decoder.enabled=true" in command
    assert "training.latent_token_decoder.rank=128" in command
    assert "training.latent_token_decoder.vocabulary_mode=low_rank" in command
    assert "training.latent_token_decoder.lr_multiplier=10.0" in command
    assert "training.latent_token_decoder.output_steps=8" in command
    assert "training.latent_token_decoder.candidate_token_mask=true" in command
    assert "training.latent_token_decoder.require_ready=true" in command
    assert "training.latent_token_decoder.eos_weight=2.0" in command
    assert "training.latent_token_decoder.margin=4.0" in command
    assert "training.latent_soft_prompt_decoder.enabled=false" in command
    assert "training.latent_soft_prompt_decoder.output_steps=0" in command


def test_mac_mps_stage2_smoke_full_decode_trains_raw_actor_steering() -> None:
    args = argparse.Namespace(
        python="venv/bin/python",
        output_dir="outputs/mac_mps",
        agent_a_model="Qwen/Qwen3.5-0.8B",
        agent_b_model="Qwen/Qwen3.5-0.8B",
        batch_size=3,
        smoke_samples=3,
        max_length=64,
        compressed_steps=8,
        epochs=20,
        baseline_few_shot_examples=4,
        eval_on_train=True,
        full_decode_eval=True,
        raw_decode_output_steps=2,
        raw_answer_loss_weight=12.0,
        raw_answer_first_token_loss_weight=4.0,
        raw_logit_steering_weight=160.0,
        raw_logit_steering_lr_multiplier=20.0,
        raw_logit_steering_generation_scale=1.0,
        raw_logit_steering_answer_token_weight=1.0,
        raw_logit_steering_later_answer_token_weight=1.0,
        raw_logit_steering_eos_weight=1.0,
        raw_smoke_max_loss=5000.0,
        allow_cpu_fallback=False,
    )

    command = build_command(args)

    assert "training.evaluation.semantic_readout_only=false" in command
    assert "training.evaluation.semantic_bridge_actor_decode=false" in command
    assert "training.evaluation.require_raw_decode_ready=true" in command
    assert "training.evaluation.raw_decode_stop_after_steering=true" in command
    assert "training.evaluation.raw_decode_stop_by_semantic_readout_length=true" in command
    assert "training.evaluation.early_stop_raw_decode_ready=true" in command
    assert "training.evaluation.smoke_max_loss=5000.0" in command
    assert "training.max_grad_norm=10.0" in command
    assert "training.lambda_answer=12.0" in command
    assert "training.lambda_answer_first_token=4.0" in command
    assert "training.lambda_logit_steering=160.0" in command
    assert "training.lambda_latent_token_decoder=0.0" in command
    assert "training.latent_logit_steering.enabled=true" in command
    assert "training.latent_logit_steering.vocabulary_mode=low_rank" in command
    assert "training.latent_logit_steering.lr_multiplier=20.0" in command
    assert "training.latent_logit_steering.output_steps=2" in command
    assert "training.latent_logit_steering.generation_scale=1.0" in command
    assert "training.latent_logit_steering.answer_token_weight=1.0" in command
    assert "training.latent_logit_steering.later_answer_token_weight=1.0" in command
    assert "training.latent_logit_steering.eos_weight=1.0" in command
    assert "training.latent_logit_steering.pooling=mean_last" in command
    assert "training.latent_token_decoder.enabled=false" in command
    assert "training.latent_token_decoder.require_ready=false" in command
