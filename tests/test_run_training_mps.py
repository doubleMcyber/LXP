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
    assert "training.evaluation.baseline_few_shot_examples=4" in command
