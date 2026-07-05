from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from train_receiver_lora import (  # noqa: E402
    build_answer_weight_vector,
    build_cosine_with_warmup_scheduler,
    cosine_with_warmup_lr,
    evaluate_kill_rules,
    load_lora_checkpoint,
    save_lora_checkpoint,
)


def _gate(
    *,
    step: int,
    latent_lora_correct: int,
    latent_base_correct: int = 20,
    text_correct: int = 20,
    text_lora_correct: int | None = None,
    copy_accuracy: float | None = None,
    latent_accuracy: float | None = None,
    dev_nll: float | None = None,
) -> dict[str, object]:
    gate: dict[str, object] = {
        "optimizer_step": step,
        "latent_lora": {
            "correct_count": latent_lora_correct,
            "sample_count": 32,
            "accuracy": latent_accuracy
            if latent_accuracy is not None
            else latent_lora_correct / 32.0,
        },
        "latent_base": {"correct_count": latent_base_correct, "sample_count": 32},
        "text": {"correct_count": text_correct, "sample_count": 32},
    }
    if text_lora_correct is not None:
        gate["text_lora_canary"] = {
            "correct_count": text_lora_correct,
            "sample_count": 32,
        }
    if copy_accuracy is not None:
        gate["copy_proof"] = {"accuracy": copy_accuracy}
    if dev_nll is not None:
        gate["dev_nll"] = dev_nll
    return gate


def test_evaluate_kill_rules_latent_lora_below_base() -> None:
    result = evaluate_kill_rules(
        [
            _gate(step=0, latent_lora_correct=20),
            _gate(step=10, latent_lora_correct=17),
        ]
    )

    assert result["kill"] is True
    assert "latent_lora_below_base" in result["rules"]


def test_evaluate_kill_rules_text_lora_canary_below_text() -> None:
    result = evaluate_kill_rules(
        [
            _gate(step=0, latent_lora_correct=20, text_correct=20),
            _gate(step=20, latent_lora_correct=20, text_lora_correct=17),
        ]
    )

    assert result["kill"] is True
    assert "text_lora_canary_below_text" in result["rules"]


def test_evaluate_kill_rules_copy_proof_drop_with_overall_rise() -> None:
    result = evaluate_kill_rules(
        [
            _gate(step=0, latent_lora_correct=18, copy_accuracy=0.75, latent_accuracy=0.50),
            _gate(step=20, latent_lora_correct=22, copy_accuracy=0.50, latent_accuracy=0.60),
        ]
    )

    assert result["kill"] is True
    assert "copy_proof_drop_with_overall_rise" in result["rules"]


def test_evaluate_kill_rules_phase0_divergence() -> None:
    result = evaluate_kill_rules(
        [
            _gate(step=0, latent_lora_correct=24, dev_nll=4.0, latent_accuracy=0.75),
            _gate(step=10, latent_lora_correct=22, dev_nll=3.0, latent_accuracy=0.65),
            _gate(step=20, latent_lora_correct=21, dev_nll=2.0, latent_accuracy=0.60),
        ]
    )

    assert result["kill"] is True
    assert "phase0_divergence" in result["rules"]


def test_evaluate_kill_rules_non_trigger_passes() -> None:
    result = evaluate_kill_rules(
        [
            _gate(step=0, latent_lora_correct=20, copy_accuracy=0.50, dev_nll=4.0),
            _gate(step=10, latent_lora_correct=19, copy_accuracy=0.50, dev_nll=3.5),
            _gate(step=20, latent_lora_correct=20, copy_accuracy=0.50, dev_nll=3.0),
        ]
    )

    assert result == {"kill": False, "rules": [], "reason": None}


def test_cosine_with_warmup_schedule_shape() -> None:
    warmup = [
        cosine_with_warmup_lr(step, total_steps=12, base_lr=1e-4, min_lr=1e-5)
        for step in range(1, 7)
    ]

    assert warmup == pytest.approx([1e-4 * step / 6.0 for step in range(1, 7)])
    assert cosine_with_warmup_lr(
        12,
        total_steps=12,
        base_lr=1e-4,
        min_lr=1e-5,
    ) == pytest.approx(1e-5)

    parameter = nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4, weight_decay=0.0)
    scheduler = build_cosine_with_warmup_scheduler(
        optimizer,
        total_steps=12,
        base_lr=1e-4,
        min_lr=1e-5,
    )
    observed = [optimizer.param_groups[0]["lr"]]
    for _ in range(11):
        optimizer.step()
        scheduler.step()
        observed.append(optimizer.param_groups[0]["lr"])

    assert observed[0] == pytest.approx(1e-4 / 6.0)
    assert observed[5] == pytest.approx(1e-4)
    assert observed[-1] == pytest.approx(1e-5)


def test_checkpoint_save_resume_round_trip_preserves_optimizer(tmp_path) -> None:
    torch.manual_seed(11)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    scheduler = build_cosine_with_warmup_scheduler(
        optimizer,
        total_steps=8,
        base_lr=1e-4,
        min_lr=1e-5,
    )
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

    expected_weight = model.weight.detach().clone()
    expected_optimizer_state = optimizer.state_dict()
    path = tmp_path / "lora_checkpoint.pt"
    save_lora_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        position_in_epoch=3,
        sample_step=11,
        optimizer_step=1,
        best_dev_correct=7,
        best_dev_step=1,
        epoch_end_no_improve=0,
        args={"objective": "test"},
    )

    restored = nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4, weight_decay=0.0)
    restored_scheduler = build_cosine_with_warmup_scheduler(
        restored_optimizer,
        total_steps=8,
        base_lr=1e-4,
        min_lr=1e-5,
    )
    snapshot = load_lora_checkpoint(
        path,
        model=restored,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )

    assert snapshot["sample_step"] == 11
    assert torch.allclose(restored.weight, expected_weight)
    assert restored_optimizer.state_dict()["state"].keys() == expected_optimizer_state["state"].keys()
    for parameter_id, state in expected_optimizer_state["state"].items():
        restored_state = restored_optimizer.state_dict()["state"][parameter_id]
        for key, value in state.items():
            if torch.is_tensor(value):
                assert torch.allclose(restored_state[key], value)
            else:
                assert restored_state[key] == value


class _CharTokenizer:
    def decode(self, token_ids, *, skip_special_tokens=True):
        assert skip_special_tokens is True
        return "".join(chr(int(token_id)) for token_id in token_ids)


def test_objective_b_weight_vector_starts_at_final_answer_line() -> None:
    text = "work\nnot final answer: 2\nFinal answer: 4"
    token_ids = [ord(character) for character in text]

    weights = build_answer_weight_vector(_CharTokenizer(), token_ids)
    expected_start = text.rfind("Final answer:")

    assert weights[:expected_start] == [1.0] * expected_start
    assert weights[expected_start:] == [4.0] * (len(text) - expected_start)
