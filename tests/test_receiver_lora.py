from __future__ import annotations

import copy
import hashlib

import pytest
import torch
from torch import nn

from src.models.receiver_lora import (
    RECEIVER_LORA_FORMAT,
    RECEIVER_LORA_TARGET_SUFFIXES,
    LoRALinear,
    apply_receiver_lora,
    load_receiver_lora,
    receiver_lora_file_sha256,
    receiver_lora_scope,
    receiver_lora_state_dict,
    set_receiver_lora_enabled,
)


class _TinySelfAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))


class _TinyLinearAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.in_proj_qkv = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.in_proj_qkv(x))


class _TinyLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.self_attn = _TinySelfAttention(hidden_size)
        self.linear_attn = _TinyLinearAttention(hidden_size)
        self.mlp = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_attn(self.self_attn(x))


class _TinyBackbone(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_TinyLayer(hidden_size)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _TinyReceiver(nn.Module):
    def __init__(self, hidden_size: int = 4) -> None:
        super().__init__()
        self.backbone = _TinyBackbone(hidden_size)
        self.visual = _TinyBackbone(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def test_zero_init_identity_when_enabled_and_disabled() -> None:
    torch.manual_seed(0)
    model = _TinyReceiver()
    base = copy.deepcopy(model).to(dtype=torch.bfloat16)
    x = torch.randn(2, 3, 4, dtype=torch.bfloat16)

    wrapped = apply_receiver_lora(
        model,
        rank=2,
        alpha=4.0,
        dropout=0.0,
        expected_count=6,
    )

    assert len(wrapped) == 6
    assert all(torch.count_nonzero(module.B).item() == 0 for module in wrapped.values())
    torch.testing.assert_close(model(x), base(x))

    assert set_receiver_lora_enabled(model, False) == 6
    torch.testing.assert_close(model(x), base(x))


def test_enable_disable_toggling_changes_lora_contribution() -> None:
    torch.manual_seed(1)
    model = _TinyReceiver()
    x = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    wrapped = apply_receiver_lora(
        model,
        rank=2,
        alpha=4.0,
        dropout=0.0,
        expected_count=6,
    )
    with torch.no_grad():
        for module in wrapped.values():
            module.A.fill_(0.25)
            module.B.fill_(0.125)

    enabled_output = model(x)
    assert set_receiver_lora_enabled(model, False) == 6
    disabled_output = model(x)
    assert set_receiver_lora_enabled(model, True) == 6

    assert not torch.allclose(enabled_output, disabled_output)
    assert all(module.lora_enabled for module in wrapped.values())


def test_exact_module_count_enforcement_raises_on_mismatch() -> None:
    model = _TinyReceiver()

    with pytest.raises(ValueError, match="Expected 5 receiver LoRA modules, found 6"):
        apply_receiver_lora(model, expected_count=5)

    assert not any(isinstance(module, LoRALinear) for module in model.modules())


def test_state_dict_save_load_round_trip_preserves_outputs(tmp_path) -> None:
    torch.manual_seed(2)
    base = _TinyReceiver()
    model = copy.deepcopy(base)
    reloaded = copy.deepcopy(base)
    x = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    wrapped = apply_receiver_lora(
        model,
        rank=2,
        alpha=4.0,
        dropout=0.0,
        expected_count=6,
    )
    with torch.no_grad():
        for index, module in enumerate(wrapped.values(), start=1):
            module.A.fill_(0.01 * index)
            module.B.fill_(0.02 * index)

    expected = model(x)
    state = receiver_lora_state_dict(model)
    assert state
    assert all(tensor.dtype == torch.float32 for tensor in state.values())
    path = tmp_path / "receiver_lora.pt"
    torch.save(
        {
            "format": RECEIVER_LORA_FORMAT,
            "rank": 2,
            "alpha": 4.0,
            "dropout": 0.0,
            "target_suffixes": RECEIVER_LORA_TARGET_SUFFIXES,
            "base_model": "tiny",
            "state": state,
            "train_args": {},
            "adapter_cache_key_digest": "digest",
            "objective": "A",
            "dev_gate_accuracy": 1.0,
        },
        path,
    )

    metadata = load_receiver_lora(reloaded, path, expected_count=6)

    assert metadata["format"] == RECEIVER_LORA_FORMAT
    torch.testing.assert_close(reloaded(x), expected)


def test_scope_context_manager_restores_prior_state() -> None:
    model = _TinyReceiver()
    wrapped = apply_receiver_lora(model, expected_count=6)
    modules = list(wrapped.values())
    modules[0].lora_enabled = False
    prior = [module.lora_enabled for module in modules]

    with receiver_lora_scope(model, True) as toggled:
        assert toggled == 6
        assert all(module.lora_enabled for module in modules)

    assert [module.lora_enabled for module in modules] == prior


def test_sha256_helper_stability(tmp_path) -> None:
    path = tmp_path / "artifact.pt"
    payload = b"receiver-lora-artifact"
    path.write_bytes(payload)

    expected = hashlib.sha256(payload).hexdigest()

    assert receiver_lora_file_sha256(path) == expected
    assert receiver_lora_file_sha256(path) == expected
