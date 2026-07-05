from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn


RECEIVER_LORA_TARGET_SUFFIXES = (
    ".linear_attn.in_proj_qkv",
    ".linear_attn.out_proj",
    ".self_attn.q_proj",
    ".self_attn.k_proj",
    ".self_attn.v_proj",
    ".self_attn.o_proj",
)
RECEIVER_LORA_FORMAT = "receiver_lora_v1"
DEFAULT_RECEIVER_LORA_EXPECTED_COUNT = 60


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if not isinstance(base, nn.Linear):
            raise TypeError("base must be an nn.Linear")
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.base = base.to(dtype=torch.bfloat16)
        self.dropout = nn.Dropout(float(dropout))
        self.lora_enabled = True
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.A = nn.Parameter(
            torch.empty(
                self.rank,
                int(base.in_features),
                dtype=torch.float32,
                device=base.weight.device,
            )
        )
        self.B = nn.Parameter(
            torch.empty(
                int(base.out_features),
                self.rank,
                dtype=torch.float32,
                device=base.weight.device,
            )
        )
        nn.init.normal_(self.A, mean=0.0, std=0.02)
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base(x)
        if not self.lora_enabled:
            return output
        delta = self.dropout(x.float()) @ self.A.T @ self.B.T
        return output + (self.scaling * delta).to(dtype=x.dtype)


def apply_receiver_lora(
    model: nn.Module,
    *,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
    expected_count: int = DEFAULT_RECEIVER_LORA_EXPECTED_COUNT,
) -> dict[str, LoRALinear]:
    if expected_count < 0:
        raise ValueError("expected_count must be non-negative")
    modules = dict(model.named_modules())
    existing = {
        name: module
        for name, module in modules.items()
        if _is_receiver_lora_target(name) and isinstance(module, LoRALinear)
    }
    to_wrap = [
        (name, module)
        for name, module in modules.items()
        if _is_receiver_lora_target(name) and isinstance(module, nn.Linear)
    ]
    found_count = len(existing) + len(to_wrap)
    if found_count != expected_count:
        raise ValueError(
            f"Expected {expected_count} receiver LoRA modules, found {found_count}"
        )

    wrapped = dict(existing)
    for name, module in to_wrap:
        parent, attribute = _parent_module(model, name, modules)
        wrapper = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, attribute, wrapper)
        wrapped[name] = wrapper
    return dict(sorted(wrapped.items()))


def receiver_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, module in _receiver_lora_modules(model).items():
        state[f"{name}.A"] = module.A.detach().float().cpu().clone()
        state[f"{name}.B"] = module.B.detach().float().cpu().clone()
    return state


def load_receiver_lora(
    model: nn.Module,
    path: str | Path,
    *,
    strict: bool = True,
    expected_count: int = DEFAULT_RECEIVER_LORA_EXPECTED_COUNT,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("receiver LoRA artifact must be a dict")
    if payload.get("format") != RECEIVER_LORA_FORMAT:
        raise ValueError(f"receiver LoRA artifact format must be {RECEIVER_LORA_FORMAT}")
    _require_artifact_keys(payload)
    state = payload["state"]
    if not isinstance(state, dict):
        raise ValueError("receiver LoRA artifact state must be a dict")

    modules = _receiver_lora_modules(model)
    if not modules:
        modules = apply_receiver_lora(
            model,
            rank=int(payload["rank"]),
            alpha=float(payload["alpha"]),
            dropout=float(payload["dropout"]),
            expected_count=expected_count,
        )
    elif len(modules) != expected_count:
        raise ValueError(
            f"Expected {expected_count} receiver LoRA modules, found {len(modules)}"
        )

    expected_keys = {
        key
        for name in modules
        for key in (
            f"{name}.A",
            f"{name}.B",
        )
    }
    state_keys = set(state)
    missing = sorted(expected_keys - state_keys)
    unexpected = sorted(state_keys - expected_keys)
    if strict and missing:
        raise ValueError(f"Missing receiver LoRA tensor(s): {missing}")
    if strict and unexpected:
        raise ValueError(f"Unexpected receiver LoRA tensor(s): {unexpected}")

    with torch.no_grad():
        for name, module in modules.items():
            _copy_lora_tensor(module.A, state, f"{name}.A")
            _copy_lora_tensor(module.B, state, f"{name}.B")

    return {key: value for key, value in payload.items() if key != "state"}


def set_receiver_lora_enabled(model: nn.Module, enabled: bool) -> int:
    modules = _receiver_lora_modules(model)
    if not modules:
        raise ValueError("No receiver LoRA modules were found")
    for module in modules.values():
        module.lora_enabled = bool(enabled)
    return len(modules)


@contextmanager
def receiver_lora_scope(model: nn.Module, enabled: bool) -> Iterator[int]:
    modules = _receiver_lora_modules(model)
    if not modules:
        raise ValueError("No receiver LoRA modules were found")
    prior = {name: module.lora_enabled for name, module in modules.items()}
    try:
        for module in modules.values():
            module.lora_enabled = bool(enabled)
        yield len(modules)
    finally:
        refreshed = _receiver_lora_modules(model)
        for name, was_enabled in prior.items():
            if name in refreshed:
                refreshed[name].lora_enabled = was_enabled


def receiver_lora_file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_receiver_lora_target(name: str) -> bool:
    return (
        ".layers." in name
        and "visual" not in name
        and name.endswith(RECEIVER_LORA_TARGET_SUFFIXES)
    )


def _receiver_lora_modules(model: nn.Module) -> dict[str, LoRALinear]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    }


def _parent_module(
    model: nn.Module,
    name: str,
    modules: Mapping[str, nn.Module],
) -> tuple[nn.Module, str]:
    del model
    parent_name, attribute = name.rsplit(".", 1)
    parent = modules.get(parent_name)
    if parent is None:
        raise ValueError(f"Could not locate parent module for {name}")
    return parent, attribute


def _require_artifact_keys(payload: dict[str, Any]) -> None:
    required = {
        "rank",
        "alpha",
        "dropout",
        "target_suffixes",
        "base_model",
        "state",
        "train_args",
        "adapter_cache_key_digest",
        "objective",
        "dev_gate_accuracy",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Missing receiver LoRA artifact metadata: {missing}")
    if tuple(payload["target_suffixes"]) != RECEIVER_LORA_TARGET_SUFFIXES:
        raise ValueError("receiver LoRA artifact target suffixes do not match")


def _copy_lora_tensor(
    target: torch.Tensor,
    state: dict[Any, Any],
    key: str,
) -> None:
    if key not in state:
        return
    value = state[key]
    if not torch.is_tensor(value):
        raise ValueError(f"Receiver LoRA tensor {key} must be a torch.Tensor")
    if tuple(value.shape) != tuple(target.shape):
        raise ValueError(
            f"Receiver LoRA tensor {key} has shape {tuple(value.shape)}, "
            f"expected {tuple(target.shape)}"
        )
    target.copy_(value.to(device=target.device, dtype=torch.float32))
