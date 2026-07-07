"""Receiver-LoRA rollout preparation and training.

Builds the same context/latent/target samples the production fused-forward path
consumes, prepares verified rollout continuations for objectives A/B, and trains
the receiver-internal LoRA adapters with the registered gate system.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from benchmark_all import _answers_match, _predicted_answer  # noqa: E402
from scripts.train_latent_bridge import (  # noqa: E402
    ALONE_INSTRUCTION,
    CONTINUATION_INSTRUCTION,
    TEXT_INSTRUCTION,
    chat_prefix_ids,
    load_bridge_samples,
)
from src.models.receiver_lora import (  # noqa: E402
    RECEIVER_LORA_FORMAT,
    RECEIVER_LORA_TARGET_SUFFIXES,
    apply_receiver_lora,
    receiver_lora_state_dict,
    set_receiver_lora_enabled,
)
from src.utils.alignment import apply_alignment  # noqa: E402
from src.utils.lm_eval import (  # noqa: E402
    _last_logit_forward_kwargs,
    _normalize_kv_cache,
    _reset_kv_cache_to_prefix,
    build_position_ids,
    greedy_decode_from_prefix,
    prepare_text_prefix_state,
    prepare_receiver_context_latent_prefix_state,
)

DEFAULT_ADAPTER_CACHE_PATH = (
    ".cache/generated_trajectory_adapter/"
    "generated_trajectory_adapter_8959603bbf6825e4dc4df3f373ca600ccd7e61425e0aa9c66d7f0adb1070b207.pt"
)
DEFAULT_ADAPTER_DIGEST = (
    "5c041b5521a4663a1a469410c7114df8753762df45f7b6fad66bb9ca2f03c02b"
)
DEFAULT_OUTPUT_DIR = "outputs/receiver_lora/objective_a"
DEFAULT_ROLLOUT_CACHE_DIR = REPO_ROOT / ".cache" / "receiver_lora_rollouts"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receiver-model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--sender-model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--trace-dtype", default="bfloat16")
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--train-rows", default="0:224")
    parser.add_argument("--dev-rows", default="224:256")
    parser.add_argument("--truncation-fraction", type=float, default=0.5)
    parser.add_argument("--validation-size", type=int, default=256)
    parser.add_argument("--adapter-cache-path", default=DEFAULT_ADAPTER_CACHE_PATH)
    parser.add_argument("--adapter-digest", default=DEFAULT_ADAPTER_DIGEST)
    parser.add_argument("--objective", choices=("A", "B", "C"), default="A")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-continuation-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--rollout-max-tries", type=int, default=4)
    parser.add_argument("--rollout-temperature", type=float, default=0.7)
    parser.add_argument("--rollout-top-p", type=float, default=0.95)
    parser.add_argument("--checkpoint-every", type=int, default=32)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prepare-rollouts", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    return parser


def parse_row_range(row_range: str) -> tuple[int, int]:
    try:
        start_text, stop_text = str(row_range).split(":", 1)
        start = int(start_text)
        stop = int(stop_text)
    except ValueError as error:
        raise ValueError(f"row range must be START:STOP, got {row_range!r}") from error
    if start < 0 or stop < start:
        raise ValueError(f"row range must satisfy 0 <= START <= STOP, got {row_range!r}")
    return start, stop


def prompt_sha(prompt: str) -> str:
    return hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()


def context_text(tokenizer: Any, question: str) -> str:
    user_message = f"{question}\n\n{CONTINUATION_INSTRUCTION}"
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:  # noqa: BLE001
        return user_message


def chat_prefix_text(
    tokenizer: Any,
    question: str,
    instruction: str,
    body: str = "",
) -> str:
    user_message = f"{question}\n\n{instruction}"
    if body:
        user_message = f"{question}\n\nUnfinished reasoning:\n{body}\n\n{instruction}"
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:  # noqa: BLE001
        return user_message


def build_context_ids(tokenizer: Any, question: str) -> list[int]:
    return chat_prefix_ids(tokenizer, question, CONTINUATION_INSTRUCTION)


def load_frozen_adapter(
    adapter_cache_path: str | Path,
    *,
    adapter_digest: str = DEFAULT_ADAPTER_DIGEST,
) -> dict[str, Any]:
    adapter_payload = torch.load(
        Path(adapter_cache_path),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(adapter_payload, dict):
        raise ValueError("adapter payload must be a dict")
    actual_digest = adapter_payload.get("adapter_cache_key_digest")
    assert actual_digest == adapter_digest, (
        "adapter digest mismatch: "
        f"expected {adapter_digest}, found {actual_digest}"
    )
    return adapter_payload


def project_sample_latents(sample: dict[str, Any], adapter_payload: dict[str, Any]) -> torch.Tensor:
    latents = sample["latents"]
    if not torch.is_tensor(latents):
        raise TypeError("sample latents must be a torch.Tensor")
    projected = apply_alignment(latents.unsqueeze(0).float(), adapter_payload)
    return projected.to(torch.bfloat16)


def assemble_training_sample(
    sample: dict[str, Any],
    *,
    tokenizer: Any,
    adapter_payload: dict[str, Any],
) -> dict[str, Any]:
    target_ids = [int(token_id) for token_id in sample["continuation_ids"]]
    return {
        "question": sample["question"],
        "answer": sample["answer"],
        "sample_index": int(sample["sample_index"]),
        "truncated_text": sample.get("truncated_text", ""),
        "context_ids": build_context_ids(tokenizer, str(sample["question"])),
        "context_text": context_text(tokenizer, str(sample["question"])),
        "latents": project_sample_latents(sample, adapter_payload),
        "target_ids": target_ids,
        "target_continuation_ids": target_ids,
    }


def _select_sample_index_range(
    samples: list[dict[str, Any]],
    row_range: tuple[int, int],
) -> list[dict[str, Any]]:
    start, stop = row_range
    return [
        sample
        for sample in samples
        if start <= int(sample["sample_index"]) < stop
    ]


def load_bridge_sample_rows(
    args: argparse.Namespace,
    *,
    tokenizer: Any,
    include_dev: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    train_range = parse_row_range(args.train_rows)
    dev_range = parse_row_range(args.dev_rows)
    load_limit = max(train_range[1], dev_range[1] if include_dev else train_range[1])
    samples = load_bridge_samples(
        dataset=args.dataset,
        split="train",
        limit=load_limit,
        model_id=args.sender_model,
        torch_dtype=args.trace_dtype,
        truncation_fraction=args.truncation_fraction,
        tokenizer=tokenizer,
        max_continuation_tokens=args.max_continuation_tokens,
        validation_size=args.validation_size,
    )
    rows = {"train": _select_sample_index_range(samples, train_range)}
    if include_dev:
        rows["dev"] = _select_sample_index_range(samples, dev_range)
    return rows


def assemble_training_samples(
    samples: list[dict[str, Any]],
    *,
    tokenizer: Any,
    adapter_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        assemble_training_sample(sample, tokenizer=tokenizer, adapter_payload=adapter_payload)
        for sample in samples
    ]


def _stable_json_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=list, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def rollout_config_digest(
    *,
    receiver_model: str,
    adapter_digest: str,
    truncation_fraction: float,
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> str:
    return _stable_json_digest(
        {
            "receiver_model": str(receiver_model),
            "adapter_digest": str(adapter_digest),
            "truncation_fraction": float(truncation_fraction),
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "seed": int(seed),
        }
    )


def rollout_cache_path(
    question: str,
    *,
    cache_dir: str | Path = DEFAULT_ROLLOUT_CACHE_DIR,
) -> Path:
    return Path(cache_dir) / f"rollout_{prompt_sha(question)}.pt"


def build_answer_weight_vector(
    tokenizer: Any,
    token_ids: list[int] | tuple[int, ...],
    *,
    final_answer_weight: float = 4.0,
) -> list[float]:
    token_ids = [int(token_id) for token_id in token_ids]
    weights = [1.0 for _ in token_ids]
    if not token_ids:
        return weights

    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    matches = list(re.finditer(r"final\s+answer\s*:", decoded, flags=re.IGNORECASE))
    if not matches:
        return weights

    marker_start = int(matches[-1].start())
    line_start = decoded.rfind("\n", 0, marker_start) + 1
    start_char = max(0, line_start)
    start_index = len(token_ids) - 1
    for index in range(len(token_ids)):
        prefix_text = tokenizer.decode(token_ids[: index + 1], skip_special_tokens=True)
        if len(prefix_text) > start_char:
            start_index = index
            break

    for index in range(start_index, len(weights)):
        weights[index] = float(final_answer_weight)
    return weights


def validate_target_token_ids(
    token_ids: Any,
    *,
    max_continuation_tokens: int,
    target_vocab_size: Optional[int] = None,
) -> tuple[Optional[list[int]], Optional[str]]:
    if not isinstance(token_ids, (list, tuple)):
        return None, "token_ids_not_sequence"
    if not token_ids:
        return None, "empty_target"
    max_tokens = int(max_continuation_tokens)
    if max_tokens <= 0:
        return None, "empty_after_truncation"
    if target_vocab_size is not None and int(target_vocab_size) <= 0:
        raise ValueError("target_vocab_size must be positive when provided")
    target_ids: list[int] = []
    for index, token_id in enumerate(token_ids[:max_tokens]):
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            return None, f"non_int_token:{index}:{type(token_id).__name__}"
        token_int = int(token_id)
        if token_int < 0:
            return None, f"negative_token:{index}:{token_int}"
        if target_vocab_size is not None and token_int >= int(target_vocab_size):
            return None, f"token_out_of_vocab:{index}:{token_int}>={int(target_vocab_size)}"
        target_ids.append(token_int)
    if not target_ids:
        return None, "empty_after_truncation"
    return target_ids, None


def validate_target_weights(
    weights: Any,
    *,
    target_len: int,
) -> tuple[Optional[list[float]], Optional[str]]:
    if not isinstance(weights, (list, tuple)):
        return None, "target_weights_not_sequence"
    if len(weights) != int(target_len):
        return None, "target_weights_length_mismatch"
    weight_values: list[float] = []
    for index, weight in enumerate(weights):
        try:
            weight_value = float(weight)
        except (TypeError, ValueError):
            return None, f"non_float_weight:{index}:{type(weight).__name__}"
        if not math.isfinite(weight_value):
            return None, f"non_finite_weight:{index}:{weight_value}"
        weight_values.append(weight_value)
    if sum(weight_values) <= 0.0:
        return None, "non_positive_weight_sum"
    return weight_values, None


def finite_loss_value(loss: torch.Tensor) -> Optional[float]:
    value = float(loss.detach().float().cpu().item())
    return value if math.isfinite(value) else None


def nonfinite_grad_names(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> list[str]:
    """Names of parameters whose accumulated .grad contains non-finite values.

    Parameters without a gradient are skipped. This is the gradient-level
    analog of the non-finite-loss guard: on MPS a backward pass can produce
    inf/nan gradients while the loss itself stays finite, and
    clip_grad_norm_ would then silently scale every gradient by nan.
    """
    names: list[str] = []
    for name, parameter in named_parameters:
        grad = parameter.grad
        if grad is not None and not bool(torch.isfinite(grad).all()):
            names.append(str(name))
    return names


def _skip_reason_key(reason: Optional[str]) -> str:
    if not reason:
        return "unknown"
    return str(reason).split(":", 1)[0]


def _record_sample_skip(
    skipped: list[int],
    skip_counts: dict[str, int],
    sample: dict[str, Any],
    reason: Optional[str],
) -> None:
    skipped.append(int(sample["sample_index"]))
    key = _skip_reason_key(reason)
    skip_counts[key] = skip_counts.get(key, 0) + 1


def _format_skip_counts(skip_counts: dict[str, int]) -> str:
    return ", ".join(f"{key}={skip_counts[key]}" for key in sorted(skip_counts))


def cosine_with_warmup_lr(
    step: int,
    *,
    total_steps: int,
    base_lr: float = 1e-4,
    min_lr: float = 1e-5,
    warmup_steps: int = 6,
) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if base_lr <= 0.0:
        raise ValueError("base_lr must be positive")
    if min_lr < 0.0:
        raise ValueError("min_lr must be non-negative")
    step = max(0, int(step))
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))
    if step <= 0:
        return 0.0
    if warmup_steps and step <= warmup_steps:
        return float(base_lr) * float(step) / float(warmup_steps)
    if total_steps <= warmup_steps:
        return float(base_lr)
    decay_step = min(step, total_steps)
    progress = (decay_step - warmup_steps) / float(total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr) + (float(base_lr) - float(min_lr)) * cosine


def build_cosine_with_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    base_lr: float = 1e-4,
    min_lr: float = 1e-5,
    warmup_steps: int = 6,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(scheduler_step: int) -> float:
        lr = cosine_with_warmup_lr(
            int(scheduler_step) + 1,
            total_steps=total_steps,
            base_lr=base_lr,
            min_lr=min_lr,
            warmup_steps=warmup_steps,
        )
        return lr / float(base_lr)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _metric_number(gate: dict[str, Any], variant: str, field: str) -> Optional[float]:
    section = gate.get(variant)
    if isinstance(section, dict) and field in section:
        value = section[field]
        if isinstance(value, (int, float)):
            return float(value)
    flat_keys = (
        f"{variant}_{field}",
        f"{variant}_{field.replace('_count', '')}",
    )
    for key in flat_keys:
        value = gate.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _metric_count(gate: dict[str, Any], variant: str) -> Optional[int]:
    value = _metric_number(gate, variant, "correct_count")
    if value is None:
        return None
    return int(value)


def _metric_accuracy(gate: dict[str, Any], variant: str) -> Optional[float]:
    value = _metric_number(gate, variant, "accuracy")
    if value is not None:
        return float(value)
    correct = _metric_number(gate, variant, "correct_count")
    total = _metric_number(gate, variant, "sample_count")
    if total is None:
        total = gate.get("sample_count")
    if isinstance(correct, (int, float)) and isinstance(total, (int, float)) and total:
        return float(correct) / float(total)
    return None


def _copy_proof_accuracy(gate: dict[str, Any]) -> Optional[float]:
    section = gate.get("copy_proof")
    if isinstance(section, dict):
        for key in ("latent_lora_accuracy", "accuracy"):
            value = section.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    for key in ("copy_proof_latent_lora_accuracy", "copy_proof_accuracy"):
        value = gate.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _first_defined(history: list[dict[str, Any]], getter: Any) -> Optional[float]:
    for gate in history:
        value = getter(gate)
        if value is not None:
            return value
    return None


def evaluate_kill_rules(history: list[dict[str, Any]]) -> dict[str, Any]:
    rules: list[str] = []
    if not history:
        return {"kill": False, "rules": rules, "reason": None}

    latent_base_correct = _first_defined(
        history,
        lambda gate: _metric_count(gate, "latent_base"),
    )
    if latent_base_correct is not None:
        for gate in history:
            optimizer_step = int(gate.get("optimizer_step", gate.get("global_step", 0)) or 0)
            latent_lora_correct = _metric_count(gate, "latent_lora")
            if (
                optimizer_step >= 10
                and latent_lora_correct is not None
                and latent_lora_correct < int(latent_base_correct) - 2
            ):
                rules.append("latent_lora_below_base")
                break

    text_base_correct = _first_defined(history, lambda gate: _metric_count(gate, "text"))
    if text_base_correct is not None:
        for gate in history:
            text_lora_correct = _metric_count(gate, "text_lora_canary")
            if text_lora_correct is not None and text_lora_correct < int(text_base_correct) - 2:
                rules.append("text_lora_canary_below_text")
                break

    step0_copy_accuracy = _first_defined(history, _copy_proof_accuracy)
    step0_latent_accuracy = _first_defined(
        history,
        lambda gate: _metric_accuracy(gate, "latent_lora"),
    )
    if step0_copy_accuracy is not None and step0_latent_accuracy is not None:
        for gate in history:
            copy_accuracy = _copy_proof_accuracy(gate)
            latent_accuracy = _metric_accuracy(gate, "latent_lora")
            if (
                copy_accuracy is not None
                and latent_accuracy is not None
                and copy_accuracy < float(step0_copy_accuracy)
                and latent_accuracy > float(step0_latent_accuracy)
            ):
                rules.append("copy_proof_drop_with_overall_rise")
                break

    nll_gates: list[tuple[float, float]] = []
    for gate in history:
        dev_nll = gate.get("dev_nll")
        latent_accuracy = _metric_accuracy(gate, "latent_lora")
        if isinstance(dev_nll, (int, float)) and latent_accuracy is not None:
            nll_gates.append((float(dev_nll), float(latent_accuracy)))
    for index in range(2, len(nll_gates)):
        previous_2, previous_1, current = (
            nll_gates[index - 2],
            nll_gates[index - 1],
            nll_gates[index],
        )
        if (
            current[0] < previous_1[0] < previous_2[0]
            and current[1] < previous_1[1] < previous_2[1]
        ):
            rules.append("phase0_divergence")
            break

    return {"kill": bool(rules), "rules": rules, "reason": rules[0] if rules else None}


def _mps_rng_state() -> Optional[torch.Tensor]:
    mps = getattr(torch, "mps", None)
    getter = getattr(mps, "get_rng_state", None)
    if getter is None:
        return None
    try:
        return getter()
    except Exception:  # noqa: BLE001
        return None


def _restore_mps_rng_state(state: Any) -> None:
    if not torch.is_tensor(state):
        return
    mps = getattr(torch, "mps", None)
    setter = getattr(mps, "set_rng_state", None)
    if setter is None:
        return
    try:
        setter(state)
    except Exception:  # noqa: BLE001
        return


def _rng_states() -> dict[str, Any]:
    states: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "random": random.getstate(),
    }
    mps_state = _mps_rng_state()
    if mps_state is not None:
        states["mps"] = mps_state
    return states


def _restore_rng_states(states: dict[str, Any]) -> None:
    torch_state = states.get("torch")
    if torch.is_tensor(torch_state):
        torch.set_rng_state(torch_state)
    random_state = states.get("random")
    if random_state is not None:
        random.setstate(random_state)
    _restore_mps_rng_state(states.get("mps"))


def _module_trainable_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    lora_state = receiver_lora_state_dict(module)
    if lora_state:
        return lora_state
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
        if torch.is_tensor(value)
    }


def save_lora_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    epoch: int,
    position_in_epoch: int,
    sample_step: int,
    optimizer_step: int,
    best_dev_correct: int,
    best_dev_step: int,
    epoch_end_no_improve: int,
    args: argparse.Namespace | dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "lora": _module_trainable_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "epoch": int(epoch),
        "position_in_epoch": int(position_in_epoch),
        "sample_step": int(sample_step),
        "global_step": int(optimizer_step),
        "optimizer_step": int(optimizer_step),
        "best_dev_correct": int(best_dev_correct),
        "best_dev_step": int(best_dev_step),
        "epoch_end_no_improve": int(epoch_end_no_improve),
        "rng_states": _rng_states(),
        "args": vars(args) if isinstance(args, argparse.Namespace) else dict(args),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def load_lora_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    snapshot = torch.load(Path(path), map_location="cpu", weights_only=False)
    state = snapshot.get("lora", snapshot.get("state", {}))
    if not isinstance(state, dict):
        raise ValueError("checkpoint LoRA state must be a dict")
    model.load_state_dict(state, strict=False)
    if optimizer is not None and snapshot.get("optimizer") is not None:
        optimizer.load_state_dict(snapshot["optimizer"])
    if scheduler is not None and snapshot.get("scheduler") is not None:
        scheduler.load_state_dict(snapshot["scheduler"])
    rng_states = snapshot.get("rng_states")
    if restore_rng and isinstance(rng_states, dict):
        _restore_rng_states(rng_states)
    return snapshot


def _json_default(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def write_lora_report(
    report_path: str | Path,
    history: list[dict[str, Any]],
    latest: dict[str, Any],
) -> None:
    Path(report_path).write_text(
        json.dumps({"history": history, "latest": latest}, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _forward_logits_to_keep_kwargs(model: Any, logits_to_keep: int) -> dict[str, int]:
    kwargs = _last_logit_forward_kwargs(model)
    if "logits_to_keep" in kwargs:
        return {"logits_to_keep": int(logits_to_keep)}
    return {}


def _top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if top_p >= 1.0:
        return probs
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove_mask = cumulative > float(top_p)
    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
    remove_mask[..., 0] = False
    filtered_sorted = sorted_probs.masked_fill(remove_mask, 0.0)
    filtered = torch.zeros_like(probs).scatter(-1, sorted_indices, filtered_sorted)
    normalizer = filtered.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(filtered.dtype).tiny)
    return filtered / normalizer


def sampled_decode_from_prefix(
    *,
    model: Any,
    tokenizer: Any,
    prefix_state: dict[str, Any],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop_regex: Optional[Any] = None,
) -> dict[str, Any]:
    if temperature <= 0:
        raise ValueError("temperature must be positive for sampled decoding")
    outputs = prefix_state["outputs"]
    attention_mask = prefix_state["attention_mask"]
    prefix_seq_len = int(prefix_state.get("prefix_seq_len", attention_mask.shape[1]))
    _reset_kv_cache_to_prefix(outputs.past_key_values, prefix_seq_len)
    generated_token_ids: list[int] = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    last_position_ids = build_position_ids(attention_mask)[:, -1:]

    for _ in range(max_new_tokens):
        logits = outputs.logits[:, -1, :].float() / float(temperature)
        probs = _top_p_filter(torch.softmax(logits, dim=-1), top_p)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        next_token_id = int(next_token.item())
        if eos_token_id is not None and next_token_id == eos_token_id:
            break
        generated_token_ids.append(next_token_id)
        if stop_regex is not None:
            decoded_text = str(prefix_state.get("decoded_text_prefix", "")) + tokenizer.decode(
                generated_token_ids,
                skip_special_tokens=True,
            )
            if stop_regex.search(decoded_text) is not None:
                break
        kv_cache = _normalize_kv_cache(outputs.past_key_values)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ],
            dim=1,
        )
        last_position_ids = last_position_ids + 1
        with torch.no_grad():
            outputs = model(
                input_ids=next_token.unsqueeze(-1),
                past_key_values=kv_cache,
                attention_mask=attention_mask,
                position_ids=last_position_ids,
                use_cache=True,
                return_dict=True,
            )

    decoded_text = str(prefix_state.get("decoded_text_prefix", "")) + tokenizer.decode(
        generated_token_ids,
        skip_special_tokens=True,
    )
    return {
        "decoded_text": decoded_text,
        "generated_tokens": len(generated_token_ids),
        "generated_token_ids": generated_token_ids,
    }


def _encoded_rollout_ids(tokenizer: Any, decoded_text: str, max_new_tokens: int) -> list[int]:
    try:
        token_ids = tokenizer.encode(decoded_text, add_special_tokens=False)
    except Exception:  # noqa: BLE001
        token_ids = []
    return [int(token_id) for token_id in token_ids[:max_new_tokens]]


def _rollout_verified(dataset: str, decoded_text: str, target_answer: Any) -> bool:
    predicted = _predicted_answer(dataset, decoded_text)
    return bool(
        predicted is not None and _answers_match(dataset, predicted, target_answer)
    )


def _current_rollout_config_digest(args: argparse.Namespace) -> str:
    return rollout_config_digest(
        receiver_model=args.receiver_model,
        adapter_digest=args.adapter_digest,
        truncation_fraction=args.truncation_fraction,
        max_new_tokens=args.max_new_tokens,
        temperature=args.rollout_temperature,
        seed=args.seed,
    )


def _cached_rollout_current(path: Path, config_digest: str) -> bool:
    if not path.exists():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001
        return False
    return isinstance(payload, dict) and payload.get("config_digest") == config_digest


def prepare_rollouts(
    *,
    args: argparse.Namespace,
    receiver: Any,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    cache_dir: str | Path = DEFAULT_ROLLOUT_CACHE_DIR,
) -> dict[str, int]:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    config_digest = _current_rollout_config_digest(args)
    target_vocab_size = int(receiver.get_input_embeddings().weight.shape[0])
    stats = {"accepted": 0, "skipped": 0, "failed": 0, "written": 0, "invalid": 0}

    for sample in samples:
        path = rollout_cache_path(str(sample["question"]), cache_dir=cache_root)
        if _cached_rollout_current(path, config_digest):
            stats["skipped"] += 1
            continue

        prefix_state = prepare_receiver_context_latent_prefix_state(
            model=receiver,
            tokenizer=tokenizer,
            context_text=str(sample["context_text"]),
            handoff_step=sample["latents"],
            kv_cache=None,
            suffix_text="",
            latent_position="after_context",
        )
        greedy = greedy_decode_from_prefix(
            model=receiver,
            tokenizer=tokenizer,
            prefix_state=prefix_state,
            max_new_tokens=args.max_new_tokens,
            stop_regex=None,
        )
        decoded_text = str(greedy["decoded_text"])
        greedy_correct = _rollout_verified(args.dataset, decoded_text, sample["answer"])
        attempt = 0
        token_ids = _encoded_rollout_ids(tokenizer, decoded_text, args.max_new_tokens)
        verified = greedy_correct

        if not verified:
            max_sampled_retries = min(4, int(args.rollout_max_tries))
            row_seed = int(args.seed) * 100_003 + int(sample["sample_index"])
            torch.manual_seed(row_seed)
            random.seed(row_seed)
            for retry_index in range(1, max_sampled_retries + 1):
                sampled = sampled_decode_from_prefix(
                    model=receiver,
                    tokenizer=tokenizer,
                    prefix_state=prefix_state,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.rollout_temperature,
                    top_p=args.rollout_top_p,
                    stop_regex=None,
                )
                decoded_text = str(sampled["decoded_text"])
                token_ids = [
                    int(token_id)
                    for token_id in sampled.get("generated_token_ids", [])[: args.max_new_tokens]
                ]
                attempt = retry_index
                verified = _rollout_verified(args.dataset, decoded_text, sample["answer"])
                if verified:
                    break

        validated_token_ids, token_error = validate_target_token_ids(
            token_ids,
            max_continuation_tokens=int(args.max_new_tokens),
            target_vocab_size=target_vocab_size,
        )
        if validated_token_ids is None:
            stats["invalid"] += 1
            verified = False
            token_ids = []
            decoded_text = f"{decoded_text}\n\n[invalid_rollout_target: {token_error}]"
        else:
            token_ids = validated_token_ids

        payload = {
            "prompt_sha": prompt_sha(str(sample["question"])),
            "sample_index": int(sample["sample_index"]),
            "split": "train",
            "token_ids": token_ids[: args.max_new_tokens],
            "decoded_text": decoded_text,
            "greedy_correct": bool(greedy_correct),
            "attempt": int(attempt),
            "verified": bool(verified),
            "config_digest": config_digest,
        }
        torch.save(payload, path)
        stats["written"] += 1
        if verified:
            stats["accepted"] += 1
        else:
            stats["failed"] += 1

    return stats


def _load_report_history(report_path: Path) -> list[dict[str, Any]]:
    if not report_path.exists():
        return []
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    history = payload.get("history", [])
    return history if isinstance(history, list) else []


def _load_cached_rollout_target(
    sample: dict[str, Any],
    *,
    config_digest: str,
    max_continuation_tokens: int,
    target_vocab_size: Optional[int] = None,
    cache_dir: str | Path = DEFAULT_ROLLOUT_CACHE_DIR,
) -> tuple[Optional[list[int]], Optional[str]]:
    path = rollout_cache_path(str(sample["question"]), cache_dir=cache_dir)
    if not path.exists():
        return None, "missing_rollout"
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001
        return None, "load_error"
    if not isinstance(payload, dict):
        return None, "payload_not_dict"
    if payload.get("config_digest") != config_digest or not bool(payload.get("verified")):
        return None, "stale_or_unverified_rollout"
    token_ids = payload.get("token_ids")
    return validate_target_token_ids(
        token_ids,
        max_continuation_tokens=max_continuation_tokens,
        target_vocab_size=target_vocab_size,
    )


def prepare_objective_samples(
    samples: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    tokenizer: Any,
    require_targets: bool,
    target_vocab_size: Optional[int] = None,
) -> tuple[list[dict[str, Any]], list[int], dict[str, int]]:
    prepared: list[dict[str, Any]] = []
    skipped: list[int] = []
    skip_counts: dict[str, int] = {}
    config_digest = _current_rollout_config_digest(args)
    for sample in samples:
        if args.objective == "C":
            target_ids, target_error = validate_target_token_ids(
                sample["target_continuation_ids"],
                max_continuation_tokens=int(args.max_continuation_tokens),
                target_vocab_size=target_vocab_size,
            )
        else:
            target_ids, target_error = _load_cached_rollout_target(
                sample,
                config_digest=config_digest,
                max_continuation_tokens=int(args.max_continuation_tokens),
                target_vocab_size=target_vocab_size,
            )
        if target_ids is None:
            _record_sample_skip(skipped, skip_counts, sample, target_error)
            if require_targets:
                continue
            target_ids = []
        elif not target_ids:
            _record_sample_skip(skipped, skip_counts, sample, "empty_target")
            if require_targets:
                continue
        weights = (
            build_answer_weight_vector(tokenizer, target_ids, final_answer_weight=4.0)
            if args.objective == "B"
            else [1.0 for _ in target_ids]
        )
        weights, weight_error = validate_target_weights(weights, target_len=len(target_ids))
        if weights is None:
            _record_sample_skip(skipped, skip_counts, sample, weight_error)
            if require_targets:
                continue
            weights = []
        item = dict(sample)
        item["target_ids"] = target_ids
        item["target_weights"] = weights
        item["objective"] = args.objective
        prepared.append(item)
    return prepared, skipped, skip_counts


def receiver_lora_sample_nll(
    *,
    receiver: Any,
    sample: dict[str, Any],
    device: torch.device,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    embedding = receiver.get_input_embeddings()
    target_vocab_size = int(getattr(embedding, "num_embeddings", embedding.weight.shape[0]))
    raw_target_ids = sample["target_ids"]
    max_target_tokens = len(raw_target_ids) if isinstance(raw_target_ids, (list, tuple)) else 0
    target_ids, target_error = validate_target_token_ids(
        raw_target_ids,
        max_continuation_tokens=max_target_tokens,
        target_vocab_size=target_vocab_size,
    )
    if target_ids is None:
        raise ValueError(f"invalid sample target_ids: {target_error}")
    target_len = len(target_ids)
    target = torch.tensor([target_ids], dtype=torch.long, device=device)
    weight_values, weight_error = validate_target_weights(
        sample.get("target_weights", [1.0] * target_len),
        target_len=target_len,
    )
    if weight_values is None:
        raise ValueError(f"invalid target_weights: {weight_error}")
    weights = torch.tensor(weight_values, dtype=torch.float32, device=device)

    context = torch.tensor([sample["context_ids"]], dtype=torch.long, device=device)
    with torch.no_grad():
        context_embeds = embedding(context).to(dtype=model_dtype)
        continuation_embeds = embedding(target).to(dtype=model_dtype)
    latents = sample["latents"].to(device=device, dtype=model_dtype)
    inputs_embeds = torch.cat([context_embeds, latents, continuation_embeds], dim=1)
    prefix_len = int(context_embeds.shape[1] + latents.shape[1])
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
    outputs = receiver(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=build_position_ids(attention_mask),
        use_cache=False,
        return_dict=True,
        **_forward_logits_to_keep_kwargs(receiver, target_len + 1),
    )
    logits = outputs.logits.float()
    if int(logits.shape[1]) == target_len + 1:
        prediction_logits = logits[:, :-1, :]
    else:
        prediction_logits = logits[:, prefix_len - 1 : prefix_len + target_len - 1, :]
    if int(prediction_logits.shape[1]) != target_len:
        raise ValueError(
            "training logits length mismatch: "
            f"got {int(prediction_logits.shape[1])}, expected {target_len}"
        )
    per_token = F.cross_entropy(
        prediction_logits.reshape(-1, prediction_logits.shape[-1]),
        target.reshape(-1),
        reduction="none",
    )
    return (per_token * weights).sum() / weights.sum()


def _score_decoded(dataset: str, decoded_text: str, answer: Any) -> dict[str, Any]:
    predicted = _predicted_answer(dataset, decoded_text)
    correct = bool(predicted is not None and _answers_match(dataset, predicted, answer))
    return {
        "predicted": predicted,
        "correct": correct,
        "decoded": decoded_text[:200],
    }


@torch.no_grad()
def _evaluate_latent_variant(
    *,
    receiver: Any,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
    lora_enabled: bool,
) -> list[dict[str, Any]]:
    set_receiver_lora_enabled(receiver, bool(lora_enabled))
    receiver.eval()
    rows: list[dict[str, Any]] = []
    for sample in samples:
        prefix_state = prepare_receiver_context_latent_prefix_state(
            model=receiver,
            tokenizer=tokenizer,
            context_text=str(sample["context_text"]),
            handoff_step=sample["latents"],
            kv_cache=None,
            suffix_text="",
            latent_position="after_context",
        )
        decoded = greedy_decode_from_prefix(
            model=receiver,
            tokenizer=tokenizer,
            prefix_state=prefix_state,
            max_new_tokens=args.max_new_tokens,
            stop_regex=None,
        )["decoded_text"]
        rows.append(
            {
                "sample_index": int(sample["sample_index"]),
                "answer": str(sample["answer"]),
                **_score_decoded(args.dataset, str(decoded), sample["answer"]),
            }
        )
    return rows


@torch.no_grad()
def _evaluate_text_variant(
    *,
    receiver: Any,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
    variant: str,
    lora_enabled: bool,
) -> list[dict[str, Any]]:
    set_receiver_lora_enabled(receiver, bool(lora_enabled))
    receiver.eval()
    rows: list[dict[str, Any]] = []
    for sample in samples:
        if variant == "text":
            prefix_text = chat_prefix_text(
                tokenizer,
                str(sample["question"]),
                TEXT_INSTRUCTION,
                body=str(sample.get("truncated_text", "")),
            )
        elif variant == "alone":
            prefix_text = chat_prefix_text(
                tokenizer,
                str(sample["question"]),
                ALONE_INSTRUCTION,
            )
        else:
            raise ValueError(f"unknown text variant: {variant}")
        prefix_state = prepare_text_prefix_state(
            model=receiver,
            tokenizer=tokenizer,
            prefix_text=prefix_text,
        )
        decoded = greedy_decode_from_prefix(
            model=receiver,
            tokenizer=tokenizer,
            prefix_state=prefix_state,
            max_new_tokens=args.max_new_tokens,
            stop_regex=None,
        )["decoded_text"]
        rows.append(
            {
                "sample_index": int(sample["sample_index"]),
                "answer": str(sample["answer"]),
                **_score_decoded(args.dataset, str(decoded), sample["answer"]),
            }
        )
    return rows


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct_count = sum(1 for row in rows if bool(row.get("correct")))
    sample_count = len(rows)
    accuracy = float(correct_count) / float(sample_count) if sample_count else 0.0
    return {
        "correct_count": int(correct_count),
        "sample_count": int(sample_count),
        "accuracy": accuracy,
        "accuracy_percentage": round(100.0 * accuracy, 2),
    }


def _normalized_number(text: Any) -> str:
    return re.sub(r"[^\d.]", "", str(text)).strip(".")


def _copy_proof_indices(samples: list[dict[str, Any]]) -> set[int]:
    indices: set[int] = set()
    for sample in samples:
        answer_token = _normalized_number(sample["answer"])
        normalized_prefix = re.sub(r"[,\s]", "", str(sample.get("truncated_text", "")))
        present = bool(answer_token) and answer_token in normalized_prefix
        if not present:
            indices.add(int(sample["sample_index"]))
    return indices


def _copy_proof_summary(
    latent_lora_rows: list[dict[str, Any]],
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    indices = _copy_proof_indices(samples)
    rows = [row for row in latent_lora_rows if int(row["sample_index"]) in indices]
    summary = _summarize_rows(rows)
    summary["stratum"] = "answer_not_in_prefix"
    return summary


@torch.no_grad()
def _mean_dev_nll(
    *,
    receiver: Any,
    samples: list[dict[str, Any]],
    device: torch.device,
    model_dtype: torch.dtype,
) -> Optional[float]:
    if not samples:
        return None
    set_receiver_lora_enabled(receiver, True)
    receiver.eval()
    losses: list[float] = []
    for sample in samples:
        loss = receiver_lora_sample_nll(
            receiver=receiver,
            sample=sample,
            device=device,
            model_dtype=model_dtype,
        )
        losses.append(float(loss.detach().cpu().item()))
    return float(sum(losses) / len(losses)) if losses else None


def _identity_mismatches(
    latent_lora_rows: list[dict[str, Any]],
    latent_base_rows: list[dict[str, Any]],
) -> list[int]:
    base_by_index = {
        int(row["sample_index"]): bool(row.get("correct"))
        for row in latent_base_rows
    }
    mismatches: list[int] = []
    for row in latent_lora_rows:
        sample_index = int(row["sample_index"])
        if bool(row.get("correct")) != base_by_index.get(sample_index):
            mismatches.append(sample_index)
    return mismatches


def _save_best_lora(
    path: Path,
    *,
    receiver: Any,
    args: argparse.Namespace,
    dev_gate: dict[str, Any],
    module_count: int,
) -> None:
    latent_summary = dev_gate["latent_lora"]
    payload = {
        "format": RECEIVER_LORA_FORMAT,
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "dropout": float(args.dropout),
        "target_suffixes": RECEIVER_LORA_TARGET_SUFFIXES,
        "base_model": str(args.receiver_model),
        "state": receiver_lora_state_dict(receiver),
        "train_args": vars(args),
        "adapter_cache_key_digest": str(args.adapter_digest),
        "objective": str(args.objective),
        "dev_gate_accuracy": float(latent_summary["accuracy"]),
        "dev_gate_correct_count": int(latent_summary["correct_count"]),
        "global_step": int(dev_gate["global_step"]),
        "optimizer_step": int(dev_gate["optimizer_step"]),
        "sample_step": int(dev_gate["sample_step"]),
        "module_count": int(module_count),
    }
    torch.save(payload, path)


# NOTE: do NOT call torch.mps.empty_cache() anywhere between training backward
# passes. On MPS (torch 2.10, bf16 Qwen3.5 GatedDeltaNet backward) emptying the
# cache deterministically poisons the gradients of the next accumulation window
# while every loss stays finite; clip_grad_norm_ then scales all LoRA params to
# nan in one optimizer step. torch.mps.synchronize() first does not help.
# Reproduced/bisected 2026-07-05 (objective A, corruption at sample_step 32).


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "lora_checkpoint.pt"
    best_path = output_dir / "best_lora.pt"
    report_path = output_dir / "lora_report.json"
    device = torch.device(args.device)
    model_dtype = _torch_dtype(args.trace_dtype)
    max_epochs = min(int(args.epochs), 1) if args.objective == "C" else min(int(args.epochs), 3)

    print(f"Loading receiver tokenizer {args.receiver_model} ...", flush=True)
    receiver_tokenizer = AutoTokenizer.from_pretrained(
        args.receiver_model,
        trust_remote_code=True,
    )
    sender_tokenizer = (
        receiver_tokenizer
        if args.sender_model == args.receiver_model
        else AutoTokenizer.from_pretrained(args.sender_model, trust_remote_code=True)
    )
    adapter_payload = load_frozen_adapter(
        _resolve_workspace_path(args.adapter_cache_path),
        adapter_digest=args.adapter_digest,
    )

    print("Building receiver-LoRA train/dev samples ...", flush=True)
    rows = load_bridge_sample_rows(args, tokenizer=sender_tokenizer, include_dev=True)
    train_samples = assemble_training_samples(
        rows["train"],
        tokenizer=receiver_tokenizer,
        adapter_payload=adapter_payload,
    )
    dev_samples = assemble_training_samples(
        rows["dev"],
        tokenizer=receiver_tokenizer,
        adapter_payload=adapter_payload,
    )
    if not train_samples or not dev_samples:
        raise SystemExit("No cached train/dev traces matched the requested row ranges.")

    print(f"Loading frozen receiver {args.receiver_model} ({model_dtype}) ...", flush=True)
    receiver = AutoModelForCausalLM.from_pretrained(
        args.receiver_model,
        dtype=model_dtype,
        trust_remote_code=True,
    ).to(device)
    receiver.eval()
    for parameter in receiver.parameters():
        parameter.requires_grad_(False)
    lora_modules = apply_receiver_lora(
        receiver,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
    )
    module_count = len(lora_modules)
    lora_params = [
        parameter
        for module in lora_modules.values()
        for parameter in (module.A, module.B)
    ]
    lora_named_params = [
        (f"{name}.{suffix}", parameter)
        for name, module in lora_modules.items()
        for suffix, parameter in (("A", module.A), ("B", module.B))
    ]
    for parameter in lora_params:
        parameter.requires_grad_(True)

    target_vocab_size = int(receiver.get_input_embeddings().weight.shape[0])
    train_samples, train_skipped, train_skip_counts = prepare_objective_samples(
        train_samples,
        args=args,
        tokenizer=receiver_tokenizer,
        require_targets=True,
        target_vocab_size=target_vocab_size,
    )
    dev_nll_samples, dev_skipped, dev_skip_counts = prepare_objective_samples(
        dev_samples,
        args=args,
        tokenizer=receiver_tokenizer,
        require_targets=True,
        target_vocab_size=target_vocab_size,
    )
    if not train_samples:
        raise SystemExit("No train samples had usable objective targets.")
    if args.objective in {"A", "B"} and train_skipped:
        print(
            f"objective {args.objective}: skipped {len(train_skipped)} train rows "
            f"without usable rollout targets ({_format_skip_counts(train_skip_counts)})",
            flush=True,
        )
    if args.objective in {"A", "B"} and dev_skipped:
        print(
            f"objective {args.objective}: skipped {len(dev_skipped)} dev rows "
            f"without usable rollout targets for dev NLL ({_format_skip_counts(dev_skip_counts)})",
            flush=True,
        )

    optimizer_steps_per_epoch = max(1, math.ceil(len(train_samples) / max(1, int(args.grad_accum))))
    total_optimizer_steps = max(1, optimizer_steps_per_epoch * max_epochs)
    if args.max_steps:
        total_optimizer_steps = min(total_optimizer_steps, int(args.max_steps))
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=float(args.lr),
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )
    scheduler = build_cosine_with_warmup_scheduler(
        optimizer,
        total_steps=total_optimizer_steps,
        base_lr=float(args.lr),
        min_lr=1e-5,
        warmup_steps=6,
    )

    history = _load_report_history(report_path)
    start_epoch = 0
    start_position = 0
    sample_step = 0
    optimizer_step = 0
    best_dev_correct = -1
    best_dev_step = -1
    epoch_end_no_improve = 0
    if checkpoint_path.exists():
        snapshot = load_lora_checkpoint(
            checkpoint_path,
            model=receiver,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        start_epoch = int(snapshot.get("epoch", 0))
        start_position = int(snapshot.get("position_in_epoch", 0))
        sample_step = int(snapshot.get("sample_step", 0))
        optimizer_step = int(snapshot.get("optimizer_step", snapshot.get("global_step", 0)))
        best_dev_correct = int(snapshot.get("best_dev_correct", -1))
        best_dev_step = int(snapshot.get("best_dev_step", -1))
        epoch_end_no_improve = int(snapshot.get("epoch_end_no_improve", 0))
        print(
            f"Resumed receiver LoRA from optimizer_step={optimizer_step} "
            f"sample_step={sample_step} epoch={start_epoch} position={start_position}",
            flush=True,
        )

    def set_lora_train_mode() -> None:
        receiver.eval()
        set_receiver_lora_enabled(receiver, True)
        for module in lora_modules.values():
            module.train()

    def append_gate(gate: dict[str, Any]) -> dict[str, Any]:
        nonlocal best_dev_correct, best_dev_step
        history.append(gate)
        latest = dict(gate)
        kill = evaluate_kill_rules(history)
        latest["kill"] = kill
        if kill["kill"]:
            latest["status"] = "FAILED"
        write_lora_report(report_path, history, latest)
        dev_correct = int(gate["latent_lora"]["correct_count"])
        if dev_correct > best_dev_correct:
            best_dev_correct = dev_correct
            best_dev_step = int(gate["optimizer_step"])
            _save_best_lora(
                best_path,
                receiver=receiver,
                args=args,
                dev_gate=gate,
                module_count=module_count,
            )
            print(
                f"[gate:{gate['tag']}] new best latent_lora "
                f"{dev_correct}/{gate['latent_lora']['sample_count']}",
                flush=True,
            )
        else:
            print(
                f"[gate:{gate['tag']}] latent_lora "
                f"{dev_correct}/{gate['latent_lora']['sample_count']} "
                f"dev_nll={gate.get('dev_nll')}",
                flush=True,
            )
        return latest

    def run_gate(
        tag: str,
        *,
        epoch: int,
        include_text_canary: bool,
        include_copy_proof: bool,
        include_constant_baselines: bool,
    ) -> dict[str, Any]:
        latent_lora_rows = _evaluate_latent_variant(
            receiver=receiver,
            tokenizer=receiver_tokenizer,
            samples=dev_samples,
            args=args,
            lora_enabled=True,
        )
        gate: dict[str, Any] = {
            "tag": tag,
            "status": "RUNNING",
            "epoch": int(epoch),
            "sample_step": int(sample_step),
            "global_step": int(optimizer_step),
            "optimizer_step": int(optimizer_step),
            "sample_count": len(dev_samples),
            "latent_lora": _summarize_rows(latent_lora_rows),
            "dev_nll": _mean_dev_nll(
                receiver=receiver,
                samples=dev_nll_samples,
                device=device,
                model_dtype=model_dtype,
            ),
            "rows": {"latent_lora": latent_lora_rows},
        }
        if include_copy_proof:
            gate["copy_proof"] = _copy_proof_summary(latent_lora_rows, dev_samples)
        if include_constant_baselines:
            latent_base_rows = _evaluate_latent_variant(
                receiver=receiver,
                tokenizer=receiver_tokenizer,
                samples=dev_samples,
                args=args,
                lora_enabled=False,
            )
            text_rows = _evaluate_text_variant(
                receiver=receiver,
                tokenizer=receiver_tokenizer,
                samples=dev_samples,
                args=args,
                variant="text",
                lora_enabled=False,
            )
            alone_rows = _evaluate_text_variant(
                receiver=receiver,
                tokenizer=receiver_tokenizer,
                samples=dev_samples,
                args=args,
                variant="alone",
                lora_enabled=False,
            )
            gate["latent_base"] = _summarize_rows(latent_base_rows)
            gate["text"] = _summarize_rows(text_rows)
            gate["alone"] = _summarize_rows(alone_rows)
            gate["rows"]["latent_base"] = latent_base_rows
            gate["rows"]["text"] = text_rows
            gate["rows"]["alone"] = alone_rows
            mismatches = _identity_mismatches(latent_lora_rows, latent_base_rows)
            gate["step0_identity_mismatches"] = mismatches
        else:
            baseline_gate = next(
                (item for item in history if "latent_base" in item and "text" in item),
                None,
            )
            if baseline_gate is not None:
                gate["latent_base"] = baseline_gate["latent_base"]
                gate["text"] = baseline_gate["text"]
                gate["alone"] = baseline_gate.get("alone")
        if include_text_canary:
            text_lora_rows = _evaluate_text_variant(
                receiver=receiver,
                tokenizer=receiver_tokenizer,
                samples=dev_samples,
                args=args,
                variant="text",
                lora_enabled=True,
            )
            gate["text_lora_canary"] = _summarize_rows(text_lora_rows)
            gate["rows"]["text_lora_canary"] = text_lora_rows
        return append_gate(gate)

    if args.eval_only:
        if not checkpoint_path.exists():
            print("eval-only: no lora_checkpoint.pt found; evaluating current init", flush=True)
        run_gate(
            "eval_only",
            epoch=start_epoch,
            include_text_canary=True,
            include_copy_proof=True,
            include_constant_baselines=not history,
        )
        return

    if optimizer_step == 0 and not history:
        print("Running step-0 identity gate ...", flush=True)
        latest = run_gate(
            "step0",
            epoch=0,
            include_text_canary=False,
            include_copy_proof=True,
            include_constant_baselines=True,
        )
        mismatches = latest.get("step0_identity_mismatches", [])
        if mismatches:
            latest["status"] = "FAILED"
            latest["failure_reason"] = "step0_identity_mismatch"
            write_lora_report(report_path, history, latest)
            raise SystemExit(
                "Step-0 identity gate failed for sample_index rows: "
                f"{','.join(str(index) for index in mismatches)}"
            )
        save_lora_checkpoint(
            checkpoint_path,
            model=receiver,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=0,
            position_in_epoch=0,
            sample_step=sample_step,
            optimizer_step=optimizer_step,
            best_dev_correct=best_dev_correct,
            best_dev_step=best_dev_step,
            epoch_end_no_improve=epoch_end_no_improve,
            args=args,
        )

    print(
        f"Training receiver LoRA objective={args.objective}: "
        f"{max_epochs} epochs, {len(train_samples)} samples, "
        f"{total_optimizer_steps} optimizer steps max",
        flush=True,
    )
    stopped = False
    stop_status = "COMPLETED"
    nonfinite_loss_skips = 0
    nonfinite_grad_skips = 0

    def apply_optimizer_step(*, epoch: int, position: int) -> bool:
        """Clip + step + schedule, unless the accumulated grads are non-finite.

        Returns True when an optimizer step was taken. On non-finite grads the
        whole accumulation window is zeroed and skipped: clip_grad_norm_ would
        otherwise compute a nan total norm and scale every LoRA gradient (and
        so every parameter and Adam moment) to nan in a single step.
        """
        nonlocal optimizer_step, nonfinite_grad_skips
        bad_names = nonfinite_grad_names(lora_named_params)
        if bad_names:
            nonfinite_grad_skips += 1
            optimizer.zero_grad(set_to_none=True)
            print(
                "skipping optimizer step with non-finite grads: "
                f"epoch={epoch} position={position} "
                f"tensors={len(bad_names)} first={bad_names[0]} "
                f"skip_count={nonfinite_grad_skips}",
                flush=True,
            )
            return False
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1
        return True

    set_lora_train_mode()
    for epoch in range(start_epoch, max_epochs):
        order = list(range(len(train_samples)))
        random.Random(int(args.seed) + epoch).shuffle(order)
        position_start = start_position if epoch == start_epoch else 0
        accumulation_count = 0
        running_loss = 0.0
        next_position = position_start
        optimizer.zero_grad(set_to_none=True)
        for position in range(position_start, len(order)):
            if args.max_steps and optimizer_step >= int(args.max_steps):
                stopped = True
                stop_status = "MAX_STEPS"
                break
            sample = train_samples[order[position]]
            loss = receiver_lora_sample_nll(
                receiver=receiver,
                sample=sample,
                device=device,
                model_dtype=model_dtype,
            )
            loss_value = finite_loss_value(loss)
            if loss_value is None:
                nonfinite_loss_skips += 1
                sample_step += 1
                next_position = position + 1
                print(
                    "skipping non-finite loss before backward: "
                    f"epoch={epoch} position={position} "
                    f"sample_index={sample.get('sample_index')} "
                    f"skip_count={nonfinite_loss_skips}",
                    flush=True,
                )
                continue
            (loss / max(1, int(args.grad_accum))).backward()
            accumulation_count += 1
            sample_step += 1
            next_position = position + 1
            running_loss += loss_value
            should_step = accumulation_count >= max(1, int(args.grad_accum))
            if should_step:
                stepped = apply_optimizer_step(epoch=epoch, position=position)
                accumulation_count = 0
                if stepped and optimizer_step % 10 == 0:
                    latest = run_gate(
                        f"step_{optimizer_step}",
                        epoch=epoch,
                        include_text_canary=False,
                        include_copy_proof=False,
                        include_constant_baselines=False,
                    )
                    set_lora_train_mode()
                    if latest.get("kill", {}).get("kill"):
                        stopped = True
                        stop_status = "FAILED"
                        break
                if stepped and optimizer_step % 5 == 0:
                    print(
                        f"epoch {epoch} optimizer_step {optimizer_step} "
                        f"sample_step {sample_step} "
                        f"loss={running_loss / max(1, accumulation_count or args.grad_accum):.4f}",
                        flush=True,
                    )
            if args.checkpoint_every and sample_step % int(args.checkpoint_every) == 0:
                save_lora_checkpoint(
                    checkpoint_path,
                    model=receiver,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    position_in_epoch=position + 1,
                    sample_step=sample_step,
                    optimizer_step=optimizer_step,
                    best_dev_correct=best_dev_correct,
                    best_dev_step=best_dev_step,
                    epoch_end_no_improve=epoch_end_no_improve,
                    args=args,
                )
            if stopped:
                break
        if not stopped and accumulation_count:
            stepped = apply_optimizer_step(epoch=epoch, position=next_position - 1)
            accumulation_count = 0
            if stepped and optimizer_step % 10 == 0:
                latest = run_gate(
                    f"step_{optimizer_step}",
                    epoch=epoch,
                    include_text_canary=False,
                    include_copy_proof=False,
                    include_constant_baselines=False,
                )
                set_lora_train_mode()
                if latest.get("kill", {}).get("kill"):
                    stopped = True
                    stop_status = "FAILED"
        if stopped:
            save_lora_checkpoint(
                checkpoint_path,
                model=receiver,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                position_in_epoch=next_position,
                sample_step=sample_step,
                optimizer_step=optimizer_step,
                best_dev_correct=best_dev_correct,
                best_dev_step=best_dev_step,
                epoch_end_no_improve=epoch_end_no_improve,
                args=args,
            )
            break

        prior_best_dev_correct = best_dev_correct
        latest = run_gate(
            f"epoch_{epoch}",
            epoch=epoch,
            include_text_canary=True,
            include_copy_proof=True,
            include_constant_baselines=False,
        )
        set_lora_train_mode()
        if latest.get("kill", {}).get("kill"):
            stopped = True
            stop_status = "FAILED"
        elif int(latest["latent_lora"]["correct_count"]) > prior_best_dev_correct:
            epoch_end_no_improve = 0
        else:
            epoch_end_no_improve += 1
            if epoch_end_no_improve >= 2:
                stopped = True
                stop_status = "EARLY_STOPPED"
        save_lora_checkpoint(
            checkpoint_path,
            model=receiver,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            position_in_epoch=0,
            sample_step=sample_step,
            optimizer_step=optimizer_step,
            best_dev_correct=best_dev_correct,
            best_dev_step=best_dev_step,
            epoch_end_no_improve=epoch_end_no_improve,
            args=args,
        )
        start_position = 0
        if stopped:
            break

    latest = history[-1] if history else {}
    latest = dict(latest)
    latest["status"] = stop_status
    latest["best_dev_correct"] = best_dev_correct
    latest["best_dev_step"] = best_dev_step
    latest["nonfinite_loss_skips"] = nonfinite_loss_skips
    latest["nonfinite_grad_skips"] = nonfinite_grad_skips
    write_lora_report(report_path, history, latest)
    print(
        f"DONE receiver LoRA status={stop_status} "
        f"best_dev_correct={best_dev_correct} "
        f"nonfinite_loss_skips={nonfinite_loss_skips} "
        f"nonfinite_grad_skips={nonfinite_grad_skips} report={report_path}",
        flush=True,
    )


def _torch_dtype(trace_dtype: str) -> torch.dtype:
    return torch.bfloat16 if str(trace_dtype) == "bfloat16" else torch.float32


def _resolve_workspace_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def run_prepare_rollouts(args: argparse.Namespace) -> dict[str, int]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    receiver_tokenizer = AutoTokenizer.from_pretrained(
        args.receiver_model,
        trust_remote_code=True,
    )
    sender_tokenizer = (
        receiver_tokenizer
        if args.sender_model == args.receiver_model
        else AutoTokenizer.from_pretrained(args.sender_model, trust_remote_code=True)
    )
    adapter_payload = load_frozen_adapter(
        _resolve_workspace_path(args.adapter_cache_path),
        adapter_digest=args.adapter_digest,
    )
    rows = load_bridge_sample_rows(args, tokenizer=sender_tokenizer, include_dev=False)
    train_samples = assemble_training_samples(
        rows["train"],
        tokenizer=receiver_tokenizer,
        adapter_payload=adapter_payload,
    )
    if not train_samples:
        raise SystemExit("No cached train traces matched the requested row range.")

    model_dtype = _torch_dtype(args.trace_dtype)
    receiver = AutoModelForCausalLM.from_pretrained(
        args.receiver_model,
        dtype=model_dtype,
        trust_remote_code=True,
    ).to(torch.device(args.device))
    receiver.eval()
    for parameter in receiver.parameters():
        parameter.requires_grad = False

    stats = prepare_rollouts(
        args=args,
        receiver=receiver,
        tokenizer=receiver_tokenizer,
        samples=train_samples,
    )
    print(
        "rollouts: "
        f"accepted={stats['accepted']} failed={stats['failed']} "
        f"skipped={stats['skipped']} invalid={stats['invalid']} "
        f"written={stats['written']}",
        flush=True,
    )
    return stats


def main() -> None:
    args = build_parser().parse_args()
    if args.prepare_rollouts:
        run_prepare_rollouts(args)
        return
    train(args)


if __name__ == "__main__":
    main()
