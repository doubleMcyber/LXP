from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from src.models.dynamics import (
    _kv_cache_compatibility_status,
    _kv_cache_layer_key_value,
    _kv_cache_seq_len,
    _move_kv_cache_to_device,
    _normalize_kv_cache,
)


def build_position_ids(attention_mask: torch.Tensor) -> torch.LongTensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    return position_ids.clamp_min_(0)


def _reset_kv_cache_to_prefix(past_key_values: Any, prefix_seq_len: int) -> None:
    if past_key_values is None:
        return
    if hasattr(past_key_values, "crop"):
        try:
            past_key_values.crop(int(prefix_seq_len))
        except Exception:  # noqa: BLE001
            pass


def _tuple_kv_cache_batch_size(kv_cache: Any) -> Optional[int]:
    if kv_cache is None:
        return None
    pair = _kv_cache_layer_key_value(kv_cache, 0)
    if pair is None:
        return None
    key_tensor, _ = pair
    if key_tensor.dim() < 1:
        return None
    return int(key_tensor.shape[0])


def prepare_text_prefix_state(
    *,
    model: AutoModelForCausalLM,
    tokenizer: Any,
    prefix_text: str,
) -> dict[str, Any]:
    model_device = next(model.parameters()).device
    encoded = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(model_device)
    attention_mask = encoded["attention_mask"].to(model_device)
    position_ids = build_position_ids(attention_mask)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )

    return {
        "outputs": outputs,
        "attention_mask": attention_mask,
        "prefix_seq_len": int(attention_mask.shape[1]),
        "kv_cache_transferred": None,
    }


def prepare_latent_prefix_state(
    *,
    model: AutoModelForCausalLM,
    handoff_step: torch.Tensor,
    kv_cache: Any = None,
) -> dict[str, Any]:
    model_device = next(model.parameters()).device
    model_dtype = model.get_input_embeddings().weight.dtype
    model_input_dim = int(model.get_input_embeddings().weight.shape[-1])
    if handoff_step.dim() != 3:
        raise ValueError(
            "latent handoff_step must have shape [batch, steps, embedding_dim]; "
            f"received {tuple(handoff_step.shape)}"
        )
    if int(handoff_step.shape[0]) < 1 or int(handoff_step.shape[1]) < 1:
        raise ValueError(
            "latent handoff_step must contain at least one batch item and one prefix step; "
            f"received {tuple(handoff_step.shape)}"
        )
    if int(handoff_step.shape[-1]) != model_input_dim:
        raise ValueError(
            "latent handoff_step embedding dimension "
            f"{handoff_step.shape[-1]} does not match model input embedding dimension "
            f"{model_input_dim}"
        )
    prefix = handoff_step.to(device=model_device, dtype=model_dtype)
    kv_cache_candidate = _move_kv_cache_to_device(kv_cache, model_device)
    kv_cache_status, kv_cache_reason = _kv_cache_compatibility_status(
        kv_cache_candidate,
        model,
    )
    kv_cache_transferred = kv_cache_status == "transferred"
    if kv_cache_transferred:
        cache_batch_size = _tuple_kv_cache_batch_size(kv_cache_candidate)
        if cache_batch_size is not None and cache_batch_size != int(prefix.shape[0]):
            kv_cache_status = "invalid_cache"
            kv_cache_reason = (
                f"batch_size_mismatch: expected {int(prefix.shape[0])}, got {cache_batch_size}"
            )
            kv_cache_transferred = False
    prefix_kv_cache = kv_cache_candidate if kv_cache_transferred else None
    attention_mask = torch.ones(
        (prefix.shape[0], _kv_cache_seq_len(prefix_kv_cache) + prefix.shape[1]),
        dtype=torch.long,
        device=model_device,
    )
    position_ids = build_position_ids(attention_mask)[:, -prefix.shape[1] :]

    with torch.no_grad():
        outputs = model(
            inputs_embeds=prefix,
            past_key_values=prefix_kv_cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )

    return {
        "outputs": outputs,
        "attention_mask": attention_mask,
        "prefix_seq_len": int(attention_mask.shape[1]),
        "kv_cache_transferred": kv_cache_transferred,
        "kv_cache_status": kv_cache_status,
        "kv_cache_reason": kv_cache_reason,
        "active_kv_cache_transferred": kv_cache_transferred,
        "active_kv_cache_status": kv_cache_status,
        "active_kv_cache_reason": kv_cache_reason,
        "active_kv_cache_source": "provided_cache" if kv_cache_transferred else "none",
        "receiver_context_status": "not_used",
        "receiver_context_reason": "latent_only",
        "receiver_context_token_count": 0,
    }


def prepare_receiver_context_latent_prefix_state(
    *,
    model: AutoModelForCausalLM,
    tokenizer: Any,
    context_text: str,
    handoff_step: torch.Tensor,
    kv_cache: Any = None,
    reason: str = "sender_kv_cache_not_transferred",
    suffix_text: str = "",
    decoded_text_prefix: str = "",
    latent_position: str = "after_context",
) -> dict[str, Any]:
    if latent_position not in {"after_context", "before_context"}:
        raise ValueError("latent_position must be one of: after_context, before_context")

    context_token_count = 0
    if latent_position == "after_context":
        context_prefix_state = prepare_text_prefix_state(
            model=model,
            tokenizer=tokenizer,
            prefix_text=context_text,
        )
        context_token_count = int(context_prefix_state["prefix_seq_len"])
        context_kv_cache = _normalize_kv_cache(context_prefix_state["outputs"].past_key_values)
        latent_prefix_state = prepare_latent_prefix_state(
            model=model,
            handoff_step=handoff_step,
            kv_cache=context_kv_cache,
        )
        active_kv_cache_source = "receiver_context"
    else:
        latent_prefix_state = prepare_latent_prefix_state(
            model=model,
            handoff_step=handoff_step,
            kv_cache=kv_cache,
        )
        active_kv_cache_source = (
            "sender_then_receiver_context"
            if latent_prefix_state["kv_cache_transferred"]
            else "latent_then_receiver_context"
        )
        latent_prefix_state = append_text_to_prefix_state(
            model=model,
            tokenizer=tokenizer,
            prefix_state=latent_prefix_state,
            suffix_text=context_text,
        )
        context_token_count = int(latent_prefix_state.get("suffix_token_count", 0))

    suffix_token_count = 0
    if suffix_text.strip():
        latent_prefix_state = append_text_to_prefix_state(
            model=model,
            tokenizer=tokenizer,
            prefix_state=latent_prefix_state,
            suffix_text=suffix_text,
            decoded_text_prefix=decoded_text_prefix or suffix_text,
        )
        suffix_token_count = int(latent_prefix_state.get("suffix_token_count", 0))
    latent_prefix_state.update(
        {
            "receiver_context_status": "used_prompt_prefix",
            "receiver_context_reason": reason,
            "receiver_context_token_count": context_token_count,
            "receiver_context_suffix_token_count": suffix_token_count,
            "receiver_context_latent_position": latent_position,
            "receiver_context_kv_cache_transferred": latent_prefix_state["kv_cache_transferred"],
            "receiver_context_kv_cache_status": latent_prefix_state["kv_cache_status"],
            "receiver_context_kv_cache_reason": latent_prefix_state["kv_cache_reason"],
            "active_kv_cache_transferred": latent_prefix_state["kv_cache_transferred"],
            "active_kv_cache_status": latent_prefix_state["kv_cache_status"],
            "active_kv_cache_reason": latent_prefix_state["kv_cache_reason"],
            "active_kv_cache_source": active_kv_cache_source,
        }
    )
    return latent_prefix_state


def append_text_to_prefix_state(
    *,
    model: AutoModelForCausalLM,
    tokenizer: Any,
    prefix_state: dict[str, Any],
    suffix_text: str,
    decoded_text_prefix: str = "",
) -> dict[str, Any]:
    model_device = next(model.parameters()).device
    encoded = tokenizer(suffix_text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(model_device)
    if input_ids.numel() == 0:
        return prefix_state

    outputs = prefix_state["outputs"]
    attention_mask = prefix_state["attention_mask"]
    kv_cache = _normalize_kv_cache(outputs.past_key_values)
    suffix_attention = torch.ones(
        (attention_mask.shape[0], input_ids.shape[1]),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    attention_mask = torch.cat([attention_mask, suffix_attention], dim=1)
    position_ids = build_position_ids(attention_mask)[:, -input_ids.shape[1] :]

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=kv_cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )

    return {
        **prefix_state,
        "outputs": outputs,
        "attention_mask": attention_mask,
        "prefix_seq_len": int(attention_mask.shape[1]),
        "decoded_text_prefix": str(prefix_state.get("decoded_text_prefix", "")) + decoded_text_prefix,
        "suffix_token_count": int(input_ids.shape[1]),
    }


def greedy_decode_from_prefix(
    *,
    model: AutoModelForCausalLM,
    tokenizer: Any,
    prefix_state: dict[str, Any],
    max_new_tokens: int,
) -> dict[str, Any]:
    outputs = prefix_state["outputs"]
    attention_mask = prefix_state["attention_mask"]
    prefix_seq_len = int(prefix_state.get("prefix_seq_len", attention_mask.shape[1]))
    _reset_kv_cache_to_prefix(outputs.past_key_values, prefix_seq_len)
    generated_token_ids: list[int] = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    for _ in range(max_new_tokens):
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        next_token_id = int(next_token.item())
        if eos_token_id is not None and next_token_id == eos_token_id:
            break
        generated_token_ids.append(next_token_id)
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
        position_ids = build_position_ids(attention_mask)[:, -1:]
        with torch.no_grad():
            outputs = model(
                input_ids=next_token.unsqueeze(-1),
                past_key_values=kv_cache,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
            )

    return {
        "decoded_text": str(prefix_state.get("decoded_text_prefix", ""))
        + tokenizer.decode(generated_token_ids, skip_special_tokens=True),
        "generated_tokens": len(generated_token_ids),
    }


def compute_answer_metrics_from_prefix(
    *,
    model: AutoModelForCausalLM,
    tokenizer: Any,
    prefix_state: dict[str, Any],
    answer_text: Optional[str],
) -> dict[str, Optional[float]]:
    if answer_text is None or not str(answer_text).strip():
        return {
            "answer_token_count": 0,
            "answer_nll": None,
            "answer_perplexity": None,
        }

    answer_token_ids = tokenizer.encode(str(answer_text), add_special_tokens=False)
    if not answer_token_ids:
        return {
            "answer_token_count": 0,
            "answer_nll": None,
            "answer_perplexity": None,
        }

    outputs = prefix_state["outputs"]
    attention_mask = prefix_state["attention_mask"]
    prefix_seq_len = int(prefix_state.get("prefix_seq_len", attention_mask.shape[1]))
    _reset_kv_cache_to_prefix(outputs.past_key_values, prefix_seq_len)
    per_token_nll: list[torch.Tensor] = []

    for token_id in answer_token_ids:
        target = torch.tensor([token_id], device=outputs.logits.device)
        logits = outputs.logits[:, -1, :].float()
        per_token_nll.append(F.cross_entropy(logits, target, reduction="mean"))

        kv_cache = _normalize_kv_cache(outputs.past_key_values)
        next_token = torch.tensor([[token_id]], device=outputs.logits.device, dtype=torch.long)
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
        position_ids = build_position_ids(attention_mask)[:, -1:]
        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                past_key_values=kv_cache,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
            )

    mean_nll = float(torch.stack(per_token_nll).mean().detach().cpu().item())
    return {
        "answer_token_count": len(answer_token_ids),
        "answer_nll": mean_nll,
        "answer_perplexity": float(math.exp(mean_nll)),
    }
