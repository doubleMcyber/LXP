from __future__ import annotations

import math
import re
from collections.abc import Sequence
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
    stop_regex: Optional[re.Pattern[str]] = None,
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
        decoded_text = str(prefix_state.get("decoded_text_prefix", "")) + tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=True,
        )
        if stop_regex is not None and stop_regex.search(decoded_text) is not None:
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


def _truncate_decoded_text_at_stop(
    decoded_text: str,
    stop_regex: Optional[re.Pattern[str]],
) -> str:
    if stop_regex is None:
        return decoded_text
    match = stop_regex.search(decoded_text)
    if match is None:
        return decoded_text
    return decoded_text[: match.end()]


def _broadcast_logits_bias(
    logits: torch.Tensor,
    logits_bias: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    bias = logits_bias.to(device=logits.device, dtype=logits.dtype)
    if bias.dim() == 1:
        bias = bias.unsqueeze(0)
    if bias.dim() != 2:
        raise ValueError(
            "logits_bias must have shape [vocab_size] or [batch, vocab_size]; "
            f"received {tuple(logits_bias.shape)}"
        )
    if bias.shape[-1] != logits.shape[-1]:
        raise ValueError(
            "logits_bias vocabulary dimension does not match logits: "
            f"{bias.shape[-1]} != {logits.shape[-1]}"
        )
    if bias.shape[0] == 1 and logits.shape[0] != 1:
        bias = bias.expand(logits.shape[0], -1)
    if bias.shape[0] != logits.shape[0]:
        raise ValueError(
            "logits_bias batch dimension does not match logits: "
            f"{bias.shape[0]} != {logits.shape[0]}"
        )
    return logits + (bias * float(scale))


def generate_from_prefix_embeddings(
    *,
    model: AutoModelForCausalLM,
    tokenizer: Any,
    prefix_embeds: torch.Tensor,
    max_new_tokens: int,
    attention_mask: Optional[torch.Tensor] = None,
    decoded_text_prefix: str = "",
    stop_regex: Optional[re.Pattern[str]] = None,
    first_step_logits_bias: Optional[torch.Tensor] = None,
    first_step_logits_bias_scale: float = 1.0,
    step_logits_bias: Optional[torch.Tensor] = None,
    step_logits_bias_scale: float = 1.0,
) -> dict[str, Any]:
    """Decode from a continuous prompt using the model's native generation path."""
    model_device = next(model.parameters()).device
    model_dtype = model.get_input_embeddings().weight.dtype
    if prefix_embeds.dim() != 3:
        raise ValueError(
            "prefix_embeds must have shape [batch, steps, embedding_dim]; "
            f"received {tuple(prefix_embeds.shape)}"
        )
    prefix = prefix_embeds.to(device=model_device, dtype=model_dtype)
    if attention_mask is None:
        attention_mask = torch.ones(
            prefix.shape[:2],
            dtype=torch.long,
            device=model_device,
        )
    else:
        attention_mask = attention_mask.to(device=model_device, dtype=torch.long)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id

    steering_bias = step_logits_bias
    steering_scale = float(step_logits_bias_scale)
    if steering_bias is None and first_step_logits_bias is not None:
        steering_bias = first_step_logits_bias.unsqueeze(1) if first_step_logits_bias.dim() == 2 else first_step_logits_bias
        steering_scale = float(first_step_logits_bias_scale)
    if steering_bias is not None:
        if steering_bias.dim() == 2:
            steering_bias = steering_bias.unsqueeze(1)
        if steering_bias.dim() != 3:
            raise ValueError(
                "step_logits_bias must have shape [batch, steps, vocab_size]; "
                f"received {tuple(steering_bias.shape)}"
            )
        with torch.no_grad():
            current_prefix = prefix
            current_attention = attention_mask
            generated_token_ids: list[int] = []
            eos_token_ids: set[int] = set()
            if eos_token_id is not None:
                if isinstance(eos_token_id, (list, tuple, set)):
                    eos_token_ids = {int(token_id) for token_id in eos_token_id}
                else:
                    eos_token_ids = {int(eos_token_id)}
            manual_steps = min(max(1, int(max_new_tokens)), int(steering_bias.shape[1]))
            stopped = False
            for step_index in range(manual_steps):
                outputs = model(
                    inputs_embeds=current_prefix,
                    attention_mask=current_attention,
                    use_cache=False,
                    return_dict=True,
                )
                step_logits = _broadcast_logits_bias(
                    outputs.logits[:, -1, :].float(),
                    steering_bias[:, step_index, :],
                    scale=steering_scale,
                )
                next_tokens = step_logits.argmax(dim=-1)
                next_token_id = int(next_tokens[0].detach().cpu().item())
                if next_token_id in eos_token_ids:
                    stopped = True
                    break
                generated_token_ids.append(next_token_id)
                next_token_embeds = model.get_input_embeddings()(next_tokens.unsqueeze(1)).to(
                    dtype=current_prefix.dtype,
                )
                current_prefix = torch.cat([current_prefix, next_token_embeds], dim=1)
                current_attention = torch.cat(
                    [
                        current_attention,
                        torch.ones(
                            (current_attention.shape[0], 1),
                            dtype=current_attention.dtype,
                            device=current_attention.device,
                        ),
                    ],
                    dim=1,
                )

        first_token_id = generated_token_ids[0] if generated_token_ids else None
        eos_token_ids: set[int] = set()
        if eos_token_id is not None:
            if isinstance(eos_token_id, (list, tuple, set)):
                eos_token_ids = {int(token_id) for token_id in eos_token_id}
            else:
                eos_token_ids = {int(eos_token_id)}
        if len(generated_token_ids) < max_new_tokens and not stopped:
            with torch.no_grad():
                generated_tail = model.generate(
                    inputs_embeds=current_prefix,
                    attention_mask=current_attention,
                    max_new_tokens=max(1, int(max_new_tokens) - len(generated_token_ids)),
                    do_sample=False,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                )
            generated_token_ids.extend(generated_tail[0].detach().cpu().tolist())
        decoded_suffix = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        decoded_text = _truncate_decoded_text_at_stop(
            str(decoded_text_prefix) + decoded_suffix,
            stop_regex,
        )
        return {
            "decoded_text": decoded_text,
            "generated_tokens": len(generated_token_ids),
            "generated_token_ids": generated_token_ids,
            "first_generated_token_id": first_token_id,
            "first_generated_token_text": None
            if first_token_id is None
            else tokenizer.decode([first_token_id], skip_special_tokens=True),
        }

    generation_kwargs: dict[str, Any] = {
        "inputs_embeds": prefix,
        "attention_mask": attention_mask,
        "max_new_tokens": max(1, int(max_new_tokens)),
        "do_sample": False,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
    }

    with torch.no_grad():
        generated = model.generate(**generation_kwargs)
    generated_token_ids = generated[0].detach().cpu().tolist()
    decoded_suffix = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    decoded_text = _truncate_decoded_text_at_stop(
        str(decoded_text_prefix) + decoded_suffix,
        stop_regex,
    )
    return {
        "decoded_text": decoded_text,
        "generated_tokens": len(generated_token_ids),
        "generated_token_ids": generated_token_ids,
        "first_generated_token_id": generated_token_ids[0] if generated_token_ids else None,
        "first_generated_token_text": None
        if not generated_token_ids
        else tokenizer.decode([generated_token_ids[0]], skip_special_tokens=True),
    }


def generate_from_text_prefix(
    *,
    model: AutoModelForCausalLM,
    tokenizer: Any,
    prefix_text: str,
    max_new_tokens: int,
    decoded_text_prefix: str = "",
    stop_regex: Optional[re.Pattern[str]] = None,
) -> dict[str, Any]:
    model_device = next(model.parameters()).device
    encoded = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(model_device)
    attention_mask = encoded["attention_mask"].to(model_device)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max(1, int(max_new_tokens)),
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
    generated_token_ids = generated[0, input_ids.shape[1] :].detach().cpu().tolist()
    decoded_suffix = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    decoded_text = _truncate_decoded_text_at_stop(
        str(decoded_text_prefix) + decoded_suffix,
        stop_regex,
    )
    return {
        "decoded_text": decoded_text,
        "generated_tokens": len(generated_token_ids),
    }


def compute_answer_metrics_from_prefix(
    *,
    model: AutoModelForCausalLM,
    tokenizer: Any,
    prefix_state: dict[str, Any],
    answer_text: Optional[str],
    answer_variants: Optional[Sequence[str]] = None,
) -> dict[str, Optional[float]]:
    candidates: list[str] = []
    if answer_text is not None and str(answer_text).strip():
        candidates.append(str(answer_text))
    for variant in answer_variants or ():
        if variant is None:
            continue
        variant_text = str(variant)
        if variant_text.strip() and variant_text not in candidates:
            candidates.append(variant_text)
    if not candidates:
        return {
            "answer_token_count": 0,
            "answer_nll": None,
            "answer_perplexity": None,
        }

    best_metrics: Optional[dict[str, Optional[float]]] = None
    for candidate in candidates:
        candidate_metrics = _compute_single_answer_metrics_from_prefix(
            model=model,
            tokenizer=tokenizer,
            prefix_state=prefix_state,
            answer_text=candidate,
        )
        if candidate_metrics["answer_nll"] is None:
            continue
        if best_metrics is None or float(candidate_metrics["answer_nll"]) < float(best_metrics["answer_nll"]):
            best_metrics = candidate_metrics
    if best_metrics is None:
        return {
            "answer_token_count": 0,
            "answer_nll": None,
            "answer_perplexity": None,
        }
    return best_metrics


def compute_answer_metrics_from_prefix_embeddings(
    *,
    model: AutoModelForCausalLM,
    tokenizer: Any,
    prefix_embeds: torch.Tensor,
    answer_text: Optional[str],
    answer_variants: Optional[Sequence[str]] = None,
    first_step_logits_bias: Optional[torch.Tensor] = None,
    first_step_logits_bias_scale: float = 1.0,
    step_logits_bias: Optional[torch.Tensor] = None,
    step_logits_bias_scale: float = 1.0,
) -> dict[str, Optional[float]]:
    candidates: list[str] = []
    if answer_text is not None and str(answer_text).strip():
        candidates.append(str(answer_text))
    for variant in answer_variants or ():
        if variant is None:
            continue
        variant_text = str(variant)
        if variant_text.strip() and variant_text not in candidates:
            candidates.append(variant_text)
    if not candidates:
        return {
            "answer_token_count": 0,
            "answer_nll": None,
            "answer_perplexity": None,
        }

    best_metrics: Optional[dict[str, Optional[float]]] = None
    for candidate in candidates:
        candidate_metrics = _compute_single_answer_metrics_from_prefix_embeddings(
            model=model,
            tokenizer=tokenizer,
            prefix_embeds=prefix_embeds,
            answer_text=candidate,
            first_step_logits_bias=first_step_logits_bias,
            first_step_logits_bias_scale=first_step_logits_bias_scale,
            step_logits_bias=step_logits_bias,
            step_logits_bias_scale=step_logits_bias_scale,
        )
        if candidate_metrics["answer_nll"] is None:
            continue
        if best_metrics is None or float(candidate_metrics["answer_nll"]) < float(best_metrics["answer_nll"]):
            best_metrics = candidate_metrics
    if best_metrics is None:
        return {
            "answer_token_count": 0,
            "answer_nll": None,
            "answer_perplexity": None,
        }
    return best_metrics


def compute_first_token_metrics_from_prefix_embeddings(
    *,
    model: AutoModelForCausalLM,
    tokenizer: Any,
    prefix_embeds: torch.Tensor,
    answer_text: Optional[str],
    answer_variants: Optional[Sequence[str]] = None,
    first_step_logits_bias: Optional[torch.Tensor] = None,
    first_step_logits_bias_scale: float = 1.0,
) -> dict[str, Optional[float | int | str | bool]]:
    candidates: list[str] = []
    if answer_text is not None and str(answer_text).strip():
        candidates.append(str(answer_text))
    for variant in answer_variants or ():
        if variant is None:
            continue
        variant_text = str(variant)
        if variant_text.strip() and variant_text not in candidates:
            candidates.append(variant_text)
    encoded_candidates = [
        _token_ids
        for candidate in candidates
        if (_token_ids := tokenizer.encode(candidate, add_special_tokens=False))
    ]
    if not encoded_candidates:
        return {
            "first_token_rank": None,
            "first_token_top1": None,
            "first_token_margin": None,
            "first_token_target_id": None,
            "first_token_predicted_id": None,
            "first_token_predicted_text": None,
        }

    model_device = next(model.parameters()).device
    embedding = model.get_input_embeddings()
    prefix = prefix_embeds.to(device=model_device, dtype=embedding.weight.dtype)
    attention_mask = torch.ones(prefix.shape[:2], dtype=torch.long, device=model_device)
    with torch.no_grad():
        outputs = model(
            inputs_embeds=prefix,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
    )
    logits = outputs.logits[:, -1, :].float()
    if first_step_logits_bias is not None:
        logits = _broadcast_logits_bias(
            logits,
            first_step_logits_bias,
            scale=first_step_logits_bias_scale,
        )
    predicted_id = int(logits.argmax(dim=-1)[0].detach().cpu().item())
    predicted_text = (
        tokenizer.decode([predicted_id], skip_special_tokens=True)
        if hasattr(tokenizer, "decode")
        else str(predicted_id)
    )
    best: dict[str, Optional[float | int | str | bool]] | None = None
    for token_ids in encoded_candidates:
        target_id = int(token_ids[0])
        target_logit = logits[:, target_id]
        other_logits = logits.clone()
        other_logits[:, target_id] = float("-inf")
        best_other = other_logits.max(dim=-1).values
        rank = int((1 + (logits > target_logit.unsqueeze(1)).sum(dim=-1))[0].detach().cpu().item())
        margin = float((target_logit - best_other)[0].detach().cpu().item())
        top1 = predicted_id == target_id
        candidate_result: dict[str, Optional[float | int | str | bool]] = {
            "first_token_rank": rank,
            "first_token_top1": top1,
            "first_token_margin": margin,
            "first_token_target_id": target_id,
            "first_token_predicted_id": predicted_id,
            "first_token_predicted_text": predicted_text,
        }
        if best is None or int(candidate_result["first_token_rank"] or 10**9) < int(
            best["first_token_rank"] or 10**9
        ):
            best = candidate_result
    return best or {
        "first_token_rank": None,
        "first_token_top1": None,
        "first_token_margin": None,
        "first_token_target_id": None,
        "first_token_predicted_id": None,
        "first_token_predicted_text": None,
    }


def _compute_single_answer_metrics_from_prefix(
    *,
    model: AutoModelForCausalLM,
    tokenizer: Any,
    prefix_state: dict[str, Any],
    answer_text: str,
) -> dict[str, Optional[float]]:
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


def _compute_single_answer_metrics_from_prefix_embeddings(
    *,
    model: AutoModelForCausalLM,
    tokenizer: Any,
    prefix_embeds: torch.Tensor,
    answer_text: str,
    first_step_logits_bias: Optional[torch.Tensor] = None,
    first_step_logits_bias_scale: float = 1.0,
    step_logits_bias: Optional[torch.Tensor] = None,
    step_logits_bias_scale: float = 1.0,
) -> dict[str, Optional[float]]:
    answer_token_ids = tokenizer.encode(str(answer_text), add_special_tokens=False)
    if not answer_token_ids:
        return {
            "answer_token_count": 0,
            "answer_nll": None,
            "answer_perplexity": None,
        }

    model_device = next(model.parameters()).device
    embedding = model.get_input_embeddings()
    prefix = prefix_embeds.to(device=model_device, dtype=embedding.weight.dtype)
    answer_tensor = torch.tensor(
        [answer_token_ids],
        dtype=torch.long,
        device=model_device,
    )
    answer_embeds = embedding(answer_tensor)
    inputs_embeds = torch.cat([prefix, answer_embeds], dim=1)
    attention_mask = torch.ones(
        inputs_embeds.shape[:2],
        dtype=torch.long,
        device=model_device,
    )
    with torch.no_grad():
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

    answer_start = prefix.shape[1]
    per_token_nll: list[torch.Tensor] = []
    steering_bias = step_logits_bias
    steering_scale = float(step_logits_bias_scale)
    if steering_bias is None and first_step_logits_bias is not None:
        steering_bias = first_step_logits_bias.unsqueeze(1) if first_step_logits_bias.dim() == 2 else first_step_logits_bias
        steering_scale = float(first_step_logits_bias_scale)
    if steering_bias is not None and steering_bias.dim() == 2:
        steering_bias = steering_bias.unsqueeze(1)
    for offset, token_id in enumerate(answer_token_ids):
        prediction_position = answer_start + offset - 1
        target = torch.tensor([token_id], dtype=torch.long, device=model_device)
        logits = outputs.logits[:, prediction_position, :].float()
        if steering_bias is not None and offset < int(steering_bias.shape[1]):
            logits = _broadcast_logits_bias(
                logits,
                steering_bias[:, offset, :],
                scale=steering_scale,
            )
        per_token_nll.append(F.cross_entropy(logits, target, reduction="mean"))

    mean_nll = float(torch.stack(per_token_nll).mean().detach().cpu().item())
    return {
        "answer_token_count": len(answer_token_ids),
        "answer_nll": mean_nll,
        "answer_perplexity": float(math.exp(mean_nll)),
    }
