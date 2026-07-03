from __future__ import annotations

import inspect
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


def _last_logit_forward_kwargs(model: Any) -> dict[str, Any]:
    """Kwargs asking the model to compute only the last position's logits.

    Prefix builders read only ``logits[:, -1, :]``, but without this the forward
    transiently materializes the full ``[1, seq, vocab]`` logits tensor
    (~4.9GB fp32 at 8k context) before ``_retain_last_position_logits`` can drop
    it. Guarded via signature inspection because duck-typed test models do not
    accept the kwarg.
    """
    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return {}
    if "logits_to_keep" in parameters:
        return {"logits_to_keep": 1}
    return {}


def _retain_last_position_logits(outputs: Any) -> Any:
    """Drop all but the last-position logits from a retained prefix forward.

    Prefix states outlive their forward pass (decode, answer scoring, entropy) but
    every consumer reads only ``logits[:, -1, :]``; keeping the full [1, seq, vocab]
    tensor alive dominates peak accelerator memory on long-context prompts. The
    clone detaches the slice from the full tensor's storage so it can be freed.
    """
    logits = getattr(outputs, "logits", None)
    if torch.is_tensor(logits) and logits.dim() == 3 and logits.shape[1] > 1:
        outputs.logits = logits[:, -1:, :].clone()
    return outputs


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
            **_last_logit_forward_kwargs(model),
        )

    return {
        "outputs": _retain_last_position_logits(outputs),
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
            **_last_logit_forward_kwargs(model),
        )

    return {
        "outputs": _retain_last_position_logits(outputs),
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
        # Fused single forward over [context tokens, latents]. Chunked prefill
        # (context forward, then latents with past_key_values) is mathematically
        # equivalent for pure attention but drifts numerically on hybrid
        # linear-attention/SSM caches — enough to deflect long greedy decodes.
        # One concatenated forward matches the latent-bridge trainer exactly.
        model_device = next(model.parameters()).device
        encoded = tokenizer(context_text, return_tensors="pt", add_special_tokens=False)
        context_ids = encoded["input_ids"].to(model_device)
        context_token_count = int(context_ids.shape[1])
        with torch.no_grad():
            context_embeds = model.get_input_embeddings()(context_ids)
        combined_prefix = torch.cat(
            [
                context_embeds,
                handoff_step.to(device=model_device, dtype=context_embeds.dtype),
            ],
            dim=1,
        )
        latent_prefix_state = prepare_latent_prefix_state(
            model=model,
            handoff_step=combined_prefix,
            kv_cache=None,
        )
        # The receiver context supplies the effective prefix cache; report it as
        # such even though it is built in the same forward as the latents.
        latent_prefix_state.update(
            {
                "kv_cache_transferred": True,
                "kv_cache_status": "transferred",
                "kv_cache_reason": "receiver_context_fused_forward",
            }
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
            **_last_logit_forward_kwargs(model),
        )

    return {
        **prefix_state,
        "outputs": _retain_last_position_logits(outputs),
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
    # Appended tokens are always attended, so each step's position is simply the
    # previous last position + 1; recomputing the cumsum over the full mask every
    # step is O(T * (prefix + T)) for the same result.
    last_position_ids = build_position_ids(attention_mask)[:, -1:]

    for _ in range(max_new_tokens):
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
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
    stop_after_steering: bool = False,
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
            stopped_by_regex = False
            tail_generation_used = False
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
                if stop_regex is not None:
                    partial_suffix = tokenizer.decode(
                        generated_token_ids,
                        skip_special_tokens=True,
                    )
                    partial_text = str(decoded_text_prefix) + partial_suffix
                    stop_match = stop_regex.search(partial_text)
                    if stop_match is not None and stop_match.end() < len(partial_text):
                        stopped = True
                        stopped_by_regex = True
                        break
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
        if len(generated_token_ids) < max_new_tokens and not stopped and not stop_after_steering:
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
            tail_generation_used = True
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
            "steering_manual_steps": len(generated_token_ids),
            "steering_stopped_by_regex": stopped_by_regex,
            "steering_tail_generation_used": tail_generation_used,
            "steering_stop_after_steering": bool(stop_after_steering),
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


def _candidate_token_sequences(
    *,
    tokenizer: Any,
    candidate_answers: Sequence[str],
    selected_answer: str | None,
    max_new_tokens: int,
) -> tuple[list[list[int]], list[int]]:
    raw_candidates = [str(candidate).strip() for candidate in candidate_answers if str(candidate).strip()]
    if selected_answer is not None and str(selected_answer).strip():
        raw_candidates.append(str(selected_answer).strip())
    sequences: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for candidate in raw_candidates:
        token_ids = tokenizer.encode(candidate, add_special_tokens=False)
        token_ids = [int(token_id) for token_id in token_ids[: max(1, int(max_new_tokens))]]
        key = tuple(token_ids)
        if not token_ids or key in seen:
            continue
        sequences.append(token_ids)
        seen.add(key)
    selected_sequence: list[int] = []
    if selected_answer is not None and str(selected_answer).strip():
        selected_sequence = [
            int(token_id)
            for token_id in tokenizer.encode(
                str(selected_answer).strip(),
                add_special_tokens=False,
            )[: max(1, int(max_new_tokens))]
        ]
    return sequences, selected_sequence


def _allowed_trie_next_tokens(
    candidate_sequences: Sequence[Sequence[int]],
    generated_token_ids: Sequence[int],
) -> tuple[set[int], bool]:
    prefix = tuple(int(token_id) for token_id in generated_token_ids)
    allowed: set[int] = set()
    completed = False
    for sequence in candidate_sequences:
        sequence_tuple = tuple(int(token_id) for token_id in sequence)
        if len(prefix) > len(sequence_tuple):
            continue
        if sequence_tuple[: len(prefix)] != prefix:
            continue
        if len(prefix) == len(sequence_tuple):
            completed = True
        else:
            allowed.add(sequence_tuple[len(prefix)])
    return allowed, completed


def generate_guided_answer_from_text_prefix(
    *,
    model: AutoModelForCausalLM,
    tokenizer: Any,
    prefix_text: str,
    candidate_answers: Sequence[str],
    selected_answer: str | None,
    max_new_tokens: int,
    selected_answer_bias: float = 100.0,
    decoded_text_prefix: str = "",
    stop_regex: Optional[re.Pattern[str]] = None,
) -> dict[str, Any]:
    """Greedy actor decoding constrained to answer candidates.

    The receiver still runs the actor model step-by-step, but the output grammar is
    restricted to the benchmark answer manifold and the latent semantic readout can
    bias the selected answer path. This is a practical receiver-side bridge for
    demos and answer-only benchmarks when raw soft-prefix generation is unstable.
    """
    candidate_sequences, selected_sequence = _candidate_token_sequences(
        tokenizer=tokenizer,
        candidate_answers=candidate_answers,
        selected_answer=selected_answer,
        max_new_tokens=max_new_tokens,
    )
    if not candidate_sequences:
        return {
            "decoded_text": str(decoded_text_prefix),
            "generated_tokens": 0,
            "generated_token_ids": [],
            "first_generated_token_id": None,
            "first_generated_token_text": None,
            "guided_decode_status": "no_candidates",
        }

    model_device = next(model.parameters()).device
    embedding = model.get_input_embeddings()
    model_dtype = embedding.weight.dtype
    encoded_prefix = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded_prefix["input_ids"].to(model_device)
    attention_mask = encoded_prefix["attention_mask"].to(model_device)
    current_embeds = embedding(input_ids).to(dtype=model_dtype)
    generated_token_ids: list[int] = []

    with torch.no_grad():
        for _step_index in range(max(1, int(max_new_tokens))):
            allowed_tokens, completed = _allowed_trie_next_tokens(
                candidate_sequences,
                generated_token_ids,
            )
            if completed and not allowed_tokens:
                break
            if not allowed_tokens:
                break
            outputs = model(
                inputs_embeds=current_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            logits = outputs.logits[:, -1, :].float()
            allowed_tensor = torch.tensor(
                sorted(allowed_tokens),
                dtype=torch.long,
                device=logits.device,
            )
            mask = torch.full_like(logits, -1.0e9)
            mask.scatter_(1, allowed_tensor.unsqueeze(0), logits.index_select(1, allowed_tensor))
            if selected_sequence and tuple(selected_sequence[: len(generated_token_ids)]) == tuple(generated_token_ids):
                if len(generated_token_ids) < len(selected_sequence):
                    selected_next = int(selected_sequence[len(generated_token_ids)])
                    if selected_next in allowed_tokens:
                        mask[:, selected_next] = mask[:, selected_next] + float(selected_answer_bias)
            next_tokens = mask.argmax(dim=-1)
            next_token_id = int(next_tokens[0].detach().cpu().item())
            generated_token_ids.append(next_token_id)
            next_embeds = embedding(next_tokens.unsqueeze(1)).to(dtype=model_dtype)
            current_embeds = torch.cat([current_embeds, next_embeds], dim=1)
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
            if selected_sequence and tuple(generated_token_ids) == tuple(selected_sequence):
                break

    decoded_suffix = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    decoded_text = _truncate_decoded_text_at_stop(
        str(decoded_text_prefix) + decoded_suffix,
        stop_regex,
    )
    first_token_id = generated_token_ids[0] if generated_token_ids else None
    return {
        "decoded_text": decoded_text,
        "generated_tokens": len(generated_token_ids),
        "generated_token_ids": generated_token_ids,
        "first_generated_token_id": first_token_id,
        "first_generated_token_text": None
        if first_token_id is None
        else tokenizer.decode([first_token_id], skip_special_tokens=True),
        "guided_decode_status": "decoded" if generated_token_ids else "empty",
        "guided_selected_answer": selected_answer,
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
