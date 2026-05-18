from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from src.models.dynamics import (
    _kv_cache_compatibility_status,
    _kv_cache_seq_len,
    _move_kv_cache_to_device,
    _normalize_kv_cache,
)


def build_position_ids(attention_mask: torch.Tensor) -> torch.LongTensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    return position_ids.clamp_min_(0)


def _tuple_kv_cache_batch_size(kv_cache: Any) -> Optional[int]:
    if not isinstance(kv_cache, (tuple, list)) or not kv_cache:
        return None
    first_layer = kv_cache[0]
    if not isinstance(first_layer, (tuple, list)) or not first_layer:
        return None
    key_tensor = first_layer[0]
    if not torch.is_tensor(key_tensor) or key_tensor.dim() < 1:
        return None
    return int(key_tensor.shape[0])


def prepare_text_prefix_state(
    *,
    model: AutoModelForCausalLM,
    tokenizer: Any,
    prefix_text: str,
) -> dict[str, Any]:
    model_device = next(model.parameters()).device
    encoded = tokenizer(prefix_text, return_tensors="pt")
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
        "kv_cache_transferred": kv_cache_transferred,
        "kv_cache_status": kv_cache_status,
        "kv_cache_reason": kv_cache_reason,
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
        "decoded_text": tokenizer.decode(generated_token_ids, skip_special_tokens=True),
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
