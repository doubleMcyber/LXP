"""Comparative benchmark runner for phase 1 and phase 3 evaluation suites."""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import torch
from omegaconf import OmegaConf

from latent_pipeline import (
    _aggregate_hidden_layers,
    _build_position_ids,
    _compute_logits_entropy,
    _cosine_distance,
    _format_receiver_context_answer_suffix,
    _format_receiver_context_prompt,
    _get_pipeline_state,
    _latent_prefix_mode,
    _latent_pooling_mode,
    _normalized_l2_distance,
    _pool_latent_handoff_step,
    _receiver_context_latent_position,
    _receiver_context_mode,
    _run_actor_handoff,
    _select_hidden_layers,
    _should_use_receiver_context,
    apply_embedding_manifold_projection,
    apply_handoff_adapter,
    run_hybrid_pipeline,
)
from src.data.loader import get_dataloader, pick_field
from src.models.dynamics import _normalize_kv_cache
from src.utils.alignment import (
    apply_alignment,
    apply_linear_mapping,
    compute_alignment_state,
    compute_orthogonal_mapping,
    compute_ridge_mapping,
)
from src.utils.benchmarking import (
    REPORT_SCHEMA_VERSION,
    STANDARD_SAMPLE_FIELDS,
    STANDARD_SUMMARY_FIELDS,
    aggregate_standard_rows,
    build_ode_scaling_report,
    build_phase1_gate_report,
    build_phase3_gate_report,
    build_runtime_smoke_report,
    build_semantic_smoke_report,
    build_standard_row_base,
    write_csv,
    write_json,
)
from src.utils.lm_eval import (
    append_text_to_prefix_state,
    compute_answer_metrics_from_prefix,
    greedy_decode_from_prefix,
    prepare_latent_prefix_state,
    prepare_receiver_context_latent_prefix_state,
    prepare_text_prefix_state,
)
from src.utils.model_compat import load_model_pair_compatibility
from src.utils.metrics import extract_boxed_text, normalize_answer

DEFAULT_SUITE = "standard"
DEFAULT_DATASET = "gsm8k"
DEFAULT_SPLIT = None
DEFAULT_LIMIT = 10
DEFAULT_REPETITIONS = 1
DEFAULT_SAMPLES_OUTPUT = Path("outputs/benchmark_samples.csv")
DEFAULT_SUMMARY_OUTPUT = Path("outputs/benchmark_summary.csv")
DEFAULT_REPORT_OUTPUT = Path("outputs/benchmark_report.json")
DEFAULT_LATENT_STEPS_SWEEP = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_SEMANTIC_SMOKE_LIMIT = 3
DEFAULT_SEMANTIC_SMOKE_MAX_NEW_TOKENS = 128
DEFAULT_SEMANTIC_SMOKE_METHODS = (
    "pure_text_cot",
    "text_text_hybrid",
    "global_anchor_hybrid_affine",
    "hybrid_hl_mas",
)
DEFAULT_SEMANTIC_SMOKE_LATENT_METHODS = (
    "global_anchor_hybrid_affine",
    "hybrid_hl_mas",
)
DEFAULT_MVP_SMOKE_SAMPLE_INDICES = (0, 2, 3)
DEFAULT_MVP_SMOKE_METHODS = (
    "pure_text_cot",
    "text_text_hybrid",
    "homogeneous_orthogonal_latent",
)
DEFAULT_MVP_SMOKE_BASELINE_METHODS = (
    "pure_text_cot",
    "text_text_hybrid",
)
DEFAULT_MVP_SMOKE_LATENT_METHODS = (
    "homogeneous_orthogonal_latent",
)
DEFAULT_HETERO_SMOKE_AGENT_A_MODEL = "LGAI-EXAONE/EXAONE-4.0-1.2B"
DEFAULT_HETERO_SMOKE_AGENT_B_MODEL = "Qwen/Qwen3.5-0.8B"
DEFAULT_HETERO_SMOKE_SAMPLE_INDICES = (0, 2, 3)
DEFAULT_HETERO_SMOKE_METHODS = (
    "pure_text_cot",
    "text_text_hybrid",
    "global_anchor_hybrid_affine",
    "hybrid_hl_mas",
)
DEFAULT_HETERO_SMOKE_BASELINE_METHODS = (
    "pure_text_cot",
    "text_text_hybrid",
)
DEFAULT_HETERO_SMOKE_LATENT_METHODS = (
    "global_anchor_hybrid_affine",
    "hybrid_hl_mas",
)
GSM8K_FINAL_ANSWER_REGEX = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")
FINAL_ANSWER_MARKER_REGEX = re.compile(
    r"final\s+answer\s*[:=]\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)
FINAL_ANSWER_COMPLETE_REGEX = re.compile(
    r"final\s+answer\s*[:=]\s*(?:[$*`_\s]+)?-?\d[\d,]*(?:\.\d+)?(?:[$*`_\s]+|[^\d,.])",
    re.IGNORECASE,
)
NUMERIC_ANSWER_REGEX = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

RunnerFn = Callable[[str, Optional[str], Any, dict[str, Any]], dict[str, Any]]


def _load_cfg() -> Any:
    return OmegaConf.load(Path(__file__).resolve().parent / "configs" / "main.yaml")


def _clone_cfg(cfg: Any) -> Any:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def _dataset_cfg(cfg: Any, dataset_name: str) -> Any:
    return getattr(getattr(cfg, "datasets", None), dataset_name, None)


def _validation_size(cfg: Any, dataset_name: str) -> Optional[int]:
    dataset_cfg = _dataset_cfg(cfg, dataset_name)
    if dataset_cfg is None:
        return None
    raw_value = getattr(dataset_cfg, "validation_size", None)
    return None if raw_value is None else int(raw_value)


def _default_split_for_dataset(dataset_name: str) -> str:
    if dataset_name == "gsm8k":
        return "validation"
    return "test"


def _gating_cfg(cfg: Any, phase_name: str) -> Any:
    return getattr(getattr(getattr(cfg, "reporting", None), "phase_gates", None), phase_name, None)


def _runtime_smoke_cfg(cfg: Any) -> Any:
    return getattr(getattr(cfg, "reporting", None), "runtime_smoke", None)


def _is_runtime_smoke_run(cfg: Any, *, limit: int, repetitions: int) -> bool:
    smoke_cfg = _runtime_smoke_cfg(cfg)
    if not bool(getattr(smoke_cfg, "enabled", False)):
        return False
    return (
        int(limit) <= int(getattr(smoke_cfg, "max_samples", 1))
        and int(repetitions) <= int(getattr(smoke_cfg, "max_repetitions", 1))
    )


def _benchmark_cfg(cfg: Any) -> Any:
    return getattr(cfg, "benchmark", None)


def _text_hybrid_reasoning_max_new_tokens(cfg: Any) -> int:
    benchmark_cfg = _benchmark_cfg(cfg)
    return int(getattr(benchmark_cfg, "text_hybrid_reasoning_max_new_tokens", cfg.max_new_tokens))


def _answer_only_final_enabled(cfg: Any) -> bool:
    return bool(getattr(_benchmark_cfg(cfg), "answer_only_final", False))


def _final_answer_instruction(cfg: Any) -> str:
    if _answer_only_final_enabled(cfg):
        return (
            "Return exactly one line in this format: Final answer: <answer>. "
            "Do not include reasoning, equations, markdown, or extra text."
        )
    return "Think step by step, then give the final answer."


def _format_reasoner_cot_prompt(prompt: str, tokenizer: Any = None) -> str:
    user_message = (
        f"{prompt}\n\n"
        "Think step by step. Write concise reasoning that another model can use, "
        "then state the final answer."
    )
    return _maybe_apply_chat_template(tokenizer, user_message)


def _format_text_cot_prompt(prompt: str, tokenizer: Any = None, cfg: Any = None) -> str:
    user_message = f"{prompt}\n\n{_final_answer_instruction(cfg)}"
    return _maybe_apply_chat_template(tokenizer, user_message)


def _decode_stop_regex(cfg: Any) -> Optional[re.Pattern[str]]:
    return FINAL_ANSWER_COMPLETE_REGEX if _answer_only_final_enabled(cfg) else None


def _serialize_text_hybrid_prompt(
    prompt: str,
    reasoning_text: str,
    tokenizer: Any = None,
    cfg: Any = None,
) -> str:
    final_instruction = (
        "Use the reasoning above. Return exactly one line in this format: "
        "Final answer: <answer>. Do not include reasoning, equations, markdown, or extra text."
        if _answer_only_final_enabled(cfg)
        else "Use the reasoning above and give the final answer."
    )
    user_message = (
        f"{prompt}\n\n"
        f"Reasoning from Agent A:\n{reasoning_text.strip()}\n\n"
        f"{final_instruction}"
    )
    return _maybe_apply_chat_template(tokenizer, user_message)


def _maybe_apply_chat_template(tokenizer: Any, user_message: str) -> str:
    if tokenizer is None or not getattr(tokenizer, "chat_template", None):
        return user_message
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:  # noqa: BLE001
        return user_message


def _extract_gsm8k_target_answer(text: str) -> Optional[str]:
    match = GSM8K_FINAL_ANSWER_REGEX.search(text)
    return None if match is None else match.group(1)


def _extract_gsm8k_predicted_answer(text: str) -> Optional[str]:
    marker_match = FINAL_ANSWER_MARKER_REGEX.search(text)
    if marker_match is not None:
        return marker_match.group(1)
    matches = NUMERIC_ANSWER_REGEX.findall(text)
    if not matches:
        return None
    return matches[-1]


def _normalize_numeric_answer(answer: Optional[str]) -> Optional[str]:
    normalized = normalize_answer(answer)
    if normalized is None:
        return None
    return normalized.replace(",", "")


def _target_answer(dataset_name: str, row: Any) -> Optional[str]:
    if dataset_name == "gsm8k":
        return _extract_gsm8k_target_answer(pick_field(row, ("answer", "solution")))
    return extract_boxed_text(pick_field(row, ("solution", "answer")))


def _predicted_answer(dataset_name: str, decoded_text: str) -> Optional[str]:
    if dataset_name == "gsm8k":
        return _extract_gsm8k_predicted_answer(decoded_text)
    return extract_boxed_text(decoded_text)


def _answers_match(dataset_name: str, predicted_answer: Optional[str], target_answer: Optional[str]) -> bool:
    if dataset_name == "gsm8k":
        return _normalize_numeric_answer(predicted_answer) == _normalize_numeric_answer(target_answer)
    return normalize_answer(predicted_answer) == normalize_answer(target_answer)


def _reasoning_alignment_metadata(state: dict[str, Any]) -> tuple[tuple[int, ...], tuple[float, ...]]:
    return (
        tuple(state["global_reasoning_layer_indices"]),
        tuple(state["global_reasoning_layer_weights"]),
    )


def _alignment_variant_cfg(
    cfg: Any,
    *,
    strategy: str,
    prompt_calibration_enabled: bool,
) -> Any:
    variant_cfg = _clone_cfg(cfg)
    variant_cfg.alignment.strategy = str(strategy)
    variant_cfg.alignment.prompt_calibration.enabled = bool(prompt_calibration_enabled)
    return variant_cfg


def _alignment_variant_state(
    cfg: Any,
    state: dict[str, Any],
    *,
    strategy: str,
    prompt_calibration_enabled: bool,
) -> tuple[Any, dict[str, Any]]:
    cache = state.setdefault("_alignment_variant_state_cache", {})
    cache_key = (str(strategy), bool(prompt_calibration_enabled))
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    variant_cfg = _alignment_variant_cfg(
        cfg,
        strategy=strategy,
        prompt_calibration_enabled=prompt_calibration_enabled,
    )
    variant_state = _get_pipeline_state(variant_cfg)
    cache[cache_key] = (variant_cfg, variant_state)
    return variant_cfg, variant_state


def _generate_reasoner_text(prompt: str, cfg: Any, state: dict[str, Any]) -> str:
    tokenizer_a = state["tokenizer_a"]
    agent_a = state["agent_a"]
    agent_a_device = next(agent_a.parameters()).device
    output_embeddings = agent_a.get_output_embeddings()
    if output_embeddings is None:
        raise ValueError("Text-text hybrid baseline requires output embeddings from Agent A")

    encoded = tokenizer_a(
        _format_reasoner_cot_prompt(prompt, tokenizer_a),
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(agent_a_device)
    attention_mask = encoded["attention_mask"].to(agent_a_device)
    position_ids = _build_position_ids(attention_mask)

    with torch.no_grad():
        outputs = agent_a.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )

    generated_ids: list[int] = []
    eos_token_id = getattr(tokenizer_a, "eos_token_id", None)
    max_new_tokens = _text_hybrid_reasoning_max_new_tokens(cfg)

    for _ in range(max_new_tokens):
        next_token_logits = output_embeddings(outputs.last_hidden_state[:, -1:, :])[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        next_token_id = int(next_token.item())
        if eos_token_id is not None and next_token_id == eos_token_id:
            break
        generated_ids.append(next_token_id)

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
        position_ids = _build_position_ids(attention_mask)[:, -1:]
        with torch.no_grad():
            outputs = agent_a.model(
                input_ids=next_token.unsqueeze(-1),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=_normalize_kv_cache(outputs.past_key_values),
                use_cache=True,
                return_dict=True,
            )

    return tokenizer_a.decode(generated_ids, skip_special_tokens=True)


def _collect_sender_consensus_state(prompt: str, state: dict[str, Any], cfg: Any = None) -> dict[str, Any]:
    tokenizer_a = state["tokenizer_a"]
    agent_a = state["agent_a"]
    agent_a_device = next(agent_a.parameters()).device
    reasoning_layer_indices, reasoning_layer_weights = _reasoning_alignment_metadata(state)

    encoded = tokenizer_a(prompt, return_tensors="pt")
    input_ids_a = encoded["input_ids"].to(agent_a_device)
    attention_mask_a = encoded["attention_mask"].to(agent_a_device)
    position_ids_a = _build_position_ids(attention_mask_a)

    with torch.no_grad():
        outputs = agent_a.model(
            input_ids=input_ids_a,
            attention_mask=attention_mask_a,
            position_ids=position_ids_a,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

    if outputs.hidden_states is None:
        raise ValueError("Benchmark requires sender hidden states from Agent A")

    consensus_hidden_states = _aggregate_hidden_layers(
        _select_hidden_layers(outputs.hidden_states, reasoning_layer_indices),
        reasoning_layer_weights,
    )
    pooling_mode = _latent_pooling_mode(cfg)
    return {
        "consensus_hidden_states": consensus_hidden_states,
        "current_latent_step": _pool_latent_handoff_step(
            consensus_hidden_states,
            attention_mask_a,
            pooling_mode=pooling_mode,
        ),
        "attention_mask": attention_mask_a,
        "latent_pooling": pooling_mode,
        "kv_cache_a": _normalize_kv_cache(outputs.past_key_values),
        "reasoning_layer_indices": reasoning_layer_indices,
        "reasoning_layer_weights": reasoning_layer_weights,
    }


def _collect_sender_last_hidden_state(prompt: str, state: dict[str, Any], cfg: Any = None) -> dict[str, Any]:
    tokenizer_a = state["tokenizer_a"]
    agent_a = state["agent_a"]
    agent_a_device = next(agent_a.parameters()).device

    encoded = tokenizer_a(prompt, return_tensors="pt")
    input_ids_a = encoded["input_ids"].to(agent_a_device)
    attention_mask_a = encoded["attention_mask"].to(agent_a_device)
    position_ids_a = _build_position_ids(attention_mask_a)

    with torch.no_grad():
        hidden_states, kv_cache_a = agent_a(
            input_ids=input_ids_a,
            attention_mask=attention_mask_a,
            position_ids=position_ids_a,
            use_cache=True,
        )

    pooling_mode = _latent_pooling_mode(cfg)
    return {
        "current_latent_step": _pool_latent_handoff_step(
            hidden_states,
            attention_mask_a,
            pooling_mode=pooling_mode,
        ),
        "full_hidden_states": hidden_states,
        "latent_pooling": pooling_mode,
        "kv_cache_a": _normalize_kv_cache(kv_cache_a),
    }


def _collect_receiver_consensus_state(prompt: str, state: dict[str, Any]) -> dict[str, torch.Tensor]:
    tokenizer_b = state["tokenizer_b"]
    agent_b = state["agent_b"]
    agent_b_device = next(agent_b.parameters()).device
    reasoning_layer_indices, reasoning_layer_weights = _reasoning_alignment_metadata(state)

    encoded = tokenizer_b(prompt, return_tensors="pt")
    input_ids_b = encoded["input_ids"].to(agent_b_device)
    attention_mask_b = encoded["attention_mask"].to(agent_b_device)
    position_ids_b = _build_position_ids(attention_mask_b)

    with torch.no_grad():
        outputs = agent_b.model(
            input_ids=input_ids_b,
            attention_mask=attention_mask_b,
            position_ids=position_ids_b,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

    if outputs.hidden_states is None:
        raise ValueError("Benchmark requires receiver hidden states from Agent B")

    consensus_hidden_states = _aggregate_hidden_layers(
        _select_hidden_layers(outputs.hidden_states, reasoning_layer_indices),
        reasoning_layer_weights,
    )
    return {
        "consensus_hidden_states": consensus_hidden_states,
        "receiver_reference_handoff": consensus_hidden_states[:, -1:, :],
    }


def _collect_receiver_input_embedding_state(prompt: str, state: dict[str, Any]) -> dict[str, torch.Tensor]:
    tokenizer_b = state["tokenizer_b"]
    agent_b = state["agent_b"]
    agent_b_device = next(agent_b.parameters()).device

    encoded = tokenizer_b(prompt, return_tensors="pt")
    input_ids_b = encoded["input_ids"].to(agent_b_device)
    with torch.no_grad():
        input_embeddings = agent_b.get_input_embeddings()(input_ids_b)

    return {
        "input_embeddings": input_embeddings,
        "attention_mask": encoded["attention_mask"].to(agent_b_device),
        "receiver_reference_handoff": input_embeddings[:, -1:, :],
    }


def _select_handoff_latent(sender_state: dict[str, Any], cfg: Any) -> torch.Tensor:
    if _latent_prefix_mode(cfg) == "sequence":
        attention_mask = sender_state.get("attention_mask")
        if attention_mask is None or int(sender_state["consensus_hidden_states"].shape[0]) != 1:
            return sender_state["consensus_hidden_states"]
        valid_tokens = attention_mask[0].to(dtype=torch.bool)
        return sender_state["consensus_hidden_states"][:, valid_tokens, :]
    return sender_state["current_latent_step"]


def _resample_sequence(reference: torch.Tensor, target_steps: int) -> torch.Tensor:
    if reference.dim() != 3:
        raise ValueError("reference must have shape [batch, steps, dim]")
    if target_steps <= 0:
        raise ValueError("target_steps must be positive")
    if target_steps == 1:
        return reference[:, -1:, :]
    if int(reference.shape[1]) == target_steps:
        return reference
    transposed = reference.float().transpose(1, 2)
    resized = torch.nn.functional.interpolate(
        transposed,
        size=int(target_steps),
        mode="linear",
        align_corners=True,
    )
    return resized.transpose(1, 2).to(dtype=reference.dtype)


def _receiver_reference_for_handoff(
    prompt: str,
    state: dict[str, Any],
    target_steps: int,
) -> torch.Tensor:
    receiver_state = _collect_receiver_input_embedding_state(prompt, state)
    receiver_embeddings = receiver_state["input_embeddings"]
    return _resample_sequence(receiver_embeddings, target_steps)


def _maybe_append_answer_suffix_to_prefix_state(
    *,
    agent_b: Any,
    tokenizer_b: Any,
    cfg: Any,
    prefix_state: dict[str, Any],
) -> dict[str, Any]:
    suffix_text = _format_receiver_context_answer_suffix(cfg)
    if not suffix_text.strip():
        return prefix_state
    return append_text_to_prefix_state(
        model=agent_b,
        tokenizer=tokenizer_b,
        prefix_state=prefix_state,
        suffix_text=suffix_text,
        decoded_text_prefix="Final answer:",
    )


def _apply_local_sequence_calibration(
    source_step: torch.Tensor,
    handoff_step: torch.Tensor,
    receiver_reference_handoff: torch.Tensor,
    *,
    strength: float,
    max_norm_ratio: float,
) -> tuple[torch.Tensor, float]:
    local_q = compute_orthogonal_mapping(
        source_step.detach().float(),
        receiver_reference_handoff.detach().float().to(handoff_step.device),
    ).to(device=handoff_step.device, dtype=handoff_step.dtype)
    locally_aligned = apply_linear_mapping(
        source_step.to(device=handoff_step.device, dtype=handoff_step.dtype),
        local_q,
    )
    correction = (locally_aligned - handoff_step).float() * float(strength)
    correction_norm = torch.linalg.vector_norm(
        correction.reshape(correction.shape[0], -1),
        dim=-1,
    ).mean()
    if max_norm_ratio > 0.0:
        handoff_norm = torch.linalg.vector_norm(
            handoff_step.float().reshape(handoff_step.shape[0], -1),
            dim=-1,
        ).mean()
        max_norm = float(handoff_norm.item()) * float(max_norm_ratio)
        if correction_norm.item() > max_norm > 0.0:
            correction = correction * (max_norm / max(float(correction_norm.item()), 1e-8))
            correction_norm = torch.as_tensor(max_norm, device=handoff_step.device)
    calibrated = handoff_step + correction.to(device=handoff_step.device, dtype=handoff_step.dtype)
    return calibrated, float(correction_norm.detach().cpu().item())


def _decode_handoff(
    *,
    agent_b: Any,
    tokenizer_b: Any,
    prompt: Optional[str] = None,
    cfg: Any = None,
    handoff_step: torch.Tensor,
    kv_cache_a: Any,
    max_new_tokens: int,
    target_answer_text: Optional[str],
    append_answer_suffix_without_context: bool = False,
) -> dict[str, Any]:
    sender_prefix_state = prepare_latent_prefix_state(
        model=agent_b,
        handoff_step=handoff_step,
        kv_cache=kv_cache_a,
    )
    kv_cache_transferred = bool(sender_prefix_state["kv_cache_transferred"])
    kv_cache_status = sender_prefix_state.get(
        "kv_cache_status",
        "transferred" if kv_cache_transferred else "unsupported",
    )
    kv_cache_reason = sender_prefix_state.get("kv_cache_reason", "")
    receiver_context_status = str(sender_prefix_state.get("receiver_context_status", "not_used"))
    receiver_context_reason = str(sender_prefix_state.get("receiver_context_reason", "latent_only"))
    receiver_context_token_count = int(sender_prefix_state.get("receiver_context_token_count", 0))
    receiver_context_latent_position = str(
        sender_prefix_state.get("receiver_context_latent_position", "not_applicable")
    )
    active_kv_cache_transferred = bool(
        sender_prefix_state.get("active_kv_cache_transferred", kv_cache_transferred)
    )
    active_kv_cache_status = str(sender_prefix_state.get("active_kv_cache_status", kv_cache_status))
    active_kv_cache_reason = str(sender_prefix_state.get("active_kv_cache_reason", kv_cache_reason))
    active_kv_cache_source = str(sender_prefix_state.get("active_kv_cache_source", "provided_cache"))
    context_mode = _receiver_context_mode(cfg) if cfg is not None else "none"
    latent_position = _receiver_context_latent_position(cfg) if cfg is not None else "after_context"

    if prompt is not None and _should_use_receiver_context(
        context_mode,
        sender_kv_cache_transferred=kv_cache_transferred,
    ):
        context_reason = (
            "forced_prompt_prefix"
            if context_mode == "prompt_prefix"
            else f"sender_kv_cache_status:{kv_cache_status}"
        )
        prefix_state = prepare_receiver_context_latent_prefix_state(
            model=agent_b,
            tokenizer=tokenizer_b,
            context_text=_format_receiver_context_prompt(prompt, tokenizer_b, cfg),
            handoff_step=handoff_step,
            kv_cache=kv_cache_a,
            reason=context_reason,
            suffix_text=_format_receiver_context_answer_suffix(cfg),
            decoded_text_prefix="Final answer:",
            latent_position=latent_position,
        )
        receiver_context_status = str(prefix_state.get("receiver_context_status", "used_prompt_prefix"))
        receiver_context_reason = str(prefix_state.get("receiver_context_reason", context_reason))
        receiver_context_token_count = int(prefix_state.get("receiver_context_token_count", 0))
        receiver_context_latent_position = str(
            prefix_state.get("receiver_context_latent_position", latent_position)
        )
        active_kv_cache_transferred = bool(prefix_state.get("active_kv_cache_transferred", False))
        active_kv_cache_status = str(prefix_state.get("active_kv_cache_status", "not_provided"))
        active_kv_cache_reason = str(prefix_state.get("active_kv_cache_reason", ""))
        active_kv_cache_source = str(prefix_state.get("active_kv_cache_source", "receiver_context"))
    else:
        prefix_state = sender_prefix_state
        if append_answer_suffix_without_context:
            prefix_state = _maybe_append_answer_suffix_to_prefix_state(
                agent_b=agent_b,
                tokenizer_b=tokenizer_b,
                cfg=cfg,
                prefix_state=prefix_state,
            )

    outputs_b = prefix_state["outputs"]
    raw_handoff_entropy = float(
        _compute_logits_entropy(outputs_b.logits[:, -1, :]).mean().detach().cpu().item()
    )
    answer_metrics = compute_answer_metrics_from_prefix(
        model=agent_b,
        tokenizer=tokenizer_b,
        prefix_state=prefix_state,
        answer_text=target_answer_text,
    )
    decode_metrics = greedy_decode_from_prefix(
        model=agent_b,
        tokenizer=tokenizer_b,
        prefix_state=prefix_state,
        max_new_tokens=max_new_tokens,
        stop_regex=_decode_stop_regex(cfg),
    )
    decoded_text = str(decode_metrics["decoded_text"])
    return {
        "decoded_text": decoded_text,
        "generated_tokens": int(decode_metrics["generated_tokens"]),
        "decode_status": "decoded" if decoded_text.strip() else "empty_decode",
        "answer_token_count": answer_metrics["answer_token_count"],
        "answer_nll": answer_metrics["answer_nll"],
        "answer_perplexity": answer_metrics["answer_perplexity"],
        "kv_cache_transferred": kv_cache_transferred,
        "kv_cache_status": kv_cache_status,
        "kv_cache_reason": kv_cache_reason,
        "active_kv_cache_transferred": active_kv_cache_transferred,
        "active_kv_cache_status": active_kv_cache_status,
        "active_kv_cache_reason": active_kv_cache_reason,
        "active_kv_cache_source": active_kv_cache_source,
        "receiver_context_status": receiver_context_status,
        "receiver_context_reason": receiver_context_reason,
        "receiver_context_token_count": receiver_context_token_count,
        "receiver_context_latent_position": receiver_context_latent_position,
        "raw_handoff_entropy": raw_handoff_entropy,
    }


def _alignment_distances(
    *,
    prompt: str,
    state: dict[str, Any],
    current_latent_step: torch.Tensor,
    alignment_state: dict[str, Any] | torch.Tensor,
    cfg: Optional[Any] = None,
    adapter_state: Optional[dict[str, Any]] = None,
    calibration_strength: float = 0.0,
    calibration_max_norm_ratio: float = 0.0,
) -> dict[str, Optional[float]]:
    receiver_reference_handoff = _receiver_reference_for_handoff(
        prompt,
        state,
        int(current_latent_step.shape[1]),
    ).to(device=current_latent_step.device, dtype=current_latent_step.dtype)
    handoff_step = apply_alignment(current_latent_step, alignment_state)
    if adapter_state is not None:
        handoff_step = apply_alignment(handoff_step, adapter_state)
    if cfg is not None:
        handoff_step, _ = apply_embedding_manifold_projection(handoff_step, cfg, state)
    if calibration_strength > 0.0:
        correction = (receiver_reference_handoff - handoff_step) * float(calibration_strength)
        correction_norm = torch.linalg.vector_norm(
            correction.float().reshape(correction.shape[0], -1),
            dim=-1,
        ).mean()
        max_norm = (
            float(
                torch.linalg.vector_norm(
                    handoff_step.float().reshape(handoff_step.shape[0], -1),
                    dim=-1,
                ).mean().item()
            )
            * float(calibration_max_norm_ratio)
        )
        if correction_norm.item() > max_norm > 0.0:
            correction = correction * (max_norm / max(float(correction_norm.item()), 1e-8))
        handoff_step = handoff_step + correction.to(device=handoff_step.device, dtype=handoff_step.dtype)
    backbone_state = alignment_state
    if isinstance(alignment_state, dict):
        backbone_matrix = alignment_state.get(
            "orthogonal_q",
            alignment_state.get("mapping_matrix"),
        )
        backbone_state = {"mapping_matrix": backbone_matrix}
    backbone_handoff = apply_alignment(current_latent_step, backbone_state)
    return {
        "pre_alignment_l2_distance": float(
            _normalized_l2_distance(backbone_handoff, receiver_reference_handoff).mean().detach().cpu().item()
        ),
        "pre_alignment_cosine_distance": float(
            _cosine_distance(backbone_handoff, receiver_reference_handoff).mean().detach().cpu().item()
        ),
        "post_alignment_l2_distance": float(
            _normalized_l2_distance(handoff_step, receiver_reference_handoff).mean().detach().cpu().item()
        ),
        "post_alignment_cosine_distance": float(
            _cosine_distance(handoff_step, receiver_reference_handoff).mean().detach().cpu().item()
        ),
    }


def run_pure_text_cot(
    prompt: str,
    target_answer_text: Optional[str],
    cfg: Any,
    state: dict[str, Any],
) -> dict[str, Any]:
    tokenizer_b = state["tokenizer_b"]
    agent_b = state["agent_b"]
    prefix_text = _format_text_cot_prompt(prompt, tokenizer_b, cfg)
    prefix_state = prepare_text_prefix_state(
        model=agent_b,
        tokenizer=tokenizer_b,
        prefix_text=prefix_text,
    )
    answer_metrics = compute_answer_metrics_from_prefix(
        model=agent_b,
        tokenizer=tokenizer_b,
        prefix_state=prefix_state,
        answer_text=target_answer_text,
    )
    decode_metrics = greedy_decode_from_prefix(
        model=agent_b,
        tokenizer=tokenizer_b,
        prefix_state=prefix_state,
        max_new_tokens=int(cfg.max_new_tokens),
        stop_regex=_decode_stop_regex(cfg),
    )
    decoded_text = str(decode_metrics["decoded_text"])
    return {
        "decoded_text": decoded_text,
        "generated_tokens": int(decode_metrics["generated_tokens"]),
        "decode_status": "decoded" if decoded_text.strip() else "empty_decode",
        "answer_token_count": answer_metrics["answer_token_count"],
        "answer_nll": answer_metrics["answer_nll"],
        "answer_perplexity": answer_metrics["answer_perplexity"],
        "alignment_mode": "text_baseline",
        "handoff_status": "not_applicable",
        "handoff_surface": "text",
        "kv_cache_transferred": None,
        "kv_cache_status": "not_applicable",
        "kv_cache_reason": "text_baseline",
        "pre_alignment_l2_distance": None,
        "pre_alignment_cosine_distance": None,
        "post_alignment_l2_distance": None,
        "post_alignment_cosine_distance": None,
        "raw_handoff_entropy": None,
        "handoff_uncertainty": None,
        "confidence_gate_triggered": None,
        "fallback_discrete_reasoning_steps": None,
        "latent_trajectory_steps": None,
        "total_reasoning_steps": None,
        "continuous_integration_seconds": None,
    }


def run_text_text_hybrid(
    prompt: str,
    target_answer_text: Optional[str],
    cfg: Any,
    state: dict[str, Any],
) -> dict[str, Any]:
    tokenizer_b = state["tokenizer_b"]
    agent_b = state["agent_b"]
    reasoning_text = _generate_reasoner_text(prompt, cfg, state)
    prefix_text = _serialize_text_hybrid_prompt(prompt, reasoning_text, tokenizer_b, cfg)
    prefix_state = prepare_text_prefix_state(
        model=agent_b,
        tokenizer=tokenizer_b,
        prefix_text=prefix_text,
    )
    answer_metrics = compute_answer_metrics_from_prefix(
        model=agent_b,
        tokenizer=tokenizer_b,
        prefix_state=prefix_state,
        answer_text=target_answer_text,
    )
    decode_metrics = greedy_decode_from_prefix(
        model=agent_b,
        tokenizer=tokenizer_b,
        prefix_state=prefix_state,
        max_new_tokens=int(cfg.max_new_tokens),
        stop_regex=_decode_stop_regex(cfg),
    )
    decoded_text = str(decode_metrics["decoded_text"])
    return {
        "decoded_text": decoded_text,
        "generated_tokens": int(decode_metrics["generated_tokens"]),
        "decode_status": "decoded" if decoded_text.strip() else "empty_decode",
        "answer_token_count": answer_metrics["answer_token_count"],
        "answer_nll": answer_metrics["answer_nll"],
        "answer_perplexity": answer_metrics["answer_perplexity"],
        "alignment_mode": "text_text_hybrid",
        "handoff_status": "not_applicable",
        "handoff_surface": "text",
        "kv_cache_transferred": None,
        "kv_cache_status": "not_applicable",
        "kv_cache_reason": "text_text_baseline",
        "pre_alignment_l2_distance": None,
        "pre_alignment_cosine_distance": None,
        "post_alignment_l2_distance": None,
        "post_alignment_cosine_distance": None,
        "raw_handoff_entropy": None,
        "handoff_uncertainty": None,
        "confidence_gate_triggered": None,
        "fallback_discrete_reasoning_steps": None,
        "latent_trajectory_steps": None,
        "total_reasoning_steps": None,
        "continuous_integration_seconds": None,
    }


def run_prompt_local_latent(
    prompt: str,
    target_answer_text: Optional[str],
    cfg: Any,
    state: dict[str, Any],
) -> dict[str, Any]:
    tokenizer_b = state["tokenizer_b"]
    agent_b = state["agent_b"]
    agent_b_device = next(agent_b.parameters()).device
    sender_state = _collect_sender_consensus_state(prompt, state, cfg)
    receiver_state = _collect_receiver_input_embedding_state(prompt, state)
    handoff_source = _select_handoff_latent(sender_state, cfg)
    sequence_prefix = _latent_prefix_mode(cfg) == "sequence"
    prompt_local_q = compute_orthogonal_mapping(
        sender_state["consensus_hidden_states"],
        receiver_state["input_embeddings"].to(
            device=sender_state["consensus_hidden_states"].device,
            dtype=sender_state["consensus_hidden_states"].dtype,
        ),
    ).to(
        device=handoff_source.device,
        dtype=handoff_source.dtype,
    )
    handoff_step = apply_linear_mapping(handoff_source, prompt_local_q).to(
        device=agent_b_device,
        dtype=agent_b.get_input_embeddings().weight.dtype,
    )
    decode_metrics = _decode_handoff(
        agent_b=agent_b,
        tokenizer_b=tokenizer_b,
        prompt=None if sequence_prefix else prompt,
        cfg=cfg,
        handoff_step=handoff_step,
        kv_cache_a=None if sequence_prefix else sender_state["kv_cache_a"],
        max_new_tokens=int(cfg.max_new_tokens),
        target_answer_text=target_answer_text,
        append_answer_suffix_without_context=sequence_prefix,
    )
    return {
        **decode_metrics,
        **_alignment_distances(
            prompt=prompt,
            state=state,
            current_latent_step=handoff_source,
            alignment_state={"mapping_matrix": prompt_local_q},
        ),
        "alignment_mode": "prompt_local_procrustes",
        "alignment_strategy": "orthogonal",
        "handoff_status": "ok",
        "handoff_surface": "input_embedding_sequence" if sequence_prefix else "input_embedding",
        "handoff_uncertainty": None,
        "confidence_gate_triggered": False,
        "fallback_discrete_reasoning_steps": 0,
        "latent_trajectory_steps": int(handoff_step.shape[1]),
        "total_reasoning_steps": int(handoff_step.shape[1]),
        "continuous_integration_seconds": 0.0,
    }


def run_global_anchor_latent(
    prompt: str,
    target_answer_text: Optional[str],
    cfg: Any,
    state: dict[str, Any],
) -> dict[str, Any]:
    return _run_global_anchor_variant(
        prompt,
        target_answer_text,
        cfg,
        state,
        strategy=str(getattr(getattr(cfg, "alignment", None), "strategy", "hybrid_affine")),
        prompt_calibration_enabled=bool(
            getattr(getattr(getattr(cfg, "alignment", None), "prompt_calibration", None), "enabled", True)
        ),
        method_alignment_mode=str(getattr(getattr(cfg, "alignment", None), "strategy", "hybrid_affine")),
    )


def _run_global_anchor_variant(
    prompt: str,
    target_answer_text: Optional[str],
    cfg: Any,
    state: dict[str, Any],
    *,
    strategy: str,
    prompt_calibration_enabled: bool,
    method_alignment_mode: str,
) -> dict[str, Any]:
    variant_cfg, variant_state = _alignment_variant_state(
        cfg,
        state,
        strategy=strategy,
        prompt_calibration_enabled=prompt_calibration_enabled,
    )
    tokenizer_b = variant_state["tokenizer_b"]
    agent_b = variant_state["agent_b"]
    agent_b_device = next(agent_b.parameters()).device
    sender_state = _collect_sender_consensus_state(prompt, variant_state, variant_cfg)
    handoff_mapping = variant_state.get(
        "handoff_alignment_q",
        variant_state["global_alignment_q"],
    )
    handoff_bias = variant_state.get(
        "handoff_alignment_bias",
        variant_state.get("global_alignment_bias"),
    )
    alignment_state = {
        "mapping_matrix": handoff_mapping,
        "mapping_bias": handoff_bias,
        "pre_projection_state": variant_state.get(
            "handoff_pre_projection_state",
            variant_state.get("pre_projection_state"),
        ),
        "post_projection_state": variant_state.get(
            "handoff_post_projection_state",
            variant_state.get("post_projection_state"),
        ),
        "alignment_strategy": variant_state.get("alignment_strategy", strategy),
        "orthogonal_q": variant_state.get(
            "handoff_alignment_backbone_q",
            variant_state.get("global_alignment_backbone_q", handoff_mapping),
        ),
        "residual_matrix": variant_state.get("handoff_alignment_residual"),
        "residual_norm_ratio": variant_state.get(
            "handoff_residual_norm_ratio",
            variant_state.get("residual_norm_ratio"),
        ),
        "bias_norm": variant_state.get("handoff_bias_norm", variant_state.get("bias_norm")),
    }
    calibration_cfg = getattr(getattr(variant_cfg, "alignment", None), "prompt_calibration", None)
    calibration_strength = float(getattr(calibration_cfg, "strength", 0.0)) if prompt_calibration_enabled else 0.0
    calibration_max_norm_ratio = float(getattr(calibration_cfg, "max_norm_ratio", 0.0)) if prompt_calibration_enabled else 0.0
    handoff_source = _select_handoff_latent(sender_state, variant_cfg)
    sequence_prefix = _latent_prefix_mode(variant_cfg) == "sequence"
    handoff_step = apply_alignment(handoff_source, alignment_state).to(
        device=agent_b_device,
        dtype=agent_b.get_input_embeddings().weight.dtype,
    )
    handoff_step, adapter_metrics = apply_handoff_adapter(
        handoff_step,
        variant_state,
    )
    handoff_step, manifold_metrics = apply_embedding_manifold_projection(
        handoff_step,
        variant_cfg,
        variant_state,
    )
    calibration_bias_norm: Optional[float] = None
    if prompt_calibration_enabled:
        receiver_reference_handoff = _receiver_reference_for_handoff(
            prompt,
            variant_state,
            int(handoff_step.shape[1]),
        ).to(device=agent_b_device, dtype=handoff_step.dtype)
        if sequence_prefix:
            handoff_step, calibration_bias_norm = _apply_local_sequence_calibration(
                handoff_source,
                handoff_step,
                receiver_reference_handoff,
                strength=calibration_strength,
                max_norm_ratio=calibration_max_norm_ratio,
            )
        else:
            correction = (receiver_reference_handoff - handoff_step) * calibration_strength
            correction_norm = torch.linalg.vector_norm(
                correction.float().reshape(correction.shape[0], -1),
                dim=-1,
            ).mean()
            max_norm = (
                float(
                    torch.linalg.vector_norm(
                        handoff_step.float().reshape(handoff_step.shape[0], -1),
                        dim=-1,
                    ).mean().item()
                )
                * calibration_max_norm_ratio
            )
            if correction_norm.item() > max_norm > 0.0:
                correction = correction * (max_norm / max(float(correction_norm.item()), 1e-8))
            handoff_step = handoff_step + correction.to(device=handoff_step.device, dtype=handoff_step.dtype)
            calibration_bias_norm = float(correction_norm.detach().cpu().item())
    decode_metrics = _decode_handoff(
        agent_b=agent_b,
        tokenizer_b=tokenizer_b,
        prompt=None if sequence_prefix else prompt,
        cfg=variant_cfg,
        handoff_step=handoff_step,
        kv_cache_a=None if sequence_prefix else sender_state["kv_cache_a"],
        max_new_tokens=int(cfg.max_new_tokens),
        target_answer_text=target_answer_text,
        append_answer_suffix_without_context=sequence_prefix,
    )
    return {
        **decode_metrics,
        **_alignment_distances(
            prompt=prompt,
            state=variant_state,
            current_latent_step=handoff_source,
            alignment_state=alignment_state,
            cfg=variant_cfg,
            adapter_state=variant_state.get("handoff_adapter_state")
            if bool(variant_state.get("handoff_adapter_enabled", False))
            else None,
            calibration_strength=calibration_strength,
            calibration_max_norm_ratio=calibration_max_norm_ratio,
        ),
        "alignment_mode": method_alignment_mode,
        "alignment_strategy": str(strategy),
        "handoff_status": "ok",
        "handoff_surface": "input_embedding_sequence"
        if sequence_prefix
        else variant_state.get("handoff_surface", "input_embedding"),
        "alignment_residual_norm_ratio": alignment_state.get("residual_norm_ratio"),
        "alignment_bias_norm": alignment_state.get("bias_norm"),
        "anchor_reconstruction_mse": variant_state.get(
            "handoff_anchor_reconstruction_mse",
            variant_state.get("anchor_reconstruction_mse"),
        ),
        "anchor_pairwise_distance_distortion": variant_state.get(
            "handoff_anchor_pairwise_distance_distortion",
            variant_state.get("anchor_pairwise_distance_distortion"),
        ),
        "anchor_cosine_structure_error": variant_state.get(
            "handoff_anchor_cosine_structure_error",
            variant_state.get("anchor_cosine_structure_error"),
        ),
        "prompt_calibration_enabled": bool(prompt_calibration_enabled),
        "prompt_calibration_bias_norm": calibration_bias_norm,
        "handoff_adapter_enabled": bool(variant_state.get("handoff_adapter_enabled", False)),
        "handoff_adapter_status": variant_state.get("handoff_adapter_status"),
        "handoff_adapter_applied": adapter_metrics["handoff_adapter_applied"],
        "handoff_adapter_delta_norm": adapter_metrics["handoff_adapter_delta_norm"],
        "handoff_adapter_cache_hit": variant_state.get("handoff_adapter_cache_hit"),
        "handoff_adapter_cache_path": variant_state.get("handoff_adapter_cache_path"),
        "handoff_adapter_training_prompt_count": variant_state.get(
            "handoff_adapter_training_prompt_count"
        ),
        "handoff_adapter_training_token_count": variant_state.get(
            "handoff_adapter_training_token_count"
        ),
        "handoff_adapter_training_reconstruction_mse": variant_state.get(
            "handoff_adapter_training_reconstruction_mse"
        ),
        "handoff_adapter_training_mean_cosine_similarity": variant_state.get(
            "handoff_adapter_training_mean_cosine_similarity"
        ),
        "embedding_manifold_enabled": bool(
            getattr(getattr(variant_cfg.handoff, "embedding_manifold", None), "enabled", False)
        ),
        "embedding_manifold_applied": manifold_metrics["embedding_manifold_applied"],
        "embedding_manifold_delta_norm": manifold_metrics["embedding_manifold_delta_norm"],
        "embedding_manifold_mean_top_similarity": manifold_metrics[
            "embedding_manifold_mean_top_similarity"
        ],
        "embedding_manifold_unique_token_count": manifold_metrics[
            "embedding_manifold_unique_token_count"
        ],
        "handoff_uncertainty": None,
        "confidence_gate_triggered": False,
        "fallback_discrete_reasoning_steps": 0,
        "latent_trajectory_steps": 1,
        "total_reasoning_steps": 1,
        "continuous_integration_seconds": 0.0,
        "global_alignment_cache_hit": variant_state["global_alignment_cache_hit"],
        "_row_cfg": variant_cfg,
        "_row_state": variant_state,
    }


def run_global_anchor_orthogonal(
    prompt: str,
    target_answer_text: Optional[str],
    cfg: Any,
    state: dict[str, Any],
) -> dict[str, Any]:
    return _run_global_anchor_variant(
        prompt,
        target_answer_text,
        cfg,
        state,
        strategy="orthogonal",
        prompt_calibration_enabled=False,
        method_alignment_mode="global_anchor_orthogonal",
    )


def run_global_anchor_ridge(
    prompt: str,
    target_answer_text: Optional[str],
    cfg: Any,
    state: dict[str, Any],
) -> dict[str, Any]:
    return _run_global_anchor_variant(
        prompt,
        target_answer_text,
        cfg,
        state,
        strategy="ridge",
        prompt_calibration_enabled=False,
        method_alignment_mode="global_anchor_ridge",
    )


def run_global_anchor_hybrid_affine(
    prompt: str,
    target_answer_text: Optional[str],
    cfg: Any,
    state: dict[str, Any],
) -> dict[str, Any]:
    return _run_global_anchor_variant(
        prompt,
        target_answer_text,
        cfg,
        state,
        strategy="hybrid_affine",
        prompt_calibration_enabled=False,
        method_alignment_mode="global_anchor_hybrid_affine",
    )


def run_global_anchor_hybrid_affine_plus_calibration(
    prompt: str,
    target_answer_text: Optional[str],
    cfg: Any,
    state: dict[str, Any],
) -> dict[str, Any]:
    return _run_global_anchor_variant(
        prompt,
        target_answer_text,
        cfg,
        state,
        strategy="hybrid_affine",
        prompt_calibration_enabled=True,
        method_alignment_mode="global_anchor_hybrid_affine_plus_calibration",
    )


def run_hybrid(
    prompt: str,
    target_answer_text: Optional[str],
    cfg: Any,
    state: dict[str, Any],
) -> dict[str, Any]:
    del state
    output = run_hybrid_pipeline(
        cfg,
        prompt=prompt,
        collect_alignment_metrics=True,
        target_answer_text=target_answer_text,
    )
    return {
        "decoded_text": str(output["decoded_text"]),
        "generated_tokens": int(output["generated_tokens"]),
        "decode_status": output["decode_status"],
        "handoff_status": output["handoff_status"],
        "handoff_surface": output["handoff_surface"],
        "answer_token_count": output["answer_token_count"],
        "answer_nll": output["answer_nll"],
        "answer_perplexity": output["answer_perplexity"],
        "alignment_mode": str(output["alignment_mode"]),
        "alignment_strategy": output.get("alignment_strategy"),
        "kv_cache_transferred": output["kv_cache_transferred"],
        "kv_cache_status": output["kv_cache_status"],
        "kv_cache_reason": output["kv_cache_reason"],
        "active_kv_cache_transferred": output.get("active_kv_cache_transferred"),
        "active_kv_cache_status": output.get("active_kv_cache_status"),
        "active_kv_cache_reason": output.get("active_kv_cache_reason"),
        "active_kv_cache_source": output.get("active_kv_cache_source"),
        "receiver_context_status": output.get("receiver_context_status"),
        "receiver_context_reason": output.get("receiver_context_reason"),
        "receiver_context_token_count": output.get("receiver_context_token_count"),
        "receiver_context_latent_position": output.get("receiver_context_latent_position"),
        "pre_alignment_l2_distance": output["pre_alignment_l2_distance"],
        "pre_alignment_cosine_distance": output["pre_alignment_cosine_distance"],
        "post_alignment_l2_distance": output["post_alignment_l2_distance"],
        "post_alignment_cosine_distance": output["post_alignment_cosine_distance"],
        "raw_handoff_entropy": output["raw_handoff_entropy"],
        "handoff_uncertainty": output["handoff_uncertainty"],
        "confidence_gate_triggered": output["confidence_gate_triggered"],
        "fallback_discrete_reasoning_steps": output["fallback_discrete_reasoning_steps"],
        "latent_trajectory_steps": output["latent_trajectory_steps"],
        "total_reasoning_steps": output["total_reasoning_steps"],
        "continuous_integration_seconds": output["continuous_integration_seconds"],
        "global_alignment_cache_hit": output["global_alignment_cache_hit"],
        "alignment_residual_norm_ratio": output.get("alignment_residual_norm_ratio"),
        "alignment_bias_norm": output.get("alignment_bias_norm"),
        "prompt_calibration_enabled": output.get("prompt_calibration_enabled"),
        "prompt_calibration_bias_norm": output.get("prompt_calibration_bias_norm"),
        "handoff_adapter_enabled": output.get("handoff_adapter_enabled"),
        "handoff_adapter_status": output.get("handoff_adapter_status"),
        "handoff_adapter_applied": output.get("handoff_adapter_applied"),
        "handoff_adapter_delta_norm": output.get("handoff_adapter_delta_norm"),
        "handoff_adapter_cache_hit": output.get("handoff_adapter_cache_hit"),
        "handoff_adapter_cache_path": output.get("handoff_adapter_cache_path"),
        "handoff_adapter_training_prompt_count": output.get(
            "handoff_adapter_training_prompt_count"
        ),
        "handoff_adapter_training_token_count": output.get(
            "handoff_adapter_training_token_count"
        ),
        "handoff_adapter_training_reconstruction_mse": output.get(
            "handoff_adapter_training_reconstruction_mse"
        ),
        "handoff_adapter_training_mean_cosine_similarity": output.get(
            "handoff_adapter_training_mean_cosine_similarity"
        ),
        "embedding_manifold_enabled": output.get("embedding_manifold_enabled"),
        "embedding_manifold_applied": output.get("embedding_manifold_applied"),
        "embedding_manifold_delta_norm": output.get("embedding_manifold_delta_norm"),
        "embedding_manifold_mean_top_similarity": output.get(
            "embedding_manifold_mean_top_similarity"
        ),
        "embedding_manifold_unique_token_count": output.get(
            "embedding_manifold_unique_token_count"
        ),
    }


def run_homogeneous_ridge_latent(
    prompt: str,
    target_answer_text: Optional[str],
    cfg: Any,
    state: dict[str, Any],
) -> dict[str, Any]:
    sender_state = _collect_sender_last_hidden_state(prompt, state, cfg)
    agent_a = state["agent_a"]
    agent_b = state["agent_b"]
    tokenizer_b = state["tokenizer_b"]
    agent_b_device = next(agent_b.parameters()).device

    ridge_mapping = state.get("_homogeneous_ridge_mapping")
    if ridge_mapping is None:
        output_embeddings = agent_a.get_output_embeddings()
        if output_embeddings is None:
            raise ValueError("Homogeneous ridge baseline requires output embeddings from Agent A")
        ridge_mapping = compute_ridge_mapping(
            output_embeddings.weight,
            agent_b.get_input_embeddings().weight,
            regularization=1e-4,
        )
        state["_homogeneous_ridge_mapping"] = ridge_mapping.cpu()
    ridge_mapping = state["_homogeneous_ridge_mapping"].to(
        device=sender_state["current_latent_step"].device,
        dtype=sender_state["current_latent_step"].dtype,
    )
    handoff_step = apply_linear_mapping(sender_state["current_latent_step"], ridge_mapping).to(
        device=agent_b_device,
        dtype=agent_b.get_input_embeddings().weight.dtype,
    )
    decode_metrics = _decode_handoff(
        agent_b=agent_b,
        tokenizer_b=tokenizer_b,
        prompt=prompt,
        cfg=cfg,
        handoff_step=handoff_step,
        kv_cache_a=sender_state["kv_cache_a"],
        max_new_tokens=int(cfg.max_new_tokens),
        target_answer_text=target_answer_text,
    )
    return {
        **decode_metrics,
        "alignment_mode": "homogeneous_ridge",
        "handoff_status": "ok",
        "handoff_surface": "input_embedding",
        "pre_alignment_l2_distance": None,
        "pre_alignment_cosine_distance": None,
        "post_alignment_l2_distance": None,
        "post_alignment_cosine_distance": None,
        "handoff_uncertainty": None,
        "confidence_gate_triggered": False,
        "fallback_discrete_reasoning_steps": 0,
        "latent_trajectory_steps": 1,
        "total_reasoning_steps": 1,
        "continuous_integration_seconds": 0.0,
    }


def run_homogeneous_orthogonal_latent(
    prompt: str,
    target_answer_text: Optional[str],
    cfg: Any,
    state: dict[str, Any],
) -> dict[str, Any]:
    sender_state = _collect_sender_last_hidden_state(prompt, state, cfg)
    tokenizer_b = state["tokenizer_b"]
    agent_b = state["agent_b"]
    agent_b_device = next(agent_b.parameters()).device
    encoded_b = tokenizer_b(prompt, return_tensors="pt")
    input_ids_b = encoded_b["input_ids"].to(agent_b_device)
    attention_mask_b = encoded_b["attention_mask"].to(agent_b_device)
    position_ids_b = _build_position_ids(attention_mask_b)

    with torch.no_grad():
        receiver_hidden_states = agent_b.model(
            input_ids=input_ids_b,
            attention_mask=attention_mask_b,
            position_ids=position_ids_b,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state

    orthogonal_q = compute_orthogonal_mapping(
        sender_state["full_hidden_states"],
        receiver_hidden_states,
    ).to(
        device=sender_state["current_latent_step"].device,
        dtype=sender_state["current_latent_step"].dtype,
    )
    handoff_step = apply_linear_mapping(sender_state["current_latent_step"], orthogonal_q).to(
        device=agent_b_device,
        dtype=agent_b.get_input_embeddings().weight.dtype,
    )
    decode_metrics = _decode_handoff(
        agent_b=agent_b,
        tokenizer_b=tokenizer_b,
        prompt=prompt,
        cfg=cfg,
        handoff_step=handoff_step,
        kv_cache_a=sender_state["kv_cache_a"],
        max_new_tokens=int(cfg.max_new_tokens),
        target_answer_text=target_answer_text,
    )
    return {
        **decode_metrics,
        "alignment_mode": "homogeneous_prompt_local_orthogonal",
        "handoff_status": "ok",
        "handoff_surface": "input_embedding",
        "pre_alignment_l2_distance": None,
        "pre_alignment_cosine_distance": None,
        "post_alignment_l2_distance": None,
        "post_alignment_cosine_distance": None,
        "handoff_uncertainty": None,
        "confidence_gate_triggered": False,
        "fallback_discrete_reasoning_steps": 0,
        "latent_trajectory_steps": 1,
        "total_reasoning_steps": 1,
        "continuous_integration_seconds": 0.0,
    }


def run_homogeneous_orthogonal_control(
    prompt: str,
    target_answer_text: Optional[str],
    cfg: Any,
    state: dict[str, Any],
) -> dict[str, Any]:
    del state
    homogeneous_cfg = _clone_cfg(cfg)
    homogeneous_cfg.agent_b_model = homogeneous_cfg.agent_a_model
    homogeneous_state = _get_pipeline_state(homogeneous_cfg)
    result = run_homogeneous_orthogonal_latent(
        prompt,
        target_answer_text,
        homogeneous_cfg,
        homogeneous_state,
    )
    result["_row_cfg"] = homogeneous_cfg
    result["_row_state"] = homogeneous_state
    return result


def _methods_for_suite(
    suite_name: str,
    selected_methods: Optional[Sequence[str]] = None,
) -> list[tuple[str, RunnerFn]]:
    if suite_name == "phase1_homogeneous":
        methods = [
            ("pure_text_cot", run_pure_text_cot),
            ("text_text_hybrid", run_text_text_hybrid),
            ("homogeneous_ridge_latent", run_homogeneous_ridge_latent),
            ("homogeneous_orthogonal_latent", run_homogeneous_orthogonal_latent),
        ]
    else:
        methods = [
            ("pure_text_cot", run_pure_text_cot),
            ("text_text_hybrid", run_text_text_hybrid),
            ("prompt_local_latent", run_prompt_local_latent),
            ("global_anchor_orthogonal", run_global_anchor_orthogonal),
            ("global_anchor_ridge", run_global_anchor_ridge),
            ("global_anchor_hybrid_affine", run_global_anchor_hybrid_affine),
            (
                "global_anchor_hybrid_affine_plus_calibration",
                run_global_anchor_hybrid_affine_plus_calibration,
            ),
            ("hybrid_hl_mas", run_hybrid),
            ("homogeneous_orthogonal_latent", run_homogeneous_orthogonal_control),
        ]

    if selected_methods is None:
        return methods

    method_map = dict(methods)
    unknown_methods = [name for name in selected_methods if name not in method_map]
    if unknown_methods:
        available = ", ".join(name for name, _ in methods)
        raise ValueError(
            f"Unknown methods for suite {suite_name!r}: {', '.join(unknown_methods)}. "
            f"Available methods: {available}"
        )
    return [(name, method_map[name]) for name in selected_methods]


def _suite_cfg(base_cfg: Any, suite_name: str) -> Any:
    suite_cfg = _clone_cfg(base_cfg)
    if suite_name == "phase1_homogeneous":
        suite_cfg.agent_b_model = suite_cfg.agent_a_model
    return suite_cfg


def _load_model_pair_compatibility_dict(agent_a_model: str, agent_b_model: str) -> dict[str, Any]:
    try:
        return load_model_pair_compatibility(agent_a_model, agent_b_model).to_dict()
    except Exception as exc:  # noqa: BLE001
        return {
            "agent_a": {"model_id": str(agent_a_model)},
            "agent_b": {"model_id": str(agent_b_model)},
            "kv_cache_compatible": False,
            "status": "config_load_error",
            "reason": str(exc),
            "mismatches": [],
            "warnings": [],
        }


def _coerce_sample_indices(values: Optional[Sequence[Any]]) -> Optional[list[int]]:
    if values is None:
        return None
    return [int(value) for value in values]


def _apply_model_profile_defaults(
    cfg: Any,
    *,
    agent_a_model: Optional[str],
    agent_b_model: Optional[str],
    hetero_smoke: bool,
) -> None:
    if hetero_smoke:
        if agent_a_model is None:
            cfg.agent_a_model = DEFAULT_HETERO_SMOKE_AGENT_A_MODEL
        if agent_b_model is None:
            cfg.agent_b_model = DEFAULT_HETERO_SMOKE_AGENT_B_MODEL
    if agent_a_model is not None:
        cfg.agent_a_model = str(agent_a_model)
    if agent_b_model is not None:
        cfg.agent_b_model = str(agent_b_model)


def run_benchmark(
    *,
    suite_name: str,
    dataset_name: str,
    dataset_split: Optional[str],
    limit: int,
    repetitions: int,
    samples_output_path: Path,
    summary_output_path: Path,
    report_output_path: Path,
    max_new_tokens: Optional[int] = None,
    latent_steps_values: Optional[list[int]] = None,
    agent_a_model: Optional[str] = None,
    agent_b_model: Optional[str] = None,
    latent_pooling: Optional[str] = None,
    receiver_context_mode: Optional[str] = None,
    receiver_context_latent_position: Optional[str] = None,
    prompt_calibration_enabled: Optional[bool] = None,
    prompt_calibration_strength: Optional[float] = None,
    prompt_calibration_max_norm_ratio: Optional[float] = None,
    handoff_adapter_enabled: Optional[bool] = None,
    handoff_adapter_train_on_missing: Optional[bool] = None,
    handoff_adapter_train_limit: Optional[int] = None,
    embedding_manifold_enabled: Optional[bool] = None,
    embedding_manifold_top_k: Optional[int] = None,
    embedding_manifold_blend: Optional[float] = None,
    seed: Optional[int] = None,
    method_names: Optional[list[str]] = None,
    sample_indices: Optional[list[int]] = None,
    semantic_smoke: bool = False,
    mvp_smoke: bool = False,
    hetero_smoke: bool = False,
    answer_only_final: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    base_cfg = _load_cfg()
    _apply_model_profile_defaults(
        base_cfg,
        agent_a_model=agent_a_model,
        agent_b_model=agent_b_model,
        hetero_smoke=hetero_smoke,
    )
    if latent_pooling is not None:
        base_cfg.handoff.latent_pooling = str(latent_pooling)
    if receiver_context_mode is not None:
        base_cfg.handoff.receiver_context.mode = str(receiver_context_mode)
    if receiver_context_latent_position is not None:
        base_cfg.handoff.receiver_context.latent_position = str(receiver_context_latent_position)
    if prompt_calibration_enabled is not None:
        base_cfg.alignment.prompt_calibration.enabled = bool(prompt_calibration_enabled)
    if prompt_calibration_strength is not None:
        base_cfg.alignment.prompt_calibration.strength = float(prompt_calibration_strength)
    if prompt_calibration_max_norm_ratio is not None:
        base_cfg.alignment.prompt_calibration.max_norm_ratio = float(prompt_calibration_max_norm_ratio)
    if handoff_adapter_enabled is not None:
        base_cfg.handoff.adapter.enabled = bool(handoff_adapter_enabled)
    if handoff_adapter_train_on_missing is not None:
        base_cfg.handoff.adapter.train_on_missing = bool(handoff_adapter_train_on_missing)
    if handoff_adapter_train_limit is not None:
        base_cfg.handoff.adapter.train_limit = int(handoff_adapter_train_limit)
    if embedding_manifold_enabled is not None:
        base_cfg.handoff.embedding_manifold.enabled = bool(embedding_manifold_enabled)
    if embedding_manifold_top_k is not None:
        base_cfg.handoff.embedding_manifold.top_k = int(embedding_manifold_top_k)
    if embedding_manifold_blend is not None:
        base_cfg.handoff.embedding_manifold.blend = float(embedding_manifold_blend)
    if seed is not None:
        base_cfg.seed = int(seed)
    if max_new_tokens is not None:
        base_cfg.max_new_tokens = int(max_new_tokens)
    if semantic_smoke or mvp_smoke or hetero_smoke or answer_only_final:
        base_cfg.benchmark.answer_only_final = True
    if mvp_smoke:
        base_cfg.reporting.semantic_smoke.baseline_methods = list(DEFAULT_MVP_SMOKE_BASELINE_METHODS)
        base_cfg.reporting.semantic_smoke.latent_methods = list(DEFAULT_MVP_SMOKE_LATENT_METHODS)
        base_cfg.reporting.semantic_smoke.require_final_answer_marker_methods = list(DEFAULT_MVP_SMOKE_METHODS)
    if hetero_smoke:
        base_cfg.reporting.semantic_smoke.baseline_methods = list(DEFAULT_HETERO_SMOKE_BASELINE_METHODS)
        base_cfg.reporting.semantic_smoke.latent_methods = list(DEFAULT_HETERO_SMOKE_LATENT_METHODS)
        base_cfg.reporting.semantic_smoke.require_final_answer_marker_methods = list(DEFAULT_HETERO_SMOKE_METHODS)
    semantic_smoke_cfg = getattr(getattr(base_cfg, "reporting", None), "semantic_smoke", None)
    if sample_indices is None and semantic_smoke_cfg is not None:
        sample_indices = _coerce_sample_indices(getattr(semantic_smoke_cfg, "sample_indices", None))
    effective_split = dataset_split or _default_split_for_dataset(dataset_name)
    validation_size = _validation_size(base_cfg, dataset_name)
    suite_cfg = _suite_cfg(base_cfg, suite_name)
    latent_step_candidates = latent_steps_values or [int(getattr(suite_cfg, "latent_steps", 0))]
    methods = _methods_for_suite(suite_name, method_names)
    compatibility_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def compatibility_for_cfg(cfg: Any) -> dict[str, Any]:
        key = (str(cfg.agent_a_model), str(cfg.agent_b_model))
        if key not in compatibility_cache:
            compatibility_cache[key] = _load_model_pair_compatibility_dict(*key)
        return compatibility_cache[key]

    suite_model_pair_compatibility = compatibility_for_cfg(suite_cfg)
    samples = get_dataloader(
        dataset_name,
        limit=limit,
        split=effective_split,
        validation_size=validation_size,
        sample_indices=sample_indices,
    )
    effective_sample_indices = list(sample_indices) if sample_indices is not None else None
    if effective_sample_indices is not None:
        effective_sample_indices = effective_sample_indices[: min(limit, len(effective_sample_indices))]

    sample_rows: list[dict[str, Any]] = []
    for latent_steps in latent_step_candidates:
        step_cfg = _clone_cfg(suite_cfg)
        step_cfg.latent_steps = int(latent_steps)
        state = _get_pipeline_state(step_cfg)
        tokenizer_b = state["tokenizer_b"]

        for repetition in range(repetitions):
            for method_name, runner in methods:
                print(
                    f"Running {suite_name}:{method_name} on {dataset_name}/{effective_split} "
                    f"with latent_steps={latent_steps} "
                    f"(repetition {repetition + 1}/{repetitions})..."
                )
                for index, row in enumerate(samples):
                    sample_index = (
                        effective_sample_indices[index]
                        if effective_sample_indices is not None and index < len(effective_sample_indices)
                        else index
                    )
                    prompt = pick_field(row, ("question", "problem"))
                    target_answer = _target_answer(dataset_name, row)
                    predicted_answer: Optional[str] = None
                    decoded_text = ""
                    result: dict[str, Any] = {}
                    error: Optional[str] = None

                    start = time.perf_counter()
                    try:
                        result = runner(prompt, target_answer, step_cfg, state)
                        decoded_text = str(result["decoded_text"])
                        predicted_answer = _predicted_answer(dataset_name, decoded_text)
                    except Exception as exc:  # noqa: BLE001
                        error = str(exc)
                    latency = time.perf_counter() - start

                    row_cfg = result.get("_row_cfg", step_cfg)
                    row_state = result.get("_row_state", state)
                    row_result = {
                        key: value
                        for key, value in result.items()
                        if not key.startswith("_")
                    }
                    reasoning_layer_weights = tuple(row_state.get("global_reasoning_layer_weights", ()))
                    model_pair_compatibility = compatibility_for_cfg(row_cfg)
                    row_base = build_standard_row_base(
                        row_cfg,
                        evaluation_surface="benchmark_all",
                        suite=suite_name,
                        method=method_name,
                        dataset=dataset_name,
                        dataset_split=effective_split,
                        repetition=repetition,
                        seed=int(getattr(row_cfg, "seed", 0)),
                        compression_steps=int(getattr(row_cfg, "latent_steps", 0)),
                        alignment_mode=row_result.get("alignment_mode", ""),
                        alignment_strategy=row_result.get("alignment_strategy", ""),
                        semantic_anchor_count=int(row_state.get("semantic_anchor_count", 0)),
                        reasoning_layer_weights=reasoning_layer_weights,
                        model_pair_kv_cache_compatible=model_pair_compatibility.get("kv_cache_compatible"),
                        model_pair_compatibility_status=str(model_pair_compatibility.get("status", "")),
                        model_pair_compatibility_reason=str(model_pair_compatibility.get("reason", "")),
                    )
                    decode_status = row_result.get(
                        "decode_status",
                        "decoded" if decoded_text.strip() else "empty_decode",
                    )
                    handoff_status = row_result.get(
                        "handoff_status",
                        "not_applicable" if method_name in {"pure_text_cot", "text_text_hybrid"} else "",
                    )
                    handoff_surface = row_result.get(
                        "handoff_surface",
                        "text" if method_name in {"pure_text_cot", "text_text_hybrid"} else "input_embedding",
                    )
                    kv_cache_transferred = row_result.get("kv_cache_transferred")
                    kv_cache_status = row_result.get(
                        "kv_cache_status",
                        "not_applicable" if kv_cache_transferred is None else (
                            "transferred" if kv_cache_transferred else "unsupported"
                        ),
                    )
                    kv_cache_reason = row_result.get(
                        "kv_cache_reason",
                        "not_applicable" if kv_cache_transferred is None else "",
                    )
                    active_kv_cache_transferred = row_result.get("active_kv_cache_transferred")
                    active_kv_cache_status = row_result.get(
                        "active_kv_cache_status",
                        "not_applicable" if active_kv_cache_transferred is None else (
                            "transferred" if active_kv_cache_transferred else "unsupported"
                        ),
                    )
                    active_kv_cache_reason = row_result.get(
                        "active_kv_cache_reason",
                        "not_applicable" if active_kv_cache_transferred is None else "",
                    )
                    active_kv_cache_source = row_result.get(
                        "active_kv_cache_source",
                        "text_baseline"
                        if method_name in {"pure_text_cot", "text_text_hybrid"}
                        else "unknown",
                    )
                    receiver_context_status = row_result.get(
                        "receiver_context_status",
                        "not_applicable" if method_name in {"pure_text_cot", "text_text_hybrid"} else "not_used",
                    )
                    receiver_context_reason = row_result.get(
                        "receiver_context_reason",
                        "text_baseline" if method_name in {"pure_text_cot", "text_text_hybrid"} else "latent_only",
                    )
                    receiver_context_token_count = int(row_result.get("receiver_context_token_count", 0) or 0)
                    receiver_context_latent_position = row_result.get(
                        "receiver_context_latent_position",
                        "not_applicable"
                        if method_name in {"pure_text_cot", "text_text_hybrid"}
                        else _receiver_context_latent_position(row_cfg),
                    )
                    sample_rows.append(
                        {
                            **row_base,
                            "sample_index": sample_index,
                            "handoff_status": handoff_status,
                            "handoff_surface": handoff_surface,
                            "kv_cache_transferred": kv_cache_transferred,
                            "kv_cache_status": kv_cache_status,
                            "kv_cache_reason": kv_cache_reason,
                            "active_kv_cache_transferred": active_kv_cache_transferred,
                            "active_kv_cache_status": active_kv_cache_status,
                            "active_kv_cache_reason": active_kv_cache_reason,
                            "active_kv_cache_source": active_kv_cache_source,
                            "receiver_context_status": receiver_context_status,
                            "receiver_context_reason": receiver_context_reason,
                            "receiver_context_token_count": receiver_context_token_count,
                            "receiver_context_latent_position": receiver_context_latent_position,
                            "decode_status": decode_status,
                            "prompt": prompt,
                            "target_answer": target_answer,
                            "predicted_answer": predicted_answer,
                            "decoded_text": decoded_text,
                            "generated_tokens": int(
                                row_result.get(
                                    "generated_tokens",
                                    len(tokenizer_b.encode(decoded_text, add_special_tokens=False)) if decoded_text else 0,
                                )
                            ),
                            "answer_token_count": int(row_result.get("answer_token_count", 0) or 0),
                            "answer_nll": row_result.get("answer_nll"),
                            "answer_perplexity": row_result.get("answer_perplexity"),
                            "correct": _answers_match(dataset_name, predicted_answer, target_answer),
                            "latency_seconds": latency,
                            "pre_alignment_l2_distance": row_result.get("pre_alignment_l2_distance"),
                            "pre_alignment_cosine_distance": row_result.get("pre_alignment_cosine_distance"),
                            "post_alignment_l2_distance": row_result.get("post_alignment_l2_distance"),
                            "post_alignment_cosine_distance": row_result.get("post_alignment_cosine_distance"),
                            "alignment_residual_norm_ratio": row_result.get("alignment_residual_norm_ratio"),
                            "alignment_bias_norm": row_result.get("alignment_bias_norm"),
                            "prompt_calibration_enabled": row_result.get("prompt_calibration_enabled"),
                            "prompt_calibration_bias_norm": row_result.get("prompt_calibration_bias_norm"),
                            "handoff_adapter_enabled": row_result.get("handoff_adapter_enabled"),
                            "handoff_adapter_status": row_result.get("handoff_adapter_status"),
                            "handoff_adapter_applied": row_result.get("handoff_adapter_applied"),
                            "handoff_adapter_delta_norm": row_result.get("handoff_adapter_delta_norm"),
                            "handoff_adapter_cache_hit": row_result.get("handoff_adapter_cache_hit"),
                            "handoff_adapter_cache_path": row_result.get("handoff_adapter_cache_path"),
                            "handoff_adapter_training_prompt_count": row_result.get(
                                "handoff_adapter_training_prompt_count"
                            ),
                            "handoff_adapter_training_token_count": row_result.get(
                                "handoff_adapter_training_token_count"
                            ),
                            "handoff_adapter_training_reconstruction_mse": row_result.get(
                                "handoff_adapter_training_reconstruction_mse"
                            ),
                            "handoff_adapter_training_mean_cosine_similarity": row_result.get(
                                "handoff_adapter_training_mean_cosine_similarity"
                            ),
                            "embedding_manifold_enabled": row_result.get("embedding_manifold_enabled"),
                            "embedding_manifold_applied": row_result.get("embedding_manifold_applied"),
                            "embedding_manifold_delta_norm": row_result.get(
                                "embedding_manifold_delta_norm"
                            ),
                            "embedding_manifold_mean_top_similarity": row_result.get(
                                "embedding_manifold_mean_top_similarity"
                            ),
                            "embedding_manifold_unique_token_count": row_result.get(
                                "embedding_manifold_unique_token_count"
                            ),
                            "raw_handoff_entropy": row_result.get("raw_handoff_entropy"),
                            "handoff_uncertainty": row_result.get("handoff_uncertainty"),
                            "confidence_gate_triggered": row_result.get("confidence_gate_triggered"),
                            "fallback_discrete_reasoning_steps": row_result.get("fallback_discrete_reasoning_steps"),
                            "latent_trajectory_steps": row_result.get("latent_trajectory_steps"),
                            "total_reasoning_steps": row_result.get("total_reasoning_steps"),
                            "continuous_integration_seconds": row_result.get("continuous_integration_seconds"),
                            "global_alignment_cache_hit": row_result.get(
                                "global_alignment_cache_hit",
                                row_state.get("global_alignment_cache_hit"),
                            ),
                            "error": error,
                        }
                    )

    summary_rows = aggregate_standard_rows(sample_rows)
    gating_cfg = _gating_cfg(base_cfg, "phase1" if suite_name == "phase1_homogeneous" else "phase3")
    runtime_smoke_cfg = _runtime_smoke_cfg(base_cfg)
    runtime_smoke_report = build_runtime_smoke_report(
        sample_rows,
        max_error_count=int(getattr(runtime_smoke_cfg, "max_error_count", 0)),
        require_explicit_statuses=bool(
            getattr(runtime_smoke_cfg, "require_explicit_statuses", True)
        ),
    )
    if semantic_smoke or mvp_smoke or hetero_smoke:
        phase_gate_report = {
            "phase": "phase_1" if suite_name == "phase1_homogeneous" else "phase_3",
            "passed": None,
            "skipped": True,
            "reason": "smoke mode uses semantic_smoke_report instead of phase gates",
        }
    elif _is_runtime_smoke_run(base_cfg, limit=limit, repetitions=repetitions):
        phase_gate_report = {
            "phase": "phase_1" if suite_name == "phase1_homogeneous" else "phase_3",
            "passed": None,
            "skipped": True,
            "reason": "runtime_smoke mode uses runtime_smoke_report instead of phase gates",
        }
    elif suite_name == "phase1_homogeneous":
        phase_gate_report = build_phase1_gate_report(
            summary_rows,
            required_repetitions=int(getattr(gating_cfg, "required_repetitions", 3)),
            max_error_rate_percentage=float(getattr(gating_cfg, "max_error_rate_percentage", 0.0)),
            min_cache_transfer_rate_percentage=float(getattr(gating_cfg, "min_cache_transfer_rate_percentage", 100.0)),
            min_non_empty_decoded_rate_percentage=float(
                getattr(gating_cfg, "min_non_empty_decoded_rate_percentage", 100.0)
            ),
        )
    else:
        phase_gate_report = build_phase3_gate_report(
            summary_rows,
            require_q_global_beats_prompt_local=bool(
                getattr(gating_cfg, "require_q_global_beats_prompt_local", True)
            ),
            min_hybrid_accuracy_gain_over_text_text=float(
                getattr(gating_cfg, "min_hybrid_accuracy_gain_over_text_text", 3.0)
            ),
            max_accuracy_gap_for_perplexity_tie=float(
                getattr(gating_cfg, "max_accuracy_gap_for_perplexity_tie", 1.0)
            ),
            min_perplexity_improvement_ratio=float(
                getattr(gating_cfg, "min_perplexity_improvement_ratio", 0.10)
            ),
        )

    semantic_smoke_report = None
    if semantic_smoke or mvp_smoke or hetero_smoke or bool(getattr(semantic_smoke_cfg, "enabled", False)):
        baseline_methods = tuple(
            getattr(semantic_smoke_cfg, "baseline_methods", ("pure_text_cot", "text_text_hybrid"))
        )
        latent_methods = tuple(
            getattr(semantic_smoke_cfg, "latent_methods", DEFAULT_SEMANTIC_SMOKE_LATENT_METHODS)
        )
        required_marker_methods = getattr(
            semantic_smoke_cfg,
            "require_final_answer_marker_methods",
            None,
        )
        if required_marker_methods is not None:
            required_marker_methods = tuple(str(method) for method in required_marker_methods)
        max_answer_perplexity = getattr(semantic_smoke_cfg, "max_answer_perplexity", None)
        semantic_smoke_report = build_semantic_smoke_report(
            sample_rows,
            baseline_methods=baseline_methods,
            latent_methods=latent_methods,
            model_pair_compatibility=suite_model_pair_compatibility,
            min_baseline_accuracy_percentage=getattr(
                semantic_smoke_cfg,
                "min_baseline_accuracy_percentage",
                None,
            ),
            min_latent_accuracy_percentage=getattr(
                semantic_smoke_cfg,
                "min_latent_accuracy_percentage",
                None,
            ),
            min_method_accuracy_percentage=getattr(
                semantic_smoke_cfg,
                "min_method_accuracy_percentage",
                None,
            ),
            min_latent_non_empty_decoded_rate_percentage=float(
                getattr(semantic_smoke_cfg, "min_latent_non_empty_decoded_rate_percentage", 100.0)
            ),
            min_compatible_cache_transfer_rate_percentage=float(
                getattr(semantic_smoke_cfg, "min_compatible_cache_transfer_rate_percentage", 100.0)
            ),
            max_answer_perplexity=(
                None if max_answer_perplexity is None else float(max_answer_perplexity)
            ),
            max_degenerate_decode_count=int(
                getattr(semantic_smoke_cfg, "max_degenerate_decode_count", 0)
            ),
            require_baseline_final_answer_marker=bool(
                getattr(semantic_smoke_cfg, "require_baseline_final_answer_marker", False)
            ),
            require_final_answer_marker_methods=required_marker_methods,
            max_diagnostic_rows=int(getattr(semantic_smoke_cfg, "max_diagnostic_rows", 5)),
        )

    write_csv(samples_output_path, sample_rows, STANDARD_SAMPLE_FIELDS)
    write_csv(summary_output_path, summary_rows, STANDARD_SUMMARY_FIELDS)
    report_payload = {
        "suite": suite_name,
        "dataset": dataset_name,
        "dataset_split": effective_split,
        "limit": limit,
        "sample_indices": effective_sample_indices,
        "repetitions": repetitions,
        "latent_steps_values": latent_step_candidates,
        "methods": [name for name, _ in methods],
        "latent_pooling": _latent_pooling_mode(base_cfg),
        "receiver_context_mode": _receiver_context_mode(base_cfg),
        "answer_only_final": _answer_only_final_enabled(base_cfg),
        "handoff_adapter": {
            "enabled": bool(getattr(getattr(base_cfg.handoff, "adapter", None), "enabled", False)),
            "train_on_missing": bool(
                getattr(getattr(base_cfg.handoff, "adapter", None), "train_on_missing", False)
            ),
            "train_limit": int(getattr(getattr(base_cfg.handoff, "adapter", None), "train_limit", 0)),
        },
        "embedding_manifold": {
            "enabled": bool(
                getattr(getattr(base_cfg.handoff, "embedding_manifold", None), "enabled", False)
            ),
            "top_k": int(getattr(getattr(base_cfg.handoff, "embedding_manifold", None), "top_k", 0)),
            "blend": float(getattr(getattr(base_cfg.handoff, "embedding_manifold", None), "blend", 0.0)),
        },
        "model_pair_compatibility": suite_model_pair_compatibility,
        "model_pair_compatibility_by_pair": list(compatibility_cache.values()),
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "runtime_smoke_report": runtime_smoke_report,
        "semantic_smoke_report": semantic_smoke_report,
        "phase_gate_report": phase_gate_report,
        "ode_scaling_report": build_ode_scaling_report(summary_rows),
        "summary_rows": summary_rows,
    }
    write_json(report_output_path, report_payload)
    return sample_rows, summary_rows, report_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark homogeneous phase-1 and heterogeneous phase-3 latent transfer suites."
    )
    parser.add_argument(
        "--suite",
        choices=("standard", "phase1_homogeneous"),
        default=DEFAULT_SUITE,
        help=f"Benchmark suite to run (default: {DEFAULT_SUITE}).",
    )
    parser.add_argument(
        "--dataset",
        choices=("gsm8k", "math"),
        default=DEFAULT_DATASET,
        help=f"Dataset to evaluate (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default=DEFAULT_SPLIT,
        help="Dataset split to evaluate. Defaults to validation for GSM8K and test for MATH.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=f"Number of samples to evaluate (default: {DEFAULT_LIMIT}).",
    )
    parser.add_argument(
        "--sample-indices",
        default="",
        help=(
            "Optional comma-separated dataset indices to evaluate after split selection. "
            "When set, --limit caps this selected list."
        ),
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
        help=f"Number of repeated benchmark passes to run (default: {DEFAULT_REPETITIONS}).",
    )
    parser.add_argument(
        "--samples-output",
        type=Path,
        default=DEFAULT_SAMPLES_OUTPUT,
        help=f"Per-sample CSV output path (default: {DEFAULT_SAMPLES_OUTPUT}).",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
        help=f"Summary CSV output path (default: {DEFAULT_SUMMARY_OUTPUT}).",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=DEFAULT_REPORT_OUTPUT,
        help=f"Phase-gate JSON report path (default: {DEFAULT_REPORT_OUTPUT}).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional override for cfg.max_new_tokens to support faster smoke tests.",
    )
    parser.add_argument(
        "--latent-steps-values",
        default="",
        help=(
            "Optional comma-separated latent step sweep. "
            f"Recommended ODE sweep: {','.join(str(value) for value in DEFAULT_LATENT_STEPS_SWEEP)}."
        ),
    )
    parser.add_argument(
        "--agent-a-model",
        default=None,
        help="Optional override for Agent A model.",
    )
    parser.add_argument(
        "--agent-b-model",
        default=None,
        help="Optional override for Agent B model.",
    )
    parser.add_argument(
        "--latent-pooling",
        choices=("last_token", "mean", "prompt_mean"),
        default=None,
        help="Optional override for handoff.latent_pooling.",
    )
    parser.add_argument(
        "--receiver-context-mode",
        choices=("none", "auto", "prompt_prefix"),
        default=None,
        help="Optional override for handoff.receiver_context.mode.",
    )
    parser.add_argument(
        "--receiver-context-latent-position",
        choices=("after_context", "before_context"),
        default=None,
        help="Optional override for handoff.receiver_context.latent_position.",
    )
    parser.add_argument(
        "--disable-prompt-calibration",
        action="store_true",
        help="Disable alignment.prompt_calibration for this benchmark run.",
    )
    parser.add_argument(
        "--prompt-calibration-strength",
        type=float,
        default=None,
        help="Optional override for alignment.prompt_calibration.strength.",
    )
    parser.add_argument(
        "--prompt-calibration-max-norm-ratio",
        type=float,
        default=None,
        help="Optional override for alignment.prompt_calibration.max_norm_ratio.",
    )
    parser.add_argument(
        "--enable-handoff-adapter",
        action="store_true",
        help="Enable the train-split handoff adapter for latent-prefix methods.",
    )
    parser.add_argument(
        "--disable-handoff-adapter",
        action="store_true",
        help="Disable the train-split handoff adapter for this benchmark run.",
    )
    parser.add_argument(
        "--handoff-adapter-train-on-missing",
        action="store_true",
        help="Fit and cache the handoff adapter if no matching adapter cache exists.",
    )
    parser.add_argument(
        "--handoff-adapter-train-limit",
        type=int,
        default=None,
        help="Optional override for handoff.adapter.train_limit.",
    )
    parser.add_argument(
        "--enable-embedding-manifold",
        action="store_true",
        help="Project latent prefix vectors onto the receiver embedding manifold.",
    )
    parser.add_argument(
        "--disable-embedding-manifold",
        action="store_true",
        help="Disable receiver embedding-manifold projection for this benchmark run.",
    )
    parser.add_argument(
        "--embedding-manifold-top-k",
        type=int,
        default=None,
        help="Optional override for handoff.embedding_manifold.top_k.",
    )
    parser.add_argument(
        "--embedding-manifold-blend",
        type=float,
        default=None,
        help="Optional override for handoff.embedding_manifold.blend.",
    )
    parser.add_argument(
        "--methods",
        default="",
        help="Optional comma-separated method names to run.",
    )
    parser.add_argument(
        "--semantic-smoke",
        action="store_true",
        help=(
            "Use the full semantic smoke profile: limit=3, max_new_tokens=128, "
            "latent_steps=1, answer-only prompting, and focused methods unless overridden."
        ),
    )
    parser.add_argument(
        "--mvp-smoke",
        action="store_true",
        help=(
            "Use the viable MVP smoke profile: selected stable GSM8K validation samples, "
            "text baselines, and homogeneous latent transfer only."
        ),
    )
    parser.add_argument(
        "--hetero-smoke",
        action="store_true",
        help=(
            "Use a heterogeneous handoff experiment profile. Defaults to "
            f"{DEFAULT_HETERO_SMOKE_AGENT_A_MODEL} -> {DEFAULT_HETERO_SMOKE_AGENT_B_MODEL} "
            "unless model overrides are supplied."
        ),
    )
    parser.add_argument(
        "--answer-only-final",
        action="store_true",
        help="Prompt text baselines to return only a final answer.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional deterministic seed to stamp into benchmark metadata.",
    )
    args = parser.parse_args()
    if args.semantic_smoke or args.mvp_smoke or args.hetero_smoke:
        if args.limit is None:
            args.limit = DEFAULT_SEMANTIC_SMOKE_LIMIT
        if args.mvp_smoke and not args.sample_indices:
            args.sample_indices = ",".join(str(index) for index in DEFAULT_MVP_SMOKE_SAMPLE_INDICES)
        if args.hetero_smoke and not args.sample_indices:
            args.sample_indices = ",".join(str(index) for index in DEFAULT_HETERO_SMOKE_SAMPLE_INDICES)
        if args.max_new_tokens is None:
            args.max_new_tokens = DEFAULT_SEMANTIC_SMOKE_MAX_NEW_TOKENS
        if not args.latent_steps_values:
            args.latent_steps_values = "1"
        if not args.methods:
            if args.mvp_smoke:
                default_methods = DEFAULT_MVP_SMOKE_METHODS
            elif args.hetero_smoke:
                default_methods = DEFAULT_HETERO_SMOKE_METHODS
            else:
                default_methods = DEFAULT_SEMANTIC_SMOKE_METHODS
            args.methods = ",".join(default_methods)
        if args.receiver_context_mode is None:
            args.receiver_context_mode = "prompt_prefix"
    if args.limit is None:
        args.limit = DEFAULT_LIMIT

    latent_steps_values = (
        [int(value.strip()) for value in args.latent_steps_values.split(",") if value.strip()]
        if args.latent_steps_values
        else None
    )
    method_names = (
        [value.strip() for value in args.methods.split(",") if value.strip()]
        if args.methods
        else None
    )
    sample_indices = (
        [int(value.strip()) for value in args.sample_indices.split(",") if value.strip()]
        if args.sample_indices
        else None
    )

    _, summary_rows, report_payload = run_benchmark(
        suite_name=args.suite,
        dataset_name=args.dataset,
        dataset_split=args.split,
        limit=args.limit,
        repetitions=args.repetitions,
        samples_output_path=args.samples_output,
        summary_output_path=args.summary_output,
        report_output_path=args.report_output,
        max_new_tokens=args.max_new_tokens,
        latent_steps_values=latent_steps_values,
        agent_a_model=args.agent_a_model,
        agent_b_model=args.agent_b_model,
        latent_pooling=args.latent_pooling,
        receiver_context_mode=args.receiver_context_mode,
        receiver_context_latent_position=args.receiver_context_latent_position,
        prompt_calibration_enabled=False if args.disable_prompt_calibration else None,
        prompt_calibration_strength=args.prompt_calibration_strength,
        prompt_calibration_max_norm_ratio=args.prompt_calibration_max_norm_ratio,
        handoff_adapter_enabled=(
            False
            if args.disable_handoff_adapter
            else True
            if args.enable_handoff_adapter
            else None
        ),
        handoff_adapter_train_on_missing=True if args.handoff_adapter_train_on_missing else None,
        handoff_adapter_train_limit=args.handoff_adapter_train_limit,
        embedding_manifold_enabled=(
            False
            if args.disable_embedding_manifold
            else True
            if args.enable_embedding_manifold
            else None
        ),
        embedding_manifold_top_k=args.embedding_manifold_top_k,
        embedding_manifold_blend=args.embedding_manifold_blend,
        seed=args.seed,
        method_names=method_names,
        sample_indices=sample_indices,
        semantic_smoke=args.semantic_smoke,
        mvp_smoke=args.mvp_smoke,
        hetero_smoke=args.hetero_smoke,
        answer_only_final=args.answer_only_final,
    )

    print(f"Wrote per-sample benchmark rows to {args.samples_output}")
    print(f"Wrote benchmark summary rows to {args.summary_output}")
    print(f"Wrote phase-gate report to {args.report_output}")
    print(f"Phase gate passed: {report_payload['phase_gate_report']['passed']}")
    if report_payload.get("semantic_smoke_report") is not None:
        print(f"Semantic smoke passed: {report_payload['semantic_smoke_report']['passed']}")
    print(f"\n{'Method':<30} {'Acc':>8} {'PPL':>10} {'Latency':>11} {'Cache %':>9}")
    print("-" * 78)
    for row in summary_rows:
        cache_rate = row["cache_transfer_rate_percentage"]
        cache_label = "n/a" if cache_rate is None else f"{float(cache_rate):.1f}%"
        perplexity = row["answer_perplexity"]
        perplexity_label = "n/a" if perplexity is None else f"{float(perplexity):.2f}"
        print(
            f"{row['method']:<30} {float(row['accuracy_percentage']):>7.1f}% "
            f"{perplexity_label:>10} {float(row['average_latency_seconds']):>10.3f}s {cache_label:>9}"
        )


if __name__ == "__main__":
    main()
