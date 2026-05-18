"""Comparative benchmark runner for phase 1 and phase 3 evaluation suites."""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from omegaconf import OmegaConf

from latent_pipeline import (
    _aggregate_hidden_layers,
    _build_position_ids,
    _compute_logits_entropy,
    _compute_receiver_reference_handoff,
    _cosine_distance,
    _get_pipeline_state,
    _normalized_l2_distance,
    _run_actor_handoff,
    _select_hidden_layers,
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
    STANDARD_SAMPLE_FIELDS,
    STANDARD_SUMMARY_FIELDS,
    aggregate_standard_rows,
    build_ode_scaling_report,
    build_phase1_gate_report,
    build_phase3_gate_report,
    build_runtime_smoke_report,
    build_standard_row_base,
    write_csv,
    write_json,
)
from src.utils.lm_eval import (
    compute_answer_metrics_from_prefix,
    greedy_decode_from_prefix,
    prepare_latent_prefix_state,
    prepare_text_prefix_state,
)
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
GSM8K_FINAL_ANSWER_REGEX = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")
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


def _format_text_cot_prompt(prompt: str) -> str:
    return f"{prompt}\n\nThink step by step, then give the final answer."


def _serialize_text_hybrid_prompt(prompt: str, reasoning_text: str) -> str:
    return (
        f"{prompt}\n\n"
        f"Reasoning from Agent A:\n{reasoning_text.strip()}\n\n"
        "Use the reasoning above and give the final answer."
    )


def _extract_gsm8k_target_answer(text: str) -> Optional[str]:
    match = GSM8K_FINAL_ANSWER_REGEX.search(text)
    return None if match is None else match.group(1)


def _extract_gsm8k_predicted_answer(text: str) -> Optional[str]:
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

    encoded = tokenizer_a(_format_text_cot_prompt(prompt), return_tensors="pt")
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


def _collect_sender_consensus_state(prompt: str, state: dict[str, Any]) -> dict[str, Any]:
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
    return {
        "consensus_hidden_states": consensus_hidden_states,
        "current_latent_step": consensus_hidden_states[:, -1:, :],
        "kv_cache_a": _normalize_kv_cache(outputs.past_key_values),
        "reasoning_layer_indices": reasoning_layer_indices,
        "reasoning_layer_weights": reasoning_layer_weights,
    }


def _collect_sender_last_hidden_state(prompt: str, state: dict[str, Any]) -> dict[str, Any]:
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

    return {
        "current_latent_step": hidden_states[:, -1:, :],
        "full_hidden_states": hidden_states,
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
        "receiver_reference_handoff": input_embeddings[:, -1:, :],
    }


def _decode_handoff(
    *,
    agent_b: Any,
    tokenizer_b: Any,
    handoff_step: torch.Tensor,
    kv_cache_a: Any,
    max_new_tokens: int,
    target_answer_text: Optional[str],
) -> dict[str, Any]:
    prefix_state = prepare_latent_prefix_state(
        model=agent_b,
        handoff_step=handoff_step,
        kv_cache=kv_cache_a,
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
    )
    decoded_text = str(decode_metrics["decoded_text"])
    return {
        "decoded_text": decoded_text,
        "generated_tokens": int(decode_metrics["generated_tokens"]),
        "decode_status": "decoded" if decoded_text.strip() else "empty_decode",
        "answer_token_count": answer_metrics["answer_token_count"],
        "answer_nll": answer_metrics["answer_nll"],
        "answer_perplexity": answer_metrics["answer_perplexity"],
        "kv_cache_transferred": prefix_state["kv_cache_transferred"],
        "kv_cache_status": prefix_state.get(
            "kv_cache_status",
            "transferred" if prefix_state["kv_cache_transferred"] else "unsupported",
        ),
        "kv_cache_reason": prefix_state.get("kv_cache_reason", ""),
        "raw_handoff_entropy": raw_handoff_entropy,
    }


def _alignment_distances(
    *,
    prompt: str,
    state: dict[str, Any],
    current_latent_step: torch.Tensor,
    alignment_state: dict[str, Any] | torch.Tensor,
    calibration_strength: float = 0.0,
    calibration_max_norm_ratio: float = 0.0,
) -> dict[str, Optional[float]]:
    reasoning_layer_indices, reasoning_layer_weights = _reasoning_alignment_metadata(state)
    receiver_reference_handoff = _compute_receiver_reference_handoff(
        prompt=prompt,
        state=state,
        reasoning_layer_indices=reasoning_layer_indices,
        reasoning_layer_weights=reasoning_layer_weights,
    ).to(device=current_latent_step.device, dtype=current_latent_step.dtype)
    handoff_step = apply_alignment(current_latent_step, alignment_state)
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
    prefix_text = _format_text_cot_prompt(prompt)
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
    prefix_text = _serialize_text_hybrid_prompt(prompt, reasoning_text)
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
    sender_state = _collect_sender_consensus_state(prompt, state)
    receiver_state = _collect_receiver_input_embedding_state(prompt, state)
    prompt_local_q = compute_orthogonal_mapping(
        sender_state["consensus_hidden_states"],
        receiver_state["input_embeddings"].to(
            device=sender_state["consensus_hidden_states"].device,
            dtype=sender_state["consensus_hidden_states"].dtype,
        ),
    ).to(
        device=sender_state["current_latent_step"].device,
        dtype=sender_state["current_latent_step"].dtype,
    )
    handoff_step = apply_linear_mapping(sender_state["current_latent_step"], prompt_local_q).to(
        device=agent_b_device,
        dtype=agent_b.get_input_embeddings().weight.dtype,
    )
    decode_metrics = _decode_handoff(
        agent_b=agent_b,
        tokenizer_b=tokenizer_b,
        handoff_step=handoff_step,
        kv_cache_a=sender_state["kv_cache_a"],
        max_new_tokens=int(cfg.max_new_tokens),
        target_answer_text=target_answer_text,
    )
    return {
        **decode_metrics,
        **_alignment_distances(
            prompt=prompt,
            state=state,
            current_latent_step=sender_state["current_latent_step"],
            alignment_state={"mapping_matrix": prompt_local_q},
        ),
        "alignment_mode": "prompt_local_procrustes",
        "alignment_strategy": "orthogonal",
        "handoff_status": "ok",
        "handoff_surface": "input_embedding",
        "handoff_uncertainty": None,
        "confidence_gate_triggered": False,
        "fallback_discrete_reasoning_steps": 0,
        "latent_trajectory_steps": 1,
        "total_reasoning_steps": 1,
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
    sender_state = _collect_sender_consensus_state(prompt, variant_state)
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
    handoff_step = apply_alignment(sender_state["current_latent_step"], alignment_state).to(
        device=agent_b_device,
        dtype=agent_b.get_input_embeddings().weight.dtype,
    )
    if prompt_calibration_enabled:
        receiver_reference_handoff = _compute_receiver_reference_handoff(
            prompt=prompt,
            state=variant_state,
            reasoning_layer_indices=sender_state["reasoning_layer_indices"],
            reasoning_layer_weights=sender_state["reasoning_layer_weights"],
        ).to(device=agent_b_device, dtype=handoff_step.dtype)
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
    decode_metrics = _decode_handoff(
        agent_b=agent_b,
        tokenizer_b=tokenizer_b,
        handoff_step=handoff_step,
        kv_cache_a=sender_state["kv_cache_a"],
        max_new_tokens=int(cfg.max_new_tokens),
        target_answer_text=target_answer_text,
    )
    return {
        **decode_metrics,
        **_alignment_distances(
            prompt=prompt,
            state=variant_state,
            current_latent_step=sender_state["current_latent_step"],
            alignment_state=alignment_state,
            calibration_strength=calibration_strength,
            calibration_max_norm_ratio=calibration_max_norm_ratio,
        ),
        "alignment_mode": method_alignment_mode,
        "alignment_strategy": str(strategy),
        "handoff_status": "ok",
        "handoff_surface": variant_state.get("handoff_surface", "input_embedding"),
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
    }


def run_homogeneous_ridge_latent(
    prompt: str,
    target_answer_text: Optional[str],
    cfg: Any,
    state: dict[str, Any],
) -> dict[str, Any]:
    sender_state = _collect_sender_last_hidden_state(prompt, state)
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
    sender_state = _collect_sender_last_hidden_state(prompt, state)
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


def _methods_for_suite(suite_name: str) -> list[tuple[str, RunnerFn]]:
    if suite_name == "phase1_homogeneous":
        return [
            ("pure_text_cot", run_pure_text_cot),
            ("text_text_hybrid", run_text_text_hybrid),
            ("homogeneous_ridge_latent", run_homogeneous_ridge_latent),
            ("homogeneous_orthogonal_latent", run_homogeneous_orthogonal_latent),
        ]
    return [
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


def _suite_cfg(base_cfg: Any, suite_name: str) -> Any:
    suite_cfg = _clone_cfg(base_cfg)
    if suite_name == "phase1_homogeneous":
        suite_cfg.agent_b_model = suite_cfg.agent_a_model
    return suite_cfg


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
    seed: Optional[int] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    base_cfg = _load_cfg()
    if agent_a_model is not None:
        base_cfg.agent_a_model = str(agent_a_model)
    if agent_b_model is not None:
        base_cfg.agent_b_model = str(agent_b_model)
    if seed is not None:
        base_cfg.seed = int(seed)
    if max_new_tokens is not None:
        base_cfg.max_new_tokens = int(max_new_tokens)
    effective_split = dataset_split or _default_split_for_dataset(dataset_name)
    validation_size = _validation_size(base_cfg, dataset_name)
    suite_cfg = _suite_cfg(base_cfg, suite_name)
    latent_step_candidates = latent_steps_values or [int(getattr(suite_cfg, "latent_steps", 0))]
    samples = get_dataloader(
        dataset_name,
        limit=limit,
        split=effective_split,
        validation_size=validation_size,
    )

    sample_rows: list[dict[str, Any]] = []
    for latent_steps in latent_step_candidates:
        step_cfg = _clone_cfg(suite_cfg)
        step_cfg.latent_steps = int(latent_steps)
        state = _get_pipeline_state(step_cfg)
        tokenizer_b = state["tokenizer_b"]

        for repetition in range(repetitions):
            for method_name, runner in _methods_for_suite(suite_name):
                print(
                    f"Running {suite_name}:{method_name} on {dataset_name}/{effective_split} "
                    f"with latent_steps={latent_steps} "
                    f"(repetition {repetition + 1}/{repetitions})..."
                )
                for index, row in enumerate(samples):
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
                    sample_rows.append(
                        {
                            **row_base,
                            "sample_index": index,
                            "handoff_status": handoff_status,
                            "handoff_surface": handoff_surface,
                            "kv_cache_transferred": kv_cache_transferred,
                            "kv_cache_status": kv_cache_status,
                            "kv_cache_reason": kv_cache_reason,
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
    if _is_runtime_smoke_run(base_cfg, limit=limit, repetitions=repetitions):
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

    write_csv(samples_output_path, sample_rows, STANDARD_SAMPLE_FIELDS)
    write_csv(summary_output_path, summary_rows, STANDARD_SUMMARY_FIELDS)
    report_payload = {
        "suite": suite_name,
        "dataset": dataset_name,
        "dataset_split": effective_split,
        "limit": limit,
        "repetitions": repetitions,
        "latent_steps_values": latent_step_candidates,
        "report_schema_version": 2,
        "runtime_smoke_report": runtime_smoke_report,
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
        default=DEFAULT_LIMIT,
        help=f"Number of samples to evaluate (default: {DEFAULT_LIMIT}).",
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
        "--seed",
        type=int,
        default=None,
        help="Optional deterministic seed to stamp into benchmark metadata.",
    )
    args = parser.parse_args()
    latent_steps_values = (
        [int(value.strip()) for value in args.latent_steps_values.split(",") if value.strip()]
        if args.latent_steps_values
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
        seed=args.seed,
    )

    print(f"Wrote per-sample benchmark rows to {args.samples_output}")
    print(f"Wrote benchmark summary rows to {args.summary_output}")
    print(f"Wrote phase-gate report to {args.report_output}")
    print(f"Phase gate passed: {report_payload['phase_gate_report']['passed']}")
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
