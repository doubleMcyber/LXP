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
    apply_linear_mapping,
    compute_orthogonal_mapping,
    compute_ridge_mapping,
)
from src.utils.benchmarking import (
    STANDARD_SAMPLE_FIELDS,
    STANDARD_SUMMARY_FIELDS,
    aggregate_standard_rows,
    build_phase1_gate_report,
    build_phase3_gate_report,
    build_standard_row_base,
    write_csv,
    write_json,
)
from src.utils.metrics import extract_boxed_text, normalize_answer

DEFAULT_SUITE = "standard"
DEFAULT_DATASET = "gsm8k"
DEFAULT_LIMIT = 10
DEFAULT_REPETITIONS = 1
DEFAULT_SAMPLES_OUTPUT = Path("benchmark_samples.csv")
DEFAULT_SUMMARY_OUTPUT = Path("benchmark_summary.csv")
DEFAULT_REPORT_OUTPUT = Path("benchmark_report.json")
GSM8K_FINAL_ANSWER_REGEX = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")
NUMERIC_ANSWER_REGEX = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

RunnerFn = Callable[[str, Any, dict[str, Any]], dict[str, Any]]


def _load_cfg() -> Any:
    return OmegaConf.load(Path(__file__).resolve().parent / "configs" / "main.yaml")


def _clone_cfg(cfg: Any) -> Any:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def _gating_cfg(cfg: Any, phase_name: str) -> Any:
    return getattr(getattr(getattr(cfg, "reporting", None), "phase_gates", None), phase_name, None)


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


def _decode_handoff(
    *,
    agent_b: Any,
    tokenizer_b: Any,
    agent_b_device: torch.device,
    handoff_step: torch.Tensor,
    kv_cache_a: Any,
    max_new_tokens: int,
) -> dict[str, Any]:
    outputs_b, attention_mask_b, kv_cache_transferred = _run_actor_handoff(
        agent_b=agent_b,
        handoff_step=handoff_step,
        kv_cache_a=kv_cache_a,
        agent_b_device=agent_b_device,
    )
    raw_handoff_entropy = float(
        _compute_logits_entropy(outputs_b.logits[:, -1, :]).mean().detach().cpu().item()
    )
    generated_token_ids: list[int] = []
    eos_token_id = tokenizer_b.eos_token_id

    for _ in range(max_new_tokens):
        next_token = torch.argmax(outputs_b.logits[:, -1, :], dim=-1)
        next_token_id = int(next_token.item())
        if eos_token_id is not None and next_token_id == eos_token_id:
            break
        generated_token_ids.append(next_token_id)
        kv_cache_b = _normalize_kv_cache(outputs_b.past_key_values)
        attention_mask_b = torch.cat(
            [
                attention_mask_b,
                torch.ones(
                    (attention_mask_b.shape[0], 1),
                    dtype=attention_mask_b.dtype,
                    device=attention_mask_b.device,
                ),
            ],
            dim=1,
        )
        position_ids_b = _build_position_ids(attention_mask_b)[:, -1:]
        with torch.no_grad():
            outputs_b = agent_b(
                input_ids=next_token.unsqueeze(-1),
                past_key_values=kv_cache_b,
                attention_mask=attention_mask_b,
                position_ids=position_ids_b,
                use_cache=True,
                return_dict=True,
            )

    decoded_text = tokenizer_b.decode(generated_token_ids, skip_special_tokens=True)
    return {
        "decoded_text": decoded_text,
        "generated_tokens": len(generated_token_ids),
        "kv_cache_transferred": kv_cache_transferred,
        "raw_handoff_entropy": raw_handoff_entropy,
    }


def _alignment_distances(
    *,
    prompt: str,
    state: dict[str, Any],
    current_latent_step: torch.Tensor,
    mapping_matrix: torch.Tensor,
) -> dict[str, Optional[float]]:
    reasoning_layer_indices, reasoning_layer_weights = _reasoning_alignment_metadata(state)
    receiver_reference_handoff = _compute_receiver_reference_handoff(
        prompt=prompt,
        state=state,
        reasoning_layer_indices=reasoning_layer_indices,
        reasoning_layer_weights=reasoning_layer_weights,
    ).to(device=current_latent_step.device, dtype=current_latent_step.dtype)
    handoff_step = apply_linear_mapping(current_latent_step, mapping_matrix)
    receiver_reference_in_sender = apply_linear_mapping(
        receiver_reference_handoff.to(device=current_latent_step.device, dtype=current_latent_step.dtype),
        mapping_matrix.transpose(0, 1),
    )
    return {
        "pre_alignment_l2_distance": float(
            _normalized_l2_distance(current_latent_step, receiver_reference_in_sender).mean().detach().cpu().item()
        ),
        "pre_alignment_cosine_distance": float(
            _cosine_distance(current_latent_step, receiver_reference_in_sender).mean().detach().cpu().item()
        ),
        "post_alignment_l2_distance": float(
            _normalized_l2_distance(handoff_step, receiver_reference_handoff).mean().detach().cpu().item()
        ),
        "post_alignment_cosine_distance": float(
            _cosine_distance(handoff_step, receiver_reference_handoff).mean().detach().cpu().item()
        ),
    }


def run_pure_text_cot(prompt: str, cfg: Any, state: dict[str, Any]) -> dict[str, Any]:
    tokenizer_b = state["tokenizer_b"]
    agent_b = state["agent_b"]
    agent_b_device = next(agent_b.parameters()).device
    encoded = tokenizer_b(prompt, return_tensors="pt").to(agent_b_device)
    with torch.no_grad():
        output_ids = agent_b.generate(
            **encoded,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
        )
    generated_ids = output_ids[0, encoded["input_ids"].shape[1] :]
    return {
        "decoded_text": tokenizer_b.decode(generated_ids, skip_special_tokens=True),
        "generated_tokens": int(generated_ids.shape[0]),
        "alignment_mode": "text_baseline",
        "kv_cache_transferred": None,
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


def run_prompt_local_latent(prompt: str, cfg: Any, state: dict[str, Any]) -> dict[str, Any]:
    tokenizer_b = state["tokenizer_b"]
    agent_b = state["agent_b"]
    agent_b_device = next(agent_b.parameters()).device
    sender_state = _collect_sender_consensus_state(prompt, state)
    receiver_state = _collect_receiver_consensus_state(prompt, state)
    prompt_local_q = compute_orthogonal_mapping(
        sender_state["consensus_hidden_states"],
        receiver_state["consensus_hidden_states"].to(
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
        agent_b_device=agent_b_device,
        handoff_step=handoff_step,
        kv_cache_a=sender_state["kv_cache_a"],
        max_new_tokens=int(cfg.max_new_tokens),
    )
    return {
        **decode_metrics,
        **_alignment_distances(
            prompt=prompt,
            state=state,
            current_latent_step=sender_state["current_latent_step"],
            mapping_matrix=prompt_local_q,
        ),
        "alignment_mode": "prompt_local_procrustes",
        "handoff_uncertainty": None,
        "confidence_gate_triggered": False,
        "fallback_discrete_reasoning_steps": 0,
        "latent_trajectory_steps": 1,
        "total_reasoning_steps": 1,
        "continuous_integration_seconds": 0.0,
    }


def run_global_anchor_latent(prompt: str, cfg: Any, state: dict[str, Any]) -> dict[str, Any]:
    tokenizer_b = state["tokenizer_b"]
    agent_b = state["agent_b"]
    agent_b_device = next(agent_b.parameters()).device
    sender_state = _collect_sender_consensus_state(prompt, state)
    q_global = state["global_alignment_q"].to(
        device=sender_state["current_latent_step"].device,
        dtype=sender_state["current_latent_step"].dtype,
    )
    handoff_step = apply_linear_mapping(sender_state["current_latent_step"], q_global).to(
        device=agent_b_device,
        dtype=agent_b.get_input_embeddings().weight.dtype,
    )
    decode_metrics = _decode_handoff(
        agent_b=agent_b,
        tokenizer_b=tokenizer_b,
        agent_b_device=agent_b_device,
        handoff_step=handoff_step,
        kv_cache_a=sender_state["kv_cache_a"],
        max_new_tokens=int(cfg.max_new_tokens),
    )
    return {
        **decode_metrics,
        **_alignment_distances(
            prompt=prompt,
            state=state,
            current_latent_step=sender_state["current_latent_step"],
            mapping_matrix=q_global,
        ),
        "alignment_mode": str(state["alignment_mode"]),
        "handoff_uncertainty": None,
        "confidence_gate_triggered": False,
        "fallback_discrete_reasoning_steps": 0,
        "latent_trajectory_steps": 1,
        "total_reasoning_steps": 1,
        "continuous_integration_seconds": 0.0,
    }


def run_hybrid(prompt: str, cfg: Any, state: dict[str, Any]) -> dict[str, Any]:
    del state
    output = run_hybrid_pipeline(cfg, prompt=prompt, collect_alignment_metrics=True)
    generated_tokens = len(str(output["decoded_text"]).split())
    return {
        "decoded_text": str(output["decoded_text"]),
        "generated_tokens": generated_tokens,
        "alignment_mode": str(output["alignment_mode"]),
        "kv_cache_transferred": output["kv_cache_transferred"],
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
    }


def run_homogeneous_ridge_latent(prompt: str, cfg: Any, state: dict[str, Any]) -> dict[str, Any]:
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
        agent_b_device=agent_b_device,
        handoff_step=handoff_step,
        kv_cache_a=sender_state["kv_cache_a"],
        max_new_tokens=int(cfg.max_new_tokens),
    )
    return {
        **decode_metrics,
        "alignment_mode": "homogeneous_ridge",
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


def run_homogeneous_orthogonal_latent(prompt: str, cfg: Any, state: dict[str, Any]) -> dict[str, Any]:
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
        agent_b_device=agent_b_device,
        handoff_step=handoff_step,
        kv_cache_a=sender_state["kv_cache_a"],
        max_new_tokens=int(cfg.max_new_tokens),
    )
    return {
        **decode_metrics,
        "alignment_mode": "homogeneous_prompt_local_orthogonal",
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


def _methods_for_suite(suite_name: str) -> list[tuple[str, RunnerFn]]:
    if suite_name == "phase1_homogeneous":
        return [
            ("pure_text_cot", run_pure_text_cot),
            ("homogeneous_ridge_latent", run_homogeneous_ridge_latent),
            ("homogeneous_orthogonal_latent", run_homogeneous_orthogonal_latent),
        ]
    return [
        ("pure_text_cot", run_pure_text_cot),
        ("prompt_local_latent", run_prompt_local_latent),
        ("global_anchor_latent", run_global_anchor_latent),
        ("hybrid_hl_mas", run_hybrid),
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
    limit: int,
    repetitions: int,
    samples_output_path: Path,
    summary_output_path: Path,
    report_output_path: Path,
    max_new_tokens: Optional[int] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    base_cfg = _load_cfg()
    if max_new_tokens is not None:
        base_cfg.max_new_tokens = int(max_new_tokens)
    suite_cfg = _suite_cfg(base_cfg, suite_name)
    samples = get_dataloader(dataset_name, limit=limit)
    state = _get_pipeline_state(suite_cfg)
    tokenizer_b = state["tokenizer_b"]
    reasoning_layer_weights = tuple(state.get("global_reasoning_layer_weights", ()))

    sample_rows: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        for method_name, runner in _methods_for_suite(suite_name):
            print(f"Running {suite_name}:{method_name} on {dataset_name} (repetition {repetition + 1}/{repetitions})...")
            for index, row in enumerate(samples):
                prompt = pick_field(row, ("question", "problem"))
                target_answer = _target_answer(dataset_name, row)
                predicted_answer: Optional[str] = None
                decoded_text = ""
                result: dict[str, Any] = {}
                error: Optional[str] = None

                start = time.perf_counter()
                try:
                    result = runner(prompt, suite_cfg, state)
                    decoded_text = str(result["decoded_text"])
                    predicted_answer = _predicted_answer(dataset_name, decoded_text)
                except Exception as exc:  # noqa: BLE001
                    error = str(exc)
                latency = time.perf_counter() - start

                row_base = build_standard_row_base(
                    suite_cfg,
                    evaluation_surface="benchmark_all",
                    suite=suite_name,
                    method=method_name,
                    dataset=dataset_name,
                    repetition=repetition,
                    compression_steps=int(getattr(suite_cfg, "latent_steps", 0)),
                    alignment_mode=result.get("alignment_mode", ""),
                    semantic_anchor_count=int(state.get("semantic_anchor_count", 0)),
                    reasoning_layer_weights=reasoning_layer_weights,
                )
                sample_rows.append(
                    {
                        **row_base,
                        "sample_index": index,
                        "kv_cache_transferred": result.get("kv_cache_transferred"),
                        "prompt": prompt,
                        "target_answer": target_answer,
                        "predicted_answer": predicted_answer,
                        "decoded_text": decoded_text,
                        "generated_tokens": len(
                            tokenizer_b.encode(decoded_text, add_special_tokens=False)
                        ) if decoded_text else 0,
                        "correct": _answers_match(dataset_name, predicted_answer, target_answer),
                        "latency_seconds": latency,
                        "pre_alignment_l2_distance": result.get("pre_alignment_l2_distance"),
                        "pre_alignment_cosine_distance": result.get("pre_alignment_cosine_distance"),
                        "post_alignment_l2_distance": result.get("post_alignment_l2_distance"),
                        "post_alignment_cosine_distance": result.get("post_alignment_cosine_distance"),
                        "raw_handoff_entropy": result.get("raw_handoff_entropy"),
                        "handoff_uncertainty": result.get("handoff_uncertainty"),
                        "confidence_gate_triggered": result.get("confidence_gate_triggered"),
                        "fallback_discrete_reasoning_steps": result.get("fallback_discrete_reasoning_steps"),
                        "latent_trajectory_steps": result.get("latent_trajectory_steps"),
                        "total_reasoning_steps": result.get("total_reasoning_steps"),
                        "continuous_integration_seconds": result.get("continuous_integration_seconds"),
                        "error": error,
                    }
                )

    summary_rows = aggregate_standard_rows(sample_rows)
    gating_cfg = _gating_cfg(base_cfg, "phase1" if suite_name == "phase1_homogeneous" else "phase3")
    if suite_name == "phase1_homogeneous":
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
        )

    write_csv(samples_output_path, sample_rows, STANDARD_SAMPLE_FIELDS)
    write_csv(summary_output_path, summary_rows, STANDARD_SUMMARY_FIELDS)
    report_payload = {
        "suite": suite_name,
        "dataset": dataset_name,
        "limit": limit,
        "repetitions": repetitions,
        "phase_gate_report": phase_gate_report,
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
    args = parser.parse_args()

    _, summary_rows, report_payload = run_benchmark(
        suite_name=args.suite,
        dataset_name=args.dataset,
        limit=args.limit,
        repetitions=args.repetitions,
        samples_output_path=args.samples_output,
        summary_output_path=args.summary_output,
        report_output_path=args.report_output,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"Wrote per-sample benchmark rows to {args.samples_output}")
    print(f"Wrote benchmark summary rows to {args.summary_output}")
    print(f"Wrote phase-gate report to {args.report_output}")
    print(f"Phase gate passed: {report_payload['phase_gate_report']['passed']}")
    print(f"\n{'Method':<30} {'Accuracy':>10} {'Avg Latency':>13} {'Cache %':>9}")
    print("-" * 72)
    for row in summary_rows:
        cache_rate = row["cache_transfer_rate_percentage"]
        cache_label = "n/a" if cache_rate is None else f"{float(cache_rate):.1f}%"
        print(
            f"{row['method']:<30} {float(row['accuracy_percentage']):>9.1f}% "
            f"{float(row['average_latency_seconds']):>12.3f}s {cache_label:>9}"
        )


if __name__ == "__main__":
    main()
