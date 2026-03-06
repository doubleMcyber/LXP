from __future__ import annotations

import math
import re
import time
from collections.abc import Sequence
from types import MethodType
from typing import Any, Optional

import hydra
import torch
from omegaconf import DictConfig
from torchdiffeq import odeint_adjoint
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.dynamics import (
    TransformerBlockDynamics,
    _is_kv_cache_compatible,
    _kv_cache_layer_count,
    _kv_cache_seq_len,
    _move_kv_cache_to_device,
    _normalize_kv_cache,
    _sync_if_cuda,
)
from src.utils.alignment import (
    apply_orthogonal_mapping,
    compute_orthogonal_mapping,
    resolve_shared_semantic_anchor_ids,
)

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

_PIPELINE_STATE: Optional[dict[str, Any]] = None
_PREFERRED_REASONING_LAYERS: tuple[int, ...] = (12, 16, 20)
_MATH_COMPLEXITY_PATTERNS: tuple[str, ...] = (
    "solve",
    "equation",
    "integral",
    "derivative",
    "matrix",
    "probability",
    "statistics",
    "geometry",
    "algebra",
    "theorem",
    "proof",
    "entropy",
    "thermodynamics",
    "physics",
)
_CODE_COMPLEXITY_PATTERNS: tuple[str, ...] = (
    "python",
    "javascript",
    "typescript",
    "function",
    "class",
    "algorithm",
    "debug",
    "refactor",
    "sql",
    "regex",
)
_REASONING_COMPLEXITY_PATTERNS: tuple[str, ...] = (
    "compare",
    "analyze",
    "reason",
    "tradeoff",
    "why",
    "how",
    "explain",
    "multi-step",
    "step by step",
)


def load_agent(model_name: str, torch_dtype: str = "bfloat16", device_map: str = "auto") -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=_DTYPE_MAP.get(torch_dtype, torch.bfloat16),
        device_map=device_map,
        trust_remote_code=True,
    )


def _build_position_ids(attention_mask: torch.Tensor) -> torch.LongTensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    return position_ids.clamp_min_(0)


def estimate_problem_complexity(
    prompt: str, cfg: Optional[DictConfig] = None
) -> float:
    thresholds = getattr(getattr(cfg, "dynamics", None), "complexity_thresholds", None)

    default_factor = float(getattr(thresholds, "default", 1.0))
    short_factor = float(getattr(thresholds, "short", 0.75))
    long_factor = float(getattr(thresholds, "long", 1.1))
    reasoning_factor = float(getattr(thresholds, "reasoning", 1.15))
    math_factor = float(getattr(thresholds, "math", 1.25))
    code_factor = float(getattr(thresholds, "code", 1.3))
    short_prompt_words = int(getattr(thresholds, "short_prompt_words", 8))
    long_prompt_words = int(getattr(thresholds, "long_prompt_words", 24))

    normalized_prompt = prompt.casefold()
    prompt_words = re.findall(r"\b\w+\b", normalized_prompt)
    prompt_word_count = len(prompt_words)
    complexity_factor = default_factor

    if 0 < prompt_word_count <= short_prompt_words:
        complexity_factor = min(complexity_factor, short_factor)
    elif prompt_word_count >= long_prompt_words:
        complexity_factor = max(complexity_factor, long_factor)

    if any(pattern in normalized_prompt for pattern in _REASONING_COMPLEXITY_PATTERNS):
        complexity_factor = max(complexity_factor, reasoning_factor)
    if any(pattern in normalized_prompt for pattern in _CODE_COMPLEXITY_PATTERNS):
        complexity_factor = max(complexity_factor, code_factor)

    math_signal_count = sum(char.isdigit() for char in prompt)
    has_math_operator = any(operator in prompt for operator in ("+", "-", "*", "/", "="))
    if (
        any(pattern in normalized_prompt for pattern in _MATH_COMPLEXITY_PATTERNS)
        or math_signal_count >= 3
        or (math_signal_count > 0 and has_math_operator)
    ):
        complexity_factor = max(complexity_factor, math_factor)

    return max(0.5, min(2.0, complexity_factor))


def _scale_integration_points(base_points: int, complexity_factor: float) -> int:
    if base_points < 2:
        raise ValueError("base_points must be at least 2 for ODE integration")
    return max(2, math.ceil(base_points * complexity_factor))


def _build_integration_time_space(
    point_count: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if point_count < 2:
        raise ValueError("point_count must be at least 2 for ODE integration")
    # Keep the time horizon fixed to avoid drifting off the learned manifold.
    return torch.linspace(0.0, 1.0, point_count, device=device, dtype=dtype)


def _resolve_reasoning_layer_indices(
    sender_hidden_states: Sequence[torch.Tensor],
    receiver_hidden_states: Sequence[torch.Tensor],
) -> tuple[int, ...]:
    shared_max_index = min(len(sender_hidden_states), len(receiver_hidden_states)) - 1
    if shared_max_index < 1:
        raise ValueError("At least one transformer layer is required for alignment")

    target_count = min(len(_PREFERRED_REASONING_LAYERS), shared_max_index)
    selected_indices: list[int] = []

    for index in _PREFERRED_REASONING_LAYERS:
        if 1 <= index <= shared_max_index and index not in selected_indices:
            selected_indices.append(index)
        if len(selected_indices) == target_count:
            break

    if len(selected_indices) < target_count:
        for index in range(shared_max_index, 0, -1):
            if index not in selected_indices:
                selected_indices.append(index)
            if len(selected_indices) == target_count:
                break

    return tuple(sorted(selected_indices))


def _select_hidden_layers(
    hidden_states: Sequence[torch.Tensor], layer_indices: Sequence[int]
) -> list[torch.Tensor]:
    return [hidden_states[index] for index in layer_indices]


def _mean_hidden_layers(hidden_layers: Sequence[torch.Tensor]) -> torch.Tensor:
    if not hidden_layers:
        raise ValueError("At least one hidden layer is required to build a consensus latent")
    return torch.stack(tuple(hidden_layers), dim=0).mean(dim=0)


def _compute_logits_entropy(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    probabilities = torch.softmax(logits.float(), dim=-1)
    return -(probabilities * torch.log(probabilities + eps)).sum(dim=-1)


def _confidence_gate_settings(cfg: Optional[DictConfig]) -> tuple[bool, float, int]:
    gate_cfg = getattr(getattr(cfg, "dynamics", None), "confidence_gate", None)
    enabled = bool(getattr(gate_cfg, "enabled", True))
    uncertainty_threshold = float(getattr(gate_cfg, "uncertainty_threshold", 8.0))
    extra_discrete_steps = int(getattr(gate_cfg, "extra_discrete_steps", 3))
    return enabled, uncertainty_threshold, extra_discrete_steps


def _run_actor_handoff(
    *,
    agent_b: AutoModelForCausalLM,
    handoff_step: torch.Tensor,
    kv_cache_a: Any,
    agent_b_device: torch.device,
) -> tuple[Any, torch.Tensor, bool]:
    kv_cache_b_candidate = _move_kv_cache_to_device(kv_cache_a, agent_b_device)
    kv_cache_transferred = _is_kv_cache_compatible(kv_cache_b_candidate, agent_b)
    kv_cache_b = kv_cache_b_candidate if kv_cache_transferred else None
    attention_mask_b = torch.ones(
        (handoff_step.shape[0], _kv_cache_seq_len(kv_cache_b) + 1),
        dtype=torch.long,
        device=agent_b_device,
    )
    position_ids_b = _build_position_ids(attention_mask_b)[:, -1:]

    with torch.no_grad():
        outputs_b = agent_b(
            inputs_embeds=handoff_step,
            past_key_values=kv_cache_b,
            attention_mask=attention_mask_b,
            position_ids=position_ids_b,
            use_cache=True,
            return_dict=True,
        )

    return outputs_b, attention_mask_b, kv_cache_transferred


def _run_discrete_reasoning_fallback(
    *,
    agent_a: AutoModelForCausalLM,
    current_latent_step: torch.Tensor,
    reasoner_last_hidden_state: torch.Tensor,
    kv_cache_a: Any,
    attention_mask_a: torch.Tensor,
    reasoning_layer_indices: Sequence[int],
    extra_discrete_steps: int,
) -> tuple[torch.Tensor, Any, torch.Tensor, list[int]]:
    if extra_discrete_steps <= 0:
        return current_latent_step, kv_cache_a, attention_mask_a, []

    output_embeddings = agent_a.get_output_embeddings()
    if output_embeddings is None:
        raise ValueError("Reasoner model does not expose output embeddings for confidence fallback")

    fallback_latents: list[torch.Tensor] = [current_latent_step]
    fallback_token_ids: list[int] = []
    next_hidden_state = reasoner_last_hidden_state
    updated_attention_mask = attention_mask_a
    updated_kv_cache = kv_cache_a

    for _ in range(extra_discrete_steps):
        next_token_logits = output_embeddings(next_hidden_state[:, -1:, :])
        next_token = torch.argmax(next_token_logits[:, -1, :], dim=-1)
        fallback_token_ids.append(int(next_token.item()))

        updated_attention_mask = torch.cat(
            [
                updated_attention_mask,
                torch.ones(
                    (updated_attention_mask.shape[0], 1),
                    dtype=updated_attention_mask.dtype,
                    device=updated_attention_mask.device,
                ),
            ],
            dim=1,
        )
        next_position_ids = _build_position_ids(updated_attention_mask)[:, -1:]

        with torch.no_grad():
            reasoner_outputs = agent_a.model(
                input_ids=next_token.unsqueeze(-1),
                attention_mask=updated_attention_mask,
                position_ids=next_position_ids,
                past_key_values=updated_kv_cache,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )

        if reasoner_outputs.hidden_states is None:
            raise ValueError("Reasoner fallback did not return hidden states")

        updated_kv_cache = _normalize_kv_cache(reasoner_outputs.past_key_values)
        next_hidden_state = reasoner_outputs.last_hidden_state
        fallback_latents.append(
            _mean_hidden_layers(
                _select_hidden_layers(reasoner_outputs.hidden_states, reasoning_layer_indices)
            )
        )

    refined_latent_step = torch.cat(fallback_latents, dim=1).mean(dim=1, keepdim=True)
    return refined_latent_step, updated_kv_cache, updated_attention_mask, fallback_token_ids


def _collect_single_token_hidden_states(
    agent_model: AutoModelForCausalLM,
    token_ids: torch.LongTensor,
    device: torch.device,
) -> Sequence[torch.Tensor]:
    input_ids = token_ids.to(device=device, dtype=torch.long).reshape(-1, 1)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    position_ids = torch.zeros_like(input_ids, dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = agent_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

    if outputs.hidden_states is None:
        raise ValueError("Semantic-anchor alignment requires hidden states from both models")

    return outputs.hidden_states


def _compute_global_semantic_alignment(
    *,
    tokenizer_a: AutoTokenizer,
    tokenizer_b: AutoTokenizer,
    agent_a: AutoModelForCausalLM,
    agent_b: AutoModelForCausalLM,
) -> dict[str, Any]:
    semantic_anchor_strings, anchor_token_ids_a, anchor_token_ids_b = (
        resolve_shared_semantic_anchor_ids(tokenizer_a, tokenizer_b, anchor_count=100)
    )
    anchor_hidden_states_a = _collect_single_token_hidden_states(
        agent_a,
        anchor_token_ids_a,
        next(agent_a.parameters()).device,
    )
    anchor_hidden_states_b = _collect_single_token_hidden_states(
        agent_b,
        anchor_token_ids_b,
        next(agent_b.parameters()).device,
    )
    reasoning_layer_indices = _resolve_reasoning_layer_indices(
        anchor_hidden_states_a,
        anchor_hidden_states_b,
    )
    q_global = compute_orthogonal_mapping(
        _select_hidden_layers(anchor_hidden_states_a, reasoning_layer_indices),
        _select_hidden_layers(anchor_hidden_states_b, reasoning_layer_indices),
    ).detach()

    return {
        "semantic_anchor_strings": semantic_anchor_strings,
        "semantic_anchor_ids_a": tuple(int(token_id) for token_id in anchor_token_ids_a.tolist()),
        "semantic_anchor_ids_b": tuple(int(token_id) for token_id in anchor_token_ids_b.tolist()),
        "global_reasoning_layer_indices": reasoning_layer_indices,
        "global_alignment_q": q_global.cpu(),
    }


def attach_latent_forward(agent_model: AutoModelForCausalLM) -> None:
    def latent_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        del return_dict
        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = model_outputs.last_hidden_state
        past_key_values = _normalize_kv_cache(model_outputs.past_key_values)
        return hidden_states, past_key_values

    agent_model.forward = MethodType(latent_forward, agent_model)


def _get_pipeline_state(cfg: DictConfig) -> dict[str, Any]:
    global _PIPELINE_STATE
    if _PIPELINE_STATE is not None:
        return _PIPELINE_STATE

    tokenizer_a = AutoTokenizer.from_pretrained(cfg.agent_a_model, trust_remote_code=True)
    tokenizer_b = AutoTokenizer.from_pretrained(cfg.agent_b_model, trust_remote_code=True)
    agent_a = load_agent(cfg.agent_a_model, torch_dtype=cfg.torch_dtype, device_map=cfg.device_map)
    agent_b = load_agent(cfg.agent_b_model, torch_dtype=cfg.torch_dtype, device_map=cfg.device_map)
    attach_latent_forward(agent_a)
    alignment_state = _compute_global_semantic_alignment(
        tokenizer_a=tokenizer_a,
        tokenizer_b=tokenizer_b,
        agent_a=agent_a,
        agent_b=agent_b,
    )

    _PIPELINE_STATE = {
        "tokenizer_a": tokenizer_a,
        "tokenizer_b": tokenizer_b,
        "agent_a": agent_a,
        "agent_b": agent_b,
        **alignment_state,
    }
    return _PIPELINE_STATE


def initialize_hybrid_pipeline(cfg: DictConfig) -> None:
    _get_pipeline_state(cfg)


def get_global_alignment_metadata(cfg: DictConfig) -> dict[str, Any]:
    state = _get_pipeline_state(cfg)
    return {
        "alignment_mode": "semantic_anchor_global",
        "semantic_anchor_count": len(state["semantic_anchor_strings"]),
        "semantic_anchor_preview": tuple(state["semantic_anchor_strings"][:10]),
        "semantic_anchor_ids_a_preview": tuple(state["semantic_anchor_ids_a"][:10]),
        "semantic_anchor_ids_b_preview": tuple(state["semantic_anchor_ids_b"][:10]),
        "reasoning_layer_indices": tuple(state["global_reasoning_layer_indices"]),
        "q_global_shape": tuple(state["global_alignment_q"].shape),
    }


def extract_reasoning_trace(cfg: DictConfig, prompt: Optional[str] = None) -> dict[str, Any]:
    if prompt is None:
        prompt = cfg.default_prompt

    complexity_factor = estimate_problem_complexity(prompt, cfg)
    state = _get_pipeline_state(cfg)
    tokenizer_a = state["tokenizer_a"]
    agent_a = state["agent_a"]
    reasoning_layer_indices = tuple(state["global_reasoning_layer_indices"])
    semantic_anchor_strings = tuple(state["semantic_anchor_strings"])

    encoded = tokenizer_a(prompt, return_tensors="pt")
    agent_a_device = next(agent_a.parameters()).device
    input_ids_a = encoded["input_ids"].to(agent_a_device)
    attention_mask_a = encoded["attention_mask"].to(agent_a_device)
    position_ids_a = _build_position_ids(attention_mask_a)

    with torch.no_grad():
        agent_a_outputs = agent_a.model(
            input_ids=input_ids_a,
            attention_mask=attention_mask_a,
            position_ids=position_ids_a,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

    sender_reasoning_stack = agent_a_outputs.hidden_states
    if sender_reasoning_stack is None:
        raise ValueError("Reasoning trace extraction requires hidden states from Agent A")
    sender_reasoning_layers = _select_hidden_layers(
        sender_reasoning_stack, reasoning_layer_indices
    )
    consensus_hidden_states = _mean_hidden_layers(sender_reasoning_layers)
    procrustes_q = state["global_alignment_q"].to(
        device=consensus_hidden_states.device,
        dtype=consensus_hidden_states.dtype,
    )

    current_latent_step = consensus_hidden_states[:, -1:, :]
    continuous_position_ids = position_ids_a[:, -1:] + 1
    effective_latent_steps = _scale_integration_points(
        int(cfg.latent_steps), complexity_factor
    )

    rotary_emb = getattr(agent_a.model, "rotary_emb", None)
    dynamics = TransformerBlockDynamics(agent_a.model.layers[0], rotary_emb=rotary_emb)
    dynamics.set_context(position_ids=continuous_position_ids)

    time_space = _build_integration_time_space(
        effective_latent_steps,
        device=current_latent_step.device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        continuous_trajectory = odeint_adjoint(
            dynamics,
            current_latent_step,
            time_space,
            method="rk4",
        )

    return {
        "prompt": prompt,
        "complexity_factor": complexity_factor,
        "alignment_mode": "semantic_anchor_global",
        "semantic_anchor_count": len(semantic_anchor_strings),
        "semantic_anchor_preview": semantic_anchor_strings[:10],
        "latent_trajectory_steps": effective_latent_steps,
        "reasoning_layer_indices": reasoning_layer_indices,
        "continuous_trajectory": continuous_trajectory.detach().cpu(),
        "handoff_latent": continuous_trajectory[-1].detach().cpu(),
        "time_space": time_space.detach().cpu(),
        "procrustes_q_shape": tuple(procrustes_q.shape),
    }


def run_hybrid_pipeline(cfg: DictConfig, prompt: Optional[str] = None) -> dict[str, Any]:
    if prompt is None:
        prompt = cfg.default_prompt

    complexity_factor = estimate_problem_complexity(prompt, cfg)
    confidence_gate_enabled, uncertainty_threshold, extra_discrete_steps = _confidence_gate_settings(
        cfg
    )
    state = _get_pipeline_state(cfg)
    tokenizer_a = state["tokenizer_a"]
    tokenizer_b = state["tokenizer_b"]
    agent_a = state["agent_a"]
    agent_b = state["agent_b"]
    reasoning_layer_indices = tuple(state["global_reasoning_layer_indices"])
    semantic_anchor_strings = tuple(state["semantic_anchor_strings"])

    encoded = tokenizer_a(prompt, return_tensors="pt")
    agent_a_device = next(agent_a.parameters()).device
    input_ids_a = encoded["input_ids"].to(agent_a_device)
    attention_mask_a = encoded["attention_mask"].to(agent_a_device)
    position_ids_a = _build_position_ids(attention_mask_a)

    with torch.no_grad():
        agent_a_outputs = agent_a.model(
            input_ids=input_ids_a,
            attention_mask=attention_mask_a,
            position_ids=position_ids_a,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

    reasoner_last_hidden_state = agent_a_outputs.last_hidden_state
    kv_cache_a = _normalize_kv_cache(agent_a_outputs.past_key_values)
    sender_reasoning_stack = agent_a_outputs.hidden_states
    if sender_reasoning_stack is None:
        raise ValueError("Agent A did not return hidden states for reasoning-layer alignment")
    agent_b_device = next(agent_b.parameters()).device
    sender_reasoning_layers = _select_hidden_layers(
        sender_reasoning_stack, reasoning_layer_indices
    )
    consensus_hidden_states = _mean_hidden_layers(sender_reasoning_layers)
    procrustes_q = state["global_alignment_q"].to(
        device=consensus_hidden_states.device, dtype=consensus_hidden_states.dtype
    )
    current_latent_step = consensus_hidden_states[:, -1:, :]
    continuous_position_ids = position_ids_a[:, -1:] + 1
    effective_latent_steps = _scale_integration_points(
        int(cfg.latent_steps), complexity_factor
    )
    effective_simulated_steps = _scale_integration_points(
        int(cfg.simulated_continuous_steps), complexity_factor
    )

    rotary_emb = getattr(agent_a.model, "rotary_emb", None)
    dynamics = TransformerBlockDynamics(agent_a.model.layers[0], rotary_emb=rotary_emb)
    dynamics.set_context(position_ids=continuous_position_ids)

    time_space = _build_integration_time_space(
        effective_latent_steps,
        device=current_latent_step.device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        continuous_trajectory = odeint_adjoint(
            dynamics,
            current_latent_step,
            time_space,
            method="rk4",
        )
    current_latent_step = continuous_trajectory[-1]

    simulated_time_space = _build_integration_time_space(
        effective_simulated_steps,
        device=current_latent_step.device,
        dtype=torch.float32,
    )
    _sync_if_cuda(current_latent_step.device)
    integration_start = time.perf_counter()
    with torch.no_grad():
        _ = odeint_adjoint(
            dynamics,
            current_latent_step,
            simulated_time_space,
            method="rk4",
        )
    _sync_if_cuda(current_latent_step.device)
    integration_duration = time.perf_counter() - integration_start

    agent_b_embed_dtype = agent_b.get_input_embeddings().weight.dtype
    agent_b_embed_dim = agent_b.get_input_embeddings().weight.shape[-1]
    candidate_handoff_step = apply_orthogonal_mapping(current_latent_step, procrustes_q)
    if candidate_handoff_step.shape[-1] != agent_b_embed_dim:
        raise ValueError(
            "Consensus latent handoff dimension "
            f"{candidate_handoff_step.shape[-1]} does not match Agent B input dimension "
            f"{agent_b_embed_dim}"
        )
    candidate_handoff_step = candidate_handoff_step.to(
        device=agent_b_device,
        dtype=agent_b_embed_dtype,
    )
    outputs_b, attention_mask_b, kv_cache_transferred = _run_actor_handoff(
        agent_b=agent_b,
        handoff_step=candidate_handoff_step,
        kv_cache_a=kv_cache_a,
        agent_b_device=agent_b_device,
    )
    raw_handoff_entropy = float(
        _compute_logits_entropy(outputs_b.logits[:, -1, :]).mean().detach().cpu().item()
    )
    handoff_uncertainty = raw_handoff_entropy / max(complexity_factor, 1e-8)
    fallback_triggered = (
        confidence_gate_enabled
        and extra_discrete_steps > 0
        and handoff_uncertainty > uncertainty_threshold
    )
    executed_discrete_fallback_steps = 0
    fallback_token_ids: list[int] = []
    handoff_step = candidate_handoff_step

    if fallback_triggered:
        current_latent_step, kv_cache_a, attention_mask_a, fallback_token_ids = _run_discrete_reasoning_fallback(
            agent_a=agent_a,
            current_latent_step=current_latent_step,
            reasoner_last_hidden_state=reasoner_last_hidden_state,
            kv_cache_a=kv_cache_a,
            attention_mask_a=attention_mask_a,
            reasoning_layer_indices=reasoning_layer_indices,
            extra_discrete_steps=extra_discrete_steps,
        )
        executed_discrete_fallback_steps = len(fallback_token_ids)
        handoff_step = apply_orthogonal_mapping(current_latent_step, procrustes_q)
        if handoff_step.shape[-1] != agent_b_embed_dim:
            raise ValueError(
                "Fallback handoff dimension "
                f"{handoff_step.shape[-1]} does not match Agent B input dimension "
                f"{agent_b_embed_dim}"
            )
        handoff_step = handoff_step.to(device=agent_b_device, dtype=agent_b_embed_dtype)
        outputs_b, attention_mask_b, kv_cache_transferred = _run_actor_handoff(
            agent_b=agent_b,
            handoff_step=handoff_step,
            kv_cache_a=kv_cache_a,
            agent_b_device=agent_b_device,
        )

    total_reasoning_steps = effective_latent_steps + executed_discrete_fallback_steps

    generated_token_ids: list[int] = []
    eos_token_id = tokenizer_b.eos_token_id

    for _ in range(cfg.max_new_tokens):
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
        "complexity_factor": complexity_factor,
        "alignment_mode": "semantic_anchor_global",
        "semantic_anchor_count": len(semantic_anchor_strings),
        "semantic_anchor_preview": semantic_anchor_strings[:10],
        "raw_handoff_entropy": raw_handoff_entropy,
        "handoff_uncertainty": handoff_uncertainty,
        "confidence_gate_enabled": confidence_gate_enabled,
        "confidence_gate_threshold": uncertainty_threshold,
        "confidence_gate_triggered": fallback_triggered,
        "fallback_discrete_reasoning_steps": executed_discrete_fallback_steps,
        "fallback_reasoning_token_ids": fallback_token_ids,
        "latent_trajectory_steps": effective_latent_steps,
        "total_reasoning_steps": total_reasoning_steps,
        "continuous_integration_steps": effective_simulated_steps,
        "reasoning_layer_indices": reasoning_layer_indices,
        "consensus_latent_shape": tuple(consensus_hidden_states.shape),
        "final_latent_shape": tuple(current_latent_step.shape),
        "final_latent_dtype": str(current_latent_step.dtype),
        "procrustes_q_shape": tuple(procrustes_q.shape),
        "kv_cache_length": _kv_cache_layer_count(kv_cache_a),
        "kv_cache_transferred": kv_cache_transferred,
        "continuous_integration_seconds": integration_duration,
        "continuous_integration_50_steps_seconds": integration_duration,
    }


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig) -> None:
    complexity_factor = estimate_problem_complexity(cfg.default_prompt, cfg)
    alignment_metadata = get_global_alignment_metadata(cfg)
    print(f"Estimated complexity factor: {complexity_factor:.2f}")
    print(f"Alignment mode: {alignment_metadata['alignment_mode']}")
    print(f"Semantic anchor count: {alignment_metadata['semantic_anchor_count']}")
    print(f"Global Q shape: {alignment_metadata['q_global_shape']}")
    outputs = run_hybrid_pipeline(cfg)
    print(f"Pipeline complexity factor: {outputs['complexity_factor']:.2f}")
    print(f"Handoff uncertainty: {outputs['handoff_uncertainty']:.4f}")
    print(f"Confidence gate triggered: {outputs['confidence_gate_triggered']}")
    print(f"Fallback discrete reasoning steps: {outputs['fallback_discrete_reasoning_steps']}")
    print(f"Latent trajectory steps: {outputs['latent_trajectory_steps']}")
    print(f"Total reasoning steps: {outputs['total_reasoning_steps']}")
    print(f"Continuous integration steps: {outputs['continuous_integration_steps']}")
    print(f"Final latent step shape: {outputs['final_latent_shape']}")
    print(f"Final latent step dtype: {outputs['final_latent_dtype']}")
    print(f"Procrustes Q shape: {outputs['procrustes_q_shape']}")
    print(f"KV cache tuple length: {outputs['kv_cache_length']}")
    print(f"KV cache transferred to Agent B: {outputs['kv_cache_transferred']}")
    print(
        f"Continuous integration time ({outputs['continuous_integration_steps']} steps): "
        f"{outputs['continuous_integration_seconds']:.4f} seconds"
    )
    print(f"Agent B decoded text: {outputs['decoded_text']}")


if __name__ == "__main__":
    main()
