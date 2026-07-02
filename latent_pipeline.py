"""
LXP Latent Pipeline Execution Engine
------------------------------------
This module acts as the core orchestrator for the Latent Exchange Protocol (LXP).
It manages the end-to-end tensor flow between heterogeneous models (Agent A to Agent B).

Key Responsibilities:
1. Model Initialization: Loading both the Sender (Reasoner) and Receiver (Actor) models.
2. Latent Extraction: Intercepting the continuous hidden states from Agent A instead of standard text generation.
3. Continuous Dynamics: Invoking the Neural ODE solver (`TransformerBlockDynamics`) to simulate multi-step reasoning in continuous time.
4. Alignment & Handoff: Routing the compressed thought vector through the Orthogonal Procrustes bridge and passing it to Agent B along with the prompt's KV cache.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import weakref
from collections.abc import Sequence
from pathlib import Path
from types import MethodType
from typing import Any, Optional

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torchdiffeq import odeint
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.loader import get_dataloader, pick_field
from src.models.dynamics import (
    TransformerBlockDynamics,
    _is_kv_cache_compatible,
    _kv_cache_layer_count,
    _kv_cache_seq_len,
    _move_kv_cache_to_device,
    _normalize_kv_cache,
    _sync_if_cuda,
)
from src.models.handoff_adapter import (
    HandoffAdapterFitConfig,
    fit_handoff_adapter_state,
    project_to_embedding_manifold,
)
from src.utils.alignment import (
    apply_alignment,
    apply_linear_mapping,
    compute_alignment_state,
    compute_orthogonal_mapping,
    score_anchor_stability,
    resolve_shared_semantic_anchor_ids,
)
from src.utils.lm_eval import (
    append_text_to_prefix_state,
    compute_answer_metrics_from_prefix,
    greedy_decode_from_prefix,
    prepare_latent_prefix_state,
    prepare_receiver_context_latent_prefix_state,
)

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

_PIPELINE_STATE: Optional[dict[str, Any]] = None
_PIPELINE_STATE_KEY: Optional[tuple[str, str, str, str]] = None
_GLOBAL_ALIGNMENT_MEMORY_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_HANDOFF_ADAPTER_MEMORY_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_PREFERRED_REASONING_LAYERS: tuple[int, ...] = (12, 16, 20)
_DEFAULT_REASONING_LAYER_WEIGHTS: tuple[float, ...] = (0.2, 0.3, 0.5)
_ALIGNMENT_CACHE_VERSION = 4
_HANDOFF_ADAPTER_CACHE_VERSION = 1
_SUPPORTED_HANDOFF_TARGETS: frozenset[str] = frozenset({"input_embedding"})
_SUPPORTED_DIAGNOSTIC_TARGETS: frozenset[str] = frozenset({"hidden_consensus"})
_SUPPORTED_LATENT_POOLING_MODES: frozenset[str] = frozenset({"last_token", "mean", "prompt_mean"})
_SUPPORTED_LATENT_PREFIX_MODES: frozenset[str] = frozenset({"scalar", "sequence"})
_SUPPORTED_RECEIVER_CONTEXT_MODES: frozenset[str] = frozenset({"none", "auto", "prompt_prefix"})
_SUPPORTED_RECEIVER_CONTEXT_LATENT_POSITIONS: frozenset[str] = frozenset(
    {"after_context", "before_context"}
)
_FINAL_ANSWER_COMPLETE_REGEX = re.compile(
    r"final\s+answer\s*[:=]\s*(?:[$*`_\s]+)?-?\d[\d,]*(?:\.\d+)?(?:[$*`_\s]+|[^\d,.]|\.(?!\d))",
    re.IGNORECASE,
)
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
    dtype = _DTYPE_MAP.get(torch_dtype, torch.bfloat16)
    normalized_device_map = "" if device_map is None else str(device_map).strip().lower()
    if normalized_device_map in {"none", "cpu", "mps"}:
        if normalized_device_map == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("device_map=mps requested, but torch MPS is not available")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            trust_remote_code=True,
        )
        if normalized_device_map == "mps":
            return model.to(torch.device("mps"))
        if normalized_device_map == "cpu":
            return model.to(torch.device("cpu"))
        return model
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
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
    if base_points < 1:
        raise ValueError("base_points must be at least 1")
    return max(1, math.ceil(base_points * complexity_factor))


def _build_integration_time_space(
    point_count: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if point_count < 2:
        raise ValueError("point_count must be at least 2 for ODE integration")
    # Keep the time horizon fixed to avoid drifting off the learned manifold.
    return torch.linspace(0.0, 1.0, point_count, device=device, dtype=dtype)


def _pipeline_state_key(cfg: DictConfig) -> tuple[str, str, str, str]:
    return (
        str(cfg.agent_a_model),
        str(cfg.agent_b_model),
        str(cfg.torch_dtype),
        str(cfg.device_map),
    )


def _alignment_cfg(cfg: Optional[DictConfig]) -> Any:
    return getattr(cfg, "alignment", None)


def _semantic_anchor_count(cfg: Optional[DictConfig]) -> int:
    alignment_cfg = _alignment_cfg(cfg)
    return int(getattr(alignment_cfg, "semantic_anchor_count", 250))


def _alignment_strategy(cfg: Optional[DictConfig]) -> str:
    alignment_cfg = _alignment_cfg(cfg)
    return str(getattr(alignment_cfg, "strategy", "hybrid_affine")).strip().lower()


def _alignment_handoff_target(cfg: Optional[DictConfig]) -> str:
    alignment_cfg = _alignment_cfg(cfg)
    target = str(getattr(alignment_cfg, "handoff_target", "input_embedding")).strip().lower()
    if target not in _SUPPORTED_HANDOFF_TARGETS:
        supported = ", ".join(sorted(_SUPPORTED_HANDOFF_TARGETS))
        raise ValueError(f"alignment.handoff_target must be one of: {supported}")
    return target


def _alignment_diagnostic_target(cfg: Optional[DictConfig]) -> str:
    alignment_cfg = _alignment_cfg(cfg)
    target = str(getattr(alignment_cfg, "diagnostic_target", "hidden_consensus")).strip().lower()
    if target not in _SUPPORTED_DIAGNOSTIC_TARGETS:
        supported = ", ".join(sorted(_SUPPORTED_DIAGNOSTIC_TARGETS))
        raise ValueError(f"alignment.diagnostic_target must be one of: {supported}")
    return target


def _dynamics_mode(cfg: Optional[DictConfig]) -> str:
    dynamics_cfg = getattr(cfg, "dynamics", None)
    mode = str(getattr(dynamics_cfg, "mode", "identity")).strip().lower()
    if mode not in {"identity", "ode"}:
        raise ValueError("dynamics.mode must be either 'identity' or 'ode'")
    return mode


def _handoff_cfg(cfg: Optional[DictConfig]) -> Any:
    return getattr(cfg, "handoff", None)


def _handoff_adapter_cfg(cfg: Optional[DictConfig]) -> Any:
    return getattr(_handoff_cfg(cfg), "adapter", None)


def _handoff_adapter_enabled(cfg: Optional[DictConfig]) -> bool:
    return bool(getattr(_handoff_adapter_cfg(cfg), "enabled", False))


def _handoff_adapter_train_on_missing(cfg: Optional[DictConfig]) -> bool:
    return bool(getattr(_handoff_adapter_cfg(cfg), "train_on_missing", False))


def _handoff_adapter_cache_dir(cfg: Optional[DictConfig]) -> Path:
    adapter_cfg = _handoff_adapter_cfg(cfg)
    return Path(str(getattr(adapter_cfg, "cache_dir", ".cache/handoff_adapter")))


def _handoff_adapter_dataset_name(cfg: Optional[DictConfig]) -> str:
    adapter_cfg = _handoff_adapter_cfg(cfg)
    return str(getattr(adapter_cfg, "dataset_name", "gsm8k"))


def _handoff_adapter_train_split(cfg: Optional[DictConfig]) -> str:
    adapter_cfg = _handoff_adapter_cfg(cfg)
    return str(getattr(adapter_cfg, "train_split", "train"))


def _handoff_adapter_train_limit(cfg: Optional[DictConfig]) -> int:
    adapter_cfg = _handoff_adapter_cfg(cfg)
    return int(getattr(adapter_cfg, "train_limit", 32))


def _handoff_adapter_max_length(cfg: Optional[DictConfig]) -> int:
    adapter_cfg = _handoff_adapter_cfg(cfg)
    return int(getattr(adapter_cfg, "max_length", 96))


def _handoff_adapter_strategy(cfg: Optional[DictConfig]) -> str:
    adapter_cfg = _handoff_adapter_cfg(cfg)
    return str(getattr(adapter_cfg, "strategy", "hybrid_affine")).strip().lower()


def _handoff_adapter_regularization(cfg: Optional[DictConfig]) -> float:
    adapter_cfg = _handoff_adapter_cfg(cfg)
    return float(getattr(adapter_cfg, "regularization", 1e-3))


def _handoff_adapter_residual_alpha(cfg: Optional[DictConfig]) -> float:
    adapter_cfg = _handoff_adapter_cfg(cfg)
    return float(getattr(adapter_cfg, "residual_alpha", 1.0))


def _handoff_adapter_residual_max_norm_ratio(cfg: Optional[DictConfig]) -> float:
    adapter_cfg = _handoff_adapter_cfg(cfg)
    value = getattr(adapter_cfg, "residual_max_norm_ratio", 0.5)
    return 0.0 if value is None else float(value)


def _handoff_adapter_center(cfg: Optional[DictConfig]) -> bool:
    adapter_cfg = _handoff_adapter_cfg(cfg)
    return bool(getattr(adapter_cfg, "center", True))


def _handoff_adapter_use_bias(cfg: Optional[DictConfig]) -> bool:
    adapter_cfg = _handoff_adapter_cfg(cfg)
    return bool(getattr(adapter_cfg, "use_bias", True))


def _embedding_manifold_cfg(cfg: Optional[DictConfig]) -> Any:
    return getattr(_handoff_cfg(cfg), "embedding_manifold", None)


def _embedding_manifold_enabled(cfg: Optional[DictConfig]) -> bool:
    return bool(getattr(_embedding_manifold_cfg(cfg), "enabled", False))


def _embedding_manifold_top_k(cfg: Optional[DictConfig]) -> int:
    return int(getattr(_embedding_manifold_cfg(cfg), "top_k", 1))


def _embedding_manifold_temperature(cfg: Optional[DictConfig]) -> float:
    return float(getattr(_embedding_manifold_cfg(cfg), "temperature", 0.05))


def _embedding_manifold_blend(cfg: Optional[DictConfig]) -> float:
    return float(getattr(_embedding_manifold_cfg(cfg), "blend", 1.0))


def _embedding_manifold_normalize(cfg: Optional[DictConfig]) -> bool:
    return bool(getattr(_embedding_manifold_cfg(cfg), "normalize", True))


def _embedding_manifold_chunk_size(cfg: Optional[DictConfig]) -> int:
    return int(getattr(_embedding_manifold_cfg(cfg), "chunk_size", 32))


def _latent_pooling_mode(cfg: Optional[DictConfig]) -> str:
    mode = str(getattr(_handoff_cfg(cfg), "latent_pooling", "last_token")).strip().lower()
    if mode not in _SUPPORTED_LATENT_POOLING_MODES:
        supported = ", ".join(sorted(_SUPPORTED_LATENT_POOLING_MODES))
        raise ValueError(f"handoff.latent_pooling must be one of: {supported}")
    return "mean" if mode == "prompt_mean" else mode


def _latent_prefix_mode(cfg: Optional[DictConfig]) -> str:
    mode = str(getattr(_handoff_cfg(cfg), "latent_prefix_mode", "sequence")).strip().lower()
    if mode not in _SUPPORTED_LATENT_PREFIX_MODES:
        supported = ", ".join(sorted(_SUPPORTED_LATENT_PREFIX_MODES))
        raise ValueError(f"handoff.latent_prefix_mode must be one of: {supported}")
    return mode


def _receiver_context_mode(cfg: Optional[DictConfig]) -> str:
    context_cfg = getattr(_handoff_cfg(cfg), "receiver_context", None)
    mode = str(getattr(context_cfg, "mode", "auto")).strip().lower()
    if mode not in _SUPPORTED_RECEIVER_CONTEXT_MODES:
        supported = ", ".join(sorted(_SUPPORTED_RECEIVER_CONTEXT_MODES))
        raise ValueError(f"handoff.receiver_context.mode must be one of: {supported}")
    return mode


def _receiver_context_latent_position(cfg: Optional[DictConfig]) -> str:
    context_cfg = getattr(_handoff_cfg(cfg), "receiver_context", None)
    position = str(getattr(context_cfg, "latent_position", "after_context")).strip().lower()
    if position not in _SUPPORTED_RECEIVER_CONTEXT_LATENT_POSITIONS:
        supported = ", ".join(sorted(_SUPPORTED_RECEIVER_CONTEXT_LATENT_POSITIONS))
        raise ValueError(f"handoff.receiver_context.latent_position must be one of: {supported}")
    return position


def _should_use_receiver_context(context_mode: str, *, sender_kv_cache_transferred: bool) -> bool:
    if context_mode == "none":
        return False
    if context_mode == "prompt_prefix":
        return True
    return not sender_kv_cache_transferred


_CHAT_TEMPLATE_FALLBACK_WARNED: set[int] = set()


def _maybe_apply_chat_template(tokenizer: Any, user_message: str) -> str:
    if tokenizer is None or not getattr(tokenizer, "chat_template", None):
        return user_message
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as error:  # noqa: BLE001
        # A silently dropped chat template degrades every handoff in the run;
        # surface the fallback once per tokenizer instead of failing or spamming.
        if id(tokenizer) not in _CHAT_TEMPLATE_FALLBACK_WARNED:
            _CHAT_TEMPLATE_FALLBACK_WARNED.add(id(tokenizer))
            # evict on GC so a recycled id() cannot suppress a new tokenizer's warning
            try:
                weakref.finalize(
                    tokenizer, _CHAT_TEMPLATE_FALLBACK_WARNED.discard, id(tokenizer)
                )
            except TypeError:
                pass
            print(
                f"Warning: chat template failed ({error!r}); "
                "falling back to raw prompts for this tokenizer.",
                flush=True,
            )
        return user_message


def _answer_only_final_enabled(cfg: Optional[DictConfig]) -> bool:
    return bool(getattr(getattr(cfg, "benchmark", None), "answer_only_final", False))


def _decode_stop_regex(cfg: Optional[DictConfig]) -> Optional[re.Pattern[str]]:
    return _FINAL_ANSWER_COMPLETE_REGEX if _answer_only_final_enabled(cfg) else None


def _answer_metric_variants(cfg: Optional[DictConfig], answer_text: Optional[str]) -> tuple[str, ...]:
    if not _answer_only_final_enabled(cfg) or answer_text is None or not str(answer_text).strip():
        return ()
    answer = str(answer_text).strip()
    return (
        f"Final answer:{answer}",
        f"Final answer: {answer}",
        f"\nFinal answer:{answer}",
        f"\nFinal answer: {answer}",
        f"\n\nFinal answer:{answer}",
        f"\n\nFinal answer: {answer}",
    )


def _sender_truncation_active(cfg: Optional[DictConfig]) -> bool:
    return (
        getattr(getattr(cfg, "benchmark", None), "sender_reasoning_truncation_fraction", None)
        is not None
    )


def _format_receiver_context_prompt(prompt: str, tokenizer: Any = None, cfg: Optional[DictConfig] = None) -> str:
    if _sender_truncation_active(cfg):
        # Mid-reasoning handoff: the latents carry unfinished work; asking for "the
        # final answer" primes a guess. Phase 0 measured this layout at +20 points
        # over receiver-alone with the same latents that scored 0% under the old one.
        instruction = (
            "An assistant started solving this problem but stopped mid-reasoning. "
            "Its partial work is handed to you as a latent summary after this message. "
            "Continue the reasoning step by step and finish with exactly one line: "
            "Final answer: <answer>."
        )
    elif _answer_only_final_enabled(cfg):
        instruction = (
            "Use the latent reasoning signal that follows. Return only the final answer "
            "in the form: Final answer: <answer>."
        )
    else:
        instruction = "Use the latent reasoning signal that follows and give the final answer."
    user_message = f"{prompt}\n\n{instruction}"
    return _maybe_apply_chat_template(tokenizer, user_message)


def _format_receiver_context_answer_suffix(cfg: Optional[DictConfig] = None) -> str:
    if not _answer_only_final_enabled(cfg):
        return ""
    return "\n\nFinal answer:"


def _resample_sequence(reference: torch.Tensor, target_steps: int) -> torch.Tensor:
    if reference.dim() != 3:
        raise ValueError("reference must have shape [batch, steps, dim]")
    if target_steps <= 0:
        raise ValueError("target_steps must be positive")
    if target_steps == 1:
        return reference[:, -1:, :]
    if int(reference.shape[1]) == target_steps:
        return reference
    resized = F.interpolate(
        reference.float().transpose(1, 2),
        size=int(target_steps),
        mode="linear",
        align_corners=True,
    )
    return resized.transpose(1, 2).to(dtype=reference.dtype)


def _compute_receiver_reference_handoff_sequence(
    *,
    prompt: str,
    state: dict[str, Any],
    target_steps: int,
) -> torch.Tensor:
    tokenizer_b = state["tokenizer_b"]
    agent_b = state["agent_b"]
    agent_b_device = next(agent_b.parameters()).device
    encoded_b = tokenizer_b(prompt, return_tensors="pt")
    input_ids_b = encoded_b["input_ids"].to(agent_b_device)
    with torch.no_grad():
        receiver_input_embeddings = agent_b.get_input_embeddings()(input_ids_b)
    return _resample_sequence(receiver_input_embeddings, target_steps)


def _maybe_append_answer_suffix_to_prefix_state(
    *,
    model: AutoModelForCausalLM,
    tokenizer: Any,
    cfg: Optional[DictConfig],
    prefix_state: dict[str, Any],
) -> dict[str, Any]:
    suffix_text = _format_receiver_context_answer_suffix(cfg)
    if not suffix_text.strip():
        return prefix_state
    return append_text_to_prefix_state(
        model=model,
        tokenizer=tokenizer,
        prefix_state=prefix_state,
        suffix_text=suffix_text,
        decoded_text_prefix="Final answer:",
    )


def _pool_latent_handoff_step(
    consensus_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    pooling_mode: str,
) -> torch.Tensor:
    if pooling_mode == "last_token":
        return consensus_hidden_states[:, -1:, :]
    if pooling_mode != "mean":
        raise ValueError("pooling_mode must be 'last_token' or 'mean'")
    weights = attention_mask.to(
        device=consensus_hidden_states.device,
        dtype=torch.float32,
    ).unsqueeze(-1)
    pooled = (consensus_hidden_states.float() * weights).sum(dim=1, keepdim=True)
    pooled = pooled / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    return pooled.to(dtype=consensus_hidden_states.dtype)


def _alignment_center_anchors(cfg: Optional[DictConfig]) -> bool:
    alignment_cfg = _alignment_cfg(cfg)
    return bool(getattr(alignment_cfg, "center_anchors", True))


def _alignment_use_bias(cfg: Optional[DictConfig]) -> bool:
    alignment_cfg = _alignment_cfg(cfg)
    return bool(getattr(alignment_cfg, "use_bias", True))


def _alignment_residual_lambda(cfg: Optional[DictConfig]) -> float:
    alignment_cfg = _alignment_cfg(cfg)
    return float(getattr(alignment_cfg, "residual_lambda", 1e-3))


def _alignment_residual_alpha(cfg: Optional[DictConfig]) -> float:
    alignment_cfg = _alignment_cfg(cfg)
    return float(getattr(alignment_cfg, "residual_alpha", 1.0))


def _alignment_residual_max_norm_ratio(cfg: Optional[DictConfig]) -> float:
    alignment_cfg = _alignment_cfg(cfg)
    return float(getattr(alignment_cfg, "residual_max_norm_ratio", 0.25))


def _alignment_anchor_selection_pool_size(
    cfg: Optional[DictConfig],
    semantic_anchor_count: int,
) -> int:
    alignment_cfg = _alignment_cfg(cfg)
    configured = getattr(alignment_cfg, "anchor_selection_pool_size", None)
    if configured is None:
        return max(int(semantic_anchor_count), min(500, int(semantic_anchor_count) * 2))
    return max(int(semantic_anchor_count), int(configured))


def _alignment_anchor_stability_bootstrap_count(cfg: Optional[DictConfig]) -> int:
    alignment_cfg = _alignment_cfg(cfg)
    return int(getattr(alignment_cfg, "anchor_stability_bootstrap_count", 3))


def _alignment_anchor_stability_bootstrap_ratio(cfg: Optional[DictConfig]) -> float:
    alignment_cfg = _alignment_cfg(cfg)
    return float(getattr(alignment_cfg, "anchor_stability_bootstrap_ratio", 0.8))


def _alignment_anchor_stability_bootstrap_weight(cfg: Optional[DictConfig]) -> float:
    alignment_cfg = _alignment_cfg(cfg)
    return float(getattr(alignment_cfg, "anchor_stability_bootstrap_weight", 0.5))


def _alignment_adaptive_projection_settings(
    cfg: Optional[DictConfig],
) -> tuple[bool, float, float]:
    projection_cfg = getattr(_alignment_cfg(cfg), "adaptive_projection", None)
    enabled = bool(getattr(projection_cfg, "enabled", True))
    strength = float(getattr(projection_cfg, "strength", 0.15))
    clip_std_multiplier = float(
        getattr(projection_cfg, "clip_std_multiplier", 4.0)
    )
    return enabled, strength, clip_std_multiplier


def _prompt_calibration_settings(
    cfg: Optional[DictConfig],
) -> tuple[bool, float, float]:
    calibration_cfg = getattr(_alignment_cfg(cfg), "prompt_calibration", None)
    enabled = bool(getattr(calibration_cfg, "enabled", True))
    strength = float(getattr(calibration_cfg, "strength", 0.2))
    max_norm_ratio = float(getattr(calibration_cfg, "max_norm_ratio", 0.15))
    return enabled, strength, max_norm_ratio


def _alignment_cache_dir(cfg: Optional[DictConfig]) -> Path:
    alignment_cfg = _alignment_cfg(cfg)
    cache_dir = Path(str(getattr(alignment_cfg, "cache_dir", ".cache/alignment")))
    if cache_dir.is_absolute():
        return cache_dir
    return Path(__file__).resolve().parent / cache_dir


def _configured_reasoning_layer_weights(cfg: Optional[DictConfig]) -> tuple[float, ...]:
    alignment_cfg = _alignment_cfg(cfg)
    raw_weights = getattr(
        alignment_cfg,
        "reasoning_layer_weights",
        _DEFAULT_REASONING_LAYER_WEIGHTS,
    )
    weights = tuple(float(weight) for weight in raw_weights)
    if len(weights) != len(_PREFERRED_REASONING_LAYERS):
        raise ValueError(
            "alignment.reasoning_layer_weights must match the preferred reasoning "
            f"layer count ({len(_PREFERRED_REASONING_LAYERS)})"
        )
    if not weights or any((not math.isfinite(weight)) or weight <= 0 for weight in weights):
        raise ValueError("alignment.reasoning_layer_weights must contain only positive finite values")
    return weights


def _resolve_reasoning_layer_indices_from_counts(
    sender_layer_count: int,
    receiver_layer_count: int,
) -> tuple[int, ...]:
    shared_max_index = min(sender_layer_count, receiver_layer_count)
    if shared_max_index < 1:
        raise ValueError("At least one transformer layer is required for alignment")

    available_preferred = tuple(
        index for index in _PREFERRED_REASONING_LAYERS if 1 <= index <= shared_max_index
    )
    if available_preferred:
        return available_preferred

    fallback_count = min(len(_PREFERRED_REASONING_LAYERS), shared_max_index)
    start_index = shared_max_index - fallback_count + 1
    return tuple(range(start_index, shared_max_index + 1))


def _resolve_reasoning_layer_indices(
    sender_hidden_states: Sequence[torch.Tensor],
    receiver_hidden_states: Sequence[torch.Tensor],
) -> tuple[int, ...]:
    return _resolve_reasoning_layer_indices_from_counts(
        len(sender_hidden_states) - 1,
        len(receiver_hidden_states) - 1,
    )


def _select_hidden_layers(
    hidden_states: Sequence[torch.Tensor], layer_indices: Sequence[int]
) -> list[torch.Tensor]:
    return [hidden_states[index] for index in layer_indices]


def _resolve_reasoning_layer_weights(
    cfg: Optional[DictConfig],
    layer_indices: Sequence[int],
) -> tuple[float, ...]:
    if not layer_indices:
        raise ValueError("At least one reasoning layer index is required")

    configured_weights = _configured_reasoning_layer_weights(cfg)
    preferred_weight_map = dict(zip(_PREFERRED_REASONING_LAYERS, configured_weights))
    fallback_weights = list(configured_weights[-len(layer_indices) :])
    fallback_cursor = 0
    resolved_weights: list[float] = []

    for layer_index in layer_indices:
        if layer_index in preferred_weight_map:
            resolved_weights.append(preferred_weight_map[layer_index])
            continue
        resolved_weights.append(fallback_weights[fallback_cursor])
        fallback_cursor += 1

    weight_sum = sum(resolved_weights)
    if weight_sum <= 0:
        raise ValueError("Reasoning layer weights must sum to a positive value")
    return tuple(weight / weight_sum for weight in resolved_weights)


def _aggregate_hidden_layers(
    hidden_layers: Sequence[torch.Tensor],
    layer_weights: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    if not hidden_layers:
        raise ValueError("At least one hidden layer is required to build a consensus latent")

    stacked = torch.stack(tuple(hidden_layers), dim=0)
    if layer_weights is None:
        normalized_weights = torch.full(
            (len(hidden_layers),),
            1.0 / len(hidden_layers),
            device=stacked.device,
            dtype=torch.float32,
        )
    else:
        normalized_weights = torch.as_tensor(
            layer_weights,
            device=stacked.device,
            dtype=torch.float32,
        ).flatten()
        if normalized_weights.numel() != len(hidden_layers):
            raise ValueError(
                f"Expected {len(hidden_layers)} layer weights, received {normalized_weights.numel()}"
            )
        if not torch.isfinite(normalized_weights).all():
            raise ValueError("layer_weights must contain only finite values")
        if torch.any(normalized_weights <= 0):
            raise ValueError("layer_weights must contain only positive values")
        normalized_weights = normalized_weights / normalized_weights.sum()

    view_shape = (len(hidden_layers),) + (1,) * (stacked.dim() - 1)
    weighted_sum = (
        stacked.to(torch.float32)
        * normalized_weights.view(view_shape)
    ).sum(dim=0)
    return weighted_sum.to(dtype=stacked.dtype)


def _mean_hidden_layers(hidden_layers: Sequence[torch.Tensor]) -> torch.Tensor:
    return _aggregate_hidden_layers(hidden_layers)


def _compute_logits_entropy(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    probabilities = torch.softmax(logits.float(), dim=-1)
    return -(probabilities * torch.log(probabilities + eps)).sum(dim=-1)


def _normalized_l2_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_flat = left.float().reshape(left.shape[0], -1)
    right_flat = right.float().reshape(right.shape[0], -1)
    if left_flat.shape != right_flat.shape:
        raise ValueError("normalized L2 distance requires matching flattened shapes")
    return torch.linalg.vector_norm(left_flat - right_flat, dim=-1) / math.sqrt(left_flat.shape[-1])


def _cosine_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_flat = left.float().reshape(left.shape[0], -1)
    right_flat = right.float().reshape(right.shape[0], -1)
    if left_flat.shape != right_flat.shape:
        raise ValueError("cosine distance requires matching flattened shapes")
    return 1.0 - F.cosine_similarity(left_flat, right_flat, dim=-1)


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
    reasoning_layer_weights: Sequence[float],
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
            _aggregate_hidden_layers(
                _select_hidden_layers(reasoner_outputs.hidden_states, reasoning_layer_indices),
                reasoning_layer_weights,
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


def _collect_single_token_input_embeddings(
    agent_model: AutoModelForCausalLM,
    token_ids: torch.LongTensor,
    device: torch.device,
) -> torch.Tensor:
    input_ids = token_ids.to(device=device, dtype=torch.long).reshape(-1, 1)
    with torch.no_grad():
        return agent_model.get_input_embeddings()(input_ids)


def _repeat_alignment_target_for_layers(
    target_hidden_states: torch.Tensor,
    layer_count: int,
) -> tuple[torch.Tensor, ...]:
    if layer_count <= 0:
        raise ValueError("layer_count must be positive")
    return tuple(target_hidden_states for _ in range(layer_count))


def _build_latent_trajectory(
    *,
    dynamics_mode: str,
    dynamics: TransformerBlockDynamics,
    current_latent_step: torch.Tensor,
    point_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if point_count < 1:
        raise ValueError("point_count must be at least 1")
    if dynamics_mode == "identity" or point_count == 1:
        time_space = torch.zeros(
            point_count,
            device=current_latent_step.device,
            dtype=torch.float32,
        )
        trajectory = current_latent_step.unsqueeze(0).expand(
            (point_count,) + tuple(current_latent_step.shape)
        ).clone()
        return trajectory, time_space

    time_space = _build_integration_time_space(
        point_count,
        device=current_latent_step.device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        trajectory = odeint(
            dynamics,
            current_latent_step,
            time_space,
            method="rk4",
        )
    return trajectory, time_space


def _time_simulated_integration(
    *,
    dynamics_mode: str,
    dynamics: TransformerBlockDynamics,
    current_latent_step: torch.Tensor,
    point_count: int,
) -> float:
    if dynamics_mode == "identity" or point_count < 2:
        return 0.0

    simulated_time_space = _build_integration_time_space(
        point_count,
        device=current_latent_step.device,
        dtype=torch.float32,
    )
    _sync_if_cuda(current_latent_step.device)
    integration_start = time.perf_counter()
    with torch.no_grad():
        _ = odeint(
            dynamics,
            current_latent_step,
            simulated_time_space,
            method="rk4",
        )
    _sync_if_cuda(current_latent_step.device)
    return time.perf_counter() - integration_start


def _trace_tensor_event(
    *,
    operation: str,
    tensor: torch.Tensor,
    source_surface: str,
    target_surface: str,
    model_id: str,
    diagnostics: Optional[dict[str, Any]] = None,
    kv_cache_status: Optional[str] = None,
    decode_status: Optional[str] = None,
) -> dict[str, Any]:
    scalar_diagnostics = {
        key: value
        for key, value in (diagnostics or {}).items()
        if isinstance(value, (bool, int, float, str)) or value is None
    }
    return {
        "operation": operation,
        "tensor_shape": tuple(int(dim) for dim in tensor.shape),
        "source_surface": source_surface,
        "target_surface": target_surface,
        "model_id": model_id,
        "kv_cache_status": kv_cache_status,
        "decode_status": decode_status,
        "scalar_diagnostics": scalar_diagnostics,
        "diagnostics": diagnostics or {},
    }


def _build_alignment_cache_key(
    *,
    agent_a_model: str,
    agent_b_model: str,
    torch_dtype: str,
    semantic_anchor_count: int,
    reasoning_layer_indices: Sequence[int],
    reasoning_layer_weights: Sequence[float],
    alignment_strategy: str,
    center_anchors: bool,
    use_bias: bool,
    residual_lambda: float,
    residual_alpha: float,
    residual_max_norm_ratio: float,
    adaptive_projection_enabled: bool,
    adaptive_projection_strength: float,
    adaptive_projection_clip_std_multiplier: float,
    handoff_target: str,
    diagnostic_target: str,
) -> tuple[Any, ...]:
    return (
        _ALIGNMENT_CACHE_VERSION,
        str(agent_a_model),
        str(agent_b_model),
        str(torch_dtype),
        int(semantic_anchor_count),
        tuple(int(index) for index in reasoning_layer_indices),
        tuple(round(float(weight), 8) for weight in reasoning_layer_weights),
        str(alignment_strategy),
        bool(center_anchors),
        bool(use_bias),
        round(float(residual_lambda), 8),
        round(float(residual_alpha), 8),
        round(float(residual_max_norm_ratio), 8),
        bool(adaptive_projection_enabled),
        round(float(adaptive_projection_strength), 8),
        round(float(adaptive_projection_clip_std_multiplier), 8),
        str(handoff_target),
        str(diagnostic_target),
    )


def _alignment_cache_path(
    cache_dir: Path,
    cache_key: tuple[Any, ...],
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(
        json.dumps(cache_key, sort_keys=False, default=list).encode("utf-8")
    ).hexdigest()
    return cache_dir / f"q_global_{digest}.pt"


def _load_alignment_state_from_disk(cache_path: Path) -> Optional[dict[str, Any]]:
    if not cache_path.is_file():
        return None
    try:
        cached_state = torch.load(cache_path, map_location="cpu")
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(cached_state, dict):
        return None
    if "global_alignment_q" not in cached_state and "mapping_matrix" not in cached_state:
        return None
    return cached_state


def _compute_global_semantic_alignment(
    *,
    cfg: DictConfig,
    tokenizer_a: AutoTokenizer,
    tokenizer_b: AutoTokenizer,
    agent_a: AutoModelForCausalLM,
    agent_b: AutoModelForCausalLM,
    reasoning_layer_indices: Sequence[int],
    reasoning_layer_weights: Sequence[float],
    semantic_anchor_count: int,
) -> dict[str, Any]:
    candidate_anchor_count = _alignment_anchor_selection_pool_size(
        cfg,
        semantic_anchor_count,
    )
    semantic_anchor_strings, candidate_anchor_ids_a, candidate_anchor_ids_b = (
        resolve_shared_semantic_anchor_ids(
            tokenizer_a,
            tokenizer_b,
            anchor_count=candidate_anchor_count,
        )
    )
    selected_indices = _select_stable_anchor_subset(
        cfg=cfg,
        agent_a=agent_a,
        agent_b=agent_b,
        candidate_anchor_ids_a=candidate_anchor_ids_a,
        candidate_anchor_ids_b=candidate_anchor_ids_b,
        reasoning_layer_indices=reasoning_layer_indices,
        reasoning_layer_weights=reasoning_layer_weights,
        target_anchor_count=semantic_anchor_count,
    )
    anchor_token_ids_a = candidate_anchor_ids_a.index_select(0, selected_indices)
    anchor_token_ids_b = candidate_anchor_ids_b.index_select(0, selected_indices)
    selected_anchor_strings = tuple(
        semantic_anchor_strings[int(index)] for index in selected_indices.tolist()
    )
    return compute_semantic_alignment_from_token_ids(
        cfg=cfg,
        tokenizer_a=tokenizer_a,
        tokenizer_b=tokenizer_b,
        agent_a=agent_a,
        agent_b=agent_b,
        sender_anchor_ids=anchor_token_ids_a,
        receiver_anchor_ids=anchor_token_ids_b,
        reasoning_layer_indices=reasoning_layer_indices,
        reasoning_layer_weights=reasoning_layer_weights,
        alignment_mode=_alignment_strategy(cfg),
        semantic_anchor_strings=selected_anchor_strings,
        candidate_anchor_strings=semantic_anchor_strings,
        candidate_anchor_ids_a=candidate_anchor_ids_a,
        candidate_anchor_ids_b=candidate_anchor_ids_b,
        selected_anchor_indices=selected_indices,
    )


def _select_stable_anchor_subset(
    *,
    cfg: Optional[DictConfig],
    agent_a: AutoModelForCausalLM,
    agent_b: AutoModelForCausalLM,
    candidate_anchor_ids_a: torch.LongTensor,
    candidate_anchor_ids_b: torch.LongTensor,
    reasoning_layer_indices: Sequence[int],
    reasoning_layer_weights: Sequence[float],
    target_anchor_count: int,
) -> torch.LongTensor:
    candidate_count = int(candidate_anchor_ids_a.numel())
    if target_anchor_count >= candidate_count:
        return torch.arange(candidate_count, dtype=torch.long)

    sender_hidden_states = _collect_single_token_hidden_states(
        agent_a,
        candidate_anchor_ids_a,
        next(agent_a.parameters()).device,
    )
    receiver_hidden_states = _collect_single_token_hidden_states(
        agent_b,
        candidate_anchor_ids_b,
        next(agent_b.parameters()).device,
    )
    sender_consensus = _aggregate_hidden_layers(
        _select_hidden_layers(sender_hidden_states, reasoning_layer_indices),
        reasoning_layer_weights,
    ).reshape(candidate_count, -1)
    receiver_consensus = _aggregate_hidden_layers(
        _select_hidden_layers(receiver_hidden_states, reasoning_layer_indices),
        reasoning_layer_weights,
    ).reshape(candidate_count, -1)
    stability = score_anchor_stability(
        sender_consensus,
        receiver_consensus,
        strategy=_alignment_strategy(cfg),
        regularization=_alignment_residual_lambda(cfg),
        residual_alpha=_alignment_residual_alpha(cfg),
        residual_max_norm_ratio=_alignment_residual_max_norm_ratio(cfg),
        center=_alignment_center_anchors(cfg),
        use_bias=_alignment_use_bias(cfg),
        bootstrap_count=_alignment_anchor_stability_bootstrap_count(cfg),
        bootstrap_ratio=_alignment_anchor_stability_bootstrap_ratio(cfg),
        bootstrap_weight=_alignment_anchor_stability_bootstrap_weight(cfg),
        seed=int(getattr(cfg, "seed", 0)),
    )
    combined_score = stability["combined_score"].float()
    sorted_indices = torch.argsort(combined_score, dim=0, descending=False)
    return sorted_indices[:target_anchor_count].cpu()


def compute_semantic_alignment_from_token_ids(
    *,
    cfg: Optional[DictConfig],
    tokenizer_a: AutoTokenizer,
    tokenizer_b: AutoTokenizer,
    agent_a: AutoModelForCausalLM,
    agent_b: AutoModelForCausalLM,
    sender_anchor_ids: torch.LongTensor,
    receiver_anchor_ids: torch.LongTensor,
    reasoning_layer_indices: Sequence[int],
    reasoning_layer_weights: Sequence[float],
    alignment_mode: str,
    semantic_anchor_strings: Optional[Sequence[str]] = None,
    candidate_anchor_strings: Optional[Sequence[str]] = None,
    candidate_anchor_ids_a: Optional[torch.LongTensor] = None,
    candidate_anchor_ids_b: Optional[torch.LongTensor] = None,
    selected_anchor_indices: Optional[torch.LongTensor] = None,
) -> dict[str, Any]:
    anchor_hidden_states_a = _collect_single_token_hidden_states(
        agent_a,
        sender_anchor_ids,
        next(agent_a.parameters()).device,
    )
    anchor_hidden_states_b = _collect_single_token_hidden_states(
        agent_b,
        receiver_anchor_ids,
        next(agent_b.parameters()).device,
    )
    sender_layers = _select_hidden_layers(anchor_hidden_states_a, reasoning_layer_indices)
    receiver_layers = _select_hidden_layers(anchor_hidden_states_b, reasoning_layer_indices)
    receiver_input_embeddings = _collect_single_token_input_embeddings(
        agent_b,
        receiver_anchor_ids,
        next(agent_b.parameters()).device,
    )
    handoff_receiver_layers = _repeat_alignment_target_for_layers(
        receiver_input_embeddings,
        len(sender_layers),
    )
    handoff_target = _alignment_handoff_target(cfg)
    diagnostic_target = _alignment_diagnostic_target(cfg)
    adaptive_projection_enabled, adaptive_projection_strength, adaptive_projection_clip_std = (
        _alignment_adaptive_projection_settings(cfg)
    )
    diagnostic_alignment_state = compute_alignment_state(
        sender_layers,
        receiver_layers,
        layer_weights=reasoning_layer_weights,
        strategy=alignment_mode,
        center=_alignment_center_anchors(cfg),
        use_bias=_alignment_use_bias(cfg),
        regularization=_alignment_residual_lambda(cfg),
        residual_alpha=_alignment_residual_alpha(cfg),
        residual_max_norm_ratio=_alignment_residual_max_norm_ratio(cfg),
        adaptive_projection_strength=(
            adaptive_projection_strength if adaptive_projection_enabled else 0.0
        ),
        adaptive_projection_clip_std_multiplier=adaptive_projection_clip_std,
    )
    handoff_alignment_state = compute_alignment_state(
        sender_layers,
        handoff_receiver_layers,
        layer_weights=reasoning_layer_weights,
        strategy=alignment_mode,
        center=_alignment_center_anchors(cfg),
        use_bias=_alignment_use_bias(cfg),
        regularization=_alignment_residual_lambda(cfg),
        residual_alpha=_alignment_residual_alpha(cfg),
        residual_max_norm_ratio=_alignment_residual_max_norm_ratio(cfg),
        adaptive_projection_strength=(
            adaptive_projection_strength if adaptive_projection_enabled else 0.0
        ),
        adaptive_projection_clip_std_multiplier=adaptive_projection_clip_std,
    )
    mapping_matrix = diagnostic_alignment_state["mapping_matrix"].detach()
    bias_vector = diagnostic_alignment_state["mapping_bias"].detach()
    orthogonal_q = diagnostic_alignment_state["orthogonal_q"].detach()
    residual_matrix = diagnostic_alignment_state["residual_matrix"].detach()
    singular_values = diagnostic_alignment_state["alignment_singular_values"].detach()
    handoff_mapping_matrix = handoff_alignment_state["mapping_matrix"].detach()
    handoff_bias_vector = handoff_alignment_state["mapping_bias"].detach()
    handoff_orthogonal_q = handoff_alignment_state["orthogonal_q"].detach()
    handoff_residual_matrix = handoff_alignment_state["residual_matrix"].detach()
    handoff_singular_values = handoff_alignment_state["alignment_singular_values"].detach()

    return {
        "alignment_mode": str(alignment_mode),
        "alignment_strategy": str(diagnostic_alignment_state["alignment_strategy"]),
        "handoff_surface": handoff_target,
        "diagnostic_surface": diagnostic_target,
        "semantic_anchor_count": int(sender_anchor_ids.numel()),
        "semantic_anchor_strings": tuple(semantic_anchor_strings or ()),
        "semantic_anchor_ids_a": tuple(int(token_id) for token_id in sender_anchor_ids.tolist()),
        "semantic_anchor_ids_b": tuple(int(token_id) for token_id in receiver_anchor_ids.tolist()),
        "candidate_anchor_strings": tuple(candidate_anchor_strings or ()),
        "candidate_anchor_ids_a": tuple()
        if candidate_anchor_ids_a is None
        else tuple(int(token_id) for token_id in candidate_anchor_ids_a.tolist()),
        "candidate_anchor_ids_b": tuple()
        if candidate_anchor_ids_b is None
        else tuple(int(token_id) for token_id in candidate_anchor_ids_b.tolist()),
        "selected_anchor_indices": tuple()
        if selected_anchor_indices is None
        else tuple(int(index) for index in selected_anchor_indices.tolist()),
        "global_reasoning_layer_indices": tuple(int(index) for index in reasoning_layer_indices),
        "global_reasoning_layer_weights": tuple(float(weight) for weight in reasoning_layer_weights),
        "global_alignment_q": mapping_matrix.cpu(),
        "global_alignment_bias": bias_vector.cpu(),
        "global_alignment_backbone_q": orthogonal_q.cpu(),
        "global_alignment_residual": residual_matrix.cpu(),
        "alignment_singular_values": singular_values.cpu(),
        "handoff_alignment_q": handoff_mapping_matrix.cpu(),
        "handoff_alignment_bias": handoff_bias_vector.cpu(),
        "handoff_alignment_backbone_q": handoff_orthogonal_q.cpu(),
        "handoff_alignment_residual": handoff_residual_matrix.cpu(),
        "handoff_alignment_singular_values": handoff_singular_values.cpu(),
        "center_anchors": bool(diagnostic_alignment_state["center_anchors"]),
        "use_bias": bool(diagnostic_alignment_state["use_bias"]),
        "residual_norm_ratio": float(diagnostic_alignment_state["residual_norm_ratio"]),
        "bias_norm": float(diagnostic_alignment_state["bias_norm"]),
        "handoff_residual_norm_ratio": float(handoff_alignment_state["residual_norm_ratio"]),
        "handoff_bias_norm": float(handoff_alignment_state["bias_norm"]),
        "anchor_reconstruction_mse": float(diagnostic_alignment_state["anchor_reconstruction_mse"]),
        "handoff_anchor_reconstruction_mse": float(
            handoff_alignment_state["anchor_reconstruction_mse"]
        ),
        "anchor_pairwise_distance_distortion": float(
            diagnostic_alignment_state["anchor_pairwise_distance_distortion"]
        ),
        "anchor_cosine_structure_error": float(
            diagnostic_alignment_state["anchor_cosine_structure_error"]
        ),
        "handoff_anchor_pairwise_distance_distortion": float(
            handoff_alignment_state["anchor_pairwise_distance_distortion"]
        ),
        "handoff_anchor_cosine_structure_error": float(
            handoff_alignment_state["anchor_cosine_structure_error"]
        ),
        "pre_projection_state": diagnostic_alignment_state["pre_projection_state"],
        "post_projection_state": diagnostic_alignment_state["post_projection_state"],
        "handoff_pre_projection_state": handoff_alignment_state["pre_projection_state"],
        "handoff_post_projection_state": handoff_alignment_state["post_projection_state"],
        "sender_anchor_mean": diagnostic_alignment_state["sender_anchor_mean"],
        "receiver_anchor_mean": diagnostic_alignment_state["receiver_anchor_mean"],
        "handoff_receiver_anchor_mean": handoff_alignment_state["receiver_anchor_mean"],
    }


def load_or_compute_global_alignment_state(
    cfg: DictConfig,
    *,
    tokenizer_a: AutoTokenizer,
    tokenizer_b: AutoTokenizer,
    agent_a: AutoModelForCausalLM,
    agent_b: AutoModelForCausalLM,
) -> dict[str, Any]:
    sender_layer_count = len(agent_a.model.layers)
    receiver_layer_count = len(agent_b.model.layers)
    reasoning_layer_indices = _resolve_reasoning_layer_indices_from_counts(
        sender_layer_count,
        receiver_layer_count,
    )
    reasoning_layer_weights = _resolve_reasoning_layer_weights(cfg, reasoning_layer_indices)
    semantic_anchor_count = _semantic_anchor_count(cfg)
    alignment_strategy = _alignment_strategy(cfg)
    center_anchors = _alignment_center_anchors(cfg)
    use_bias = _alignment_use_bias(cfg)
    residual_lambda = _alignment_residual_lambda(cfg)
    residual_alpha = _alignment_residual_alpha(cfg)
    residual_max_norm_ratio = _alignment_residual_max_norm_ratio(cfg)
    handoff_target = _alignment_handoff_target(cfg)
    diagnostic_target = _alignment_diagnostic_target(cfg)
    adaptive_projection_enabled, adaptive_projection_strength, adaptive_projection_clip_std = (
        _alignment_adaptive_projection_settings(cfg)
    )
    alignment_cache_key = _build_alignment_cache_key(
        agent_a_model=str(cfg.agent_a_model),
        agent_b_model=str(cfg.agent_b_model),
        torch_dtype=str(cfg.torch_dtype),
        semantic_anchor_count=semantic_anchor_count,
        reasoning_layer_indices=reasoning_layer_indices,
        reasoning_layer_weights=reasoning_layer_weights,
        alignment_strategy=alignment_strategy,
        center_anchors=center_anchors,
        use_bias=use_bias,
        residual_lambda=residual_lambda,
        residual_alpha=residual_alpha,
        residual_max_norm_ratio=residual_max_norm_ratio,
        adaptive_projection_enabled=adaptive_projection_enabled,
        adaptive_projection_strength=adaptive_projection_strength,
        adaptive_projection_clip_std_multiplier=adaptive_projection_clip_std,
        handoff_target=handoff_target,
        diagnostic_target=diagnostic_target,
    )

    cached_alignment_state = _GLOBAL_ALIGNMENT_MEMORY_CACHE.get(alignment_cache_key)
    cache_hit = cached_alignment_state is not None
    cache_path = _alignment_cache_path(_alignment_cache_dir(cfg), alignment_cache_key)

    if cached_alignment_state is None:
        cached_alignment_state = _load_alignment_state_from_disk(cache_path)
        cache_hit = cached_alignment_state is not None

    if cached_alignment_state is None:
        cached_alignment_state = _compute_global_semantic_alignment(
            cfg=cfg,
            tokenizer_a=tokenizer_a,
            tokenizer_b=tokenizer_b,
            agent_a=agent_a,
            agent_b=agent_b,
            reasoning_layer_indices=reasoning_layer_indices,
            reasoning_layer_weights=reasoning_layer_weights,
            semantic_anchor_count=semantic_anchor_count,
        )
        torch.save(cached_alignment_state, cache_path)

    _GLOBAL_ALIGNMENT_MEMORY_CACHE[alignment_cache_key] = cached_alignment_state
    return {
        **cached_alignment_state,
        "global_alignment_cache_key": alignment_cache_key,
        "global_alignment_cache_hit": cache_hit,
        "global_alignment_cache_path": str(cache_path),
    }


def _dataset_validation_size(cfg: Optional[DictConfig], dataset_name: str) -> Optional[int]:
    datasets_cfg = getattr(cfg, "datasets", None)
    dataset_cfg = getattr(datasets_cfg, str(dataset_name).lower(), None)
    value = getattr(dataset_cfg, "validation_size", None)
    return None if value is None else int(value)


def _build_handoff_adapter_cache_key(
    cfg: DictConfig,
    *,
    alignment_cache_key: Any,
    reasoning_layer_indices: Sequence[int],
    reasoning_layer_weights: Sequence[float],
) -> tuple[Any, ...]:
    adaptive_projection_enabled, adaptive_projection_strength, adaptive_projection_clip_std = (
        _alignment_adaptive_projection_settings(cfg)
    )
    return (
        _HANDOFF_ADAPTER_CACHE_VERSION,
        str(cfg.agent_a_model),
        str(cfg.agent_b_model),
        str(cfg.torch_dtype),
        alignment_cache_key,
        tuple(int(index) for index in reasoning_layer_indices),
        tuple(round(float(weight), 8) for weight in reasoning_layer_weights),
        _handoff_adapter_dataset_name(cfg),
        _handoff_adapter_train_split(cfg),
        _handoff_adapter_train_limit(cfg),
        _handoff_adapter_max_length(cfg),
        _handoff_adapter_strategy(cfg),
        round(_handoff_adapter_regularization(cfg), 10),
        round(_handoff_adapter_residual_alpha(cfg), 8),
        round(_handoff_adapter_residual_max_norm_ratio(cfg), 8),
        _handoff_adapter_center(cfg),
        _handoff_adapter_use_bias(cfg),
        bool(adaptive_projection_enabled),
        round(float(adaptive_projection_strength), 8),
        round(float(adaptive_projection_clip_std), 8),
    )


def _handoff_adapter_cache_path(cache_dir: Path, cache_key: tuple[Any, ...]) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(
        json.dumps(cache_key, sort_keys=False, default=list).encode("utf-8")
    ).hexdigest()
    return cache_dir / f"handoff_adapter_{digest}.pt"


def _load_handoff_adapter_state_from_disk(cache_path: Path) -> Optional[dict[str, Any]]:
    if not cache_path.is_file():
        return None
    try:
        cached_state = torch.load(cache_path, map_location="cpu")
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(cached_state, dict):
        return None
    if "mapping_matrix" not in cached_state:
        return None
    return cached_state


def _handoff_adapter_fit_config(cfg: DictConfig) -> HandoffAdapterFitConfig:
    adaptive_projection_enabled, adaptive_projection_strength, adaptive_projection_clip_std = (
        _alignment_adaptive_projection_settings(cfg)
    )
    return HandoffAdapterFitConfig(
        strategy=_handoff_adapter_strategy(cfg),
        regularization=_handoff_adapter_regularization(cfg),
        residual_alpha=_handoff_adapter_residual_alpha(cfg),
        residual_max_norm_ratio=_handoff_adapter_residual_max_norm_ratio(cfg),
        center=_handoff_adapter_center(cfg),
        use_bias=_handoff_adapter_use_bias(cfg),
        max_length=_handoff_adapter_max_length(cfg),
        adaptive_projection_strength=(
            adaptive_projection_strength if adaptive_projection_enabled else 0.0
        ),
        adaptive_projection_clip_std_multiplier=adaptive_projection_clip_std,
    )


def _handoff_adapter_base_alignment_state(state: dict[str, Any]) -> dict[str, Any]:
    return _handoff_alignment_state_from_pipeline_state(state)


def load_or_train_handoff_adapter_state(
    cfg: DictConfig,
    state: dict[str, Any],
) -> dict[str, Any]:
    if not _handoff_adapter_enabled(cfg):
        return {
            "handoff_adapter_enabled": False,
            "handoff_adapter_status": "disabled",
            "handoff_adapter_cache_hit": None,
            "handoff_adapter_cache_path": "",
            "handoff_adapter_state": None,
        }

    reasoning_layer_indices = tuple(state["global_reasoning_layer_indices"])
    reasoning_layer_weights = tuple(state["global_reasoning_layer_weights"])
    cache_key = _build_handoff_adapter_cache_key(
        cfg,
        alignment_cache_key=state.get("global_alignment_cache_key"),
        reasoning_layer_indices=reasoning_layer_indices,
        reasoning_layer_weights=reasoning_layer_weights,
    )
    cache_path = _handoff_adapter_cache_path(_handoff_adapter_cache_dir(cfg), cache_key)
    cached_state = _HANDOFF_ADAPTER_MEMORY_CACHE.get(cache_key)
    cache_hit = cached_state is not None
    if cached_state is None:
        cached_state = _load_handoff_adapter_state_from_disk(cache_path)
        cache_hit = cached_state is not None

    if cached_state is None and _handoff_adapter_train_on_missing(cfg):
        dataset_name = _handoff_adapter_dataset_name(cfg)
        train_split = _handoff_adapter_train_split(cfg)
        train_rows = get_dataloader(
            dataset_name,
            limit=_handoff_adapter_train_limit(cfg),
            split=train_split,
            validation_size=_dataset_validation_size(cfg, dataset_name),
        )
        prompts = [pick_field(row, ("question", "problem")) for row in train_rows]
        cached_state = fit_handoff_adapter_state(
            prompts=prompts,
            tokenizer_a=state["tokenizer_a"],
            tokenizer_b=state["tokenizer_b"],
            agent_a=state["agent_a"],
            agent_b=state["agent_b"],
            base_alignment_state=_handoff_adapter_base_alignment_state(state),
            reasoning_layer_indices=reasoning_layer_indices,
            reasoning_layer_weights=reasoning_layer_weights,
            fit_config=_handoff_adapter_fit_config(cfg),
        )
        torch.save(cached_state, cache_path)
        cache_hit = False

    if cached_state is None:
        return {
            "handoff_adapter_enabled": False,
            "handoff_adapter_status": "missing",
            "handoff_adapter_cache_hit": False,
            "handoff_adapter_cache_path": str(cache_path),
            "handoff_adapter_state": None,
        }

    _HANDOFF_ADAPTER_MEMORY_CACHE[cache_key] = cached_state
    return {
        "handoff_adapter_enabled": True,
        "handoff_adapter_status": "loaded" if cache_hit else "trained",
        "handoff_adapter_cache_hit": cache_hit,
        "handoff_adapter_cache_path": str(cache_path),
        "handoff_adapter_state": cached_state,
        "handoff_adapter_training_prompt_count": cached_state.get("training_prompt_count"),
        "handoff_adapter_training_token_count": cached_state.get("training_token_count"),
        "handoff_adapter_training_reconstruction_mse": cached_state.get(
            "training_reconstruction_mse"
        ),
        "handoff_adapter_training_mean_cosine_similarity": cached_state.get(
            "training_mean_cosine_similarity"
        ),
    }


def apply_handoff_adapter(
    handoff_step: torch.Tensor,
    state: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    adapter_state = state.get("handoff_adapter_state")
    if adapter_state is None or not bool(state.get("handoff_adapter_enabled", False)):
        return handoff_step, {
            "handoff_adapter_applied": False,
            "handoff_adapter_delta_norm": None,
        }
    adapted = apply_alignment(handoff_step, adapter_state)
    delta = adapted.float() - handoff_step.float()
    delta_norm = float(
        torch.linalg.vector_norm(delta.reshape(delta.shape[0], -1), dim=-1)
        .mean()
        .detach()
        .cpu()
        .item()
    )
    return adapted.to(dtype=handoff_step.dtype), {
        "handoff_adapter_applied": True,
        "handoff_adapter_delta_norm": delta_norm,
    }


def apply_embedding_manifold_projection(
    handoff_step: torch.Tensor,
    cfg: DictConfig,
    state: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not _embedding_manifold_enabled(cfg):
        return handoff_step, {
            "embedding_manifold_applied": False,
            "embedding_manifold_delta_norm": None,
            "embedding_manifold_mean_top_similarity": None,
            "embedding_manifold_unique_token_count": None,
        }
    agent_b = state["agent_b"]
    projected, projection_metrics = project_to_embedding_manifold(
        handoff_step,
        agent_b.get_input_embeddings().weight,
        top_k=_embedding_manifold_top_k(cfg),
        temperature=_embedding_manifold_temperature(cfg),
        blend=_embedding_manifold_blend(cfg),
        normalize=_embedding_manifold_normalize(cfg),
        chunk_size=_embedding_manifold_chunk_size(cfg),
    )
    delta = projected.float() - handoff_step.float()
    delta_norm = float(
        torch.linalg.vector_norm(delta.reshape(delta.shape[0], -1), dim=-1)
        .mean()
        .detach()
        .cpu()
        .item()
    )
    return projected.to(dtype=handoff_step.dtype), {
        "embedding_manifold_applied": True,
        "embedding_manifold_delta_norm": delta_norm,
        **projection_metrics,
    }


def attach_latent_forward(agent_model: AutoModelForCausalLM) -> None:
    if bool(getattr(agent_model, "_lxp_latent_forward_attached", False)):
        return
    agent_model._lxp_original_forward = agent_model.forward

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
    agent_model._lxp_latent_forward_attached = True


def _get_pipeline_state(cfg: DictConfig) -> dict[str, Any]:
    global _PIPELINE_STATE, _PIPELINE_STATE_KEY

    state_key = _pipeline_state_key(cfg)
    if _PIPELINE_STATE is None or _PIPELINE_STATE_KEY != state_key:
        tokenizer_a = AutoTokenizer.from_pretrained(cfg.agent_a_model, trust_remote_code=True)
        tokenizer_b = AutoTokenizer.from_pretrained(cfg.agent_b_model, trust_remote_code=True)
        agent_a = load_agent(
            cfg.agent_a_model,
            torch_dtype=cfg.torch_dtype,
            device_map=cfg.device_map,
        )
        agent_b = load_agent(
            cfg.agent_b_model,
            torch_dtype=cfg.torch_dtype,
            device_map=cfg.device_map,
        )
        attach_latent_forward(agent_a)
        _PIPELINE_STATE = {
            "tokenizer_a": tokenizer_a,
            "tokenizer_b": tokenizer_b,
            "agent_a": agent_a,
            "agent_b": agent_b,
        }
        _PIPELINE_STATE_KEY = state_key

    if _PIPELINE_STATE is None:
        raise RuntimeError("Pipeline state failed to initialize")

    agent_a = _PIPELINE_STATE["agent_a"]
    agent_b = _PIPELINE_STATE["agent_b"]
    cached_alignment_state = load_or_compute_global_alignment_state(
        cfg,
        tokenizer_a=_PIPELINE_STATE["tokenizer_a"],
        tokenizer_b=_PIPELINE_STATE["tokenizer_b"],
        agent_a=agent_a,
        agent_b=agent_b,
    )
    state_with_alignment = {
        **_PIPELINE_STATE,
        **cached_alignment_state,
    }
    handoff_adapter_state = load_or_train_handoff_adapter_state(
        cfg,
        state_with_alignment,
    )
    return {
        **state_with_alignment,
        **handoff_adapter_state,
    }


def initialize_hybrid_pipeline(cfg: DictConfig) -> None:
    _get_pipeline_state(cfg)


def _compute_receiver_reference_handoff(
    *,
    prompt: str,
    state: dict[str, Any],
    reasoning_layer_indices: Sequence[int],
    reasoning_layer_weights: Sequence[float],
) -> torch.Tensor:
    del reasoning_layer_indices, reasoning_layer_weights
    tokenizer_b = state["tokenizer_b"]
    agent_b = state["agent_b"]
    agent_b_device = next(agent_b.parameters()).device
    encoded_b = tokenizer_b(prompt, return_tensors="pt")
    input_ids_b = encoded_b["input_ids"].to(agent_b_device)
    with torch.no_grad():
        receiver_input_embeddings = agent_b.get_input_embeddings()(input_ids_b)
    return receiver_input_embeddings[:, -1:, :]


def _alignment_state_from_pipeline_state(
    state: dict[str, Any],
) -> dict[str, Any]:
    mapping_matrix = state.get("global_alignment_q", state.get("mapping_matrix"))
    mapping_bias = state.get("global_alignment_bias", state.get("mapping_bias"))
    return {
        "mapping_matrix": mapping_matrix,
        "mapping_bias": mapping_bias,
        "pre_projection_state": state.get("pre_projection_state"),
        "post_projection_state": state.get("post_projection_state"),
        "alignment_strategy": state.get("alignment_strategy", state.get("alignment_mode", "unknown")),
        "orthogonal_q": state.get("global_alignment_backbone_q", state.get("orthogonal_q", mapping_matrix)),
        "residual_matrix": state.get("global_alignment_residual", state.get("residual_matrix")),
        "residual_norm_ratio": state.get("residual_norm_ratio"),
        "bias_norm": state.get("bias_norm"),
    }


def _handoff_alignment_state_from_pipeline_state(
    state: dict[str, Any],
) -> dict[str, Any]:
    mapping_matrix = state.get("handoff_alignment_q", state.get("global_alignment_q"))
    mapping_bias = state.get("handoff_alignment_bias", state.get("global_alignment_bias"))
    return {
        "mapping_matrix": mapping_matrix,
        "mapping_bias": mapping_bias,
        "pre_projection_state": state.get(
            "handoff_pre_projection_state",
            state.get("pre_projection_state"),
        ),
        "post_projection_state": state.get(
            "handoff_post_projection_state",
            state.get("post_projection_state"),
        ),
        "alignment_strategy": state.get("alignment_strategy", state.get("alignment_mode")),
        "orthogonal_q": state.get(
            "handoff_alignment_backbone_q",
            state.get("global_alignment_backbone_q", mapping_matrix),
        ),
        "residual_matrix": state.get("handoff_alignment_residual"),
        "residual_norm_ratio": state.get(
            "handoff_residual_norm_ratio",
            state.get("residual_norm_ratio"),
        ),
        "bias_norm": state.get("handoff_bias_norm", state.get("bias_norm")),
    }


def _apply_prompt_calibration(
    handoff_step: torch.Tensor,
    receiver_reference_handoff: torch.Tensor,
    *,
    strength: float,
    max_norm_ratio: float,
) -> tuple[torch.Tensor, float]:
    correction = (receiver_reference_handoff - handoff_step).float()
    correction = correction * float(strength)
    correction_norm = float(
        torch.linalg.vector_norm(correction.reshape(correction.shape[0], -1), dim=-1)
        .mean()
        .item()
    )
    if max_norm_ratio > 0.0:
        handoff_norm = torch.linalg.vector_norm(
            handoff_step.float().reshape(handoff_step.shape[0], -1),
            dim=-1,
        ).mean()
        max_norm = float(handoff_norm.item()) * float(max_norm_ratio)
        if correction_norm > max_norm > 0.0:
            correction = correction * (max_norm / max(correction_norm, 1e-8))
            correction_norm = max_norm
    calibrated = handoff_step + correction.to(
        device=handoff_step.device,
        dtype=handoff_step.dtype,
    )
    return calibrated, correction_norm


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
    correction_norm = float(
        torch.linalg.vector_norm(correction.reshape(correction.shape[0], -1), dim=-1)
        .mean()
        .item()
    )
    if max_norm_ratio > 0.0:
        handoff_norm = torch.linalg.vector_norm(
            handoff_step.float().reshape(handoff_step.shape[0], -1),
            dim=-1,
        ).mean()
        max_norm = float(handoff_norm.item()) * float(max_norm_ratio)
        if correction_norm > max_norm > 0.0:
            correction = correction * (max_norm / max(correction_norm, 1e-8))
            correction_norm = max_norm
    calibrated = handoff_step + correction.to(
        device=handoff_step.device,
        dtype=handoff_step.dtype,
    )
    return calibrated, correction_norm


def get_global_alignment_metadata(cfg: DictConfig) -> dict[str, Any]:
    state = _get_pipeline_state(cfg)
    return {
        "alignment_mode": state["alignment_mode"],
        "alignment_strategy": state.get("alignment_strategy", state["alignment_mode"]),
        "semantic_anchor_count": int(state["semantic_anchor_count"]),
        "semantic_anchor_preview": tuple(state["semantic_anchor_strings"][:10]),
        "semantic_anchor_ids_a_preview": tuple(state["semantic_anchor_ids_a"][:10]),
        "semantic_anchor_ids_b_preview": tuple(state["semantic_anchor_ids_b"][:10]),
        "candidate_anchor_count": len(state.get("candidate_anchor_strings", ())),
        "selected_anchor_indices_preview": tuple(state.get("selected_anchor_indices", ())[:10]),
        "reasoning_layer_indices": tuple(state["global_reasoning_layer_indices"]),
        "reasoning_layer_weights": tuple(state["global_reasoning_layer_weights"]),
        "q_global_shape": tuple(state["global_alignment_q"].shape),
        "handoff_q_shape": tuple(
            state.get("handoff_alignment_q", state["global_alignment_q"]).shape
        ),
        "global_alignment_bias_shape": tuple(state["global_alignment_bias"].shape),
        "handoff_alignment_bias_shape": tuple(
            state.get("handoff_alignment_bias", state["global_alignment_bias"]).shape
        ),
        "residual_norm_ratio": float(state.get("residual_norm_ratio", 0.0)),
        "handoff_residual_norm_ratio": float(
            state.get("handoff_residual_norm_ratio", state.get("residual_norm_ratio", 0.0))
        ),
        "bias_norm": float(state.get("bias_norm", 0.0)),
        "handoff_bias_norm": float(state.get("handoff_bias_norm", state.get("bias_norm", 0.0))),
        "handoff_surface": state.get("handoff_surface", _alignment_handoff_target(cfg)),
        "diagnostic_surface": state.get("diagnostic_surface", _alignment_diagnostic_target(cfg)),
        "global_alignment_cache_hit": bool(state["global_alignment_cache_hit"]),
        "global_alignment_cache_path": str(state["global_alignment_cache_path"]),
    }


def extract_reasoning_trace(cfg: DictConfig, prompt: Optional[str] = None) -> dict[str, Any]:
    if prompt is None:
        prompt = cfg.default_prompt

    complexity_factor = estimate_problem_complexity(prompt, cfg)
    state = _get_pipeline_state(cfg)
    tokenizer_a = state["tokenizer_a"]
    agent_a = state["agent_a"]
    reasoning_layer_indices = tuple(state["global_reasoning_layer_indices"])
    reasoning_layer_weights = tuple(state["global_reasoning_layer_weights"])
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
    consensus_hidden_states = _aggregate_hidden_layers(
        sender_reasoning_layers,
        reasoning_layer_weights,
    )
    alignment_state = _handoff_alignment_state_from_pipeline_state(state)

    latent_pooling_mode = _latent_pooling_mode(cfg)
    current_latent_step = _pool_latent_handoff_step(
        consensus_hidden_states,
        attention_mask_a,
        pooling_mode=latent_pooling_mode,
    )
    continuous_position_ids = position_ids_a[:, -1:] + 1
    effective_latent_steps = _scale_integration_points(
        int(cfg.latent_steps), complexity_factor
    )

    dynamics_mode = _dynamics_mode(cfg)
    rotary_emb = getattr(agent_a.model, "rotary_emb", None)
    dynamics = TransformerBlockDynamics(agent_a.model.layers[0], rotary_emb=rotary_emb)
    dynamics.set_context(position_ids=continuous_position_ids)

    continuous_trajectory, time_space = _build_latent_trajectory(
        dynamics_mode=dynamics_mode,
        dynamics=dynamics,
        current_latent_step=current_latent_step,
        point_count=effective_latent_steps,
    )
    aligned_continuous_trajectory = apply_alignment(
        continuous_trajectory,
        alignment_state,
    )

    return {
        "prompt": prompt,
        "complexity_factor": complexity_factor,
        "alignment_mode": state["alignment_mode"],
        "alignment_strategy": state.get("alignment_strategy", state["alignment_mode"]),
        "handoff_surface": state.get("handoff_surface", _alignment_handoff_target(cfg)),
        "diagnostic_surface": state.get("diagnostic_surface", _alignment_diagnostic_target(cfg)),
        "dynamics_mode": dynamics_mode,
        "latent_pooling": latent_pooling_mode,
        "semantic_anchor_count": int(state["semantic_anchor_count"]),
        "semantic_anchor_preview": semantic_anchor_strings[:10],
        "reasoning_layer_weights": reasoning_layer_weights,
        "latent_trajectory_steps": effective_latent_steps,
        "reasoning_layer_indices": reasoning_layer_indices,
        "continuous_trajectory": continuous_trajectory.detach().cpu(),
        "aligned_continuous_trajectory": aligned_continuous_trajectory.detach().cpu(),
        "handoff_latent": continuous_trajectory[-1].detach().cpu(),
        "pre_alignment_handoff": continuous_trajectory[-1].detach().cpu(),
        "post_alignment_handoff": aligned_continuous_trajectory[-1].detach().cpu(),
        "time_space": time_space.detach().cpu(),
        "procrustes_q_shape": tuple(state["global_alignment_q"].shape),
        "handoff_q_shape": tuple(
            state.get("handoff_alignment_q", state["global_alignment_q"]).shape
        ),
        "alignment_bias_shape": tuple(state["global_alignment_bias"].shape),
        "global_alignment_cache_hit": bool(state["global_alignment_cache_hit"]),
    }


def run_hybrid_pipeline(
    cfg: DictConfig,
    prompt: Optional[str] = None,
    *,
    collect_alignment_metrics: bool = False,
    target_answer_text: Optional[str] = None,
    alignment_q_override: Optional[Any] = None,
    alignment_mode_override: Optional[str] = None,
) -> dict[str, Any]:
    """
    Executes the end-to-end Latent Exchange Protocol (LXP) pipeline for a single prompt.

    This function handles the entire lifecycle of a multi-agent latent handoff:
    1. Agent A encodes the prompt and its internal reasoning hidden states are captured.
    2. A continuous trajectory is generated using Neural ODEs (`TransformerBlockDynamics`).
    3. The continuous thoughts are aligned using an Orthogonal Procrustes geometric bridge.
    4. The compressed, aligned thought vector (along with Agent A's KV cache) is injected
       into Agent B (the Actor) as a latent prefix.
    5. Agent B decodes the task.

    Args:
        cfg: The main configuration dictionary specifying models, latent steps, and alignment strategies.
        prompt: The input math or coding problem string.
        collect_alignment_metrics: If true, runs expensive baseline checks for tensor drift.
        target_answer_text: Optional ground truth string.
        alignment_q_override: Pre-computed Procrustes matrix Q to use instead of calculating on the fly.
        alignment_mode_override: Bypass config strategy (e.g., force 'orthogonal' or 'ridge').

    Returns:
        A dictionary containing the generated metrics, trajectories, handoff tensors, and final decoded output.
    """
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
    reasoning_layer_weights = tuple(state["global_reasoning_layer_weights"])
    semantic_anchor_strings = tuple(state["semantic_anchor_strings"])
    prompt_calibration_enabled, prompt_calibration_strength, prompt_calibration_max_norm = (
        _prompt_calibration_settings(cfg)
    )
    dynamics_mode = _dynamics_mode(cfg)
    latent_pooling_mode = _latent_pooling_mode(cfg)
    latent_prefix_mode = _latent_prefix_mode(cfg)
    sequence_prefix = latent_prefix_mode == "sequence"
    receiver_context_mode = _receiver_context_mode(cfg)
    receiver_context_latent_position = _receiver_context_latent_position(cfg)
    trace_events: list[dict[str, Any]] = []

    encoded = tokenizer_a(prompt, return_tensors="pt")
    agent_a_device = next(agent_a.parameters()).device
    input_ids_a = encoded["input_ids"].to(agent_a_device)
    attention_mask_a = encoded["attention_mask"].to(agent_a_device)
    position_ids_a = _build_position_ids(attention_mask_a)

    # The sender KV cache is consumed only by the scalar-prefix path (direct cache
    # transfer plus the discrete fallback) and by the before_context receiver layout;
    # in sequence mode with after_context it is dead weight that scales with prompt
    # length, so skip materializing it.
    sender_kv_needed = (not sequence_prefix) or (
        receiver_context_mode != "none" and receiver_context_latent_position == "before_context"
    )
    with torch.no_grad():
        agent_a_outputs = agent_a.model(
            input_ids=input_ids_a,
            attention_mask=attention_mask_a,
            position_ids=position_ids_a,
            use_cache=sender_kv_needed,
            output_hidden_states=True,
            return_dict=True,
        )

    reasoner_last_hidden_state = agent_a_outputs.last_hidden_state
    kv_cache_a = _normalize_kv_cache(agent_a_outputs.past_key_values)
    kv_cache_a_layer_count = _kv_cache_layer_count(kv_cache_a)
    sender_reasoning_stack = agent_a_outputs.hidden_states
    if sender_reasoning_stack is None:
        raise ValueError("Agent A did not return hidden states for reasoning-layer alignment")
    agent_b_device = next(agent_b.parameters()).device
    sender_reasoning_layers = _select_hidden_layers(
        sender_reasoning_stack, reasoning_layer_indices
    )
    consensus_hidden_states = _aggregate_hidden_layers(
        sender_reasoning_layers,
        reasoning_layer_weights,
    )
    if sequence_prefix:
        # The fallback path that reads the last hidden state is scalar-mode only;
        # drop the full per-layer stack before receiver work to cap peak memory.
        reasoner_last_hidden_state = None
    del sender_reasoning_layers, sender_reasoning_stack, agent_a_outputs
    base_handoff_alignment_state = _handoff_alignment_state_from_pipeline_state(state)
    if alignment_q_override is None:
        alignment_state: Any = base_handoff_alignment_state
    elif isinstance(alignment_q_override, torch.Tensor):
        alignment_state = {
            "mapping_matrix": alignment_q_override.detach().cpu(),
            "mapping_bias": torch.zeros(
                (1, int(alignment_q_override.shape[-1])),
                dtype=torch.float32,
            ),
            "alignment_strategy": "override_matrix",
        }
    elif isinstance(alignment_q_override, dict):
        alignment_state = _handoff_alignment_state_from_pipeline_state(alignment_q_override)
    else:
        raise TypeError("alignment_q_override must be None, a tensor, or an alignment-state dictionary")
    alignment_mode = (
        str(alignment_mode_override)
        if alignment_mode_override is not None
        else str(
            alignment_state.get(
                "alignment_strategy",
                state.get("alignment_strategy", state["alignment_mode"]),
            )
        )
    )
    current_latent_step = _pool_latent_handoff_step(
        consensus_hidden_states,
        attention_mask_a,
        pooling_mode=latent_pooling_mode,
    )
    if sequence_prefix and int(consensus_hidden_states.shape[0]) == 1:
        valid_tokens = attention_mask_a[0].to(dtype=torch.bool)
        handoff_latent_source = consensus_hidden_states[:, valid_tokens, :]
    elif sequence_prefix:
        handoff_latent_source = consensus_hidden_states
    else:
        handoff_latent_source = current_latent_step
    trace_events.append(
        _trace_tensor_event(
            operation="sender_consensus",
            tensor=handoff_latent_source,
            source_surface="reasoner_hidden_consensus",
            target_surface="reasoner_hidden_consensus",
            model_id=str(cfg.agent_a_model),
            diagnostics={
                "reasoning_layer_indices": reasoning_layer_indices,
                "latent_pooling": latent_pooling_mode,
                "latent_prefix_mode": latent_prefix_mode,
            },
        )
    )
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

    continuous_trajectory, time_space = _build_latent_trajectory(
        dynamics_mode=dynamics_mode,
        dynamics=dynamics,
        current_latent_step=current_latent_step,
        point_count=effective_latent_steps,
    )
    trace_events.append(
        _trace_tensor_event(
            operation="latent_dynamics",
            tensor=continuous_trajectory,
            source_surface="reasoner_hidden_consensus",
            target_surface="reasoner_hidden_consensus",
            model_id=str(cfg.agent_a_model),
            diagnostics={
                "dynamics_mode": dynamics_mode,
                "latent_trajectory_steps": effective_latent_steps,
            },
        )
    )
    aligned_continuous_trajectory = apply_alignment(
        continuous_trajectory,
        alignment_state,
    )
    current_latent_step = continuous_trajectory[-1]
    if not sequence_prefix:
        handoff_latent_source = current_latent_step

    integration_duration = _time_simulated_integration(
        dynamics_mode=dynamics_mode,
        dynamics=dynamics,
        current_latent_step=current_latent_step,
        point_count=effective_simulated_steps,
    )

    agent_b_embed_dtype = agent_b.get_input_embeddings().weight.dtype
    agent_b_embed_dim = agent_b.get_input_embeddings().weight.shape[-1]
    aligned_handoff_step = apply_alignment(handoff_latent_source, alignment_state)
    aligned_handoff_step, adapter_metrics = apply_handoff_adapter(
        aligned_handoff_step,
        state,
    )
    aligned_handoff_step, manifold_metrics = apply_embedding_manifold_projection(
        aligned_handoff_step,
        cfg,
        state,
    )
    handoff_status = "ok"
    handoff_surface = (
        "input_embedding_sequence"
        if sequence_prefix
        else state.get("handoff_surface", _alignment_handoff_target(cfg))
    )
    if aligned_handoff_step.shape[-1] != agent_b_embed_dim:
        handoff_status = "dimension_mismatch"
        raise ValueError(
            "Consensus latent handoff dimension "
            f"{aligned_handoff_step.shape[-1]} does not match Agent B input dimension "
            f"{agent_b_embed_dim}"
        )
    trace_events.append(
        _trace_tensor_event(
            operation="handoff_alignment",
            tensor=aligned_handoff_step,
            source_surface="reasoner_hidden_consensus",
            target_surface=handoff_surface,
            model_id=f"{cfg.agent_a_model} -> {cfg.agent_b_model}",
            diagnostics={
                "alignment_strategy": alignment_state.get("alignment_strategy", alignment_mode),
                "residual_norm_ratio": alignment_state.get("residual_norm_ratio"),
                "bias_norm": alignment_state.get("bias_norm"),
                "handoff_adapter_applied": adapter_metrics["handoff_adapter_applied"],
                "handoff_adapter_delta_norm": adapter_metrics["handoff_adapter_delta_norm"],
                "embedding_manifold_applied": manifold_metrics["embedding_manifold_applied"],
                "embedding_manifold_delta_norm": manifold_metrics["embedding_manifold_delta_norm"],
            },
        )
    )
    calibration_bias_norm = 0.0
    receiver_reference_handoff_cpu: Optional[torch.Tensor] = None
    receiver_reference_handoff_for_metrics: Optional[torch.Tensor] = None
    if prompt_calibration_enabled or collect_alignment_metrics:
        receiver_reference_handoff = _compute_receiver_reference_handoff_sequence(
            prompt=prompt,
            state=state,
            target_steps=int(aligned_handoff_step.shape[1]),
        ).to(device=agent_b_device, dtype=aligned_handoff_step.dtype)
        receiver_reference_handoff_for_metrics = receiver_reference_handoff
        receiver_reference_handoff_cpu = receiver_reference_handoff.detach().cpu()
        if prompt_calibration_enabled:
            if sequence_prefix:
                aligned_handoff_step, calibration_bias_norm = _apply_local_sequence_calibration(
                    handoff_latent_source,
                    aligned_handoff_step,
                    receiver_reference_handoff,
                    strength=prompt_calibration_strength,
                    max_norm_ratio=prompt_calibration_max_norm,
                )
            else:
                aligned_handoff_step, calibration_bias_norm = _apply_prompt_calibration(
                    aligned_handoff_step,
                    receiver_reference_handoff,
                    strength=prompt_calibration_strength,
                    max_norm_ratio=prompt_calibration_max_norm,
                )
    handoff_step = aligned_handoff_step.to(
        device=agent_b_device,
        dtype=agent_b_embed_dtype,
    )
    prefix_kv_cache_a = None if sequence_prefix else kv_cache_a
    if prefix_kv_cache_a is None and _should_use_receiver_context(
        receiver_context_mode,
        sender_kv_cache_transferred=False,
    ):
        # With no sender cache offered, the latent-only prefix statuses are fixed and
        # the receiver-context path below rebuilds the prefix anyway — skip the
        # otherwise-discarded full receiver forward over the latent prefix.
        sender_prefix_state = None
        kv_cache_transferred = False
        kv_cache_status = "not_provided"
        kv_cache_reason = "no_cache_provided"
        receiver_context_status = "not_used"
        receiver_context_reason = "latent_only"
        receiver_context_token_count = 0
        receiver_context_latent_position_value = "not_applicable"
        active_kv_cache_transferred = False
        active_kv_cache_status = "not_provided"
        active_kv_cache_reason = "no_cache_provided"
        active_kv_cache_source = "none"
    else:
        sender_prefix_state = prepare_latent_prefix_state(
            model=agent_b,
            handoff_step=handoff_step,
            kv_cache=prefix_kv_cache_a,
        )
        kv_cache_transferred = bool(sender_prefix_state["kv_cache_transferred"])
        kv_cache_status = str(
            sender_prefix_state.get(
                "kv_cache_status",
                "transferred" if kv_cache_transferred else "unsupported",
            )
        )
        kv_cache_reason = str(sender_prefix_state.get("kv_cache_reason", ""))
        receiver_context_status = str(sender_prefix_state.get("receiver_context_status", "not_used"))
        receiver_context_reason = str(sender_prefix_state.get("receiver_context_reason", "latent_only"))
        receiver_context_token_count = int(sender_prefix_state.get("receiver_context_token_count", 0))
        receiver_context_latent_position_value = str(
            sender_prefix_state.get("receiver_context_latent_position", "not_applicable")
        )
        active_kv_cache_transferred = bool(
            sender_prefix_state.get("active_kv_cache_transferred", kv_cache_transferred)
        )
        active_kv_cache_status = str(sender_prefix_state.get("active_kv_cache_status", kv_cache_status))
        active_kv_cache_reason = str(sender_prefix_state.get("active_kv_cache_reason", kv_cache_reason))
        active_kv_cache_source = str(sender_prefix_state.get("active_kv_cache_source", "provided_cache"))

    if _should_use_receiver_context(
        receiver_context_mode,
        sender_kv_cache_transferred=kv_cache_transferred,
    ):
        context_reason = (
            "forced_prompt_prefix"
            if receiver_context_mode == "prompt_prefix"
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
            latent_position=receiver_context_latent_position,
        )
        receiver_context_status = str(prefix_state.get("receiver_context_status", "used_prompt_prefix"))
        receiver_context_reason = str(prefix_state.get("receiver_context_reason", context_reason))
        receiver_context_token_count = int(prefix_state.get("receiver_context_token_count", 0))
        receiver_context_latent_position_value = str(
            prefix_state.get("receiver_context_latent_position", receiver_context_latent_position)
        )
        active_kv_cache_transferred = bool(prefix_state.get("active_kv_cache_transferred", False))
        active_kv_cache_status = str(prefix_state.get("active_kv_cache_status", "not_provided"))
        active_kv_cache_reason = str(prefix_state.get("active_kv_cache_reason", ""))
        active_kv_cache_source = str(prefix_state.get("active_kv_cache_source", "receiver_context"))
    else:
        prefix_state = sender_prefix_state
        if sequence_prefix:
            prefix_state = _maybe_append_answer_suffix_to_prefix_state(
                model=agent_b,
                tokenizer=tokenizer_b,
                cfg=cfg,
                prefix_state=prefix_state,
            )

    outputs_b = prefix_state["outputs"]
    raw_handoff_entropy = float(
        _compute_logits_entropy(outputs_b.logits[:, -1, :]).mean().detach().cpu().item()
    )
    handoff_uncertainty = raw_handoff_entropy / max(complexity_factor, 1e-8)
    fallback_triggered = (
        (not sequence_prefix)
        and confidence_gate_enabled
        and extra_discrete_steps > 0
        and handoff_uncertainty > uncertainty_threshold
    )
    executed_discrete_fallback_steps = 0
    fallback_token_ids: list[int] = []

    if fallback_triggered:
        current_latent_step, kv_cache_a, attention_mask_a, fallback_token_ids = _run_discrete_reasoning_fallback(
            agent_a=agent_a,
            current_latent_step=current_latent_step,
            reasoner_last_hidden_state=reasoner_last_hidden_state,
            kv_cache_a=kv_cache_a,
            attention_mask_a=attention_mask_a,
            reasoning_layer_indices=reasoning_layer_indices,
            reasoning_layer_weights=reasoning_layer_weights,
            extra_discrete_steps=extra_discrete_steps,
        )
        executed_discrete_fallback_steps = len(fallback_token_ids)
        kv_cache_a_layer_count = _kv_cache_layer_count(kv_cache_a)
        aligned_handoff_step = apply_alignment(current_latent_step, alignment_state)
        aligned_handoff_step, adapter_metrics = apply_handoff_adapter(
            aligned_handoff_step,
            state,
        )
        aligned_handoff_step, manifold_metrics = apply_embedding_manifold_projection(
            aligned_handoff_step,
            cfg,
            state,
        )
        if aligned_handoff_step.shape[-1] != agent_b_embed_dim:
            raise ValueError(
                "Fallback handoff dimension "
                f"{aligned_handoff_step.shape[-1]} does not match Agent B input dimension "
                f"{agent_b_embed_dim}"
            )
        if prompt_calibration_enabled and receiver_reference_handoff_for_metrics is not None:
            aligned_handoff_step, calibration_bias_norm = _apply_prompt_calibration(
                aligned_handoff_step,
                receiver_reference_handoff_for_metrics,
                strength=prompt_calibration_strength,
                max_norm_ratio=prompt_calibration_max_norm,
            )
        handoff_step = aligned_handoff_step.to(device=agent_b_device, dtype=agent_b_embed_dtype)
        sender_prefix_state = prepare_latent_prefix_state(
            model=agent_b,
            handoff_step=handoff_step,
            kv_cache=kv_cache_a,
        )
        kv_cache_transferred = bool(sender_prefix_state["kv_cache_transferred"])
        kv_cache_status = str(sender_prefix_state.get("kv_cache_status", kv_cache_status))
        kv_cache_reason = str(sender_prefix_state.get("kv_cache_reason", kv_cache_reason))
        active_kv_cache_transferred = bool(
            sender_prefix_state.get("active_kv_cache_transferred", kv_cache_transferred)
        )
        active_kv_cache_status = str(sender_prefix_state.get("active_kv_cache_status", kv_cache_status))
        active_kv_cache_reason = str(sender_prefix_state.get("active_kv_cache_reason", kv_cache_reason))
        active_kv_cache_source = str(sender_prefix_state.get("active_kv_cache_source", "provided_cache"))
        if _should_use_receiver_context(
            receiver_context_mode,
            sender_kv_cache_transferred=kv_cache_transferred,
        ):
            context_reason = (
                "forced_prompt_prefix"
                if receiver_context_mode == "prompt_prefix"
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
                latent_position=receiver_context_latent_position,
            )
            receiver_context_status = str(prefix_state.get("receiver_context_status", "used_prompt_prefix"))
            receiver_context_reason = str(prefix_state.get("receiver_context_reason", context_reason))
            receiver_context_token_count = int(prefix_state.get("receiver_context_token_count", 0))
            receiver_context_latent_position_value = str(
                prefix_state.get("receiver_context_latent_position", receiver_context_latent_position)
            )
            active_kv_cache_transferred = bool(prefix_state.get("active_kv_cache_transferred", False))
            active_kv_cache_status = str(prefix_state.get("active_kv_cache_status", "not_provided"))
            active_kv_cache_reason = str(prefix_state.get("active_kv_cache_reason", ""))
            active_kv_cache_source = str(prefix_state.get("active_kv_cache_source", "receiver_context"))
        else:
            prefix_state = sender_prefix_state
        outputs_b = prefix_state["outputs"]

    total_reasoning_steps = effective_latent_steps + executed_discrete_fallback_steps
    pre_alignment_l2_distance: Optional[float] = None
    pre_alignment_cosine_distance: Optional[float] = None
    post_alignment_l2_distance: Optional[float] = None
    post_alignment_cosine_distance: Optional[float] = None

    if collect_alignment_metrics:
        if receiver_reference_handoff_for_metrics is None:
            receiver_reference_handoff_for_metrics = _compute_receiver_reference_handoff_sequence(
                prompt=prompt,
                state=state,
                target_steps=int(handoff_step.shape[1]),
            ).to(device=agent_b_device, dtype=handoff_step.dtype)
            receiver_reference_handoff_cpu = receiver_reference_handoff_for_metrics.detach().cpu()
        backbone_mapping = alignment_state.get(
            "orthogonal_q",
            alignment_state.get("mapping_matrix"),
        )
        backbone_alignment_state = {
            "mapping_matrix": backbone_mapping,
        }
        backbone_handoff_step = apply_alignment(handoff_latent_source, backbone_alignment_state).to(
            device=agent_b_device,
            dtype=handoff_step.dtype,
        )
        pre_alignment_l2_distance = float(
            _normalized_l2_distance(backbone_handoff_step, receiver_reference_handoff_for_metrics)
            .mean()
            .detach()
            .cpu()
            .item()
        )
        pre_alignment_cosine_distance = float(
            _cosine_distance(backbone_handoff_step, receiver_reference_handoff_for_metrics)
            .mean()
            .detach()
            .cpu()
            .item()
        )
        post_alignment_l2_distance = float(
            _normalized_l2_distance(handoff_step, receiver_reference_handoff_for_metrics)
            .mean()
            .detach()
            .cpu()
            .item()
        )
        post_alignment_cosine_distance = float(
            _cosine_distance(handoff_step, receiver_reference_handoff_for_metrics)
            .mean()
            .detach()
            .cpu()
            .item()
        )

    decode_metrics = greedy_decode_from_prefix(
        model=agent_b,
        tokenizer=tokenizer_b,
        prefix_state=prefix_state,
        max_new_tokens=int(cfg.max_new_tokens),
        stop_regex=_decode_stop_regex(cfg),
    )
    decoded_text = str(decode_metrics["decoded_text"])
    answer_metrics = compute_answer_metrics_from_prefix(
        model=agent_b,
        tokenizer=tokenizer_b,
        prefix_state=prefix_state,
        answer_text=target_answer_text,
        answer_variants=_answer_metric_variants(cfg, target_answer_text),
    )
    decode_status = "decoded" if decoded_text.strip() else "empty_decode"
    trace_events.append(
        _trace_tensor_event(
            operation="actor_decode",
            tensor=handoff_step,
            source_surface=handoff_surface,
            target_surface="text",
            model_id=str(cfg.agent_b_model),
            kv_cache_status=kv_cache_status,
            decode_status=decode_status,
            diagnostics={
                "kv_cache_status": kv_cache_status,
                "kv_cache_reason": kv_cache_reason,
                "active_kv_cache_status": active_kv_cache_status,
                "active_kv_cache_reason": active_kv_cache_reason,
                "active_kv_cache_source": active_kv_cache_source,
                "decode_status": decode_status,
                "generated_tokens": int(decode_metrics["generated_tokens"]),
                "receiver_context_status": receiver_context_status,
                "receiver_context_reason": receiver_context_reason,
                "receiver_context_token_count": receiver_context_token_count,
                "receiver_context_latent_position": receiver_context_latent_position_value,
            },
        )
    )
    return {
        "decoded_text": decoded_text,
        "generated_tokens": int(decode_metrics["generated_tokens"]),
        "decode_status": decode_status,
        "handoff_status": handoff_status,
        "handoff_surface": handoff_surface,
        "kv_cache_status": kv_cache_status,
        "kv_cache_reason": kv_cache_reason,
        "active_kv_cache_transferred": active_kv_cache_transferred,
        "active_kv_cache_status": active_kv_cache_status,
        "active_kv_cache_reason": active_kv_cache_reason,
        "active_kv_cache_source": active_kv_cache_source,
        "receiver_context_status": receiver_context_status,
        "receiver_context_reason": receiver_context_reason,
        "receiver_context_token_count": receiver_context_token_count,
        "trace_events": trace_events,
        "complexity_factor": complexity_factor,
        "dynamics_mode": dynamics_mode,
        "latent_pooling": latent_pooling_mode,
        "latent_prefix_mode": latent_prefix_mode,
        "receiver_context_mode": receiver_context_mode,
        "receiver_context_latent_position": receiver_context_latent_position,
        "alignment_mode": alignment_mode,
        "alignment_strategy": alignment_state.get("alignment_strategy", alignment_mode),
        "semantic_anchor_count": int(state["semantic_anchor_count"]),
        "semantic_anchor_preview": semantic_anchor_strings[:10],
        "reasoning_layer_weights": reasoning_layer_weights,
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
        "final_latent_shape": tuple(handoff_latent_source.shape),
        "final_latent_dtype": str(handoff_latent_source.dtype),
        "procrustes_q_shape": tuple(state["global_alignment_q"].shape),
        "handoff_q_shape": tuple(
            state.get("handoff_alignment_q", state["global_alignment_q"]).shape
        ),
        "alignment_bias_shape": tuple(state["global_alignment_bias"].shape),
        "alignment_residual_norm_ratio": alignment_state.get("residual_norm_ratio"),
        "alignment_bias_norm": alignment_state.get("bias_norm"),
        "prompt_calibration_enabled": prompt_calibration_enabled,
        "prompt_calibration_bias_norm": calibration_bias_norm,
        "handoff_adapter_enabled": bool(state.get("handoff_adapter_enabled", False)),
        "handoff_adapter_status": state.get("handoff_adapter_status"),
        "handoff_adapter_applied": adapter_metrics["handoff_adapter_applied"],
        "handoff_adapter_delta_norm": adapter_metrics["handoff_adapter_delta_norm"],
        "handoff_adapter_cache_hit": state.get("handoff_adapter_cache_hit"),
        "handoff_adapter_cache_path": state.get("handoff_adapter_cache_path"),
        "handoff_adapter_training_prompt_count": state.get(
            "handoff_adapter_training_prompt_count"
        ),
        "handoff_adapter_training_token_count": state.get(
            "handoff_adapter_training_token_count"
        ),
        "handoff_adapter_training_reconstruction_mse": state.get(
            "handoff_adapter_training_reconstruction_mse"
        ),
        "handoff_adapter_training_mean_cosine_similarity": state.get(
            "handoff_adapter_training_mean_cosine_similarity"
        ),
        "embedding_manifold_enabled": _embedding_manifold_enabled(cfg),
        "embedding_manifold_applied": manifold_metrics["embedding_manifold_applied"],
        "embedding_manifold_delta_norm": manifold_metrics["embedding_manifold_delta_norm"],
        "embedding_manifold_mean_top_similarity": manifold_metrics[
            "embedding_manifold_mean_top_similarity"
        ],
        "embedding_manifold_unique_token_count": manifold_metrics[
            "embedding_manifold_unique_token_count"
        ],
        "pre_alignment_handoff": handoff_latent_source.detach().cpu(),
        "post_alignment_handoff": handoff_step.detach().cpu(),
        "aligned_continuous_trajectory": aligned_continuous_trajectory.detach().cpu(),
        "alignment_metrics_collected": collect_alignment_metrics,
        "receiver_reference_handoff": receiver_reference_handoff_cpu,
        "pre_alignment_l2_distance": pre_alignment_l2_distance,
        "pre_alignment_cosine_distance": pre_alignment_cosine_distance,
        "post_alignment_l2_distance": post_alignment_l2_distance,
        "post_alignment_cosine_distance": post_alignment_cosine_distance,
        "answer_token_count": answer_metrics["answer_token_count"],
        "answer_nll": answer_metrics["answer_nll"],
        "answer_perplexity": answer_metrics["answer_perplexity"],
        "kv_cache_length": kv_cache_a_layer_count,
        "kv_cache_transferred": kv_cache_transferred,
        "continuous_integration_seconds": integration_duration,
        "continuous_integration_50_steps_seconds": integration_duration,
        "global_alignment_cache_hit": bool(state["global_alignment_cache_hit"]),
        "global_alignment_cache_path": str(state["global_alignment_cache_path"]),
        "alignment_override_applied": alignment_q_override is not None,
    }


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig) -> None:
    complexity_factor = estimate_problem_complexity(cfg.default_prompt, cfg)
    alignment_metadata = get_global_alignment_metadata(cfg)
    print(f"Estimated complexity factor: {complexity_factor:.2f}")
    print(f"Alignment mode: {alignment_metadata['alignment_mode']}")
    print(f"Semantic anchor count: {alignment_metadata['semantic_anchor_count']}")
    print(f"Reasoning layer weights: {alignment_metadata['reasoning_layer_weights']}")
    print(f"Global Q shape: {alignment_metadata['q_global_shape']}")
    print(f"Handoff Q shape: {alignment_metadata['handoff_q_shape']}")
    print(f"Handoff surface: {alignment_metadata['handoff_surface']}")
    print(f"Global alignment cache hit: {alignment_metadata['global_alignment_cache_hit']}")
    outputs = run_hybrid_pipeline(cfg)
    print(f"Pipeline complexity factor: {outputs['complexity_factor']:.2f}")
    print(f"Dynamics mode: {outputs['dynamics_mode']}")
    print(f"Latent pooling: {outputs['latent_pooling']}")
    print(f"Handoff status: {outputs['handoff_status']}")
    print(
        f"Receiver context: {outputs['receiver_context_status']} "
        f"({outputs['receiver_context_reason']})"
    )
    print(f"KV cache status: {outputs['kv_cache_status']} ({outputs['kv_cache_reason']})")
    print(f"Decode status: {outputs['decode_status']}")
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
