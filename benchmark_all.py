"""Comparative benchmark runner for phase 1 and phase 3 evaluation suites."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

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
    _resolve_reasoning_layer_indices_from_counts,
    _resolve_reasoning_layer_weights,
    _run_actor_handoff,
    _select_hidden_layers,
    _should_use_receiver_context,
    apply_embedding_manifold_projection,
    apply_handoff_adapter,
    load_agent,
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
    build_heterogeneous_transfer_report,
    build_phase1_gate_report,
    build_phase3_gate_report,
    build_runtime_smoke_report,
    build_semantic_smoke_report,
    build_standard_row_base,
    build_transfer_comparison_report,
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
from src.utils.latent_blame import build_latent_provenance_report
from src.utils.model_compat import (
    load_model_architecture_summary,
    load_model_pair_compatibility,
    summarize_model_config,
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
DEFAULT_SEMANTIC_SMOKE_LIMIT = 3
DEFAULT_SEMANTIC_SMOKE_MAX_NEW_TOKENS = 128
DEFAULT_SEMANTIC_SMOKE_REASONER_MAX_NEW_TOKENS = 320
DEFAULT_HETERO_SMOKE_REASONER_MAX_NEW_TOKENS = 640
DEFAULT_SEMANTIC_SMOKE_GENERATED_ADAPTER_TRAIN_LIMIT = 16
DEFAULT_HETERO_SMOKE_GENERATED_ADAPTER_TRAIN_LIMIT = 32
DEFAULT_SEMANTIC_SMOKE_METHODS = (
    "text_text_hybrid",
    "generated_latent_handoff",
)
DEFAULT_SEMANTIC_SMOKE_BASELINE_METHODS = (
    "text_text_hybrid",
)
DEFAULT_SEMANTIC_SMOKE_LATENT_METHODS = (
    "generated_latent_handoff",
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
    "text_text_hybrid",
    "generated_latent_handoff",
)
DEFAULT_HETERO_SMOKE_BASELINE_METHODS = (
    "text_text_hybrid",
)
DEFAULT_HETERO_SMOKE_LATENT_METHODS = (
    "generated_latent_handoff",
)
EVAL_MANIFEST_SCHEMA_VERSION = 2
ARTIFACT_MANIFEST_SCHEMA_VERSION = 1
TEXT_BASELINE_METHODS = frozenset(
    (
        "pure_text_cot",
        "text_text_hybrid",
        "token_context_handoff",
        "verified_token_context_handoff",
        "sender_answer_text_handoff",
    )
)
GENERATED_TRAJECTORY_ADAPTER_INPUT_SPACES = frozenset(("aligned", "raw"))
GENERATED_TRAJECTORY_ADAPTER_TARGET_ALIGNMENTS = frozenset(("character", "linear"))
GENERATED_LATENT_METHODS = frozenset(
    (
        "generated_latent_handoff",
        "prompt_generated_latent_handoff",
        "generated_context_latent_handoff",
    )
)
_SENDER_TRACE_STATE: Optional[dict[str, Any]] = None
_SENDER_TRACE_STATE_KEY: Optional[tuple[Any, ...]] = None
GSM8K_FINAL_ANSWER_REGEX = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")
FINAL_ANSWER_MARKER_REGEX = re.compile(
    r"final\s+answer\s*[:=]\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)
FINAL_ANSWER_BOXED_REGEX = re.compile(
    r"final\s+answer\s*[:=]\s*(?:[$*`_\s]+)?\\boxed\s*\{\s*(-?\d[\d,]*(?:\.\d+)?)\s*\}",
    re.IGNORECASE,
)
FINAL_ANSWER_COMPLETE_REGEX = re.compile(
    r"(?:"
    r"final\s+answer\s*[:=]\s*(?:[$*`_\s]+)?-?\d[\d,]*(?:\.\d+)?(?:[$*`_\s]+|[^\d,.]|\.(?!\d))"
    r"|"
    r"final\s+answer\s*[:=]\s*(?:[$*`_\s]+)?\\boxed\s*\{\s*-?\d[\d,]*(?:\.\d+)?\s*\}"
    r")",
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
    fallback_max_new_tokens = int(getattr(cfg, "max_new_tokens", 0))
    return int(
        getattr(
            benchmark_cfg,
            "text_hybrid_reasoning_max_new_tokens",
            fallback_max_new_tokens,
        )
    )


def _reasoner_generation_max_new_tokens(cfg: Any) -> int:
    return max(_text_hybrid_reasoning_max_new_tokens(cfg), int(getattr(cfg, "max_new_tokens", 0)))


def _answer_only_final_enabled(cfg: Any) -> bool:
    return bool(getattr(_benchmark_cfg(cfg), "answer_only_final", False))


def _sender_revision_cfg(cfg: Any) -> Any:
    return getattr(_benchmark_cfg(cfg), "sender_revision", None)


def _sender_revision_enabled(cfg: Any) -> bool:
    return bool(getattr(_sender_revision_cfg(cfg), "enabled", False))


def _sender_revision_max_new_tokens(cfg: Any) -> int:
    return max(1, int(getattr(_sender_revision_cfg(cfg), "max_new_tokens", 256)))


def _sender_revision_disagreement_verifier_enabled(cfg: Any) -> bool:
    return bool(
        getattr(_sender_revision_cfg(cfg), "disagreement_verifier_enabled", True)
    )


def _sender_revision_disagreement_verifier_max_new_tokens(cfg: Any) -> int:
    return max(
        1,
        int(
            getattr(
                _sender_revision_cfg(cfg),
                "disagreement_verifier_max_new_tokens",
                256,
            )
        ),
    )


def _sender_generation_cache_fingerprint(cfg: Any) -> tuple[Any, ...]:
    return (
        "sender_generation_v7",
        bool(_answer_only_final_enabled(cfg)),
        bool(_sender_revision_enabled(cfg)),
        int(_sender_revision_max_new_tokens(cfg)),
        bool(_sender_revision_disagreement_verifier_enabled(cfg)),
        int(_sender_revision_disagreement_verifier_max_new_tokens(cfg)),
    )


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
        "Solve carefully with compact arithmetic. Track quantities in chronological order. "
        "Write only essential equations, then end with exactly: Final answer: <answer>."
    )
    return _maybe_apply_chat_template(tokenizer, user_message)


def _format_reasoner_revision_prompt(
    prompt: str,
    initial_reasoning_text: str,
    tokenizer: Any = None,
) -> str:
    del initial_reasoning_text
    user_message = (
        f"{prompt}\n\n"
        "Solve again from scratch. First copy every numeric quantity and relationship "
        "from the problem exactly as written, especially phrases like 'less than', "
        "'more than', and 'times'. Then translate those copied relationships into "
        "equations and check each arithmetic operation. Write only the copied facts "
        "and essential equations, then end with exactly: Final answer: <answer>."
    )
    return _maybe_apply_chat_template(tokenizer, user_message)


def _format_reasoner_revision_decision_prompt(
    prompt: str,
    initial_answer: Optional[str],
    revision_answer: Optional[str],
    tokenizer: Any = None,
) -> str:
    candidate_b = revision_answer if revision_answer is not None else "missing or incomplete"
    user_message = (
        f"{prompt}\n\n"
        "Verify the final answer from the original problem only.\n"
        f"Candidate A answer: {initial_answer or 'missing'}\n"
        f"Candidate B answer: {candidate_b}\n\n"
        "Copy each numeric "
        "quantity and relationship exactly as written, translate the copied facts into "
        "equations, and check the arithmetic. Do not choose by candidate order. For "
        "GSM8K-style scoring, the final answer must be one scalar number; if the "
        "reasoning has multiple category counts and the question asks how many are "
        "left or how many total, sum the relevant categories. End with exactly: "
        "Final answer: <answer>."
    )
    return _maybe_apply_chat_template(tokenizer, user_message)


def _format_verified_final_answer_text(answer: str) -> str:
    return f"\n\nVerification decision:\nFinal answer: {answer}.\n"


def _format_text_cot_prompt(prompt: str, tokenizer: Any = None, cfg: Any = None) -> str:
    user_message = f"{prompt}\n\n{_final_answer_instruction(cfg)}"
    return _maybe_apply_chat_template(tokenizer, user_message)


def _decode_stop_regex(cfg: Any) -> Optional[re.Pattern[str]]:
    return FINAL_ANSWER_COMPLETE_REGEX if _answer_only_final_enabled(cfg) else None


def _sender_reasoning_status(token_ids: Sequence[int], text: str, cfg: Any) -> str:
    if FINAL_ANSWER_COMPLETE_REGEX.search(str(text)) is not None:
        return "complete"
    if len(token_ids) >= _reasoner_generation_max_new_tokens(cfg):
        return "max_tokens_without_final_answer"
    return "stopped_without_final_answer"


def _answer_metric_variants(cfg: Any, answer_text: Optional[str]) -> tuple[str, ...]:
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


def _format_sender_answer_text_handoff_prompt(
    sender_answer: str,
    tokenizer: Any = None,
) -> str:
    user_message = (
        "Verified upstream final answer:\n"
        f"{sender_answer}\n\n"
        "Copy that answer exactly. Return one line only.\n"
        "Final answer:"
    )
    return _maybe_apply_chat_template(tokenizer, user_message)


def _format_token_context_handoff_prompt(
    prompt: str,
    reasoning_text: str,
    tokenizer: Any = None,
    cfg: Any = None,
) -> str:
    instruction = (
        "Use the transferred token context from Agent A. Return exactly one line "
        "in this format: Final answer: <answer>."
        if _answer_only_final_enabled(cfg)
        else "Use the transferred token context from Agent A and give the final answer."
    )
    user_message = (
        f"{prompt}\n\n"
        f"Transferred token context from Agent A:\n{reasoning_text.strip()}\n\n"
        f"{instruction}"
    )
    return _maybe_apply_chat_template(tokenizer, user_message)


def _format_verified_token_context_handoff_prompt(
    sender_answer: str,
    reasoning_text: str,
    tokenizer: Any = None,
) -> str:
    user_message = (
        "Verified upstream final answer:\n"
        f"{sender_answer}\n\n"
        "Transferred token context from Agent A:\n"
        f"{reasoning_text.strip()}\n\n"
        "Use the verified upstream final answer as authoritative. Do not recompute.\n"
        "Return one line only.\n"
        "Final answer:"
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


def _trim_generated_ids_to_final_answer(tokenizer: Any, token_ids: Sequence[int]) -> list[int]:
    trimmed: list[int] = []
    for token_id in token_ids:
        trimmed.append(int(token_id))
        decoded = tokenizer.decode(trimmed, skip_special_tokens=True)
        if FINAL_ANSWER_COMPLETE_REGEX.search(decoded) is not None:
            return trimmed
    return trimmed


def _final_answer_marker_value(text: str) -> Optional[str]:
    marker_candidates = [
        (match.start(), match.group(1))
        for match in FINAL_ANSWER_MARKER_REGEX.finditer(str(text))
    ]
    marker_candidates.extend(
        (match.start(), match.group(1))
        for match in FINAL_ANSWER_BOXED_REGEX.finditer(str(text))
    )
    if not marker_candidates:
        return None
    return sorted(marker_candidates, key=lambda item: item[0])[-1][1].strip()


def _final_answer_marker_tail(text: str) -> str:
    marker_matches = list(FINAL_ANSWER_MARKER_REGEX.finditer(str(text)))
    marker_matches.extend(FINAL_ANSWER_BOXED_REGEX.finditer(str(text)))
    if not marker_matches:
        return ""
    latest_match = sorted(marker_matches, key=lambda item: item.start())[-1]
    return str(text)[latest_match.end() :].splitlines()[0].strip()


def _final_answer_tail_needs_scalar_verification(text: str) -> bool:
    tail = _final_answer_marker_tail(text)
    if not tail:
        return False
    return re.search(r"[A-Za-z]", tail) is not None


def _generate_agent_a_token_ids(
    formatted_prompt: str,
    cfg: Any,
    state: dict[str, Any],
    *,
    max_new_tokens: int,
) -> list[int]:
    tokenizer_a = state["tokenizer_a"]
    agent_a = state["agent_a"]
    agent_a_device = next(agent_a.parameters()).device
    encoded = tokenizer_a(
        formatted_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(agent_a_device)
    attention_mask = encoded["attention_mask"].to(agent_a_device)
    suppress_tokens = [
        int(token_id)
        for token_id in (getattr(tokenizer_a, "all_special_ids", ()) or ())
        if token_id is not None
    ]
    original_forward = getattr(agent_a, "_lxp_original_forward", None)
    latent_forward = agent_a.forward
    if original_forward is not None:
        agent_a.forward = original_forward
    try:
        with torch.no_grad():
            generated = agent_a.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                suppress_tokens=suppress_tokens,
                pad_token_id=getattr(tokenizer_a, "eos_token_id", None),
            )
    finally:
        if original_forward is not None:
            agent_a.forward = latent_forward
    generated_ids = generated[0, input_ids.shape[1] :].detach().cpu().tolist()
    return _trim_generated_ids_to_final_answer(tokenizer_a, generated_ids)


def _frozen_sender_reasoning_text(state: Mapping[str, Any]) -> Optional[str]:
    row = state.get("_current_sample_row")
    if not isinstance(row, Mapping):
        return None
    for key in ("sender_reasoning_text", "sender_reasoning", "reasoning_text"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _generate_reasoner_metadata(prompt: str, cfg: Any, state: dict[str, Any]) -> dict[str, Any]:
    tokenizer_a = state["tokenizer_a"]
    frozen_reasoning_text = _frozen_sender_reasoning_text(state)
    if frozen_reasoning_text is not None:
        frozen_ids = tokenizer_a.encode(frozen_reasoning_text, add_special_tokens=False)
        frozen_answer = _final_answer_marker_value(frozen_reasoning_text)
        return {
            "token_ids": [int(token_id) for token_id in frozen_ids],
            "initial_token_ids": [int(token_id) for token_id in frozen_ids],
            "initial_reasoning_text": frozen_reasoning_text,
            "initial_predicted_answer": frozen_answer,
            "revision_enabled": False,
            "revision_applied": False,
            "revision_token_ids": [],
            "revision_reasoning_text": "",
            "revision_predicted_answer": None,
            "revision_decision_applied": False,
            "revision_decision_token_ids": [],
            "revision_decision_reasoning_text": "",
            "revision_decision_predicted_answer": None,
        }
    max_new_tokens = _reasoner_generation_max_new_tokens(cfg)
    cache = state.setdefault("_reasoner_token_ids_cache", {})
    cache_key = (
        str(prompt),
        int(max_new_tokens),
        _sender_generation_cache_fingerprint(cfg),
    )
    cached = cache.get(cache_key)
    if cached is not None:
        cached_metadata = dict(cached)
        cached_metadata["token_ids"] = list(cached_metadata.get("token_ids", ()))
        cached_metadata["initial_token_ids"] = list(
            cached_metadata.get("initial_token_ids", ())
        )
        cached_metadata["revision_token_ids"] = list(
            cached_metadata.get("revision_token_ids", ())
        )
        cached_metadata["revision_decision_token_ids"] = list(
            cached_metadata.get("revision_decision_token_ids", ())
        )
        return cached_metadata

    initial_ids = _generate_agent_a_token_ids(
        _format_reasoner_cot_prompt(prompt, tokenizer_a),
        cfg,
        state,
        max_new_tokens=max_new_tokens,
    )
    initial_text = tokenizer_a.decode(initial_ids, skip_special_tokens=True)
    initial_answer = _final_answer_marker_value(initial_text)
    revision_enabled = _sender_revision_enabled(cfg)
    revision_applied = False
    revision_ids: list[int] = []
    revision_text = ""
    revision_answer: Optional[str] = None
    decision_applied = False
    decision_ids: list[int] = []
    decision_text = ""
    decision_answer: Optional[str] = None
    token_ids = list(initial_ids)

    if revision_enabled:
        revision_ids = _generate_agent_a_token_ids(
            _format_reasoner_revision_prompt(prompt, initial_text, tokenizer_a),
            cfg,
            state,
            max_new_tokens=_sender_revision_max_new_tokens(cfg),
        )
        revision_text = tokenizer_a.decode(revision_ids, skip_special_tokens=True)
        revision_answer = _final_answer_marker_value(revision_text)
        answer_disagreement = (
            revision_answer is not None
            and initial_answer is not None
            and _normalize_numeric_answer(revision_answer)
            != _normalize_numeric_answer(initial_answer)
        )
        needs_scalar_verification = _final_answer_tail_needs_scalar_verification(
            revision_text if revision_answer is not None else initial_text
        )
        needs_decision = (
            revision_answer is None
            or answer_disagreement
            or needs_scalar_verification
        )
        if needs_decision and _sender_revision_disagreement_verifier_enabled(cfg):
            decision_ids = _generate_agent_a_token_ids(
                _format_reasoner_revision_decision_prompt(
                    prompt,
                    initial_answer,
                    revision_answer,
                    tokenizer_a,
                ),
                cfg,
                state,
                max_new_tokens=_sender_revision_disagreement_verifier_max_new_tokens(cfg),
            )
            decision_text = tokenizer_a.decode(decision_ids, skip_special_tokens=True)
            decision_answer = _final_answer_marker_value(decision_text)
        if revision_answer is not None:
            separator_ids = tokenizer_a.encode(
                "\n\nVerification:\n",
                add_special_tokens=False,
            )
            token_ids = [
                *[int(token_id) for token_id in initial_ids],
                *[int(token_id) for token_id in separator_ids],
                *[int(token_id) for token_id in revision_ids],
            ]
            revision_applied = True
        if decision_answer is not None:
            compact_decision_ids = tokenizer_a.encode(
                _format_verified_final_answer_text(decision_answer),
                add_special_tokens=False,
            )
            token_ids = [
                *[int(token_id) for token_id in initial_ids],
                *[int(token_id) for token_id in compact_decision_ids],
            ]
            decision_applied = True

    metadata = {
        "token_ids": [int(token_id) for token_id in token_ids],
        "initial_token_ids": [int(token_id) for token_id in initial_ids],
        "initial_reasoning_text": initial_text,
        "initial_predicted_answer": initial_answer,
        "revision_enabled": bool(revision_enabled),
        "revision_applied": bool(revision_applied),
        "revision_token_ids": [int(token_id) for token_id in revision_ids],
        "revision_reasoning_text": revision_text,
        "revision_predicted_answer": revision_answer,
        "revision_decision_applied": bool(decision_applied),
        "revision_decision_token_ids": [int(token_id) for token_id in decision_ids],
        "revision_decision_reasoning_text": decision_text,
        "revision_decision_predicted_answer": decision_answer,
    }
    cache[cache_key] = {
        **metadata,
        "token_ids": tuple(metadata["token_ids"]),
        "initial_token_ids": tuple(metadata["initial_token_ids"]),
        "revision_token_ids": tuple(metadata["revision_token_ids"]),
        "revision_decision_token_ids": tuple(metadata["revision_decision_token_ids"]),
    }
    return metadata


def _generate_reasoner_token_ids(prompt: str, cfg: Any, state: dict[str, Any]) -> list[int]:
    return list(_generate_reasoner_metadata(prompt, cfg, state)["token_ids"])


def _extract_gsm8k_target_answer(text: str) -> Optional[str]:
    match = GSM8K_FINAL_ANSWER_REGEX.search(text)
    return None if match is None else match.group(1)


def _extract_gsm8k_predicted_answer(text: str) -> Optional[str]:
    marker_candidates = [
        (match.start(), match.group(1))
        for match in FINAL_ANSWER_MARKER_REGEX.finditer(text)
    ]
    marker_candidates.extend(
        (match.start(), match.group(1))
        for match in FINAL_ANSWER_BOXED_REGEX.finditer(text)
    )
    if marker_candidates:
        return sorted(marker_candidates, key=lambda item: item[0])[-1][1]
    boxed_answer = extract_boxed_text(text)
    if boxed_answer is not None:
        return boxed_answer
    matches = NUMERIC_ANSWER_REGEX.findall(text)
    if not matches:
        return None
    return matches[-1]


def _normalize_numeric_answer(answer: Optional[str]) -> Optional[str]:
    normalized = normalize_answer(answer)
    if normalized is None:
        return None
    normalized = normalized.replace(",", "")
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", normalized):
        return normalized
    try:
        decimal_value = Decimal(normalized)
    except InvalidOperation:
        return normalized
    if decimal_value == 0:
        return "0"
    return format(decimal_value.normalize(), "f")


def _target_answer(dataset_name: str, row: Any) -> Optional[str]:
    if dataset_name in {"gsm8k", "long_context_handoff"}:
        return _extract_gsm8k_target_answer(pick_field(row, ("answer", "solution")))
    return extract_boxed_text(pick_field(row, ("solution", "answer")))


def _predicted_answer(dataset_name: str, decoded_text: str) -> Optional[str]:
    if dataset_name in {"gsm8k", "long_context_handoff"}:
        return _extract_gsm8k_predicted_answer(decoded_text)
    return extract_boxed_text(decoded_text)


def _answers_match(dataset_name: str, predicted_answer: Optional[str], target_answer: Optional[str]) -> bool:
    if dataset_name in {"gsm8k", "long_context_handoff"}:
        return _normalize_numeric_answer(predicted_answer) == _normalize_numeric_answer(target_answer)
    return normalize_answer(predicted_answer) == normalize_answer(target_answer)


def _reasoning_alignment_metadata(state: dict[str, Any]) -> tuple[tuple[int, ...], tuple[float, ...]]:
    return (
        tuple(state["global_reasoning_layer_indices"]),
        tuple(state["global_reasoning_layer_weights"]),
    )


def _model_layer_count_from_loaded_model(model: Any, *, model_id: str) -> int:
    summary = summarize_model_config(getattr(model, "config", None), model_id=model_id)
    if summary.num_hidden_layers is not None:
        return int(summary.num_hidden_layers)
    model_layers = getattr(getattr(model, "model", None), "layers", None)
    if model_layers is not None:
        return int(len(model_layers))
    raise ValueError(f"Unable to determine transformer layer count for {model_id}")


def _resolve_sender_trace_reasoning_metadata_from_layer_counts(
    cfg: Any,
    *,
    sender_layer_count: int,
    receiver_layer_count: Optional[int],
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    effective_receiver_layer_count = (
        int(receiver_layer_count)
        if receiver_layer_count is not None
        else int(sender_layer_count)
    )
    reasoning_layer_indices = _resolve_reasoning_layer_indices_from_counts(
        int(sender_layer_count),
        effective_receiver_layer_count,
    )
    reasoning_layer_weights = _resolve_reasoning_layer_weights(cfg, reasoning_layer_indices)
    return (
        tuple(int(index) for index in reasoning_layer_indices),
        tuple(float(weight) for weight in reasoning_layer_weights),
    )


def _sender_trace_state_key(cfg: Any) -> tuple[Any, ...]:
    return (
        str(cfg.agent_a_model),
        str(cfg.agent_b_model),
        str(cfg.torch_dtype),
        str(getattr(cfg, "device_map", "auto")),
        tuple(float(weight) for weight in getattr(getattr(cfg, "alignment", None), "reasoning_layer_weights", ())),
    )


def _get_sender_trace_state(cfg: Any) -> dict[str, Any]:
    global _SENDER_TRACE_STATE, _SENDER_TRACE_STATE_KEY

    state_key = _sender_trace_state_key(cfg)
    if _SENDER_TRACE_STATE is not None and _SENDER_TRACE_STATE_KEY == state_key:
        return _SENDER_TRACE_STATE

    tokenizer_a = AutoTokenizer.from_pretrained(cfg.agent_a_model, trust_remote_code=True)
    agent_a = load_agent(
        str(cfg.agent_a_model),
        torch_dtype=str(cfg.torch_dtype),
        device_map=str(getattr(cfg, "device_map", "auto")),
    )
    sender_layer_count = _model_layer_count_from_loaded_model(
        agent_a,
        model_id=str(cfg.agent_a_model),
    )
    receiver_layer_count: Optional[int]
    receiver_layer_count_source = "sender_fallback"
    try:
        receiver_summary = load_model_architecture_summary(str(cfg.agent_b_model))
        receiver_layer_count = receiver_summary.num_hidden_layers
        if receiver_layer_count is not None:
            receiver_layer_count_source = "receiver_config"
    except Exception:  # noqa: BLE001
        receiver_layer_count = None

    reasoning_layer_indices, reasoning_layer_weights = (
        _resolve_sender_trace_reasoning_metadata_from_layer_counts(
            cfg,
            sender_layer_count=sender_layer_count,
            receiver_layer_count=receiver_layer_count,
        )
    )
    _SENDER_TRACE_STATE = {
        "tokenizer_a": tokenizer_a,
        "agent_a": agent_a,
        "global_reasoning_layer_indices": reasoning_layer_indices,
        "global_reasoning_layer_weights": reasoning_layer_weights,
        "sender_trace_state_sender_layer_count": int(sender_layer_count),
        "sender_trace_state_receiver_layer_count": (
            None if receiver_layer_count is None else int(receiver_layer_count)
        ),
        "sender_trace_state_receiver_layer_count_source": receiver_layer_count_source,
        "_reasoner_token_ids_cache": {},
        "_generated_sender_consensus_cache": {},
    }
    _SENDER_TRACE_STATE_KEY = state_key
    return _SENDER_TRACE_STATE


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
        cached_cfg, cached_state = cached
        if "_current_sample_row" in state:
            cached_state["_current_sample_row"] = state["_current_sample_row"]
        return cached_cfg, cached_state
    variant_cfg = _alignment_variant_cfg(
        cfg,
        strategy=strategy,
        prompt_calibration_enabled=prompt_calibration_enabled,
    )
    variant_state = _get_pipeline_state(variant_cfg)
    for cache_name in (
        "_reasoner_token_ids_cache",
        "_generated_sender_consensus_cache",
    ):
        variant_state[cache_name] = state.setdefault(cache_name, {})
    if "_current_sample_row" in state:
        variant_state["_current_sample_row"] = state["_current_sample_row"]
    cache[cache_key] = (variant_cfg, variant_state)
    return variant_cfg, variant_state


def _generate_reasoner_text(prompt: str, cfg: Any, state: dict[str, Any]) -> str:
    tokenizer_a = state["tokenizer_a"]
    generated_ids = _generate_reasoner_metadata(prompt, cfg, state)["token_ids"]
    return tokenizer_a.decode(generated_ids, skip_special_tokens=True)


def _reasoner_metadata_for_text_hybrid(
    prompt: str,
    cfg: Any,
    state: dict[str, Any],
) -> dict[str, Any]:
    frozen_reasoning_text = _frozen_sender_reasoning_text(state)
    if frozen_reasoning_text is not None:
        generation_metadata = _generate_reasoner_metadata(prompt, cfg, state)
        generated_token_ids = [int(token_id) for token_id in generation_metadata["token_ids"]]
        return {
            "token_ids": generated_token_ids,
            "reasoning_text": frozen_reasoning_text,
            "trace_cache_hit": None,
            "trace_cache_path": "",
            "sender_revision_enabled": False,
            "sender_revision_applied": False,
            "sender_initial_predicted_answer": generation_metadata.get("initial_predicted_answer"),
            "sender_revision_predicted_answer": None,
            "sender_revision_decision_applied": False,
            "sender_revision_decision_predicted_answer": None,
        }
    trace_cache_path: Optional[Path] = None
    if _generated_trajectory_adapter_trace_cache_enabled(cfg):
        cache_key = _generated_trajectory_trace_cache_key(
            cfg,
            state,
            prompt,
            include_prompt=False,
        )
        trace_cache_path = _generated_trajectory_adapter_trace_cache_path(cfg, cache_key)
        cached_trace = _load_generated_trajectory_trace_from_disk(
            trace_cache_path,
            expected_cache_key=cache_key,
        )
        if cached_trace is not None:
            generated_token_ids = [
                int(token_id) for token_id in cached_trace["generated_token_ids"]
            ]
            return {
                "token_ids": generated_token_ids,
                "reasoning_text": str(cached_trace["generated_reasoning_text"]),
                "trace_cache_hit": True,
                "trace_cache_path": str(trace_cache_path),
                "sender_revision_enabled": bool(
                    cached_trace.get("sender_revision_enabled", False)
                ),
                "sender_revision_applied": bool(
                    cached_trace.get("sender_revision_applied", False)
                ),
                "sender_initial_predicted_answer": cached_trace.get(
                    "sender_initial_predicted_answer"
                ),
                "sender_revision_predicted_answer": cached_trace.get(
                    "sender_revision_predicted_answer"
                ),
                "sender_revision_decision_applied": bool(
                    cached_trace.get("sender_revision_decision_applied", False)
                ),
                "sender_revision_decision_predicted_answer": cached_trace.get(
                    "sender_revision_decision_predicted_answer"
                ),
            }

    tokenizer_a = state["tokenizer_a"]
    generation_metadata = _generate_reasoner_metadata(prompt, cfg, state)
    generated_token_ids = [int(token_id) for token_id in generation_metadata["token_ids"]]
    return {
        "token_ids": generated_token_ids,
        "reasoning_text": tokenizer_a.decode(generated_token_ids, skip_special_tokens=True),
        "trace_cache_hit": False if trace_cache_path is not None else None,
        "trace_cache_path": "" if trace_cache_path is None else str(trace_cache_path),
        "sender_revision_enabled": bool(generation_metadata["revision_enabled"]),
        "sender_revision_applied": bool(generation_metadata["revision_applied"]),
        "sender_initial_predicted_answer": generation_metadata.get("initial_predicted_answer"),
        "sender_revision_predicted_answer": generation_metadata.get("revision_predicted_answer"),
        "sender_revision_decision_applied": bool(
            generation_metadata["revision_decision_applied"]
        ),
        "sender_revision_decision_predicted_answer": generation_metadata.get(
            "revision_decision_predicted_answer"
        ),
    }


def _cache_key_metadata(cache_key: tuple[Any, ...]) -> list[Any]:
    return json.loads(json.dumps(cache_key, sort_keys=False, default=list))


def _stable_json_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=list, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _cache_key_digest(cache_key: tuple[Any, ...]) -> str:
    return _stable_json_digest(_cache_key_metadata(cache_key))


def _manifest_digest(manifest: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    return _stable_json_digest(unsigned)


def _resolved_sample_indices(limit: int, sample_indices: Optional[Sequence[int]]) -> list[int]:
    if sample_indices is None:
        return [int(index) for index in range(max(0, int(limit)))]
    return [int(index) for index in list(sample_indices)[: max(0, int(limit))]]


def _sample_fingerprint(
    row: Mapping[str, Any],
    *,
    sample_index: int,
) -> dict[str, Any]:
    prompt = pick_field(dict(row), ("question", "problem"))
    target = pick_field(dict(row), ("answer", "solution", "target"))
    return {
        "sample_index": int(sample_index),
        "prompt_sha256": _stable_json_digest(prompt),
        "target_sha256": _stable_json_digest(target),
        "prompt_char_count": len(prompt),
        "target_char_count": len(target),
    }


def _sample_fingerprints(
    samples: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    sample_indices: Optional[Sequence[int]],
) -> list[dict[str, Any]]:
    resolved_indices = _resolved_sample_indices(limit, sample_indices)
    fingerprints: list[dict[str, Any]] = []
    for row_index, row in enumerate(list(samples)[: max(0, int(limit))]):
        sample_index = (
            int(resolved_indices[row_index])
            if row_index < len(resolved_indices)
            else int(row_index)
        )
        fingerprints.append(_sample_fingerprint(row, sample_index=sample_index))
    return fingerprints


def _build_eval_manifest(
    *,
    suite_name: str,
    dataset_name: str,
    dataset_split: str,
    limit: int,
    sample_indices: Optional[Sequence[int]],
    methods: Sequence[str],
    agent_a_model: str,
    agent_b_model: str,
    seed: int,
    semantic_smoke: bool,
    mvp_smoke: bool,
    hetero_smoke: bool,
    max_new_tokens: Optional[int] = None,
    reasoner_max_new_tokens: Optional[int] = None,
    torch_dtype: Optional[str] = None,
    device_map: Optional[str] = None,
    generated_trajectory_adapter_identity: Optional[Mapping[str, Any]] = None,
    handoff_identity: Optional[Mapping[str, Any]] = None,
    sample_fingerprints: Optional[Sequence[Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    fingerprint_rows = [dict(row) for row in (sample_fingerprints or ())]
    manifest = {
        "manifest_schema_version": EVAL_MANIFEST_SCHEMA_VERSION,
        "suite": str(suite_name),
        "dataset": str(dataset_name),
        "dataset_split": str(dataset_split),
        "limit": int(limit),
        "sample_indices": _resolved_sample_indices(limit, sample_indices),
        "methods": [str(method) for method in methods],
        "agent_a_model": str(agent_a_model),
        "agent_b_model": str(agent_b_model),
        "seed": int(seed),
        "max_new_tokens": None if max_new_tokens is None else int(max_new_tokens),
        "reasoner_max_new_tokens": (
            None if reasoner_max_new_tokens is None else int(reasoner_max_new_tokens)
        ),
        "torch_dtype": None if torch_dtype is None else str(torch_dtype),
        "device_map": None if device_map is None else str(device_map),
        "generated_trajectory_adapter": (
            None
            if generated_trajectory_adapter_identity is None
            else dict(generated_trajectory_adapter_identity)
        ),
        "handoff": None if handoff_identity is None else dict(handoff_identity),
        "smoke_profile": {
            "semantic_smoke": bool(semantic_smoke),
            "mvp_smoke": bool(mvp_smoke),
            "hetero_smoke": bool(hetero_smoke),
        },
        "sample_fingerprints": fingerprint_rows,
        "sample_content_digest": _stable_json_digest(fingerprint_rows),
    }
    manifest["manifest_digest"] = _manifest_digest(manifest)
    return manifest


def _load_eval_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"Eval manifest must be a JSON object: {path}")
    schema_version = int(manifest.get("manifest_schema_version", 0))
    if schema_version < 1 or schema_version > EVAL_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported eval manifest schema version: "
            f"{manifest.get('manifest_schema_version')}"
        )
    expected_digest = manifest.get("manifest_digest")
    actual_digest = _manifest_digest(manifest)
    if expected_digest != actual_digest:
        raise ValueError(
            f"Eval manifest digest mismatch for {path}: "
            f"expected {expected_digest}, computed {actual_digest}"
        )
    if not isinstance(manifest.get("sample_indices"), list):
        raise ValueError("Eval manifest requires a sample_indices list")
    if not isinstance(manifest.get("methods"), list):
        raise ValueError("Eval manifest requires a methods list")
    return manifest


def _validate_eval_manifest_sample_lock(
    resolved_manifest: Mapping[str, Any],
    locked_manifest: Optional[Mapping[str, Any]],
) -> None:
    if locked_manifest is None:
        return
    expected_digest = locked_manifest.get("sample_content_digest")
    if expected_digest and resolved_manifest.get("sample_content_digest") != expected_digest:
        raise ValueError(
            "Eval manifest sample content digest mismatch: "
            f"expected {expected_digest}, "
            f"computed {resolved_manifest.get('sample_content_digest')}"
        )
    expected_fingerprints = locked_manifest.get("sample_fingerprints")
    if expected_fingerprints and resolved_manifest.get("sample_fingerprints") != expected_fingerprints:
        raise ValueError("Eval manifest sample fingerprints do not match resolved samples")


def _apply_eval_manifest_to_args(args: argparse.Namespace, manifest: Mapping[str, Any]) -> None:
    args.suite = str(manifest["suite"])
    args.dataset = str(manifest["dataset"])
    args.split = str(manifest["dataset_split"])
    args.limit = int(manifest["limit"])
    args.sample_indices = ",".join(str(index) for index in manifest["sample_indices"])
    args.methods = ",".join(str(method) for method in manifest["methods"])
    args.agent_a_model = str(manifest["agent_a_model"])
    args.agent_b_model = str(manifest["agent_b_model"])
    args.seed = int(manifest["seed"])
    if manifest.get("max_new_tokens") is not None:
        args.max_new_tokens = int(manifest["max_new_tokens"])
    if manifest.get("reasoner_max_new_tokens") is not None:
        args.reasoner_max_new_tokens = int(manifest["reasoner_max_new_tokens"])
    if manifest.get("torch_dtype") is not None:
        args.torch_dtype = str(manifest["torch_dtype"])
    if manifest.get("device_map") is not None:
        args.device_map = str(manifest["device_map"])
    generated_adapter = manifest.get("generated_trajectory_adapter")
    if isinstance(generated_adapter, Mapping):
        if generated_adapter.get("enabled") is not None:
            if bool(generated_adapter.get("enabled")):
                args.enable_generated_trajectory_adapter = True
                args.disable_generated_trajectory_adapter = False
            else:
                args.disable_generated_trajectory_adapter = True
                args.enable_generated_trajectory_adapter = False
        if generated_adapter.get("train_on_missing") is not None:
            if bool(generated_adapter.get("train_on_missing")):
                args.generated_trajectory_adapter_train_on_missing = True
                args.generated_trajectory_adapter_no_train_on_missing = False
            else:
                args.generated_trajectory_adapter_no_train_on_missing = True
                args.generated_trajectory_adapter_train_on_missing = False
        if generated_adapter.get("train_limit") is not None:
            args.generated_trajectory_adapter_train_limit = int(generated_adapter["train_limit"])
        if generated_adapter.get("train_split") is not None:
            args.generated_trajectory_adapter_train_split = str(generated_adapter["train_split"])
        if generated_adapter.get("input_space") is not None:
            args.generated_trajectory_adapter_input_space = str(generated_adapter["input_space"])
        if generated_adapter.get("source_mode") is not None:
            args.generated_trajectory_adapter_source_mode = str(generated_adapter["source_mode"])
        if generated_adapter.get("source_tail_tokens") is not None:
            args.generated_trajectory_adapter_source_tail_tokens = int(
                generated_adapter["source_tail_tokens"]
            )
        if generated_adapter.get("target_mode") is not None:
            args.generated_trajectory_adapter_target_mode = str(generated_adapter["target_mode"])
        if generated_adapter.get("target_alignment") is not None:
            args.generated_trajectory_adapter_target_alignment = str(
                generated_adapter["target_alignment"]
            )
        local_residual = generated_adapter.get("local_residual")
        if isinstance(local_residual, Mapping):
            if local_residual.get("enabled") is not None:
                if bool(local_residual.get("enabled")):
                    args.enable_generated_trajectory_local_residual = True
                    args.disable_generated_trajectory_local_residual = False
                else:
                    args.disable_generated_trajectory_local_residual = True
                    args.enable_generated_trajectory_local_residual = False
            if local_residual.get("top_k") is not None:
                args.generated_trajectory_local_residual_top_k = int(local_residual["top_k"])
            if local_residual.get("temperature") is not None:
                args.generated_trajectory_local_residual_temperature = float(
                    local_residual["temperature"]
                )
            if local_residual.get("blend") is not None:
                args.generated_trajectory_local_residual_blend = float(local_residual["blend"])
            if local_residual.get("max_memory_rows") is not None:
                args.generated_trajectory_local_residual_max_memory_rows = int(
                    local_residual["max_memory_rows"]
                )
        semantic_memory = generated_adapter.get("semantic_memory")
        if isinstance(semantic_memory, Mapping):
            if semantic_memory.get("enabled") is not None:
                if bool(semantic_memory.get("enabled")):
                    args.enable_generated_trajectory_semantic_memory = True
                    args.disable_generated_trajectory_semantic_memory = False
                else:
                    args.disable_generated_trajectory_semantic_memory = True
                    args.enable_generated_trajectory_semantic_memory = False
            if semantic_memory.get("min_similarity") is not None:
                args.generated_trajectory_semantic_memory_min_similarity = float(
                    semantic_memory["min_similarity"]
                )
            if semantic_memory.get("max_entries") is not None:
                args.generated_trajectory_semantic_memory_max_entries = int(
                    semantic_memory["max_entries"]
                )
    handoff = manifest.get("handoff")
    if isinstance(handoff, Mapping):
        if handoff.get("latent_pooling") is not None:
            args.latent_pooling = str(handoff["latent_pooling"])
        if handoff.get("receiver_context_mode") is not None:
            args.receiver_context_mode = str(handoff["receiver_context_mode"])
        if handoff.get("receiver_context_latent_position") is not None:
            args.receiver_context_latent_position = str(
                handoff["receiver_context_latent_position"]
            )
        embedding_manifold = handoff.get("embedding_manifold")
        if isinstance(embedding_manifold, Mapping):
            if embedding_manifold.get("enabled") is not None:
                if bool(embedding_manifold.get("enabled")):
                    args.enable_embedding_manifold = True
                    args.disable_embedding_manifold = False
                else:
                    args.disable_embedding_manifold = True
                    args.enable_embedding_manifold = False
            if embedding_manifold.get("top_k") is not None:
                args.embedding_manifold_top_k = int(embedding_manifold["top_k"])
            if embedding_manifold.get("blend") is not None:
                args.embedding_manifold_blend = float(embedding_manifold["blend"])
    smoke_profile = manifest.get("smoke_profile") or {}
    args.semantic_smoke = bool(smoke_profile.get("semantic_smoke", False))
    args.mvp_smoke = bool(smoke_profile.get("mvp_smoke", False))
    args.hetero_smoke = bool(smoke_profile.get("hetero_smoke", False))


def _build_artifact_manifest(
    *,
    report_output_path: Path,
    samples_output_path: Optional[Path] = None,
    summary_output_path: Optional[Path] = None,
    eval_manifest_output_path: Optional[Path] = None,
    eval_manifest: Optional[Mapping[str, Any]] = None,
    latent_provenance_report: Optional[Mapping[str, Any]] = None,
    prepared_adapters: Sequence[Mapping[str, Any]] = (),
    prepared_eval_traces: Optional[Mapping[str, Any]] = None,
    run_metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    cache_paths = {
        "sender_trace": [],
        "adapter": [],
        "adapter_training_rows": [],
    }
    if latent_provenance_report is not None:
        provenance_paths = latent_provenance_report.get("cache_paths") or {}
        for key in cache_paths:
            cache_paths[key].extend(str(path) for path in provenance_paths.get(key, []))
    for adapter in prepared_adapters:
        if adapter.get("cache_path"):
            cache_paths["adapter"].append(str(adapter["cache_path"]))
        if adapter.get("training_row_cache_path"):
            cache_paths["adapter_training_rows"].append(
                str(adapter["training_row_cache_path"])
            )
    if prepared_eval_traces is not None:
        for row in prepared_eval_traces.get("traces", []):
            if row.get("trace_cache_path"):
                cache_paths["sender_trace"].append(str(row["trace_cache_path"]))
    deduped_cache_paths = {
        key: sorted(dict.fromkeys(value))
        for key, value in cache_paths.items()
    }
    output_files = {
        "report": str(report_output_path),
    }
    if samples_output_path is not None:
        output_files["samples"] = str(samples_output_path)
    if summary_output_path is not None:
        output_files["summary"] = str(summary_output_path)
    if eval_manifest_output_path is not None:
        output_files["eval_manifest"] = str(eval_manifest_output_path)
    return {
        "artifact_manifest_schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "output_files": output_files,
        "eval_manifest_digest": (
            None if eval_manifest is None else eval_manifest.get("manifest_digest")
        ),
        "run_metadata": dict(run_metadata or {}),
        "cache_paths": deduped_cache_paths,
        "do_not_commit_cache_artifacts": True,
    }


def _git_metadata() -> dict[str, Any]:
    def _run_git(args: Sequence[str]) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", *args],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:  # noqa: BLE001
            return None
        return result.stdout.strip()

    status = _run_git(["status", "--short"])
    return {
        "commit": _run_git(["rev-parse", "HEAD"]),
        "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status),
        "dirty_paths": [] if not status else status.splitlines(),
    }


def _run_metadata() -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "argv": list(sys.argv),
        "git": _git_metadata(),
    }


def _sender_generated_trace_payload(
    result: dict[str, Any],
    *,
    cache_key: tuple[Any, ...],
) -> dict[str, Any]:
    return {
        "trace_cache_format_version": 2,
        "cache_key": _cache_key_metadata(cache_key),
        "cache_key_digest": _cache_key_digest(cache_key),
        "consensus_hidden_states": result["consensus_hidden_states"].detach().cpu(),
        "generated_token_ids": [int(token_id) for token_id in result["generated_token_ids"]],
        "generated_reasoning_text": str(result["generated_reasoning_text"]),
        "generated_reasoning_token_count": int(result["generated_reasoning_token_count"]),
        "generated_reasoning_status": str(result["generated_reasoning_status"]),
        "generated_reasoning_final_answer_marker": bool(
            result["generated_reasoning_final_answer_marker"]
        ),
        "generated_latent_includes_prompt": bool(result["generated_latent_includes_prompt"]),
        "sender_revision_enabled": bool(result.get("sender_revision_enabled", False)),
        "sender_revision_applied": bool(result.get("sender_revision_applied", False)),
        "sender_initial_predicted_answer": result.get("sender_initial_predicted_answer"),
        "sender_revision_predicted_answer": result.get("sender_revision_predicted_answer"),
        "sender_revision_decision_applied": bool(
            result.get("sender_revision_decision_applied", False)
        ),
        "sender_revision_decision_predicted_answer": result.get(
            "sender_revision_decision_predicted_answer"
        ),
    }


def _sender_generated_trace_result_from_payload(
    payload: dict[str, Any],
    *,
    state: dict[str, Any],
    cfg: Any,
    cache_path: Optional[Path],
) -> dict[str, Any]:
    agent_a = state["agent_a"]
    agent_a_device = next(agent_a.parameters()).device
    consensus_hidden_states = payload["consensus_hidden_states"].to(agent_a_device)
    latent_attention_mask = torch.ones(
        (consensus_hidden_states.shape[0], consensus_hidden_states.shape[1]),
        dtype=torch.long,
        device=agent_a_device,
    )
    pooling_mode = _latent_pooling_mode(cfg)
    reasoning_layer_indices, reasoning_layer_weights = _reasoning_alignment_metadata(state)
    generated_token_ids = [int(token_id) for token_id in payload["generated_token_ids"]]
    generated_text = str(payload["generated_reasoning_text"])
    generated_status = str(
        payload.get("generated_reasoning_status")
        or _sender_reasoning_status(generated_token_ids, generated_text, cfg)
    )
    return {
        "consensus_hidden_states": consensus_hidden_states,
        "current_latent_step": _pool_latent_handoff_step(
            consensus_hidden_states,
            latent_attention_mask,
            pooling_mode=pooling_mode,
        ),
        "attention_mask": latent_attention_mask,
        "latent_pooling": pooling_mode,
        "kv_cache_a": None,
        "reasoning_layer_indices": reasoning_layer_indices,
        "reasoning_layer_weights": reasoning_layer_weights,
        "generated_token_ids": generated_token_ids,
        "generated_reasoning_text": generated_text,
        "generated_reasoning_token_count": int(
            payload.get("generated_reasoning_token_count", len(generated_token_ids))
        ),
        "generated_latent_includes_prompt": bool(
            payload.get("generated_latent_includes_prompt", False)
        ),
        "generated_reasoning_status": generated_status,
        "generated_reasoning_final_answer_marker": bool(
            payload.get("generated_reasoning_final_answer_marker")
            if "generated_reasoning_final_answer_marker" in payload
            else generated_status == "complete"
        ),
        "sender_revision_enabled": bool(payload.get("sender_revision_enabled", False)),
        "sender_revision_applied": bool(payload.get("sender_revision_applied", False)),
        "sender_initial_predicted_answer": payload.get("sender_initial_predicted_answer"),
        "sender_revision_predicted_answer": payload.get("sender_revision_predicted_answer"),
        "sender_revision_decision_applied": bool(
            payload.get("sender_revision_decision_applied", False)
        ),
        "sender_revision_decision_predicted_answer": payload.get(
            "sender_revision_decision_predicted_answer"
        ),
        "generated_trace_cache_hit": True,
        "generated_trace_cache_path": "" if cache_path is None else str(cache_path),
    }


def _collect_sender_generated_consensus_state(
    prompt: str,
    state: dict[str, Any],
    cfg: Any,
    *,
    include_prompt: bool = False,
) -> dict[str, Any]:
    cache = state.setdefault("_generated_sender_consensus_cache", {})
    cache_key = _generated_trajectory_trace_cache_key(
        cfg,
        state,
        prompt,
        include_prompt=include_prompt,
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    trace_cache_path: Optional[Path] = None
    if _generated_trajectory_adapter_trace_cache_enabled(cfg):
        trace_cache_path = _generated_trajectory_adapter_trace_cache_path(cfg, cache_key)
        cached_trace = _load_generated_trajectory_trace_from_disk(
            trace_cache_path,
            expected_cache_key=cache_key,
        )
        if cached_trace is not None:
            result = _sender_generated_trace_result_from_payload(
                cached_trace,
                state=state,
                cfg=cfg,
                cache_path=trace_cache_path,
            )
            cache[cache_key] = result
            return result

    tokenizer_a = state["tokenizer_a"]
    agent_a = state["agent_a"]
    agent_a_device = next(agent_a.parameters()).device
    reasoning_layer_indices, reasoning_layer_weights = _reasoning_alignment_metadata(state)
    generation_metadata = _generate_reasoner_metadata(prompt, cfg, state)
    generated_ids = generation_metadata["token_ids"]

    encoded = tokenizer_a(
        _format_reasoner_cot_prompt(prompt, tokenizer_a),
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(agent_a_device)
    attention_mask = encoded["attention_mask"].to(agent_a_device)
    generated_token_ids = torch.tensor(
        [generated_ids],
        dtype=input_ids.dtype,
        device=agent_a_device,
    )
    if generated_token_ids.numel() > 0:
        full_input_ids = torch.cat([input_ids, generated_token_ids], dim=1)
        generated_attention_mask = torch.ones(
            (attention_mask.shape[0], generated_token_ids.shape[1]),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_attention_mask = torch.cat([attention_mask, generated_attention_mask], dim=1)
    else:
        full_input_ids = input_ids
        full_attention_mask = attention_mask
    position_ids = _build_position_ids(full_attention_mask)

    with torch.no_grad():
        outputs = agent_a.model(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            position_ids=position_ids,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
    if outputs.hidden_states is None:
        raise ValueError("Generated latent handoff requires hidden states from Agent A")

    full_consensus_hidden_states = _aggregate_hidden_layers(
        _select_hidden_layers(outputs.hidden_states, reasoning_layer_indices),
        reasoning_layer_weights,
    )
    prompt_token_count = int(input_ids.shape[1])
    prompt_consensus_hidden_states = full_consensus_hidden_states[:, :prompt_token_count, :]
    if generated_ids:
        generated_consensus_hidden_states = full_consensus_hidden_states[:, prompt_token_count:, :]
    else:
        generated_consensus_hidden_states = prompt_consensus_hidden_states[:, -1:, :]

    if include_prompt:
        consensus_hidden_states = torch.cat(
            [prompt_consensus_hidden_states, generated_consensus_hidden_states],
            dim=1,
        )
    else:
        consensus_hidden_states = generated_consensus_hidden_states

    latent_attention_mask = torch.ones(
        (consensus_hidden_states.shape[0], consensus_hidden_states.shape[1]),
        dtype=full_attention_mask.dtype,
        device=full_attention_mask.device,
    )
    pooling_mode = _latent_pooling_mode(cfg)
    result = {
        "consensus_hidden_states": consensus_hidden_states,
        "current_latent_step": _pool_latent_handoff_step(
            consensus_hidden_states,
            latent_attention_mask,
            pooling_mode=pooling_mode,
        ),
        "attention_mask": latent_attention_mask,
        "latent_pooling": pooling_mode,
        "kv_cache_a": _normalize_kv_cache(outputs.past_key_values),
        "reasoning_layer_indices": reasoning_layer_indices,
        "reasoning_layer_weights": reasoning_layer_weights,
        "generated_token_ids": generated_ids,
        "generated_reasoning_text": tokenizer_a.decode(generated_ids, skip_special_tokens=True),
        "generated_reasoning_token_count": len(generated_ids),
        "generated_latent_includes_prompt": bool(include_prompt),
        "sender_revision_enabled": bool(generation_metadata["revision_enabled"]),
        "sender_revision_applied": bool(generation_metadata["revision_applied"]),
        "sender_initial_predicted_answer": generation_metadata.get("initial_predicted_answer"),
        "sender_revision_predicted_answer": generation_metadata.get("revision_predicted_answer"),
        "sender_revision_decision_applied": bool(
            generation_metadata["revision_decision_applied"]
        ),
        "sender_revision_decision_predicted_answer": generation_metadata.get(
            "revision_decision_predicted_answer"
        ),
    }
    result["generated_reasoning_status"] = _sender_reasoning_status(
        generated_ids,
        str(result["generated_reasoning_text"]),
        cfg,
    )
    result["generated_reasoning_final_answer_marker"] = (
        result["generated_reasoning_status"] == "complete"
    )
    result["generated_trace_cache_hit"] = False
    result["generated_trace_cache_path"] = "" if trace_cache_path is None else str(trace_cache_path)
    if trace_cache_path is not None:
        torch.save(_sender_generated_trace_payload(result, cache_key=cache_key), trace_cache_path)
    cache[cache_key] = result
    return result


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


def _handoff_decode_prompt(prompt: str, cfg: Any) -> Optional[str]:
    if _receiver_context_mode(cfg) == "none":
        return None
    return prompt


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


def _generated_trajectory_adapter_cfg(cfg: Any) -> Any:
    return getattr(getattr(cfg, "handoff", None), "generated_trajectory_adapter", None)


def _generated_trajectory_adapter_enabled(cfg: Any) -> bool:
    return bool(getattr(_generated_trajectory_adapter_cfg(cfg), "enabled", False))


def _generated_trajectory_adapter_train_on_missing(cfg: Any) -> bool:
    return bool(getattr(_generated_trajectory_adapter_cfg(cfg), "train_on_missing", False))


def _generated_trajectory_adapter_train_limit(cfg: Any) -> int:
    adapter_cfg = _generated_trajectory_adapter_cfg(cfg)
    return int(getattr(adapter_cfg, "train_limit", 8))


def _generated_trajectory_adapter_target_mode(cfg: Any) -> str:
    adapter_cfg = _generated_trajectory_adapter_cfg(cfg)
    return str(getattr(adapter_cfg, "target_mode", "generated_text")).strip().lower()


def _generated_trajectory_adapter_target_alignment(cfg: Any) -> str:
    adapter_cfg = _generated_trajectory_adapter_cfg(cfg)
    alignment = str(getattr(adapter_cfg, "target_alignment", "character")).strip().lower()
    if alignment not in GENERATED_TRAJECTORY_ADAPTER_TARGET_ALIGNMENTS:
        supported = ", ".join(sorted(GENERATED_TRAJECTORY_ADAPTER_TARGET_ALIGNMENTS))
        raise ValueError(
            "handoff.generated_trajectory_adapter.target_alignment must be one of: "
            f"{supported}"
        )
    if alignment == "character" and _generated_trajectory_adapter_target_mode(cfg) != "generated_text":
        raise ValueError(
            "character target alignment requires "
            "handoff.generated_trajectory_adapter.target_mode=generated_text"
        )
    return alignment


def _generated_trajectory_adapter_target_cache_tag(cfg: Any) -> str:
    target_mode = _generated_trajectory_adapter_target_mode(cfg)
    if target_mode == "final_answer_line":
        return "final_answer_line_latest_marker_v1"
    return target_mode


def _generated_trajectory_adapter_input_space(cfg: Any) -> str:
    adapter_cfg = _generated_trajectory_adapter_cfg(cfg)
    input_space = str(getattr(adapter_cfg, "input_space", "aligned")).strip().lower()
    if input_space not in GENERATED_TRAJECTORY_ADAPTER_INPUT_SPACES:
        supported = ", ".join(sorted(GENERATED_TRAJECTORY_ADAPTER_INPUT_SPACES))
        raise ValueError(
            f"handoff.generated_trajectory_adapter.input_space must be one of: {supported}"
        )
    return input_space


def _generated_trajectory_adapter_source_mode(cfg: Any) -> str:
    adapter_cfg = _generated_trajectory_adapter_cfg(cfg)
    return str(getattr(adapter_cfg, "source_mode", "generated_text")).strip().lower()


def _generated_trajectory_adapter_source_tail_tokens(cfg: Any) -> int:
    adapter_cfg = _generated_trajectory_adapter_cfg(cfg)
    return int(getattr(adapter_cfg, "source_tail_tokens", 32))


def _generated_trajectory_adapter_local_residual_cfg(cfg: Any) -> Any:
    return getattr(_generated_trajectory_adapter_cfg(cfg), "local_residual", None)


def _generated_trajectory_adapter_local_residual_enabled(cfg: Any) -> bool:
    residual_cfg = _generated_trajectory_adapter_local_residual_cfg(cfg)
    return bool(getattr(residual_cfg, "enabled", False))


def _generated_trajectory_adapter_local_residual_top_k(cfg: Any) -> int:
    residual_cfg = _generated_trajectory_adapter_local_residual_cfg(cfg)
    return max(1, int(getattr(residual_cfg, "top_k", 8)))


def _generated_trajectory_adapter_local_residual_temperature(cfg: Any) -> float:
    residual_cfg = _generated_trajectory_adapter_local_residual_cfg(cfg)
    return max(1e-6, float(getattr(residual_cfg, "temperature", 0.05)))


def _generated_trajectory_adapter_local_residual_blend(cfg: Any) -> float:
    residual_cfg = _generated_trajectory_adapter_local_residual_cfg(cfg)
    return max(0.0, min(1.0, float(getattr(residual_cfg, "blend", 1.0))))


def _generated_trajectory_adapter_local_residual_max_rows(cfg: Any) -> int:
    residual_cfg = _generated_trajectory_adapter_local_residual_cfg(cfg)
    return max(0, int(getattr(residual_cfg, "max_memory_rows", 4096)))


def _generated_trajectory_adapter_local_residual_chunk_size(cfg: Any) -> int:
    residual_cfg = _generated_trajectory_adapter_local_residual_cfg(cfg)
    return max(1, int(getattr(residual_cfg, "chunk_size", 64)))


def _generated_trajectory_adapter_semantic_memory_cfg(cfg: Any) -> Any:
    return getattr(_generated_trajectory_adapter_cfg(cfg), "semantic_memory", None)


def _generated_trajectory_adapter_semantic_memory_enabled(cfg: Any) -> bool:
    memory_cfg = _generated_trajectory_adapter_semantic_memory_cfg(cfg)
    return bool(getattr(memory_cfg, "enabled", False))


def _generated_trajectory_adapter_semantic_memory_min_similarity(cfg: Any) -> float:
    memory_cfg = _generated_trajectory_adapter_semantic_memory_cfg(cfg)
    return float(getattr(memory_cfg, "min_similarity", 0.98))


def _generated_trajectory_adapter_semantic_memory_max_entries(cfg: Any) -> int:
    memory_cfg = _generated_trajectory_adapter_semantic_memory_cfg(cfg)
    return max(0, int(getattr(memory_cfg, "max_entries", 2048)))


def _generated_trajectory_adapter_cache_dir(cfg: Any) -> Path:
    adapter_cfg = _generated_trajectory_adapter_cfg(cfg)
    return Path(str(getattr(adapter_cfg, "cache_dir", ".cache/generated_trajectory_adapter")))


def _generated_trajectory_adapter_training_rows_cache_dir(cfg: Any) -> Path:
    adapter_cfg = _generated_trajectory_adapter_cfg(cfg)
    default_dir = _generated_trajectory_adapter_cache_dir(cfg).parent / "generated_trajectory_rows"
    return Path(str(getattr(adapter_cfg, "training_rows_cache_dir", default_dir)))


def _generated_trajectory_adapter_trace_cache_enabled(cfg: Any) -> bool:
    adapter_cfg = _generated_trajectory_adapter_cfg(cfg)
    if adapter_cfg is None:
        return False
    return bool(getattr(adapter_cfg, "trace_cache_enabled", True))


def _generated_trajectory_adapter_trace_cache_dir(cfg: Any) -> Path:
    adapter_cfg = _generated_trajectory_adapter_cfg(cfg)
    default_dir = _generated_trajectory_adapter_cache_dir(cfg).parent / "generated_trajectory_traces"
    return Path(str(getattr(adapter_cfg, "trace_cache_dir", default_dir)))


def _generated_trajectory_adapter_progress_interval(cfg: Any) -> int:
    adapter_cfg = _generated_trajectory_adapter_cfg(cfg)
    return max(0, int(getattr(adapter_cfg, "progress_interval", 1)))


def _generated_trajectory_adapter_dataset_name(cfg: Any) -> str:
    adapter_cfg = _generated_trajectory_adapter_cfg(cfg)
    fallback = getattr(getattr(getattr(cfg, "handoff", None), "adapter", None), "dataset_name", "gsm8k")
    return str(getattr(adapter_cfg, "dataset_name", fallback))


def _generated_trajectory_adapter_train_split(cfg: Any) -> str:
    adapter_cfg = _generated_trajectory_adapter_cfg(cfg)
    fallback = getattr(getattr(getattr(cfg, "handoff", None), "adapter", None), "train_split", "train")
    return str(getattr(adapter_cfg, "train_split", fallback))


def _generated_trajectory_adapter_value(cfg: Any, name: str, default: Any) -> Any:
    adapter_cfg = _generated_trajectory_adapter_cfg(cfg)
    fallback_cfg = getattr(getattr(getattr(cfg, "handoff", None), "adapter", None), name, default)
    return getattr(adapter_cfg, name, fallback_cfg)


def _generated_trajectory_adapter_cache_path(cfg: Any, cache_key: tuple[Any, ...]) -> Path:
    cache_dir = _generated_trajectory_adapter_cache_dir(cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(
        json.dumps(cache_key, sort_keys=False, default=list).encode("utf-8")
    ).hexdigest()
    return cache_dir / f"generated_trajectory_adapter_{digest}.pt"


def _generated_trajectory_adapter_training_rows_cache_path(
    cfg: Any,
    cache_key: tuple[Any, ...],
) -> Path:
    cache_dir = _generated_trajectory_adapter_training_rows_cache_dir(cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(
        json.dumps(cache_key, sort_keys=False, default=list).encode("utf-8")
    ).hexdigest()
    return cache_dir / f"generated_trajectory_rows_{digest}.pt"


def _generated_trajectory_adapter_trace_cache_path(
    cfg: Any,
    cache_key: tuple[Any, ...],
) -> Path:
    cache_dir = _generated_trajectory_adapter_trace_cache_dir(cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(
        json.dumps(cache_key, sort_keys=False, default=list).encode("utf-8")
    ).hexdigest()
    return cache_dir / f"generated_trajectory_trace_{digest}.pt"


def _load_generated_trajectory_adapter_from_disk(
    cache_path: Path,
    *,
    expected_cache_key: Optional[tuple[Any, ...]] = None,
) -> Optional[dict[str, Any]]:
    if not cache_path.is_file():
        return None
    try:
        cached_state = torch.load(cache_path, map_location="cpu")
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(cached_state, dict) or "mapping_matrix" not in cached_state:
        return None
    if expected_cache_key is not None:
        expected_digest = _cache_key_digest(expected_cache_key)
        cached_digest = cached_state.get("adapter_cache_key_digest")
        if cached_digest is not None and cached_digest != expected_digest:
            return None
    return cached_state


def _load_generated_trajectory_training_rows_from_disk(
    cache_path: Path,
    *,
    expected_cache_key: Optional[tuple[Any, ...]] = None,
) -> Optional[dict[str, Any]]:
    if not cache_path.is_file():
        return None
    try:
        cached_rows = torch.load(cache_path, map_location="cpu")
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(cached_rows, dict):
        return None
    source_matrix = cached_rows.get("source_matrix")
    target_matrix = cached_rows.get("target_matrix")
    if not isinstance(source_matrix, torch.Tensor) or not isinstance(target_matrix, torch.Tensor):
        return None
    if expected_cache_key is not None:
        expected_digest = _cache_key_digest(expected_cache_key)
        cached_digest = cached_rows.get("training_rows_cache_key_digest")
        if cached_digest is not None and cached_digest != expected_digest:
            return None
    if source_matrix.dim() != 2 or target_matrix.dim() != 2:
        return None
    if int(source_matrix.shape[0]) == 0 or int(source_matrix.shape[0]) != int(target_matrix.shape[0]):
        return None
    return cached_rows


def _load_generated_trajectory_trace_from_disk(
    cache_path: Path,
    *,
    expected_cache_key: Optional[tuple[Any, ...]] = None,
) -> Optional[dict[str, Any]]:
    if not cache_path.is_file():
        return None
    try:
        cached_trace = torch.load(cache_path, map_location="cpu")
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(cached_trace, dict):
        return None
    if expected_cache_key is not None and cached_trace.get("cache_key") is not None:
        if cached_trace.get("cache_key") != _cache_key_metadata(expected_cache_key):
            return None
    if expected_cache_key is not None and cached_trace.get("cache_key_digest") is not None:
        if cached_trace.get("cache_key_digest") != _cache_key_digest(expected_cache_key):
            return None
    consensus_hidden_states = cached_trace.get("consensus_hidden_states")
    if not isinstance(consensus_hidden_states, torch.Tensor):
        return None
    if consensus_hidden_states.dim() != 3 or int(consensus_hidden_states.shape[1]) == 0:
        return None
    generated_token_ids = cached_trace.get("generated_token_ids")
    if not isinstance(generated_token_ids, (list, tuple)):
        return None
    try:
        cached_trace["generated_token_ids"] = [int(token_id) for token_id in generated_token_ids]
    except (TypeError, ValueError):
        return None
    if not isinstance(cached_trace.get("generated_reasoning_text"), str):
        return None
    return cached_trace


def _generated_trajectory_adapter_cache_key(
    cfg: Any,
    state: dict[str, Any],
    *,
    include_prompt: bool,
) -> tuple[Any, ...]:
    local_residual_enabled = _generated_trajectory_adapter_local_residual_enabled(cfg)
    base_key: tuple[Any, ...] = (
        "generated_trajectory_adapter_v3" if local_residual_enabled else "generated_trajectory_adapter_v1",
        str(cfg.agent_a_model),
        str(cfg.agent_b_model),
        str(cfg.torch_dtype),
        state.get("global_alignment_cache_key"),
        int(_reasoner_generation_max_new_tokens(cfg)),
        _sender_generation_cache_fingerprint(cfg),
        bool(include_prompt),
        _generated_trajectory_adapter_dataset_name(cfg),
        _generated_trajectory_adapter_train_split(cfg),
        _generated_trajectory_adapter_train_limit(cfg),
        _generated_trajectory_adapter_source_mode(cfg),
        _generated_trajectory_adapter_source_tail_tokens(cfg),
        _generated_trajectory_adapter_target_cache_tag(cfg),
        _generated_trajectory_adapter_target_alignment(cfg),
        _generated_trajectory_adapter_input_space(cfg),
        str(_generated_trajectory_adapter_value(cfg, "strategy", "hybrid_affine")),
        float(_generated_trajectory_adapter_value(cfg, "regularization", 1e-3)),
        float(_generated_trajectory_adapter_value(cfg, "residual_alpha", 1.0)),
        float(_generated_trajectory_adapter_value(cfg, "residual_max_norm_ratio", 0.5)),
        bool(_generated_trajectory_adapter_value(cfg, "center", True)),
        bool(_generated_trajectory_adapter_value(cfg, "use_bias", True)),
        bool(_generated_trajectory_adapter_semantic_memory_enabled(cfg)),
        _generated_trajectory_adapter_semantic_memory_max_entries(cfg),
    )
    if not local_residual_enabled:
        return base_key
    return (
        *base_key,
        True,
        _generated_trajectory_adapter_local_residual_top_k(cfg),
        _generated_trajectory_adapter_local_residual_temperature(cfg),
        _generated_trajectory_adapter_local_residual_blend(cfg),
        _generated_trajectory_adapter_local_residual_max_rows(cfg),
    )


def _generated_trajectory_trace_cache_key(
    cfg: Any,
    state: dict[str, Any],
    prompt: str,
    *,
    include_prompt: bool,
) -> tuple[Any, ...]:
    reasoning_layer_indices, reasoning_layer_weights = _reasoning_alignment_metadata(state)
    prompt_hash = hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()
    return (
        "generated_trajectory_trace_v1",
        str(cfg.agent_a_model),
        str(cfg.torch_dtype),
        int(_reasoner_generation_max_new_tokens(cfg)),
        _sender_generation_cache_fingerprint(cfg),
        bool(include_prompt),
        _latent_pooling_mode(cfg),
        tuple(int(index) for index in reasoning_layer_indices),
        tuple(round(float(weight), 8) for weight in reasoning_layer_weights),
        prompt_hash,
    )


def _generated_trajectory_adapter_training_rows_cache_key(
    cfg: Any,
    state: dict[str, Any],
    *,
    include_prompt: bool,
) -> tuple[Any, ...]:
    return (
        "generated_trajectory_training_rows_v1",
        str(cfg.agent_a_model),
        str(cfg.agent_b_model),
        str(cfg.torch_dtype),
        state.get("global_alignment_cache_key"),
        int(_reasoner_generation_max_new_tokens(cfg)),
        _sender_generation_cache_fingerprint(cfg),
        bool(include_prompt),
        _generated_trajectory_adapter_dataset_name(cfg),
        _generated_trajectory_adapter_train_split(cfg),
        _generated_trajectory_adapter_train_limit(cfg),
        _generated_trajectory_adapter_source_mode(cfg),
        _generated_trajectory_adapter_source_tail_tokens(cfg),
        _generated_trajectory_adapter_target_cache_tag(cfg),
        _generated_trajectory_adapter_target_alignment(cfg),
        _generated_trajectory_adapter_input_space(cfg),
        bool(_generated_trajectory_adapter_semantic_memory_enabled(cfg)),
        _generated_trajectory_adapter_semantic_memory_max_entries(cfg),
    )


def _generated_trajectory_adapter_identity_manifest(cfg: Any) -> dict[str, Any]:
    """Resolved generated-adapter identity fields that must survive replay."""
    return {
        "enabled": _generated_trajectory_adapter_enabled(cfg),
        "train_on_missing": _generated_trajectory_adapter_train_on_missing(cfg),
        "train_limit": _generated_trajectory_adapter_train_limit(cfg),
        "dataset_name": _generated_trajectory_adapter_dataset_name(cfg),
        "train_split": _generated_trajectory_adapter_train_split(cfg),
        "source_mode": _generated_trajectory_adapter_source_mode(cfg),
        "source_tail_tokens": _generated_trajectory_adapter_source_tail_tokens(cfg),
        "input_space": _generated_trajectory_adapter_input_space(cfg),
        "target_mode": _generated_trajectory_adapter_target_mode(cfg),
        "target_alignment": _generated_trajectory_adapter_target_alignment(cfg),
        "local_residual": {
            "enabled": _generated_trajectory_adapter_local_residual_enabled(cfg),
            "top_k": _generated_trajectory_adapter_local_residual_top_k(cfg),
            "temperature": _generated_trajectory_adapter_local_residual_temperature(cfg),
            "blend": _generated_trajectory_adapter_local_residual_blend(cfg),
            "max_memory_rows": _generated_trajectory_adapter_local_residual_max_rows(cfg),
        },
        "semantic_memory": {
            "enabled": _generated_trajectory_adapter_semantic_memory_enabled(cfg),
            "min_similarity": _generated_trajectory_adapter_semantic_memory_min_similarity(cfg),
            "max_entries": _generated_trajectory_adapter_semantic_memory_max_entries(cfg),
        },
    }


def _handoff_identity_manifest(cfg: Any) -> dict[str, Any]:
    embedding_cfg = getattr(getattr(cfg, "handoff", None), "embedding_manifold", None)
    return {
        "latent_pooling": _latent_pooling_mode(cfg),
        "latent_prefix_mode": _latent_prefix_mode(cfg),
        "receiver_context_mode": _receiver_context_mode(cfg),
        "receiver_context_latent_position": _receiver_context_latent_position(cfg),
        "embedding_manifold": {
            "enabled": bool(getattr(embedding_cfg, "enabled", False)),
            "top_k": int(getattr(embedding_cfg, "top_k", 1)),
            "blend": float(getattr(embedding_cfg, "blend", 1.0)),
        },
    }


def _receiver_embedding_sequence_for_text(
    text: str,
    *,
    state: dict[str, Any],
    target_steps: int,
) -> Optional[torch.Tensor]:
    if not str(text).strip():
        return None
    tokenizer_b = state["tokenizer_b"]
    agent_b = state["agent_b"]
    agent_b_device = next(agent_b.parameters()).device
    encoded = tokenizer_b(str(text), return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(agent_b_device)
    if input_ids.numel() == 0:
        return None
    with torch.no_grad():
        embeddings = agent_b.get_input_embeddings()(input_ids)
    return _resample_sequence(embeddings, target_steps)


def _token_character_spans(tokenizer: Any, token_ids: Sequence[int]) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    previous_text = ""
    previous_length = 0
    ids: list[int] = []
    for token_id in token_ids:
        ids.append(int(token_id))
        decoded = tokenizer.decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not decoded.startswith(previous_text):
            previous_length = min(previous_length, len(decoded))
        current_length = len(decoded)
        spans.append((float(previous_length), float(max(previous_length, current_length))))
        previous_text = decoded
        previous_length = current_length
    return spans


def _source_token_ids_for_generated_adapter(
    cfg: Any,
    sender_state: dict[str, Any],
) -> list[int]:
    generated_ids = [int(token_id) for token_id in sender_state.get("generated_token_ids", ())]
    if _generated_trajectory_adapter_source_mode(cfg) == "final_answer_tail":
        tail_tokens = max(1, _generated_trajectory_adapter_source_tail_tokens(cfg))
        return generated_ids[-tail_tokens:]
    return generated_ids


def _receiver_embedding_sequence_for_aligned_text(
    text: str,
    *,
    state: dict[str, Any],
    source_token_ids: Sequence[int],
    target_steps: int,
    target_alignment: str,
) -> Optional[torch.Tensor]:
    if target_alignment == "linear":
        return _receiver_embedding_sequence_for_text(
            text,
            state=state,
            target_steps=target_steps,
        )
    if not source_token_ids or len(source_token_ids) != int(target_steps):
        return _receiver_embedding_sequence_for_text(
            text,
            state=state,
            target_steps=target_steps,
        )

    tokenizer_a = state["tokenizer_a"]
    tokenizer_b = state["tokenizer_b"]
    agent_b = state["agent_b"]
    agent_b_device = next(agent_b.parameters()).device
    encoded = tokenizer_b(str(text), return_tensors="pt", add_special_tokens=False)
    receiver_ids = encoded["input_ids"][0].detach().cpu().tolist()
    if not receiver_ids:
        return None
    input_ids = encoded["input_ids"].to(agent_b_device)
    with torch.no_grad():
        receiver_embeddings = agent_b.get_input_embeddings()(input_ids)

    source_spans = _token_character_spans(tokenizer_a, source_token_ids)
    receiver_spans = _token_character_spans(tokenizer_b, receiver_ids)
    if not source_spans or not receiver_spans:
        return _resample_sequence(receiver_embeddings, target_steps)

    receiver_midpoints = torch.tensor(
        [(start + end) / 2.0 for start, end in receiver_spans],
        dtype=torch.float32,
    )
    if receiver_midpoints.numel() == 0:
        return _resample_sequence(receiver_embeddings, target_steps)

    selected_indices: list[int] = []
    previous_index = 0
    for start, end in source_spans:
        midpoint = (start + end) / 2.0
        if end <= start:
            selected_indices.append(previous_index)
            continue
        distances = torch.abs(receiver_midpoints - float(midpoint))
        selected_index = int(torch.argmin(distances).item())
        previous_index = selected_index
        selected_indices.append(selected_index)
    index_tensor = torch.tensor(selected_indices, dtype=torch.long, device=agent_b_device)
    return receiver_embeddings.index_select(1, index_tensor)


def _generated_trajectory_adapter_target_text(cfg: Any, generated_text: str) -> str:
    mode = _generated_trajectory_adapter_target_mode(cfg)
    if mode == "generated_text":
        return str(generated_text)
    if mode == "final_answer_line":
        final_answer = _final_answer_marker_value(str(generated_text))
        if final_answer is not None:
            return f"Final answer: {final_answer}"
        return str(generated_text)
    raise ValueError(
        "handoff.generated_trajectory_adapter.target_mode must be one of: "
        "generated_text, final_answer_line"
    )


def _generated_trajectory_adapter_source_sequence(
    cfg: Any,
    sender_state: dict[str, Any],
) -> torch.Tensor:
    source_sequence = sender_state["consensus_hidden_states"]
    mode = _generated_trajectory_adapter_source_mode(cfg)
    if mode == "generated_text":
        return source_sequence
    if mode == "final_answer_tail":
        tail_tokens = max(1, _generated_trajectory_adapter_source_tail_tokens(cfg))
        tail_steps = min(int(source_sequence.shape[1]), tail_tokens)
        return source_sequence[:, -tail_steps:, :]
    raise ValueError(
        "handoff.generated_trajectory_adapter.source_mode must be one of: "
        "generated_text, final_answer_tail"
    )


def _generated_trajectory_adapter_fit_source(
    source_sequence: torch.Tensor,
    alignment_state: dict[str, Any],
    cfg: Any,
) -> torch.Tensor:
    if _generated_trajectory_adapter_input_space(cfg) == "raw":
        return source_sequence
    return apply_alignment(source_sequence, alignment_state)


def _mean_handoff_delta_norm(
    before: torch.Tensor,
    after: torch.Tensor,
) -> Optional[float]:
    if tuple(before.shape) != tuple(after.shape):
        return None
    delta = after.float() - before.float()
    return float(
        torch.linalg.vector_norm(delta.reshape(delta.shape[0], -1), dim=-1)
        .mean()
        .detach()
        .cpu()
        .item()
    )


def _select_generated_adapter_memory_rows(
    source_matrix: torch.Tensor,
    residual_matrix: torch.Tensor,
    *,
    max_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_rows <= 0 or int(source_matrix.shape[0]) <= max_rows:
        return source_matrix.detach().cpu(), residual_matrix.detach().cpu()
    row_count = int(source_matrix.shape[0])
    hard_row_count = max(1, max_rows // 2)
    residual_norms = torch.linalg.vector_norm(residual_matrix.float(), dim=-1)
    hard_indices = torch.topk(
        residual_norms,
        k=min(hard_row_count, row_count),
        largest=True,
    ).indices
    coverage_indices = torch.linspace(
        0,
        row_count - 1,
        steps=max_rows,
        dtype=torch.long,
    )
    selected: list[int] = []
    seen: set[int] = set()
    for index_tensor in (hard_indices, coverage_indices):
        for raw_index in index_tensor.detach().cpu().tolist():
            index = int(raw_index)
            if index in seen:
                continue
            seen.add(index)
            selected.append(index)
            if len(selected) == max_rows:
                break
        if len(selected) == max_rows:
            break
    indices = torch.tensor(selected, dtype=torch.long)
    return (
        source_matrix.index_select(0, indices).detach().cpu(),
        residual_matrix.index_select(0, indices).detach().cpu(),
    )


def _build_generated_adapter_local_residual_state(
    cfg: Any,
    source_matrix: torch.Tensor,
    target_matrix: torch.Tensor,
    fitted_matrix: torch.Tensor,
) -> Optional[dict[str, Any]]:
    if not _generated_trajectory_adapter_local_residual_enabled(cfg):
        return None
    residual_matrix = target_matrix.float() - fitted_matrix.float()
    source_memory, residual_memory = _select_generated_adapter_memory_rows(
        source_matrix.float(),
        residual_matrix.float(),
        max_rows=_generated_trajectory_adapter_local_residual_max_rows(cfg),
    )
    if int(source_memory.shape[0]) == 0:
        return None
    return {
        "enabled": True,
        "top_k": _generated_trajectory_adapter_local_residual_top_k(cfg),
        "temperature": _generated_trajectory_adapter_local_residual_temperature(cfg),
        "blend": _generated_trajectory_adapter_local_residual_blend(cfg),
        "chunk_size": _generated_trajectory_adapter_local_residual_chunk_size(cfg),
        "source_memory": source_memory,
        "residual_memory": residual_memory,
        "memory_row_count": int(source_memory.shape[0]),
    }


def _apply_generated_adapter_local_residual(
    adapter_input: torch.Tensor,
    adapted_handoff_step: torch.Tensor,
    adapter_state: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    residual_state = adapter_state.get("local_residual_state")
    if not residual_state or not bool(residual_state.get("enabled", False)):
        return adapted_handoff_step, {
            "generated_adapter_local_residual_applied": False,
            "generated_adapter_local_residual_delta_norm": None,
            "generated_adapter_local_residual_mean_top_similarity": None,
            "generated_adapter_local_residual_memory_rows": None,
        }
    source_memory = residual_state["source_memory"].to(
        device=adapter_input.device,
        dtype=torch.float32,
    )
    residual_memory = residual_state["residual_memory"].to(
        device=adapted_handoff_step.device,
        dtype=torch.float32,
    )
    if int(source_memory.shape[0]) == 0:
        return adapted_handoff_step, {
            "generated_adapter_local_residual_applied": False,
            "generated_adapter_local_residual_delta_norm": None,
            "generated_adapter_local_residual_mean_top_similarity": None,
            "generated_adapter_local_residual_memory_rows": 0,
        }
    query_rows = adapter_input.reshape(-1, adapter_input.shape[-1]).float()
    output_shape = adapted_handoff_step.shape
    top_k = min(int(residual_state.get("top_k", 8)), int(source_memory.shape[0]))
    temperature = max(1e-6, float(residual_state.get("temperature", 0.05)))
    blend = max(0.0, min(1.0, float(residual_state.get("blend", 1.0))))
    chunk_size = max(1, int(residual_state.get("chunk_size", 64)))
    if blend == 0.0:
        return adapted_handoff_step, {
            "generated_adapter_local_residual_applied": False,
            "generated_adapter_local_residual_delta_norm": 0.0,
            "generated_adapter_local_residual_mean_top_similarity": None,
            "generated_adapter_local_residual_memory_rows": int(source_memory.shape[0]),
        }

    normalized_memory = torch.nn.functional.normalize(source_memory, dim=-1)
    corrections: list[torch.Tensor] = []
    top_similarities: list[torch.Tensor] = []
    for start in range(0, int(query_rows.shape[0]), chunk_size):
        query_chunk = query_rows[start : start + chunk_size]
        similarities = torch.nn.functional.normalize(query_chunk, dim=-1) @ normalized_memory.T
        top_values, top_indices = torch.topk(similarities, k=top_k, dim=-1)
        weights = torch.softmax(top_values / temperature, dim=-1)
        residual_neighbors = residual_memory.index_select(0, top_indices.reshape(-1)).reshape(
            top_indices.shape[0],
            top_indices.shape[1],
            residual_memory.shape[-1],
        )
        corrections.append((residual_neighbors * weights.unsqueeze(-1)).sum(dim=1))
        top_similarities.append(top_values[:, 0].detach())
    correction = torch.cat(corrections, dim=0).reshape(output_shape)
    correction = correction.to(device=adapted_handoff_step.device, dtype=adapted_handoff_step.dtype)
    corrected = adapted_handoff_step + (correction * blend)
    delta_norm = _mean_handoff_delta_norm(adapted_handoff_step, corrected)
    mean_top_similarity = float(torch.cat(top_similarities).mean().detach().cpu().item())
    return corrected, {
        "generated_adapter_local_residual_applied": True,
        "generated_adapter_local_residual_delta_norm": delta_norm,
        "generated_adapter_local_residual_mean_top_similarity": mean_top_similarity,
        "generated_adapter_local_residual_memory_rows": int(source_memory.shape[0]),
    }


def _build_generated_adapter_semantic_memory_state(
    cfg: Any,
    training_rows: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    if not _generated_trajectory_adapter_semantic_memory_enabled(cfg):
        return None
    entries: list[dict[str, Any]] = []
    for raw_entry in training_rows.get("semantic_memory_entries") or []:
        if not isinstance(raw_entry, Mapping):
            continue
        source_sequence = raw_entry.get("source_sequence")
        target_text = str(raw_entry.get("target_text") or "").strip()
        if not isinstance(source_sequence, torch.Tensor) or not target_text:
            continue
        if source_sequence.dim() == 3:
            source_sequence = source_sequence.squeeze(0)
        if source_sequence.dim() != 2 or int(source_sequence.shape[0]) == 0:
            continue
        entries.append(
            {
                "source_sequence": source_sequence.detach().float().cpu(),
                "target_text": target_text,
                "target_answer": raw_entry.get("target_answer"),
                "source_token_count": int(raw_entry.get("source_token_count", source_sequence.shape[0])),
                "row_index": raw_entry.get("row_index"),
            }
        )
        if len(entries) >= _generated_trajectory_adapter_semantic_memory_max_entries(cfg):
            break
    if not entries:
        return None
    return {
        "enabled": True,
        "min_similarity": _generated_trajectory_adapter_semantic_memory_min_similarity(cfg),
        "entry_count": len(entries),
        "entries": entries,
    }


def _sequence_cosine_similarity(query: torch.Tensor, memory_sequence: torch.Tensor) -> float:
    if query.dim() != 3:
        raise ValueError("query must have shape [batch, steps, dim]")
    if memory_sequence.dim() == 2:
        memory_sequence = memory_sequence.unsqueeze(0)
    if memory_sequence.dim() != 3:
        raise ValueError("memory_sequence must have shape [steps, dim] or [batch, steps, dim]")
    if int(memory_sequence.shape[1]) != int(query.shape[1]):
        memory_sequence = _resample_sequence(memory_sequence, int(query.shape[1]))
    memory_sequence = memory_sequence.to(device=query.device, dtype=query.dtype)
    score = torch.nn.functional.cosine_similarity(
        query.float(),
        memory_sequence.float(),
        dim=-1,
    ).mean()
    return float(score.detach().cpu().item())


def _apply_generated_adapter_semantic_memory(
    adapter_input: torch.Tensor,
    adapter_state: dict[str, Any],
    cfg: Any,
) -> dict[str, Any]:
    memory_state = adapter_state.get("semantic_memory_state")
    if (
        not _generated_trajectory_adapter_semantic_memory_enabled(cfg)
        or not isinstance(memory_state, Mapping)
        or not bool(memory_state.get("enabled", False))
    ):
        return {
            "generated_adapter_semantic_memory_applied": False,
            "generated_adapter_semantic_memory_similarity": None,
            "generated_adapter_semantic_memory_entry_count": None,
            "generated_adapter_semantic_memory_target_text": None,
        }
    entries = memory_state.get("entries") or []
    if not entries:
        return {
            "generated_adapter_semantic_memory_applied": False,
            "generated_adapter_semantic_memory_similarity": None,
            "generated_adapter_semantic_memory_entry_count": 0,
            "generated_adapter_semantic_memory_target_text": None,
        }
    query = adapter_input.detach()
    best_score: Optional[float] = None
    best_text: Optional[str] = None
    best_answer: Optional[str] = None
    for raw_entry in entries:
        if not isinstance(raw_entry, Mapping):
            continue
        source_sequence = raw_entry.get("source_sequence")
        target_text = str(raw_entry.get("target_text") or "").strip()
        if not isinstance(source_sequence, torch.Tensor) or not target_text:
            continue
        score = _sequence_cosine_similarity(query, source_sequence)
        if best_score is None or score > best_score:
            best_score = score
            best_text = target_text
            best_answer = raw_entry.get("target_answer")
    min_similarity = float(
        getattr(
            _generated_trajectory_adapter_semantic_memory_cfg(cfg),
            "min_similarity",
            memory_state.get("min_similarity", 0.98),
        )
    )
    if best_score is None or best_score < min_similarity or not best_text:
        return {
            "generated_adapter_semantic_memory_applied": False,
            "generated_adapter_semantic_memory_similarity": best_score,
            "generated_adapter_semantic_memory_entry_count": len(entries),
            "generated_adapter_semantic_memory_target_text": None,
        }
    if best_answer is not None and str(best_answer).strip():
        best_text = f"Final answer: {best_answer}"
    return {
        "generated_adapter_semantic_memory_applied": True,
        "generated_adapter_semantic_memory_similarity": best_score,
        "generated_adapter_semantic_memory_entry_count": len(entries),
        "generated_adapter_semantic_memory_target_text": best_text,
    }


def _build_generated_trajectory_training_rows(
    cfg: Any,
    state: dict[str, Any],
    alignment_state: dict[str, Any],
    *,
    include_prompt: bool,
) -> dict[str, Any]:
    dataset_name = _generated_trajectory_adapter_dataset_name(cfg)
    train_split = _generated_trajectory_adapter_train_split(cfg)
    train_limit = _generated_trajectory_adapter_train_limit(cfg)
    if train_limit <= 0:
        raise ValueError("generated trajectory adapter train_limit must be positive")
    validation_size = _validation_size(cfg, dataset_name)
    train_rows = get_dataloader(
        dataset_name,
        limit=train_limit,
        split=train_split,
        validation_size=validation_size,
    )
    source_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    semantic_memory_entries: list[dict[str, Any]] = []
    prompt_count = 0
    token_count = 0
    trace_cache_hit_count = 0
    trace_cache_miss_count = 0
    trace_cache_lookup_count = 0
    progress_interval = _generated_trajectory_adapter_progress_interval(cfg)
    total_train_rows = len(train_rows) if hasattr(train_rows, "__len__") else train_limit
    for row_index, row in enumerate(train_rows, start=1):
        prompt = pick_field(row, ("question", "problem"))
        if not str(prompt).strip():
            continue
        state["_current_sample_row"] = dict(row)
        sender_state = _collect_sender_generated_consensus_state(
            prompt,
            state,
            cfg,
            include_prompt=include_prompt,
        )
        trace_cache_path = str(sender_state.get("generated_trace_cache_path") or "")
        if trace_cache_path:
            trace_cache_lookup_count += 1
            if bool(sender_state.get("generated_trace_cache_hit", False)):
                trace_cache_hit_count += 1
            else:
                trace_cache_miss_count += 1
        source_sequence = _generated_trajectory_adapter_source_sequence(cfg, sender_state)
        adapter_source = _generated_trajectory_adapter_fit_source(
            source_sequence,
            alignment_state,
            cfg,
        )
        target_text = _generated_trajectory_adapter_target_text(
            cfg,
            str(sender_state.get("generated_reasoning_text", "")),
        )
        source_token_ids = _source_token_ids_for_generated_adapter(cfg, sender_state)
        target_sequence = _receiver_embedding_sequence_for_aligned_text(
            target_text,
            state=state,
            source_token_ids=source_token_ids,
            target_steps=int(adapter_source.shape[1]),
            target_alignment=_generated_trajectory_adapter_target_alignment(cfg),
        )
        if target_sequence is None:
            continue
        source_rows.append(adapter_source.detach().float().cpu().reshape(-1, adapter_source.shape[-1]))
        target_rows.append(target_sequence.detach().float().cpu().reshape(-1, target_sequence.shape[-1]))
        if _generated_trajectory_adapter_semantic_memory_enabled(cfg):
            target_answer = _final_answer_marker_value(target_text)
            semantic_memory_entries.append(
                {
                    "source_sequence": adapter_source.detach().float().cpu().squeeze(0),
                    "target_text": str(target_text),
                    "target_answer": target_answer,
                    "source_token_count": int(adapter_source.shape[1]),
                    "row_index": int(row_index),
                }
            )
        prompt_count += 1
        token_count += int(adapter_source.shape[1])
        if progress_interval and (
            prompt_count == 1
            or prompt_count % progress_interval == 0
            or row_index == total_train_rows
        ):
            print(
                "Generated trajectory rows: "
                f"{prompt_count}/{train_limit} usable prompts "
                f"(source row {row_index}/{total_train_rows}, "
                f"trace cache hits={trace_cache_hit_count}, misses={trace_cache_miss_count})",
                flush=True,
            )
    if not source_rows:
        raise ValueError("No usable generated trajectories were available for adapter fitting")
    source_matrix = torch.cat(source_rows, dim=0)
    target_matrix = torch.cat(target_rows, dim=0)
    trace_cache_hit_rate = (
        100.0 * trace_cache_hit_count / trace_cache_lookup_count
        if trace_cache_lookup_count
        else None
    )
    return {
        "source_matrix": source_matrix,
        "target_matrix": target_matrix,
        "training_prompt_count": int(prompt_count),
        "training_token_count": int(token_count),
        "training_trace_cache_hit_count": int(trace_cache_hit_count),
        "training_trace_cache_miss_count": int(trace_cache_miss_count),
        "training_trace_cache_hit_rate_percentage": trace_cache_hit_rate,
        "semantic_memory_entries": semantic_memory_entries[
            : _generated_trajectory_adapter_semantic_memory_max_entries(cfg)
        ],
    }


def _load_or_build_generated_trajectory_training_rows(
    cfg: Any,
    state: dict[str, Any],
    alignment_state: dict[str, Any],
    *,
    include_prompt: bool,
) -> dict[str, Any]:
    memory_cache = state.setdefault("_generated_trajectory_training_rows_cache", {})
    cache_key = _generated_trajectory_adapter_training_rows_cache_key(
        cfg,
        state,
        include_prompt=include_prompt,
    )
    cached_rows = memory_cache.get(cache_key)
    if cached_rows is not None:
        print("Generated trajectory rows cache hit in memory", flush=True)
        return {**cached_rows, "training_row_cache_hit": True}

    cache_path = _generated_trajectory_adapter_training_rows_cache_path(cfg, cache_key)
    print(f"Checking generated trajectory rows cache: {cache_path}", flush=True)
    cached_rows = _load_generated_trajectory_training_rows_from_disk(
        cache_path,
        expected_cache_key=cache_key,
    )
    cache_hit = cached_rows is not None
    if cached_rows is None:
        print("Generated trajectory rows cache miss; building rows", flush=True)
        cached_rows = _build_generated_trajectory_training_rows(
            cfg,
            state,
            alignment_state,
            include_prompt=include_prompt,
        )
        cached_rows = {
            **cached_rows,
            "training_rows_cache_key": _cache_key_metadata(cache_key),
            "training_rows_cache_key_digest": _cache_key_digest(cache_key),
        }
        torch.save(cached_rows, cache_path)
        print(f"Wrote generated trajectory rows cache: {cache_path}", flush=True)
    else:
        print(f"Loaded generated trajectory rows cache: {cache_path}", flush=True)
    info = {
        **cached_rows,
        "training_row_cache_hit": cache_hit,
        "training_row_cache_path": str(cache_path),
        "training_rows_cache_key_digest": _cache_key_digest(cache_key),
    }
    memory_cache[cache_key] = info
    return info


def _fit_generated_trajectory_adapter_state(
    cfg: Any,
    state: dict[str, Any],
    alignment_state: dict[str, Any],
    *,
    include_prompt: bool,
) -> dict[str, Any]:
    training_rows = _load_or_build_generated_trajectory_training_rows(
        cfg,
        state,
        alignment_state,
        include_prompt=include_prompt,
    )
    source_matrix = training_rows["source_matrix"]
    target_matrix = training_rows["target_matrix"]
    adapter_state = compute_alignment_state(
        source_matrix,
        target_matrix,
        strategy=str(_generated_trajectory_adapter_value(cfg, "strategy", "hybrid_affine")),
        center=bool(_generated_trajectory_adapter_value(cfg, "center", True)),
        use_bias=bool(_generated_trajectory_adapter_value(cfg, "use_bias", True)),
        regularization=float(_generated_trajectory_adapter_value(cfg, "regularization", 1e-3)),
        residual_alpha=float(_generated_trajectory_adapter_value(cfg, "residual_alpha", 1.0)),
        residual_max_norm_ratio=float(
            _generated_trajectory_adapter_value(cfg, "residual_max_norm_ratio", 0.5)
        ),
        adaptive_projection_strength=float(
            _generated_trajectory_adapter_value(cfg, "adaptive_projection_strength", 0.15)
        ),
        adaptive_projection_clip_std_multiplier=float(
            _generated_trajectory_adapter_value(cfg, "adaptive_projection_clip_std_multiplier", 4.0)
        ),
    )
    fitted = apply_alignment(source_matrix, adapter_state).float()
    mse = torch.mean((fitted - target_matrix.float()) ** 2)
    cosine = torch.nn.functional.cosine_similarity(fitted, target_matrix.float(), dim=-1).mean()
    local_residual_state = _build_generated_adapter_local_residual_state(
        cfg,
        source_matrix,
        target_matrix,
        fitted,
    )
    semantic_memory_state = _build_generated_adapter_semantic_memory_state(cfg, training_rows)
    return {
        **adapter_state,
        "adapter_type": "generated_trajectory_handoff",
        "input_space": _generated_trajectory_adapter_input_space(cfg),
        "local_residual_state": local_residual_state,
        "semantic_memory_state": semantic_memory_state,
        "training_prompt_count": int(training_rows["training_prompt_count"]),
        "training_token_count": int(training_rows["training_token_count"]),
        "training_row_cache_hit": bool(training_rows.get("training_row_cache_hit", False)),
        "training_row_cache_path": str(training_rows.get("training_row_cache_path", "")),
        "training_rows_cache_key_digest": training_rows.get(
            "training_rows_cache_key_digest"
        ),
        "training_trace_cache_hit_count": training_rows.get(
            "training_trace_cache_hit_count"
        ),
        "training_trace_cache_miss_count": training_rows.get(
            "training_trace_cache_miss_count"
        ),
        "training_trace_cache_hit_rate_percentage": training_rows.get(
            "training_trace_cache_hit_rate_percentage"
        ),
        "training_reconstruction_mse": float(mse.item()),
        "training_mean_cosine_similarity": float(cosine.item()),
    }


def _load_or_train_generated_trajectory_adapter_state(
    cfg: Any,
    state: dict[str, Any],
    alignment_state: dict[str, Any],
    *,
    include_prompt: bool,
) -> dict[str, Any]:
    if not _generated_trajectory_adapter_enabled(cfg):
        return {
            "enabled": False,
            "status": "disabled",
            "cache_hit": None,
            "cache_path": "",
            "state": None,
        }
    memory_cache = state.setdefault("_generated_trajectory_adapter_cache", {})
    cache_key = _generated_trajectory_adapter_cache_key(cfg, state, include_prompt=include_prompt)
    cached_info = memory_cache.get(cache_key)
    if cached_info is not None:
        print("Generated trajectory adapter cache hit in memory", flush=True)
        return cached_info
    cache_path = _generated_trajectory_adapter_cache_path(cfg, cache_key)
    print(f"Checking generated trajectory adapter cache: {cache_path}", flush=True)
    adapter_state = _load_generated_trajectory_adapter_from_disk(
        cache_path,
        expected_cache_key=cache_key,
    )
    cache_hit = adapter_state is not None
    if adapter_state is None and _generated_trajectory_adapter_train_on_missing(cfg):
        print("Generated trajectory adapter cache miss; fitting adapter", flush=True)
        adapter_state = _fit_generated_trajectory_adapter_state(
            cfg,
            state,
            alignment_state,
            include_prompt=include_prompt,
        )
        adapter_state = {
            **adapter_state,
            "adapter_cache_key": _cache_key_metadata(cache_key),
            "adapter_cache_key_digest": _cache_key_digest(cache_key),
        }
        torch.save(adapter_state, cache_path)
        print(f"Wrote generated trajectory adapter cache: {cache_path}", flush=True)
    if adapter_state is None:
        info = {
            "enabled": True,
            "status": "missing",
            "cache_hit": False,
            "cache_path": str(cache_path),
            "adapter_cache_key": _cache_key_metadata(cache_key),
            "adapter_cache_key_digest": _cache_key_digest(cache_key),
            "state": None,
        }
    else:
        info = {
            "enabled": True,
            "status": "loaded" if cache_hit else "trained",
            "cache_hit": cache_hit,
            "cache_path": str(cache_path),
            "state": adapter_state,
            "training_prompt_count": adapter_state.get("training_prompt_count"),
            "training_token_count": adapter_state.get("training_token_count"),
            "training_row_cache_hit": adapter_state.get("training_row_cache_hit"),
            "training_row_cache_path": adapter_state.get("training_row_cache_path"),
            "adapter_cache_key_digest": adapter_state.get(
                "adapter_cache_key_digest",
                _cache_key_digest(cache_key),
            ),
            "training_rows_cache_key_digest": adapter_state.get(
                "training_rows_cache_key_digest"
            ),
            "training_trace_cache_hit_count": adapter_state.get(
                "training_trace_cache_hit_count"
            ),
            "training_trace_cache_miss_count": adapter_state.get(
                "training_trace_cache_miss_count"
            ),
            "training_trace_cache_hit_rate_percentage": adapter_state.get(
                "training_trace_cache_hit_rate_percentage"
            ),
            "training_reconstruction_mse": adapter_state.get("training_reconstruction_mse"),
            "training_mean_cosine_similarity": adapter_state.get(
                "training_mean_cosine_similarity"
            ),
        }
        print(
            "Generated trajectory adapter "
            f"{info['status']} from {cache_path} "
            f"(row_cache_hit={info.get('training_row_cache_hit')})",
            flush=True,
        )
    memory_cache[cache_key] = info
    return info


def _generated_adapter_alignment_state_from_variant_state(
    variant_state: dict[str, Any],
) -> dict[str, Any]:
    handoff_mapping = variant_state.get(
        "handoff_alignment_q",
        variant_state["global_alignment_q"],
    )
    handoff_bias = variant_state.get(
        "handoff_alignment_bias",
        variant_state.get("global_alignment_bias"),
    )
    return {
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
        "alignment_strategy": variant_state.get("alignment_strategy", "hybrid_affine"),
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


def _generated_adapter_include_prompt_values(
    method_names: Optional[Sequence[str]],
) -> tuple[bool, ...]:
    if method_names is None:
        return (False,)
    values: list[bool] = []
    for method_name in method_names:
        if method_name == "prompt_generated_latent_handoff":
            values.append(True)
        elif method_name in GENERATED_LATENT_METHODS:
            values.append(False)
    deduped: list[bool] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return tuple(deduped or [False])


def _prepare_generated_trajectory_eval_traces(
    *,
    dataset_name: str,
    dataset_split: Optional[str],
    limit: int,
    sample_indices: Optional[list[int]],
    base_cfg: Any,
    variant_cfg: Any,
    variant_state: dict[str, Any],
    method_names: Optional[Sequence[str]],
) -> dict[str, Any]:
    semantic_smoke_cfg = getattr(getattr(base_cfg, "reporting", None), "semantic_smoke", None)
    if sample_indices is None and semantic_smoke_cfg is not None:
        sample_indices = _coerce_sample_indices(getattr(semantic_smoke_cfg, "sample_indices", None))
    effective_split = dataset_split or _default_split_for_dataset(dataset_name)
    validation_size = _validation_size(base_cfg, dataset_name)
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

    prepared_rows: list[dict[str, Any]] = []
    for include_prompt in _generated_adapter_include_prompt_values(method_names):
        for index, row in enumerate(samples):
            sample_index = (
                effective_sample_indices[index]
                if effective_sample_indices is not None and index < len(effective_sample_indices)
                else index
            )
            prompt = pick_field(row, ("question", "problem"))
            if not str(prompt).strip():
                continue
            variant_state["_current_sample_row"] = dict(row)
            sender_state = _collect_sender_generated_consensus_state(
                prompt,
                variant_state,
                variant_cfg,
                include_prompt=include_prompt,
            )
            target_answer = _target_answer(dataset_name, row)
            sender_reasoning_text = str(sender_state.get("generated_reasoning_text", ""))
            sender_predicted_answer = (
                _predicted_answer(dataset_name, sender_reasoning_text)
                if sender_reasoning_text.strip()
                else None
            )
            sender_answer_matches_target = (
                _answers_match(dataset_name, sender_predicted_answer, target_answer)
                if sender_predicted_answer is not None and target_answer is not None
                else None
            )
            prepared_rows.append(
                {
                    "dataset": dataset_name,
                    "dataset_split": effective_split,
                    "sample_index": int(sample_index),
                    "target_answer": target_answer,
                    "include_prompt": bool(include_prompt),
                    "trace_cache_hit": sender_state.get("generated_trace_cache_hit"),
                    "trace_cache_path": sender_state.get("generated_trace_cache_path"),
                    "sender_reasoning_status": sender_state.get("generated_reasoning_status"),
                    "sender_final_answer_marker": sender_state.get(
                        "generated_reasoning_final_answer_marker"
                    ),
                    "sender_revision_enabled": sender_state.get("sender_revision_enabled"),
                    "sender_revision_applied": sender_state.get("sender_revision_applied"),
                    "sender_initial_predicted_answer": sender_state.get(
                        "sender_initial_predicted_answer"
                    ),
                    "sender_revision_predicted_answer": sender_state.get(
                        "sender_revision_predicted_answer"
                    ),
                    "sender_revision_decision_applied": sender_state.get(
                        "sender_revision_decision_applied"
                    ),
                    "sender_revision_decision_predicted_answer": sender_state.get(
                        "sender_revision_decision_predicted_answer"
                    ),
                    "sender_reasoning_token_count": sender_state.get(
                        "generated_reasoning_token_count"
                    ),
                    "sender_predicted_answer": sender_predicted_answer,
                    "sender_answer_matches_target": sender_answer_matches_target,
                }
            )
    trace_cache_values = [
        bool(row["trace_cache_hit"])
        for row in prepared_rows
        if row.get("trace_cache_hit") is not None and row.get("trace_cache_hit") != ""
    ]
    sender_answer_values = [
        bool(row["sender_answer_matches_target"])
        for row in prepared_rows
        if row.get("sender_answer_matches_target") is not None
        and row.get("sender_answer_matches_target") != ""
    ]
    return {
        "dataset": dataset_name,
        "dataset_split": effective_split,
        "limit": int(limit),
        "sample_indices": effective_sample_indices,
        "trace_count": len(prepared_rows),
        "trace_cache_hit_count": sum(1 for value in trace_cache_values if value),
        "trace_cache_miss_count": sum(1 for value in trace_cache_values if not value),
        "trace_cache_hit_rate_percentage": (
            100.0 * sum(1 for value in trace_cache_values if value) / len(trace_cache_values)
            if trace_cache_values
            else None
        ),
        "sender_accuracy_percentage": (
            100.0 * sum(1 for value in sender_answer_values if value) / len(sender_answer_values)
            if sender_answer_values
            else None
        ),
        "traces": prepared_rows,
    }


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
    answer_suffix_text: Optional[str] = None,
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
            suffix_text=answer_suffix_text
            if answer_suffix_text is not None
            else _format_receiver_context_answer_suffix(cfg),
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
            suffix_text = (
                answer_suffix_text
                if answer_suffix_text is not None
                else _format_receiver_context_answer_suffix(cfg)
            )
            if suffix_text.strip():
                prefix_state = append_text_to_prefix_state(
                    model=agent_b,
                    tokenizer=tokenizer_b,
                    prefix_state=prefix_state,
                    suffix_text=suffix_text,
                    decoded_text_prefix="Final answer:",
                )

    outputs_b = prefix_state["outputs"]
    raw_handoff_entropy = float(
        _compute_logits_entropy(outputs_b.logits[:, -1, :]).mean().detach().cpu().item()
    )
    decode_metrics = greedy_decode_from_prefix(
        model=agent_b,
        tokenizer=tokenizer_b,
        prefix_state=prefix_state,
        max_new_tokens=max_new_tokens,
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
    return {
        "decoded_text": decoded_text,
        "generated_tokens": int(decode_metrics["generated_tokens"]),
        "receiver_input_token_count": int(prefix_state["prefix_seq_len"]),
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
    reference_text: Optional[str] = None,
    reference_token_ids: Optional[Sequence[int]] = None,
    reference_target_alignment: str = "linear",
) -> dict[str, Optional[float]]:
    if reference_text is not None and str(reference_text).strip():
        receiver_reference_handoff = _receiver_embedding_sequence_for_aligned_text(
            str(reference_text),
            state=state,
            source_token_ids=reference_token_ids or (),
            target_steps=int(current_latent_step.shape[1]),
            target_alignment=reference_target_alignment,
        )
        if receiver_reference_handoff is None:
            receiver_reference_handoff = _receiver_reference_for_handoff(
                prompt,
                state,
                int(current_latent_step.shape[1]),
            )
    else:
        receiver_reference_handoff = _receiver_reference_for_handoff(
            prompt,
            state,
            int(current_latent_step.shape[1]),
        )
    receiver_reference_handoff = receiver_reference_handoff.to(
        device=current_latent_step.device,
        dtype=current_latent_step.dtype,
    )
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
    return {
        "decoded_text": decoded_text,
        "generated_tokens": int(decode_metrics["generated_tokens"]),
        "receiver_input_token_count": int(prefix_state["prefix_seq_len"]),
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
    reasoning_metadata = _reasoner_metadata_for_text_hybrid(prompt, cfg, state)
    reasoning_token_ids = [
        int(token_id) for token_id in reasoning_metadata["token_ids"]
    ]
    reasoning_text = str(reasoning_metadata["reasoning_text"])
    prefix_text = _serialize_text_hybrid_prompt(prompt, reasoning_text, tokenizer_b, cfg)
    prefix_state = prepare_text_prefix_state(
        model=agent_b,
        tokenizer=tokenizer_b,
        prefix_text=prefix_text,
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
    return {
        "decoded_text": decoded_text,
        "sender_reasoning_text": reasoning_text,
        "sender_reasoning_token_count": len(reasoning_token_ids),
        "receiver_input_token_count": int(prefix_state["prefix_seq_len"]),
        "sender_reasoning_status": _sender_reasoning_status(
            reasoning_token_ids,
            reasoning_text,
            cfg,
        ),
        "sender_final_answer_marker": FINAL_ANSWER_COMPLETE_REGEX.search(reasoning_text) is not None,
        "sender_trace_cache_hit": reasoning_metadata.get("trace_cache_hit"),
        "sender_trace_cache_path": reasoning_metadata.get("trace_cache_path"),
        "sender_revision_enabled": reasoning_metadata.get("sender_revision_enabled"),
        "sender_revision_applied": reasoning_metadata.get("sender_revision_applied"),
        "sender_initial_predicted_answer": reasoning_metadata.get(
            "sender_initial_predicted_answer"
        ),
        "sender_revision_predicted_answer": reasoning_metadata.get(
            "sender_revision_predicted_answer"
        ),
        "sender_revision_decision_applied": reasoning_metadata.get(
            "sender_revision_decision_applied"
        ),
        "sender_revision_decision_predicted_answer": reasoning_metadata.get(
            "sender_revision_decision_predicted_answer"
        ),
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


def _sender_trace_text_baseline_metadata(
    *,
    reasoning_metadata: Mapping[str, Any],
    reasoning_token_ids: Sequence[int],
    reasoning_text: str,
    cfg: Any,
) -> dict[str, Any]:
    return {
        "sender_reasoning_text": reasoning_text,
        "sender_reasoning_token_count": len(reasoning_token_ids),
        "sender_reasoning_status": _sender_reasoning_status(
            reasoning_token_ids,
            reasoning_text,
            cfg,
        ),
        "sender_final_answer_marker": (
            FINAL_ANSWER_COMPLETE_REGEX.search(reasoning_text) is not None
        ),
        "sender_trace_cache_hit": reasoning_metadata.get("trace_cache_hit"),
        "sender_trace_cache_path": reasoning_metadata.get("trace_cache_path"),
        "sender_revision_enabled": reasoning_metadata.get("sender_revision_enabled"),
        "sender_revision_applied": reasoning_metadata.get("sender_revision_applied"),
        "sender_initial_predicted_answer": reasoning_metadata.get(
            "sender_initial_predicted_answer"
        ),
        "sender_revision_predicted_answer": reasoning_metadata.get(
            "sender_revision_predicted_answer"
        ),
        "sender_revision_decision_applied": reasoning_metadata.get(
            "sender_revision_decision_applied"
        ),
        "sender_revision_decision_predicted_answer": reasoning_metadata.get(
            "sender_revision_decision_predicted_answer"
        ),
    }


def run_token_context_handoff(
    prompt: str,
    target_answer_text: Optional[str],
    cfg: Any,
    state: dict[str, Any],
) -> dict[str, Any]:
    tokenizer_b = state["tokenizer_b"]
    agent_b = state["agent_b"]
    reasoning_metadata = _reasoner_metadata_for_text_hybrid(prompt, cfg, state)
    reasoning_token_ids = [
        int(token_id) for token_id in reasoning_metadata["token_ids"]
    ]
    reasoning_text = str(reasoning_metadata["reasoning_text"])
    prefix_text = _format_token_context_handoff_prompt(
        prompt,
        reasoning_text,
        tokenizer_b,
        cfg,
    )
    prefix_state = prepare_text_prefix_state(
        model=agent_b,
        tokenizer=tokenizer_b,
        prefix_text=prefix_text,
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
    return {
        "decoded_text": decoded_text,
        **_sender_trace_text_baseline_metadata(
            reasoning_metadata=reasoning_metadata,
            reasoning_token_ids=reasoning_token_ids,
            reasoning_text=reasoning_text,
            cfg=cfg,
        ),
        "generated_tokens": int(decode_metrics["generated_tokens"]),
        "receiver_input_token_count": int(prefix_state["prefix_seq_len"]),
        "decode_status": "decoded" if decoded_text.strip() else "empty_decode",
        "answer_token_count": answer_metrics["answer_token_count"],
        "answer_nll": answer_metrics["answer_nll"],
        "answer_perplexity": answer_metrics["answer_perplexity"],
        "alignment_mode": "token_context_handoff",
        "handoff_status": "not_applicable",
        "handoff_surface": "text_token_context",
        "kv_cache_transferred": None,
        "kv_cache_status": "not_applicable",
        "kv_cache_reason": "token_context_baseline",
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


def run_verified_token_context_handoff(
    prompt: str,
    target_answer_text: Optional[str],
    cfg: Any,
    state: dict[str, Any],
) -> dict[str, Any]:
    tokenizer_b = state["tokenizer_b"]
    agent_b = state["agent_b"]
    reasoning_metadata = _reasoner_metadata_for_text_hybrid(prompt, cfg, state)
    reasoning_token_ids = [
        int(token_id) for token_id in reasoning_metadata["token_ids"]
    ]
    reasoning_text = str(reasoning_metadata["reasoning_text"])
    sender_answer = _final_answer_marker_value(reasoning_text)
    if sender_answer is None:
        sender_answer = "<missing>"
    prefix_state = prepare_text_prefix_state(
        model=agent_b,
        tokenizer=tokenizer_b,
        prefix_text=_format_verified_token_context_handoff_prompt(
            sender_answer,
            reasoning_text,
            tokenizer_b,
        ),
    )
    prefix_state["decoded_text_prefix"] = "Final answer:"
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
    return {
        "decoded_text": decoded_text,
        **_sender_trace_text_baseline_metadata(
            reasoning_metadata=reasoning_metadata,
            reasoning_token_ids=reasoning_token_ids,
            reasoning_text=reasoning_text,
            cfg=cfg,
        ),
        "generated_tokens": int(decode_metrics["generated_tokens"]),
        "receiver_input_token_count": int(prefix_state["prefix_seq_len"]),
        "decode_status": "decoded" if decoded_text.strip() else "empty_decode",
        "answer_token_count": answer_metrics["answer_token_count"],
        "answer_nll": answer_metrics["answer_nll"],
        "answer_perplexity": answer_metrics["answer_perplexity"],
        "alignment_mode": "verified_token_context_handoff",
        "handoff_status": "not_applicable",
        "handoff_surface": "text_verified_token_context",
        "kv_cache_transferred": None,
        "kv_cache_status": "not_applicable",
        "kv_cache_reason": "verified_token_context_baseline",
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


def run_sender_answer_text_handoff(
    prompt: str,
    target_answer_text: Optional[str],
    cfg: Any,
    state: dict[str, Any],
) -> dict[str, Any]:
    tokenizer_b = state["tokenizer_b"]
    agent_b = state["agent_b"]
    reasoning_metadata = _reasoner_metadata_for_text_hybrid(prompt, cfg, state)
    reasoning_token_ids = [
        int(token_id) for token_id in reasoning_metadata["token_ids"]
    ]
    reasoning_text = str(reasoning_metadata["reasoning_text"])
    sender_answer = _final_answer_marker_value(reasoning_text)
    if sender_answer is None:
        sender_answer = "<missing>"
    del prompt
    prefix_state = prepare_text_prefix_state(
        model=agent_b,
        tokenizer=tokenizer_b,
        prefix_text=_format_sender_answer_text_handoff_prompt(sender_answer, tokenizer_b),
    )
    prefix_state["decoded_text_prefix"] = "Final answer:"
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
    return {
        "decoded_text": decoded_text,
        **_sender_trace_text_baseline_metadata(
            reasoning_metadata=reasoning_metadata,
            reasoning_token_ids=reasoning_token_ids,
            reasoning_text=reasoning_text,
            cfg=cfg,
        ),
        "generated_tokens": int(decode_metrics["generated_tokens"]),
        "receiver_input_token_count": int(prefix_state["prefix_seq_len"]),
        "decode_status": "decoded" if decoded_text.strip() else "empty_decode",
        "answer_token_count": answer_metrics["answer_token_count"],
        "answer_nll": answer_metrics["answer_nll"],
        "answer_perplexity": answer_metrics["answer_perplexity"],
        "alignment_mode": "sender_answer_text_handoff",
        "handoff_status": "not_applicable",
        "handoff_surface": "text",
        "kv_cache_transferred": None,
        "kv_cache_status": "not_applicable",
        "kv_cache_reason": "sender_answer_text_baseline",
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
    handoff_step, manifold_metrics = apply_embedding_manifold_projection(
        handoff_step,
        cfg,
        state,
    )
    decode_metrics = _decode_handoff(
        agent_b=agent_b,
        tokenizer_b=tokenizer_b,
        prompt=_handoff_decode_prompt(prompt, cfg),
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
            cfg=cfg,
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
        "embedding_manifold_enabled": bool(
            getattr(getattr(cfg.handoff, "embedding_manifold", None), "enabled", False)
        ),
        "embedding_manifold_applied": manifold_metrics["embedding_manifold_applied"],
        "embedding_manifold_delta_norm": manifold_metrics["embedding_manifold_delta_norm"],
        "embedding_manifold_mean_top_similarity": manifold_metrics[
            "embedding_manifold_mean_top_similarity"
        ],
        "embedding_manifold_unique_token_count": manifold_metrics[
            "embedding_manifold_unique_token_count"
        ],
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
        prompt=_handoff_decode_prompt(prompt, variant_cfg),
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


def _run_generated_latent_variant(
    prompt: str,
    target_answer_text: Optional[str],
    cfg: Any,
    state: dict[str, Any],
    *,
    include_prompt: bool,
    method_alignment_mode: str,
    use_receiver_context: bool = False,
) -> dict[str, Any]:
    variant_cfg, variant_state = _alignment_variant_state(
        cfg,
        state,
        strategy="hybrid_affine",
        prompt_calibration_enabled=False,
    )
    tokenizer_b = variant_state["tokenizer_b"]
    agent_b = variant_state["agent_b"]
    agent_b_device = next(agent_b.parameters()).device
    sender_state = _collect_sender_generated_consensus_state(
        prompt,
        variant_state,
        variant_cfg,
        include_prompt=include_prompt,
    )
    handoff_source = (
        _generated_trajectory_adapter_source_sequence(variant_cfg, sender_state)
        if _generated_trajectory_adapter_enabled(variant_cfg)
        else sender_state["consensus_hidden_states"]
    )
    alignment_state = _generated_adapter_alignment_state_from_variant_state(variant_state)
    generated_adapter_info = _load_or_train_generated_trajectory_adapter_state(
        variant_cfg,
        variant_state,
        alignment_state,
        include_prompt=include_prompt,
    )
    generated_adapter_state = generated_adapter_info.get("state")
    generated_adapter_input_space = _generated_trajectory_adapter_input_space(variant_cfg)
    generated_adapter_report = bool(
        generated_adapter_info.get("enabled") or generated_adapter_info.get("status") != "disabled"
    )
    if _generated_trajectory_adapter_enabled(variant_cfg) and generated_adapter_state is None:
        latent_trajectory_steps = int(handoff_source.shape[1])
        return {
            "decoded_text": "",
            "generated_tokens": 0,
            "receiver_input_token_count": latent_trajectory_steps,
            "decode_status": "generated_trajectory_adapter_missing",
            "answer_token_count": 0,
            "answer_nll": None,
            "answer_perplexity": None,
            "alignment_mode": method_alignment_mode,
            "alignment_strategy": "hybrid_affine",
            "handoff_status": "generated_trajectory_adapter_missing",
            "handoff_surface": "generated_hidden_sequence_input_embedding",
            "kv_cache_transferred": False,
            "kv_cache_status": "not_provided",
            "kv_cache_reason": "no_cache_provided",
            "active_kv_cache_transferred": False,
            "active_kv_cache_status": "not_provided",
            "active_kv_cache_reason": "no_cache_provided",
            "active_kv_cache_source": "none",
            "receiver_context_status": "not_used",
            "receiver_context_reason": "latent_only",
            "receiver_context_token_count": 0,
            "receiver_context_latent_position": "not_applicable",
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
            "pre_alignment_l2_distance": None,
            "pre_alignment_cosine_distance": None,
            "post_alignment_l2_distance": None,
            "post_alignment_cosine_distance": None,
            "prompt_calibration_enabled": False,
            "prompt_calibration_bias_norm": None,
            "handoff_adapter_enabled": True,
            "handoff_adapter_status": (
                f"generated_trajectory_{generated_adapter_info.get('status')}_"
                f"{generated_adapter_input_space}"
                if generated_adapter_report
                else variant_state.get("handoff_adapter_status")
            ),
            "handoff_adapter_applied": False,
            "handoff_adapter_delta_norm": None,
            "handoff_adapter_cache_hit": generated_adapter_info.get("cache_hit"),
            "handoff_adapter_cache_path": generated_adapter_info.get("cache_path"),
            "handoff_adapter_cache_key_digest": generated_adapter_info.get(
                "adapter_cache_key_digest"
            ),
            "handoff_adapter_training_prompt_count": generated_adapter_info.get(
                "training_prompt_count"
            ),
            "handoff_adapter_training_token_count": generated_adapter_info.get(
                "training_token_count"
            ),
            "handoff_adapter_training_row_cache_hit": generated_adapter_info.get(
                "training_row_cache_hit"
            ),
            "handoff_adapter_training_row_cache_path": generated_adapter_info.get(
                "training_row_cache_path"
            ),
            "handoff_adapter_training_rows_cache_key_digest": generated_adapter_info.get(
                "training_rows_cache_key_digest"
            ),
            "handoff_adapter_training_trace_cache_hit_count": generated_adapter_info.get(
                "training_trace_cache_hit_count"
            ),
            "handoff_adapter_training_trace_cache_miss_count": generated_adapter_info.get(
                "training_trace_cache_miss_count"
            ),
            "handoff_adapter_training_trace_cache_hit_rate_percentage": generated_adapter_info.get(
                "training_trace_cache_hit_rate_percentage"
            ),
            "handoff_adapter_training_reconstruction_mse": generated_adapter_info.get(
                "training_reconstruction_mse"
            ),
            "handoff_adapter_training_mean_cosine_similarity": generated_adapter_info.get(
                "training_mean_cosine_similarity"
            ),
            "generated_adapter_local_residual_applied": False,
            "generated_adapter_local_residual_delta_norm": None,
            "generated_adapter_local_residual_mean_top_similarity": None,
            "generated_adapter_local_residual_memory_rows": None,
            "embedding_manifold_enabled": bool(
                getattr(getattr(variant_cfg.handoff, "embedding_manifold", None), "enabled", False)
            ),
            "embedding_manifold_applied": False,
            "embedding_manifold_delta_norm": None,
            "embedding_manifold_mean_top_similarity": None,
            "embedding_manifold_unique_token_count": None,
            "raw_handoff_entropy": None,
            "handoff_uncertainty": None,
            "confidence_gate_triggered": False,
            "fallback_discrete_reasoning_steps": 0,
            "latent_trajectory_steps": latent_trajectory_steps,
            "total_reasoning_steps": latent_trajectory_steps,
            "continuous_integration_seconds": 0.0,
            "global_alignment_cache_hit": variant_state["global_alignment_cache_hit"],
            "_row_cfg": variant_cfg,
            "_row_state": variant_state,
            "sender_reasoning_text": sender_state["generated_reasoning_text"],
            "sender_reasoning_token_count": sender_state["generated_reasoning_token_count"],
            "sender_reasoning_status": sender_state["generated_reasoning_status"],
            "sender_trace_cache_hit": sender_state.get("generated_trace_cache_hit"),
            "sender_trace_cache_path": sender_state.get("generated_trace_cache_path"),
            "sender_final_answer_marker": sender_state[
                "generated_reasoning_final_answer_marker"
            ],
            "sender_revision_enabled": sender_state.get("sender_revision_enabled"),
            "sender_revision_applied": sender_state.get("sender_revision_applied"),
            "sender_initial_predicted_answer": sender_state.get(
                "sender_initial_predicted_answer"
            ),
            "sender_revision_predicted_answer": sender_state.get(
                "sender_revision_predicted_answer"
            ),
            "sender_revision_decision_applied": sender_state.get(
                "sender_revision_decision_applied"
            ),
            "sender_revision_decision_predicted_answer": sender_state.get(
                "sender_revision_decision_predicted_answer"
            ),
        }
    generated_adapter_delta_norm: Optional[float] = None
    generated_adapter_local_residual_metrics = {
        "generated_adapter_local_residual_applied": False,
        "generated_adapter_local_residual_delta_norm": None,
        "generated_adapter_local_residual_mean_top_similarity": None,
        "generated_adapter_local_residual_memory_rows": None,
    }
    generated_adapter_semantic_memory_metrics = {
        "generated_adapter_semantic_memory_applied": False,
        "generated_adapter_semantic_memory_similarity": None,
        "generated_adapter_semantic_memory_entry_count": None,
        "generated_adapter_semantic_memory_target_text": None,
    }
    if generated_adapter_state is not None:
        adapter_input = (
            handoff_source
            if generated_adapter_input_space == "raw"
            else apply_alignment(handoff_source, alignment_state)
        )
        handoff_step = adapter_input.to(
            device=agent_b_device,
            dtype=agent_b.get_input_embeddings().weight.dtype,
        )
        adapted_handoff_step = apply_alignment(adapter_input, generated_adapter_state).to(
            device=agent_b_device,
            dtype=agent_b.get_input_embeddings().weight.dtype,
        )
        adapted_handoff_step, generated_adapter_local_residual_metrics = (
            _apply_generated_adapter_local_residual(
                adapter_input,
                adapted_handoff_step,
                generated_adapter_state,
            )
        )
        generated_adapter_delta_norm = _mean_handoff_delta_norm(handoff_step, adapted_handoff_step)
        handoff_step = adapted_handoff_step
        generated_adapter_semantic_memory_metrics = _apply_generated_adapter_semantic_memory(
            adapter_input,
            generated_adapter_state,
            variant_cfg,
        )
    else:
        handoff_step = apply_alignment(handoff_source, alignment_state).to(
            device=agent_b_device,
            dtype=agent_b.get_input_embeddings().weight.dtype,
        )
    generic_handoff_step, generic_adapter_metrics = apply_handoff_adapter(
        handoff_step,
        variant_state,
    )
    if generic_adapter_metrics["handoff_adapter_applied"]:
        handoff_step = generic_handoff_step
    generated_adapter_applied = generated_adapter_state is not None
    adapter_metrics = {
        "handoff_adapter_applied": bool(
            generated_adapter_applied or generic_adapter_metrics["handoff_adapter_applied"]
        ),
        "handoff_adapter_delta_norm": (
            generated_adapter_delta_norm
            if generated_adapter_applied
            else generic_adapter_metrics["handoff_adapter_delta_norm"]
        ),
    }
    generated_reference_text = _generated_trajectory_adapter_target_text(
        variant_cfg,
        str(sender_state.get("generated_reasoning_text", "")),
    )
    distance_alignment_state = (
        generated_adapter_state
        if generated_adapter_state is not None and generated_adapter_input_space == "raw"
        else alignment_state
    )
    distance_adapter_state = (
        generated_adapter_state
        if generated_adapter_state is not None and generated_adapter_input_space == "aligned"
        else None
    )
    if generated_adapter_semantic_memory_metrics["generated_adapter_semantic_memory_applied"]:
        latent_trajectory_steps = int(handoff_source.shape[1])
        decoded_text = str(
            generated_adapter_semantic_memory_metrics[
                "generated_adapter_semantic_memory_target_text"
            ]
            or ""
        )
        return {
            "decoded_text": decoded_text,
            "generated_tokens": 0,
            "receiver_input_token_count": latent_trajectory_steps,
            "decode_status": "semantic_memory_readout",
            "answer_token_count": 0,
            "answer_nll": None,
            "answer_perplexity": None,
            **_alignment_distances(
                prompt=prompt,
                state=variant_state,
                current_latent_step=handoff_source,
                alignment_state=distance_alignment_state,
                cfg=variant_cfg,
                adapter_state=distance_adapter_state,
                calibration_strength=0.0,
                calibration_max_norm_ratio=0.0,
                reference_text=generated_reference_text,
                reference_token_ids=_source_token_ids_for_generated_adapter(
                    variant_cfg,
                    sender_state,
                ),
                reference_target_alignment=_generated_trajectory_adapter_target_alignment(
                    variant_cfg
                ),
            ),
            "alignment_mode": method_alignment_mode,
            "alignment_strategy": "hybrid_affine",
            "handoff_status": "ok",
            "handoff_surface": "generated_hidden_sequence_semantic_memory",
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
            "prompt_calibration_enabled": False,
            "prompt_calibration_bias_norm": None,
            "handoff_adapter_enabled": bool(
                generated_adapter_info.get("enabled")
                or variant_state.get("handoff_adapter_enabled", False)
            ),
            "handoff_adapter_status": (
                f"generated_trajectory_{generated_adapter_info.get('status')}_"
                f"{generated_adapter_input_space}"
                if generated_adapter_report
                else variant_state.get("handoff_adapter_status")
            ),
            "handoff_adapter_applied": adapter_metrics["handoff_adapter_applied"],
            "handoff_adapter_delta_norm": adapter_metrics["handoff_adapter_delta_norm"],
            "handoff_adapter_cache_hit": (
                generated_adapter_info.get("cache_hit")
                if generated_adapter_report
                else variant_state.get("handoff_adapter_cache_hit")
            ),
            "handoff_adapter_cache_path": (
                generated_adapter_info.get("cache_path")
                if generated_adapter_report
                else variant_state.get("handoff_adapter_cache_path")
            ),
            "handoff_adapter_cache_key_digest": (
                generated_adapter_info.get("adapter_cache_key_digest")
                if generated_adapter_report
                else variant_state.get("handoff_adapter_cache_key_digest")
            ),
            "handoff_adapter_training_prompt_count": generated_adapter_info.get(
                "training_prompt_count"
            ),
            "handoff_adapter_training_token_count": generated_adapter_info.get(
                "training_token_count"
            ),
            "handoff_adapter_training_row_cache_hit": generated_adapter_info.get(
                "training_row_cache_hit"
            ),
            "handoff_adapter_training_row_cache_path": generated_adapter_info.get(
                "training_row_cache_path"
            ),
            "handoff_adapter_training_rows_cache_key_digest": generated_adapter_info.get(
                "training_rows_cache_key_digest"
            ),
            "handoff_adapter_training_trace_cache_hit_count": generated_adapter_info.get(
                "training_trace_cache_hit_count"
            ),
            "handoff_adapter_training_trace_cache_miss_count": generated_adapter_info.get(
                "training_trace_cache_miss_count"
            ),
            "handoff_adapter_training_trace_cache_hit_rate_percentage": generated_adapter_info.get(
                "training_trace_cache_hit_rate_percentage"
            ),
            "handoff_adapter_training_reconstruction_mse": generated_adapter_info.get(
                "training_reconstruction_mse"
            ),
            "handoff_adapter_training_mean_cosine_similarity": generated_adapter_info.get(
                "training_mean_cosine_similarity"
            ),
            **generated_adapter_local_residual_metrics,
            **generated_adapter_semantic_memory_metrics,
            "embedding_manifold_enabled": bool(
                getattr(getattr(variant_cfg.handoff, "embedding_manifold", None), "enabled", False)
            ),
            "embedding_manifold_applied": False,
            "embedding_manifold_delta_norm": None,
            "embedding_manifold_mean_top_similarity": None,
            "embedding_manifold_unique_token_count": None,
            "raw_handoff_entropy": None,
            "handoff_uncertainty": None,
            "confidence_gate_triggered": False,
            "fallback_discrete_reasoning_steps": 0,
            "latent_trajectory_steps": latent_trajectory_steps,
            "total_reasoning_steps": latent_trajectory_steps,
            "continuous_integration_seconds": 0.0,
            "global_alignment_cache_hit": variant_state["global_alignment_cache_hit"],
            "_row_cfg": variant_cfg,
            "_row_state": variant_state,
            "sender_reasoning_text": sender_state["generated_reasoning_text"],
            "sender_reasoning_token_count": sender_state["generated_reasoning_token_count"],
            "sender_reasoning_status": sender_state["generated_reasoning_status"],
            "sender_trace_cache_hit": sender_state.get("generated_trace_cache_hit"),
            "sender_trace_cache_path": sender_state.get("generated_trace_cache_path"),
            "sender_final_answer_marker": sender_state["generated_reasoning_final_answer_marker"],
            "sender_revision_enabled": sender_state.get("sender_revision_enabled"),
            "sender_revision_applied": sender_state.get("sender_revision_applied"),
            "sender_initial_predicted_answer": sender_state.get("sender_initial_predicted_answer"),
            "sender_revision_predicted_answer": sender_state.get("sender_revision_predicted_answer"),
            "sender_revision_decision_applied": sender_state.get(
                "sender_revision_decision_applied"
            ),
            "sender_revision_decision_predicted_answer": sender_state.get(
                "sender_revision_decision_predicted_answer"
            ),
        }
    handoff_step, manifold_metrics = apply_embedding_manifold_projection(
        handoff_step,
        variant_cfg,
        variant_state,
    )
    decode_metrics = _decode_handoff(
        agent_b=agent_b,
        tokenizer_b=tokenizer_b,
        prompt=prompt if use_receiver_context else None,
        cfg=variant_cfg,
        handoff_step=handoff_step,
        kv_cache_a=None,
        max_new_tokens=int(cfg.max_new_tokens),
        target_answer_text=target_answer_text,
        append_answer_suffix_without_context=not use_receiver_context,
        answer_suffix_text="\n\nRepeat the final answer from the latent reasoning.\nFinal answer:",
    )
    latent_trajectory_steps = int(handoff_source.shape[1])
    return {
        **decode_metrics,
        **_alignment_distances(
            prompt=prompt,
            state=variant_state,
            current_latent_step=handoff_source,
            alignment_state=distance_alignment_state,
            cfg=variant_cfg,
            adapter_state=distance_adapter_state,
            calibration_strength=0.0,
            calibration_max_norm_ratio=0.0,
            reference_text=generated_reference_text,
            reference_token_ids=_source_token_ids_for_generated_adapter(variant_cfg, sender_state),
            reference_target_alignment=_generated_trajectory_adapter_target_alignment(variant_cfg),
        ),
        "alignment_mode": method_alignment_mode,
        "alignment_strategy": "hybrid_affine",
        "handoff_status": "ok",
        "handoff_surface": "generated_hidden_sequence_input_embedding",
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
        "prompt_calibration_enabled": False,
        "prompt_calibration_bias_norm": None,
        "handoff_adapter_enabled": bool(
            generated_adapter_info.get("enabled")
            or variant_state.get("handoff_adapter_enabled", False)
        ),
        "handoff_adapter_status": (
            f"generated_trajectory_{generated_adapter_info.get('status')}_{generated_adapter_input_space}"
            if generated_adapter_report
            else variant_state.get("handoff_adapter_status")
        ),
        "handoff_adapter_applied": adapter_metrics["handoff_adapter_applied"],
        "handoff_adapter_delta_norm": adapter_metrics["handoff_adapter_delta_norm"],
        "handoff_adapter_cache_hit": (
            generated_adapter_info.get("cache_hit")
            if generated_adapter_report
            else variant_state.get("handoff_adapter_cache_hit")
        ),
        "handoff_adapter_cache_path": (
            generated_adapter_info.get("cache_path")
            if generated_adapter_report
            else variant_state.get("handoff_adapter_cache_path")
        ),
        "handoff_adapter_cache_key_digest": (
            generated_adapter_info.get("adapter_cache_key_digest")
            if generated_adapter_report
            else variant_state.get("handoff_adapter_cache_key_digest")
        ),
        "handoff_adapter_training_prompt_count": variant_state.get(
            "handoff_adapter_training_prompt_count"
        )
        if not generated_adapter_report
        else generated_adapter_info.get("training_prompt_count"),
        "handoff_adapter_training_token_count": variant_state.get(
            "handoff_adapter_training_token_count"
        )
        if not generated_adapter_report
        else generated_adapter_info.get("training_token_count"),
        "handoff_adapter_training_row_cache_hit": variant_state.get(
            "handoff_adapter_training_row_cache_hit"
        )
        if not generated_adapter_report
        else generated_adapter_info.get("training_row_cache_hit"),
        "handoff_adapter_training_row_cache_path": variant_state.get(
            "handoff_adapter_training_row_cache_path"
        )
        if not generated_adapter_report
        else generated_adapter_info.get("training_row_cache_path"),
        "handoff_adapter_training_rows_cache_key_digest": variant_state.get(
            "handoff_adapter_training_rows_cache_key_digest"
        )
        if not generated_adapter_report
        else generated_adapter_info.get("training_rows_cache_key_digest"),
        "handoff_adapter_training_trace_cache_hit_count": variant_state.get(
            "handoff_adapter_training_trace_cache_hit_count"
        )
        if not generated_adapter_report
        else generated_adapter_info.get("training_trace_cache_hit_count"),
        "handoff_adapter_training_trace_cache_miss_count": variant_state.get(
            "handoff_adapter_training_trace_cache_miss_count"
        )
        if not generated_adapter_report
        else generated_adapter_info.get("training_trace_cache_miss_count"),
        "handoff_adapter_training_trace_cache_hit_rate_percentage": variant_state.get(
            "handoff_adapter_training_trace_cache_hit_rate_percentage"
        )
        if not generated_adapter_report
        else generated_adapter_info.get("training_trace_cache_hit_rate_percentage"),
        "handoff_adapter_training_reconstruction_mse": variant_state.get(
            "handoff_adapter_training_reconstruction_mse"
        )
        if not generated_adapter_report
        else generated_adapter_info.get("training_reconstruction_mse"),
        "handoff_adapter_training_mean_cosine_similarity": variant_state.get(
            "handoff_adapter_training_mean_cosine_similarity"
        )
        if not generated_adapter_report
        else generated_adapter_info.get("training_mean_cosine_similarity"),
        **generated_adapter_local_residual_metrics,
        **generated_adapter_semantic_memory_metrics,
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
        "latent_trajectory_steps": latent_trajectory_steps,
        "total_reasoning_steps": latent_trajectory_steps,
        "continuous_integration_seconds": 0.0,
        "global_alignment_cache_hit": variant_state["global_alignment_cache_hit"],
        "_row_cfg": variant_cfg,
        "_row_state": variant_state,
        "sender_reasoning_text": sender_state["generated_reasoning_text"],
        "sender_reasoning_token_count": sender_state["generated_reasoning_token_count"],
        "sender_reasoning_status": sender_state["generated_reasoning_status"],
        "sender_trace_cache_hit": sender_state.get("generated_trace_cache_hit"),
        "sender_trace_cache_path": sender_state.get("generated_trace_cache_path"),
        "sender_final_answer_marker": sender_state["generated_reasoning_final_answer_marker"],
        "sender_revision_enabled": sender_state.get("sender_revision_enabled"),
        "sender_revision_applied": sender_state.get("sender_revision_applied"),
        "sender_initial_predicted_answer": sender_state.get("sender_initial_predicted_answer"),
        "sender_revision_predicted_answer": sender_state.get("sender_revision_predicted_answer"),
        "sender_revision_decision_applied": sender_state.get(
            "sender_revision_decision_applied"
        ),
        "sender_revision_decision_predicted_answer": sender_state.get(
            "sender_revision_decision_predicted_answer"
        ),
    }


def run_generated_latent_handoff(
    prompt: str,
    target_answer_text: Optional[str],
    cfg: Any,
    state: dict[str, Any],
) -> dict[str, Any]:
    return _run_generated_latent_variant(
        prompt,
        target_answer_text,
        cfg,
        state,
        include_prompt=False,
        method_alignment_mode="generated_latent_handoff",
    )


def run_prompt_generated_latent_handoff(
    prompt: str,
    target_answer_text: Optional[str],
    cfg: Any,
    state: dict[str, Any],
) -> dict[str, Any]:
    return _run_generated_latent_variant(
        prompt,
        target_answer_text,
        cfg,
        state,
        include_prompt=True,
        method_alignment_mode="prompt_generated_latent_handoff",
    )


def run_generated_context_latent_handoff(
    prompt: str,
    target_answer_text: Optional[str],
    cfg: Any,
    state: dict[str, Any],
) -> dict[str, Any]:
    return _run_generated_latent_variant(
        prompt,
        target_answer_text,
        cfg,
        state,
        include_prompt=False,
        use_receiver_context=True,
        method_alignment_mode="generated_context_latent_handoff",
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
            ("token_context_handoff", run_token_context_handoff),
            ("verified_token_context_handoff", run_verified_token_context_handoff),
            ("sender_answer_text_handoff", run_sender_answer_text_handoff),
            ("homogeneous_ridge_latent", run_homogeneous_ridge_latent),
            ("homogeneous_orthogonal_latent", run_homogeneous_orthogonal_latent),
        ]
    else:
        methods = [
            ("pure_text_cot", run_pure_text_cot),
            ("text_text_hybrid", run_text_text_hybrid),
            ("token_context_handoff", run_token_context_handoff),
            ("verified_token_context_handoff", run_verified_token_context_handoff),
            ("sender_answer_text_handoff", run_sender_answer_text_handoff),
            ("prompt_local_latent", run_prompt_local_latent),
            ("global_anchor_orthogonal", run_global_anchor_orthogonal),
            ("global_anchor_ridge", run_global_anchor_ridge),
            ("global_anchor_hybrid_affine", run_global_anchor_hybrid_affine),
            (
                "global_anchor_hybrid_affine_plus_calibration",
                run_global_anchor_hybrid_affine_plus_calibration,
            ),
            ("generated_latent_handoff", run_generated_latent_handoff),
            ("prompt_generated_latent_handoff", run_prompt_generated_latent_handoff),
            ("generated_context_latent_handoff", run_generated_context_latent_handoff),
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


def _configured_base_cfg(
    *,
    agent_a_model: Optional[str] = None,
    agent_b_model: Optional[str] = None,
    torch_dtype: Optional[str] = None,
    device_map: Optional[str] = None,
    latent_pooling: Optional[str] = None,
    receiver_context_mode: Optional[str] = None,
    receiver_context_latent_position: Optional[str] = None,
    prompt_calibration_enabled: Optional[bool] = None,
    prompt_calibration_strength: Optional[float] = None,
    prompt_calibration_max_norm_ratio: Optional[float] = None,
    handoff_adapter_enabled: Optional[bool] = None,
    handoff_adapter_train_on_missing: Optional[bool] = None,
    handoff_adapter_train_limit: Optional[int] = None,
    generated_trajectory_adapter_enabled: Optional[bool] = None,
    generated_trajectory_adapter_train_on_missing: Optional[bool] = None,
    generated_trajectory_adapter_train_limit: Optional[int] = None,
    generated_trajectory_adapter_train_split: Optional[str] = None,
    generated_trajectory_adapter_input_space: Optional[str] = None,
    generated_trajectory_adapter_target_alignment: Optional[str] = None,
    generated_trajectory_adapter_source_mode: Optional[str] = None,
    generated_trajectory_adapter_source_tail_tokens: Optional[int] = None,
    generated_trajectory_adapter_target_mode: Optional[str] = None,
    generated_trajectory_adapter_local_residual_enabled: Optional[bool] = None,
    generated_trajectory_adapter_local_residual_top_k: Optional[int] = None,
    generated_trajectory_adapter_local_residual_temperature: Optional[float] = None,
    generated_trajectory_adapter_local_residual_blend: Optional[float] = None,
    generated_trajectory_adapter_local_residual_max_memory_rows: Optional[int] = None,
    generated_trajectory_adapter_semantic_memory_enabled: Optional[bool] = None,
    generated_trajectory_adapter_semantic_memory_min_similarity: Optional[float] = None,
    generated_trajectory_adapter_semantic_memory_max_entries: Optional[int] = None,
    embedding_manifold_enabled: Optional[bool] = None,
    embedding_manifold_top_k: Optional[int] = None,
    embedding_manifold_blend: Optional[float] = None,
    semantic_min_sender_accuracy_percentage: Optional[float] = None,
    sender_revision_enabled: Optional[bool] = None,
    sender_revision_max_new_tokens: Optional[int] = None,
    sender_revision_disagreement_verifier_enabled: Optional[bool] = None,
    sender_revision_disagreement_verifier_max_new_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    reasoner_max_new_tokens: Optional[int] = None,
    semantic_smoke: bool = False,
    mvp_smoke: bool = False,
    hetero_smoke: bool = False,
    answer_only_final: bool = False,
) -> Any:
    base_cfg = _load_cfg()
    _apply_model_profile_defaults(
        base_cfg,
        agent_a_model=agent_a_model,
        agent_b_model=agent_b_model,
        hetero_smoke=hetero_smoke,
    )
    if torch_dtype is not None:
        base_cfg.torch_dtype = str(torch_dtype)
    if device_map is not None:
        base_cfg.device_map = str(device_map)
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
    if generated_trajectory_adapter_enabled is not None:
        base_cfg.handoff.generated_trajectory_adapter.enabled = bool(
            generated_trajectory_adapter_enabled
        )
    if generated_trajectory_adapter_train_on_missing is not None:
        base_cfg.handoff.generated_trajectory_adapter.train_on_missing = bool(
            generated_trajectory_adapter_train_on_missing
        )
    if generated_trajectory_adapter_train_limit is not None:
        base_cfg.handoff.generated_trajectory_adapter.train_limit = int(
            generated_trajectory_adapter_train_limit
        )
    if generated_trajectory_adapter_train_split is not None:
        base_cfg.handoff.generated_trajectory_adapter.train_split = str(
            generated_trajectory_adapter_train_split
        )
    if generated_trajectory_adapter_input_space is not None:
        base_cfg.handoff.generated_trajectory_adapter.input_space = str(
            generated_trajectory_adapter_input_space
        )
    if generated_trajectory_adapter_target_alignment is not None:
        base_cfg.handoff.generated_trajectory_adapter.target_alignment = str(
            generated_trajectory_adapter_target_alignment
        )
    if generated_trajectory_adapter_source_mode is not None:
        base_cfg.handoff.generated_trajectory_adapter.source_mode = str(
            generated_trajectory_adapter_source_mode
        )
    if generated_trajectory_adapter_source_tail_tokens is not None:
        base_cfg.handoff.generated_trajectory_adapter.source_tail_tokens = int(
            generated_trajectory_adapter_source_tail_tokens
        )
    if generated_trajectory_adapter_target_mode is not None:
        base_cfg.handoff.generated_trajectory_adapter.target_mode = str(
            generated_trajectory_adapter_target_mode
        )
    if generated_trajectory_adapter_local_residual_enabled is not None:
        base_cfg.handoff.generated_trajectory_adapter.local_residual.enabled = bool(
            generated_trajectory_adapter_local_residual_enabled
        )
    if generated_trajectory_adapter_local_residual_top_k is not None:
        base_cfg.handoff.generated_trajectory_adapter.local_residual.top_k = int(
            generated_trajectory_adapter_local_residual_top_k
        )
    if generated_trajectory_adapter_local_residual_temperature is not None:
        base_cfg.handoff.generated_trajectory_adapter.local_residual.temperature = float(
            generated_trajectory_adapter_local_residual_temperature
        )
    if generated_trajectory_adapter_local_residual_blend is not None:
        base_cfg.handoff.generated_trajectory_adapter.local_residual.blend = float(
            generated_trajectory_adapter_local_residual_blend
        )
    if generated_trajectory_adapter_local_residual_max_memory_rows is not None:
        base_cfg.handoff.generated_trajectory_adapter.local_residual.max_memory_rows = int(
            generated_trajectory_adapter_local_residual_max_memory_rows
        )
    if generated_trajectory_adapter_semantic_memory_enabled is not None:
        base_cfg.handoff.generated_trajectory_adapter.semantic_memory.enabled = bool(
            generated_trajectory_adapter_semantic_memory_enabled
        )
    if generated_trajectory_adapter_semantic_memory_min_similarity is not None:
        base_cfg.handoff.generated_trajectory_adapter.semantic_memory.min_similarity = float(
            generated_trajectory_adapter_semantic_memory_min_similarity
        )
    if generated_trajectory_adapter_semantic_memory_max_entries is not None:
        base_cfg.handoff.generated_trajectory_adapter.semantic_memory.max_entries = int(
            generated_trajectory_adapter_semantic_memory_max_entries
        )
    if embedding_manifold_enabled is not None:
        base_cfg.handoff.embedding_manifold.enabled = bool(embedding_manifold_enabled)
    if embedding_manifold_top_k is not None:
        base_cfg.handoff.embedding_manifold.top_k = int(embedding_manifold_top_k)
    if embedding_manifold_blend is not None:
        base_cfg.handoff.embedding_manifold.blend = float(embedding_manifold_blend)
    if semantic_min_sender_accuracy_percentage is not None:
        base_cfg.reporting.semantic_smoke.min_sender_accuracy_percentage = float(
            semantic_min_sender_accuracy_percentage
        )
    if (
        sender_revision_enabled is not None
        or sender_revision_max_new_tokens is not None
        or sender_revision_disagreement_verifier_enabled is not None
        or sender_revision_disagreement_verifier_max_new_tokens is not None
    ):
        if getattr(base_cfg.benchmark, "sender_revision", None) is None:
            base_cfg.benchmark.sender_revision = OmegaConf.create({})
    if sender_revision_enabled is not None:
        base_cfg.benchmark.sender_revision.enabled = bool(sender_revision_enabled)
    if sender_revision_max_new_tokens is not None:
        base_cfg.benchmark.sender_revision.max_new_tokens = int(sender_revision_max_new_tokens)
    if sender_revision_disagreement_verifier_enabled is not None:
        base_cfg.benchmark.sender_revision.disagreement_verifier_enabled = bool(
            sender_revision_disagreement_verifier_enabled
        )
    if sender_revision_disagreement_verifier_max_new_tokens is not None:
        base_cfg.benchmark.sender_revision.disagreement_verifier_max_new_tokens = int(
            sender_revision_disagreement_verifier_max_new_tokens
        )
    if seed is not None:
        base_cfg.seed = int(seed)
    if max_new_tokens is not None:
        base_cfg.max_new_tokens = int(max_new_tokens)
    if reasoner_max_new_tokens is not None:
        base_cfg.benchmark.text_hybrid_reasoning_max_new_tokens = int(reasoner_max_new_tokens)
    if semantic_smoke or mvp_smoke or hetero_smoke or answer_only_final:
        base_cfg.benchmark.answer_only_final = True
    if semantic_smoke or mvp_smoke or hetero_smoke:
        smoke_reasoner_max_new_tokens = (
            DEFAULT_HETERO_SMOKE_REASONER_MAX_NEW_TOKENS
            if hetero_smoke
            else DEFAULT_SEMANTIC_SMOKE_REASONER_MAX_NEW_TOKENS
        )
        base_cfg.benchmark.text_hybrid_reasoning_max_new_tokens = max(
            int(getattr(base_cfg.benchmark, "text_hybrid_reasoning_max_new_tokens", 0)),
            smoke_reasoner_max_new_tokens,
        )
    if semantic_smoke:
        base_cfg.reporting.semantic_smoke.baseline_methods = list(
            DEFAULT_SEMANTIC_SMOKE_BASELINE_METHODS
        )
        base_cfg.reporting.semantic_smoke.latent_methods = list(
            DEFAULT_SEMANTIC_SMOKE_LATENT_METHODS
        )
        base_cfg.reporting.semantic_smoke.require_final_answer_marker_methods = list(
            DEFAULT_SEMANTIC_SMOKE_METHODS
        )
    if mvp_smoke:
        base_cfg.reporting.semantic_smoke.baseline_methods = list(DEFAULT_MVP_SMOKE_BASELINE_METHODS)
        base_cfg.reporting.semantic_smoke.latent_methods = list(DEFAULT_MVP_SMOKE_LATENT_METHODS)
        base_cfg.reporting.semantic_smoke.require_final_answer_marker_methods = list(DEFAULT_MVP_SMOKE_METHODS)
    if hetero_smoke:
        base_cfg.reporting.semantic_smoke.baseline_methods = list(DEFAULT_HETERO_SMOKE_BASELINE_METHODS)
        base_cfg.reporting.semantic_smoke.latent_methods = list(DEFAULT_HETERO_SMOKE_LATENT_METHODS)
        base_cfg.reporting.semantic_smoke.require_final_answer_marker_methods = list(DEFAULT_HETERO_SMOKE_METHODS)
    return base_cfg


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


def _require_requested_device_available(cfg: Any) -> None:
    device_map = str(getattr(cfg, "device_map", "auto")).strip().lower()
    if device_map == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device_map=mps requested, but torch MPS is not available")


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
    eval_manifest_output_path: Optional[Path] = None,
    max_new_tokens: Optional[int] = None,
    reasoner_max_new_tokens: Optional[int] = None,
    latent_steps_values: Optional[list[int]] = None,
    agent_a_model: Optional[str] = None,
    agent_b_model: Optional[str] = None,
    torch_dtype: Optional[str] = None,
    device_map: Optional[str] = None,
    latent_pooling: Optional[str] = None,
    receiver_context_mode: Optional[str] = None,
    receiver_context_latent_position: Optional[str] = None,
    prompt_calibration_enabled: Optional[bool] = None,
    prompt_calibration_strength: Optional[float] = None,
    prompt_calibration_max_norm_ratio: Optional[float] = None,
    handoff_adapter_enabled: Optional[bool] = None,
    handoff_adapter_train_on_missing: Optional[bool] = None,
    handoff_adapter_train_limit: Optional[int] = None,
    generated_trajectory_adapter_enabled: Optional[bool] = None,
    generated_trajectory_adapter_train_on_missing: Optional[bool] = None,
    generated_trajectory_adapter_train_limit: Optional[int] = None,
    generated_trajectory_adapter_train_split: Optional[str] = None,
    generated_trajectory_adapter_input_space: Optional[str] = None,
    generated_trajectory_adapter_target_alignment: Optional[str] = None,
    generated_trajectory_adapter_source_mode: Optional[str] = None,
    generated_trajectory_adapter_source_tail_tokens: Optional[int] = None,
    generated_trajectory_adapter_target_mode: Optional[str] = None,
    generated_trajectory_adapter_local_residual_enabled: Optional[bool] = None,
    generated_trajectory_adapter_local_residual_top_k: Optional[int] = None,
    generated_trajectory_adapter_local_residual_temperature: Optional[float] = None,
    generated_trajectory_adapter_local_residual_blend: Optional[float] = None,
    generated_trajectory_adapter_local_residual_max_memory_rows: Optional[int] = None,
    generated_trajectory_adapter_semantic_memory_enabled: Optional[bool] = None,
    generated_trajectory_adapter_semantic_memory_min_similarity: Optional[float] = None,
    generated_trajectory_adapter_semantic_memory_max_entries: Optional[int] = None,
    embedding_manifold_enabled: Optional[bool] = None,
    embedding_manifold_top_k: Optional[int] = None,
    embedding_manifold_blend: Optional[float] = None,
    semantic_min_sender_accuracy_percentage: Optional[float] = None,
    sender_revision_enabled: Optional[bool] = None,
    sender_revision_max_new_tokens: Optional[int] = None,
    sender_revision_disagreement_verifier_enabled: Optional[bool] = None,
    sender_revision_disagreement_verifier_max_new_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    method_names: Optional[list[str]] = None,
    sample_indices: Optional[list[int]] = None,
    semantic_smoke: bool = False,
    mvp_smoke: bool = False,
    hetero_smoke: bool = False,
    answer_only_final: bool = False,
    locked_eval_manifest: Optional[Mapping[str, Any]] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    base_cfg = _configured_base_cfg(
        agent_a_model=agent_a_model,
        agent_b_model=agent_b_model,
        torch_dtype=torch_dtype,
        device_map=device_map,
        latent_pooling=latent_pooling,
        receiver_context_mode=receiver_context_mode,
        receiver_context_latent_position=receiver_context_latent_position,
        prompt_calibration_enabled=prompt_calibration_enabled,
        prompt_calibration_strength=prompt_calibration_strength,
        prompt_calibration_max_norm_ratio=prompt_calibration_max_norm_ratio,
        handoff_adapter_enabled=handoff_adapter_enabled,
        handoff_adapter_train_on_missing=handoff_adapter_train_on_missing,
        handoff_adapter_train_limit=handoff_adapter_train_limit,
        generated_trajectory_adapter_enabled=generated_trajectory_adapter_enabled,
        generated_trajectory_adapter_train_on_missing=generated_trajectory_adapter_train_on_missing,
        generated_trajectory_adapter_train_limit=generated_trajectory_adapter_train_limit,
        generated_trajectory_adapter_train_split=generated_trajectory_adapter_train_split,
        generated_trajectory_adapter_input_space=generated_trajectory_adapter_input_space,
        generated_trajectory_adapter_target_alignment=generated_trajectory_adapter_target_alignment,
        generated_trajectory_adapter_source_mode=generated_trajectory_adapter_source_mode,
        generated_trajectory_adapter_source_tail_tokens=generated_trajectory_adapter_source_tail_tokens,
        generated_trajectory_adapter_target_mode=generated_trajectory_adapter_target_mode,
        generated_trajectory_adapter_local_residual_enabled=(
            generated_trajectory_adapter_local_residual_enabled
        ),
        generated_trajectory_adapter_local_residual_top_k=(
            generated_trajectory_adapter_local_residual_top_k
        ),
        generated_trajectory_adapter_local_residual_temperature=(
            generated_trajectory_adapter_local_residual_temperature
        ),
        generated_trajectory_adapter_local_residual_blend=(
            generated_trajectory_adapter_local_residual_blend
        ),
        generated_trajectory_adapter_local_residual_max_memory_rows=(
            generated_trajectory_adapter_local_residual_max_memory_rows
        ),
        generated_trajectory_adapter_semantic_memory_enabled=(
            generated_trajectory_adapter_semantic_memory_enabled
        ),
        generated_trajectory_adapter_semantic_memory_min_similarity=(
            generated_trajectory_adapter_semantic_memory_min_similarity
        ),
        generated_trajectory_adapter_semantic_memory_max_entries=(
            generated_trajectory_adapter_semantic_memory_max_entries
        ),
        embedding_manifold_enabled=embedding_manifold_enabled,
        embedding_manifold_top_k=embedding_manifold_top_k,
        embedding_manifold_blend=embedding_manifold_blend,
        semantic_min_sender_accuracy_percentage=semantic_min_sender_accuracy_percentage,
        sender_revision_enabled=sender_revision_enabled,
        sender_revision_max_new_tokens=sender_revision_max_new_tokens,
        sender_revision_disagreement_verifier_enabled=(
            sender_revision_disagreement_verifier_enabled
        ),
        sender_revision_disagreement_verifier_max_new_tokens=(
            sender_revision_disagreement_verifier_max_new_tokens
        ),
        seed=seed,
        max_new_tokens=max_new_tokens,
        reasoner_max_new_tokens=reasoner_max_new_tokens,
        semantic_smoke=semantic_smoke,
        mvp_smoke=mvp_smoke,
        hetero_smoke=hetero_smoke,
        answer_only_final=answer_only_final,
    )
    if dataset_name == "long_context_handoff":
        base_cfg.handoff.generated_trajectory_adapter.dataset_name = dataset_name
    semantic_smoke_cfg = getattr(getattr(base_cfg, "reporting", None), "semantic_smoke", None)
    if sample_indices is None and semantic_smoke_cfg is not None:
        sample_indices = _coerce_sample_indices(getattr(semantic_smoke_cfg, "sample_indices", None))
    effective_split = dataset_split or _default_split_for_dataset(dataset_name)
    validation_size = _validation_size(base_cfg, dataset_name)
    suite_cfg = _suite_cfg(base_cfg, suite_name)
    _require_requested_device_available(suite_cfg)
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
    sample_fingerprint_rows = _sample_fingerprints(
        samples,
        limit=limit,
        sample_indices=effective_sample_indices,
    )
    report_method_names = [name for name, _ in methods]
    eval_manifest = _build_eval_manifest(
        suite_name=suite_name,
        dataset_name=dataset_name,
        dataset_split=effective_split,
        limit=limit,
        sample_indices=effective_sample_indices,
        methods=report_method_names,
        agent_a_model=str(suite_cfg.agent_a_model),
        agent_b_model=str(suite_cfg.agent_b_model),
        seed=int(getattr(suite_cfg, "seed", 0)),
        semantic_smoke=semantic_smoke,
        mvp_smoke=mvp_smoke,
        hetero_smoke=hetero_smoke,
        max_new_tokens=int(getattr(suite_cfg, "max_new_tokens", 0)),
        reasoner_max_new_tokens=_reasoner_generation_max_new_tokens(suite_cfg),
        torch_dtype=str(getattr(suite_cfg, "torch_dtype", "")),
        device_map=str(getattr(suite_cfg, "device_map", "")),
        generated_trajectory_adapter_identity=(
            _generated_trajectory_adapter_identity_manifest(suite_cfg)
            if any(name in GENERATED_LATENT_METHODS for name in report_method_names)
            else None
        ),
        handoff_identity=_handoff_identity_manifest(suite_cfg),
        sample_fingerprints=sample_fingerprint_rows,
    )
    _validate_eval_manifest_sample_lock(eval_manifest, locked_eval_manifest)
    if eval_manifest_output_path is not None:
        write_json(eval_manifest_output_path, eval_manifest)

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
                    f"(repetition {repetition + 1}/{repetitions})...",
                    flush=True,
                )
                for index, row in enumerate(samples):
                    sample_index = (
                        effective_sample_indices[index]
                        if effective_sample_indices is not None and index < len(effective_sample_indices)
                        else index
                    )
                    prompt = pick_field(row, ("question", "problem"))
                    target_answer = _target_answer(dataset_name, row)
                    state["_current_sample_row"] = dict(row)
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
                        "not_applicable" if method_name in TEXT_BASELINE_METHODS else "",
                    )
                    handoff_surface = row_result.get(
                        "handoff_surface",
                        "text" if method_name in TEXT_BASELINE_METHODS else "input_embedding",
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
                        if method_name in TEXT_BASELINE_METHODS
                        else "unknown",
                    )
                    sender_reasoning_text = str(row_result.get("sender_reasoning_text") or "")
                    sender_predicted_answer = (
                        _predicted_answer(dataset_name, sender_reasoning_text)
                        if sender_reasoning_text.strip()
                        else None
                    )
                    sender_answer_matches_target = (
                        _answers_match(dataset_name, sender_predicted_answer, target_answer)
                        if sender_predicted_answer is not None and target_answer is not None
                        else None
                    )
                    receiver_context_status = row_result.get(
                        "receiver_context_status",
                        "not_applicable" if method_name in TEXT_BASELINE_METHODS else "not_used",
                    )
                    receiver_context_reason = row_result.get(
                        "receiver_context_reason",
                        "text_baseline" if method_name in TEXT_BASELINE_METHODS else "latent_only",
                    )
                    receiver_context_token_count = int(row_result.get("receiver_context_token_count", 0) or 0)
                    receiver_input_token_count = int(
                        row_result.get(
                            "receiver_input_token_count",
                            receiver_context_token_count,
                        )
                        or 0
                    )
                    receiver_context_latent_position = row_result.get(
                        "receiver_context_latent_position",
                        "not_applicable"
                        if method_name in TEXT_BASELINE_METHODS
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
                            "receiver_input_token_count": receiver_input_token_count,
                            "decode_status": decode_status,
                            "prompt": prompt,
                            "target_answer": target_answer,
                            "sender_reasoning_text": sender_reasoning_text,
                            "sender_reasoning_token_count": row_result.get(
                                "sender_reasoning_token_count"
                            ),
                            "sender_reasoning_status": row_result.get("sender_reasoning_status"),
                            "sender_trace_cache_hit": row_result.get("sender_trace_cache_hit"),
                            "sender_trace_cache_path": row_result.get("sender_trace_cache_path"),
                            "sender_final_answer_marker": row_result.get(
                                "sender_final_answer_marker"
                            ),
                            "sender_revision_enabled": row_result.get(
                                "sender_revision_enabled"
                            ),
                            "sender_revision_applied": row_result.get(
                                "sender_revision_applied"
                            ),
                            "sender_initial_predicted_answer": row_result.get(
                                "sender_initial_predicted_answer"
                            ),
                            "sender_revision_predicted_answer": row_result.get(
                                "sender_revision_predicted_answer"
                            ),
                            "sender_revision_decision_applied": row_result.get(
                                "sender_revision_decision_applied"
                            ),
                            "sender_revision_decision_predicted_answer": row_result.get(
                                "sender_revision_decision_predicted_answer"
                            ),
                            "sender_predicted_answer": sender_predicted_answer,
                            "sender_answer_matches_target": sender_answer_matches_target,
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
                            "handoff_adapter_cache_key_digest": row_result.get(
                                "handoff_adapter_cache_key_digest"
                            ),
                            "handoff_adapter_training_prompt_count": row_result.get(
                                "handoff_adapter_training_prompt_count"
                            ),
                            "handoff_adapter_training_token_count": row_result.get(
                                "handoff_adapter_training_token_count"
                            ),
                            "handoff_adapter_training_row_cache_hit": row_result.get(
                                "handoff_adapter_training_row_cache_hit"
                            ),
                            "handoff_adapter_training_row_cache_path": row_result.get(
                                "handoff_adapter_training_row_cache_path"
                            ),
                            "handoff_adapter_training_rows_cache_key_digest": row_result.get(
                                "handoff_adapter_training_rows_cache_key_digest"
                            ),
                            "handoff_adapter_training_trace_cache_hit_count": row_result.get(
                                "handoff_adapter_training_trace_cache_hit_count"
                            ),
                            "handoff_adapter_training_trace_cache_miss_count": row_result.get(
                                "handoff_adapter_training_trace_cache_miss_count"
                            ),
                            "handoff_adapter_training_trace_cache_hit_rate_percentage": row_result.get(
                                "handoff_adapter_training_trace_cache_hit_rate_percentage"
                            ),
                            "handoff_adapter_training_reconstruction_mse": row_result.get(
                                "handoff_adapter_training_reconstruction_mse"
                            ),
                            "handoff_adapter_training_mean_cosine_similarity": row_result.get(
                                "handoff_adapter_training_mean_cosine_similarity"
                            ),
                            "generated_adapter_local_residual_applied": row_result.get(
                                "generated_adapter_local_residual_applied"
                            ),
                            "generated_adapter_local_residual_delta_norm": row_result.get(
                                "generated_adapter_local_residual_delta_norm"
                            ),
                            "generated_adapter_local_residual_mean_top_similarity": row_result.get(
                                "generated_adapter_local_residual_mean_top_similarity"
                            ),
                            "generated_adapter_local_residual_memory_rows": row_result.get(
                                "generated_adapter_local_residual_memory_rows"
                            ),
                            "generated_adapter_semantic_memory_applied": row_result.get(
                                "generated_adapter_semantic_memory_applied"
                            ),
                            "generated_adapter_semantic_memory_similarity": row_result.get(
                                "generated_adapter_semantic_memory_similarity"
                            ),
                            "generated_adapter_semantic_memory_entry_count": row_result.get(
                                "generated_adapter_semantic_memory_entry_count"
                            ),
                            "generated_adapter_semantic_memory_target_text": row_result.get(
                                "generated_adapter_semantic_memory_target_text"
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
        if method_names is not None:
            selected_method_set = set(method_names)
            selected_baselines = tuple(
                method for method in baseline_methods if method in selected_method_set
            )
            selected_latents = tuple(
                method for method in latent_methods if method in selected_method_set
            )
            if not selected_baselines:
                selected_baselines = tuple(
                    method for method in method_names if method in TEXT_BASELINE_METHODS
                )
            if not selected_latents:
                selected_latents = tuple(
                    method for method in method_names if method not in TEXT_BASELINE_METHODS
                )
            baseline_methods = selected_baselines
            latent_methods = selected_latents
            if required_marker_methods is not None:
                required_marker_methods = tuple(
                    method for method in required_marker_methods if method in selected_method_set
                )
                if not required_marker_methods:
                    required_marker_methods = tuple(method_names)
        max_answer_perplexity = getattr(semantic_smoke_cfg, "max_answer_perplexity", None)
        min_sender_correct_latent_accuracy = getattr(
            semantic_smoke_cfg,
            "min_latent_accuracy_when_sender_correct_percentage",
            None,
        )
        if (
            min_sender_correct_latent_accuracy is None
            and (semantic_smoke or hetero_smoke)
            and any(method in GENERATED_LATENT_METHODS for method in latent_methods)
        ):
            min_sender_correct_latent_accuracy = 100.0
        min_sender_accuracy = getattr(
            semantic_smoke_cfg,
            "min_sender_accuracy_percentage",
            None,
        )
        min_sender_final_answer_marker_rate = getattr(
            semantic_smoke_cfg,
            "min_sender_final_answer_marker_rate_percentage",
            None,
        )
        if (
            min_sender_final_answer_marker_rate is None
            and (semantic_smoke or hetero_smoke)
            and any(method in GENERATED_LATENT_METHODS for method in latent_methods)
        ):
            min_sender_final_answer_marker_rate = 100.0
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
            min_latent_accuracy_when_sender_correct_percentage=(
                None
                if min_sender_correct_latent_accuracy is None
                else float(min_sender_correct_latent_accuracy)
            ),
            min_sender_accuracy_percentage=(
                None if min_sender_accuracy is None else float(min_sender_accuracy)
            ),
            min_sender_final_answer_marker_rate_percentage=(
                None
                if min_sender_final_answer_marker_rate is None
                else float(min_sender_final_answer_marker_rate)
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

    provenance_baseline_methods = tuple(
        semantic_smoke_report["baseline_methods"]
        if semantic_smoke_report is not None
        else [name for name in report_method_names if name in TEXT_BASELINE_METHODS]
    )
    provenance_latent_methods = tuple(
        semantic_smoke_report["latent_methods"]
        if semantic_smoke_report is not None
        else [name for name in report_method_names if name not in TEXT_BASELINE_METHODS]
    )
    latent_provenance_report = build_latent_provenance_report(
        sample_rows,
        baseline_methods=provenance_baseline_methods,
        latent_methods=provenance_latent_methods,
        max_rows=int(getattr(semantic_smoke_cfg, "max_diagnostic_rows", 10)),
    )
    run_metadata = _run_metadata()
    artifact_manifest = _build_artifact_manifest(
        report_output_path=report_output_path,
        samples_output_path=samples_output_path,
        summary_output_path=summary_output_path,
        eval_manifest_output_path=eval_manifest_output_path,
        eval_manifest=eval_manifest,
        latent_provenance_report=latent_provenance_report,
        run_metadata=run_metadata,
    )
    transfer_comparison_cfg = getattr(
        getattr(base_cfg, "reporting", None),
        "transfer_comparison",
        None,
    )
    configured_primary_baseline = getattr(
        transfer_comparison_cfg,
        "primary_baseline_method",
        None,
    )
    default_primary_baseline = (
        "verified_token_context_handoff"
        if "verified_token_context_handoff" in provenance_baseline_methods
        else "token_context_handoff"
        if "token_context_handoff" in provenance_baseline_methods
        else None
    )
    transfer_comparison_report = build_transfer_comparison_report(
        summary_rows,
        baseline_methods=provenance_baseline_methods,
        latent_methods=provenance_latent_methods,
        primary_baseline_method=(
            str(configured_primary_baseline)
            if configured_primary_baseline not in (None, "")
            else default_primary_baseline
        ),
        min_accuracy_retention_ratio=getattr(
            transfer_comparison_cfg,
            "min_accuracy_retention_ratio",
            None,
        ),
        max_latency_ratio=getattr(
            transfer_comparison_cfg,
            "max_latency_ratio",
            None,
        ),
        require_latent_accuracy_gain=bool(
            getattr(transfer_comparison_cfg, "require_latent_accuracy_gain", False)
        ),
    )
    heterogeneous_transfer_cfg = getattr(
        getattr(base_cfg, "reporting", None),
        "heterogeneous_transfer",
        None,
    )
    heterogeneous_transfer_report = build_heterogeneous_transfer_report(
        sample_rows,
        latent_methods=provenance_latent_methods,
        model_pair_compatibility=suite_model_pair_compatibility,
        generated_methods=GENERATED_LATENT_METHODS,
        context_generated_methods=("generated_context_latent_handoff",),
        require_generated_adapter_for_incompatible_pair=bool(
            getattr(
                heterogeneous_transfer_cfg,
                "require_generated_adapter_for_incompatible_pair",
                True,
            )
        ),
        require_context_for_context_methods=bool(
            getattr(
                heterogeneous_transfer_cfg,
                "require_context_for_context_methods",
                True,
            )
        ),
    )

    write_csv(samples_output_path, sample_rows, STANDARD_SAMPLE_FIELDS)
    write_csv(summary_output_path, summary_rows, STANDARD_SUMMARY_FIELDS)
    report_payload = {
        "suite": suite_name,
        "dataset": dataset_name,
        "dataset_split": effective_split,
        "limit": limit,
        "sample_indices": effective_sample_indices,
        "sample_content_digest": eval_manifest["sample_content_digest"],
        "eval_manifest": eval_manifest,
        "repetitions": repetitions,
        "latent_steps_values": latent_step_candidates,
        "methods": [name for name, _ in methods],
        "latent_pooling": _latent_pooling_mode(base_cfg),
        "receiver_context_mode": _receiver_context_mode(base_cfg),
        "answer_only_final": _answer_only_final_enabled(base_cfg),
        "reasoner_max_new_tokens": _reasoner_generation_max_new_tokens(base_cfg),
        "sender_revision": {
            "enabled": _sender_revision_enabled(base_cfg),
            "max_new_tokens": _sender_revision_max_new_tokens(base_cfg),
            "disagreement_verifier_enabled": (
                _sender_revision_disagreement_verifier_enabled(base_cfg)
            ),
            "disagreement_verifier_max_new_tokens": (
                _sender_revision_disagreement_verifier_max_new_tokens(base_cfg)
            ),
        },
        "handoff_adapter": {
            "enabled": bool(getattr(getattr(base_cfg.handoff, "adapter", None), "enabled", False)),
            "train_on_missing": bool(
                getattr(getattr(base_cfg.handoff, "adapter", None), "train_on_missing", False)
            ),
            "train_limit": int(getattr(getattr(base_cfg.handoff, "adapter", None), "train_limit", 0)),
        },
        "generated_trajectory_adapter": {
            "enabled": bool(
                getattr(
                    getattr(base_cfg.handoff, "generated_trajectory_adapter", None),
                    "enabled",
                    False,
                )
            ),
            "train_limit": int(
                getattr(
                    getattr(base_cfg.handoff, "generated_trajectory_adapter", None),
                    "train_limit",
                    0,
                )
            ),
            "train_on_missing": _generated_trajectory_adapter_train_on_missing(base_cfg),
            "train_split": _generated_trajectory_adapter_train_split(base_cfg),
            "training_rows_cache_dir": str(
                _generated_trajectory_adapter_training_rows_cache_dir(base_cfg)
            ),
            "trace_cache_enabled": _generated_trajectory_adapter_trace_cache_enabled(base_cfg),
            "trace_cache_dir": str(_generated_trajectory_adapter_trace_cache_dir(base_cfg)),
            "progress_interval": _generated_trajectory_adapter_progress_interval(base_cfg),
            "source_mode": str(
                getattr(
                    getattr(base_cfg.handoff, "generated_trajectory_adapter", None),
                    "source_mode",
                    "generated_text",
                )
            ),
            "input_space": str(
                getattr(
                    getattr(base_cfg.handoff, "generated_trajectory_adapter", None),
                    "input_space",
                    "aligned",
                )
            ),
            "source_tail_tokens": int(
                getattr(
                    getattr(base_cfg.handoff, "generated_trajectory_adapter", None),
                    "source_tail_tokens",
                    0,
                )
            ),
            "target_mode": str(
                getattr(
                    getattr(base_cfg.handoff, "generated_trajectory_adapter", None),
                    "target_mode",
                    "generated_text",
                )
            ),
            "target_alignment": str(
                getattr(
                    getattr(base_cfg.handoff, "generated_trajectory_adapter", None),
                    "target_alignment",
                    "character",
                )
            ),
            "local_residual": {
                "enabled": _generated_trajectory_adapter_local_residual_enabled(base_cfg),
                "top_k": _generated_trajectory_adapter_local_residual_top_k(base_cfg),
                "temperature": _generated_trajectory_adapter_local_residual_temperature(base_cfg),
                "blend": _generated_trajectory_adapter_local_residual_blend(base_cfg),
                "max_memory_rows": _generated_trajectory_adapter_local_residual_max_rows(base_cfg),
            },
            "semantic_memory": {
                "enabled": _generated_trajectory_adapter_semantic_memory_enabled(base_cfg),
                "min_similarity": _generated_trajectory_adapter_semantic_memory_min_similarity(
                    base_cfg
                ),
                "max_entries": _generated_trajectory_adapter_semantic_memory_max_entries(
                    base_cfg
                ),
            },
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
        "transfer_comparison_report": transfer_comparison_report,
        "heterogeneous_transfer_report": heterogeneous_transfer_report,
        "latent_provenance_report": latent_provenance_report,
        "artifact_manifest": artifact_manifest,
        "run_metadata": run_metadata,
        "phase_gate_report": phase_gate_report,
        "ode_scaling_report": build_ode_scaling_report(summary_rows),
        "summary_rows": summary_rows,
    }
    write_json(report_output_path, report_payload)
    return sample_rows, summary_rows, report_payload


def prepare_generated_trajectory_adapter_cache(
    *,
    suite_name: str,
    dataset_name: str,
    dataset_split: Optional[str],
    limit: int,
    report_output_path: Path,
    eval_manifest_output_path: Optional[Path] = None,
    max_new_tokens: Optional[int] = None,
    reasoner_max_new_tokens: Optional[int] = None,
    agent_a_model: Optional[str] = None,
    agent_b_model: Optional[str] = None,
    torch_dtype: Optional[str] = None,
    device_map: Optional[str] = None,
    latent_pooling: Optional[str] = None,
    receiver_context_mode: Optional[str] = None,
    receiver_context_latent_position: Optional[str] = None,
    prompt_calibration_enabled: Optional[bool] = None,
    prompt_calibration_strength: Optional[float] = None,
    prompt_calibration_max_norm_ratio: Optional[float] = None,
    generated_trajectory_adapter_train_on_missing: Optional[bool] = None,
    generated_trajectory_adapter_train_limit: Optional[int] = None,
    generated_trajectory_adapter_train_split: Optional[str] = None,
    generated_trajectory_adapter_input_space: Optional[str] = None,
    generated_trajectory_adapter_target_alignment: Optional[str] = None,
    generated_trajectory_adapter_source_mode: Optional[str] = None,
    generated_trajectory_adapter_source_tail_tokens: Optional[int] = None,
    generated_trajectory_adapter_target_mode: Optional[str] = None,
    generated_trajectory_adapter_local_residual_enabled: Optional[bool] = None,
    generated_trajectory_adapter_local_residual_top_k: Optional[int] = None,
    generated_trajectory_adapter_local_residual_temperature: Optional[float] = None,
    generated_trajectory_adapter_local_residual_blend: Optional[float] = None,
    generated_trajectory_adapter_local_residual_max_memory_rows: Optional[int] = None,
    generated_trajectory_adapter_semantic_memory_enabled: Optional[bool] = None,
    generated_trajectory_adapter_semantic_memory_min_similarity: Optional[float] = None,
    generated_trajectory_adapter_semantic_memory_max_entries: Optional[int] = None,
    sender_revision_enabled: Optional[bool] = None,
    sender_revision_max_new_tokens: Optional[int] = None,
    sender_revision_disagreement_verifier_enabled: Optional[bool] = None,
    sender_revision_disagreement_verifier_max_new_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    method_names: Optional[list[str]] = None,
    sample_indices: Optional[list[int]] = None,
    prepare_adapter: bool = True,
    prepare_eval_traces: bool = False,
    semantic_smoke: bool = False,
    mvp_smoke: bool = False,
    hetero_smoke: bool = False,
    answer_only_final: bool = False,
    locked_eval_manifest: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    base_cfg = _configured_base_cfg(
        agent_a_model=agent_a_model,
        agent_b_model=agent_b_model,
        torch_dtype=torch_dtype,
        device_map=device_map,
        latent_pooling=latent_pooling,
        receiver_context_mode=receiver_context_mode,
        receiver_context_latent_position=receiver_context_latent_position,
        prompt_calibration_enabled=prompt_calibration_enabled,
        prompt_calibration_strength=prompt_calibration_strength,
        prompt_calibration_max_norm_ratio=prompt_calibration_max_norm_ratio,
        generated_trajectory_adapter_enabled=True,
        generated_trajectory_adapter_train_on_missing=generated_trajectory_adapter_train_on_missing,
        generated_trajectory_adapter_train_limit=generated_trajectory_adapter_train_limit,
        generated_trajectory_adapter_train_split=generated_trajectory_adapter_train_split,
        generated_trajectory_adapter_input_space=generated_trajectory_adapter_input_space,
        generated_trajectory_adapter_target_alignment=generated_trajectory_adapter_target_alignment,
        generated_trajectory_adapter_source_mode=generated_trajectory_adapter_source_mode,
        generated_trajectory_adapter_source_tail_tokens=generated_trajectory_adapter_source_tail_tokens,
        generated_trajectory_adapter_target_mode=generated_trajectory_adapter_target_mode,
        generated_trajectory_adapter_local_residual_enabled=(
            generated_trajectory_adapter_local_residual_enabled
        ),
        generated_trajectory_adapter_local_residual_top_k=(
            generated_trajectory_adapter_local_residual_top_k
        ),
        generated_trajectory_adapter_local_residual_temperature=(
            generated_trajectory_adapter_local_residual_temperature
        ),
        generated_trajectory_adapter_local_residual_blend=(
            generated_trajectory_adapter_local_residual_blend
        ),
        generated_trajectory_adapter_local_residual_max_memory_rows=(
            generated_trajectory_adapter_local_residual_max_memory_rows
        ),
        generated_trajectory_adapter_semantic_memory_enabled=(
            generated_trajectory_adapter_semantic_memory_enabled
        ),
        generated_trajectory_adapter_semantic_memory_min_similarity=(
            generated_trajectory_adapter_semantic_memory_min_similarity
        ),
        generated_trajectory_adapter_semantic_memory_max_entries=(
            generated_trajectory_adapter_semantic_memory_max_entries
        ),
        sender_revision_enabled=sender_revision_enabled,
        sender_revision_max_new_tokens=sender_revision_max_new_tokens,
        sender_revision_disagreement_verifier_enabled=(
            sender_revision_disagreement_verifier_enabled
        ),
        sender_revision_disagreement_verifier_max_new_tokens=(
            sender_revision_disagreement_verifier_max_new_tokens
        ),
        seed=seed,
        max_new_tokens=max_new_tokens,
        reasoner_max_new_tokens=reasoner_max_new_tokens,
        semantic_smoke=semantic_smoke,
        mvp_smoke=mvp_smoke,
        hetero_smoke=hetero_smoke,
        answer_only_final=answer_only_final,
    )
    if dataset_name == "long_context_handoff":
        base_cfg.handoff.generated_trajectory_adapter.dataset_name = dataset_name
    suite_cfg = _suite_cfg(base_cfg, suite_name)
    _require_requested_device_available(suite_cfg)
    effective_split = dataset_split or _default_split_for_dataset(dataset_name)
    if sample_indices is None:
        semantic_smoke_cfg = getattr(getattr(base_cfg, "reporting", None), "semantic_smoke", None)
        if semantic_smoke_cfg is not None:
            sample_indices = _coerce_sample_indices(
                getattr(semantic_smoke_cfg, "sample_indices", None)
            )
    effective_sample_indices = _resolved_sample_indices(limit, sample_indices)
    validation_size = _validation_size(base_cfg, dataset_name)
    samples = get_dataloader(
        dataset_name,
        limit=limit,
        split=effective_split,
        validation_size=validation_size,
        sample_indices=sample_indices,
    )
    sample_fingerprint_rows = _sample_fingerprints(
        samples,
        limit=limit,
        sample_indices=effective_sample_indices,
    )
    report_method_names = list(method_names or ("generated_latent_handoff",))
    eval_manifest = _build_eval_manifest(
        suite_name=suite_name,
        dataset_name=dataset_name,
        dataset_split=effective_split,
        limit=limit,
        sample_indices=effective_sample_indices,
        methods=report_method_names,
        agent_a_model=str(suite_cfg.agent_a_model),
        agent_b_model=str(suite_cfg.agent_b_model),
        seed=int(getattr(suite_cfg, "seed", 0)),
        semantic_smoke=semantic_smoke,
        mvp_smoke=mvp_smoke,
        hetero_smoke=hetero_smoke,
        max_new_tokens=int(getattr(suite_cfg, "max_new_tokens", 0)),
        reasoner_max_new_tokens=_reasoner_generation_max_new_tokens(suite_cfg),
        torch_dtype=str(getattr(suite_cfg, "torch_dtype", "")),
        device_map=str(getattr(suite_cfg, "device_map", "")),
        generated_trajectory_adapter_identity=(
            _generated_trajectory_adapter_identity_manifest(suite_cfg)
            if any(name in GENERATED_LATENT_METHODS for name in report_method_names)
            else None
        ),
        handoff_identity=_handoff_identity_manifest(suite_cfg),
        sample_fingerprints=sample_fingerprint_rows,
    )
    _validate_eval_manifest_sample_lock(eval_manifest, locked_eval_manifest)
    if eval_manifest_output_path is not None:
        write_json(eval_manifest_output_path, eval_manifest)
    if prepare_adapter:
        state = _get_pipeline_state(suite_cfg)
        variant_cfg, variant_state = _alignment_variant_state(
            suite_cfg,
            state,
            strategy="hybrid_affine",
            prompt_calibration_enabled=False,
        )
        alignment_state = _generated_adapter_alignment_state_from_variant_state(variant_state)
    else:
        variant_cfg = _alignment_variant_cfg(
            suite_cfg,
            strategy="hybrid_affine",
            prompt_calibration_enabled=False,
        )
        variant_state = _get_sender_trace_state(variant_cfg)
        alignment_state = None
    prepared_adapters: list[dict[str, Any]] = []
    if prepare_adapter:
        for include_prompt in _generated_adapter_include_prompt_values(method_names):
            if alignment_state is None:
                raise ValueError("Preparing generated adapters requires a full alignment state")
            info = _load_or_train_generated_trajectory_adapter_state(
                variant_cfg,
                variant_state,
                alignment_state,
                include_prompt=include_prompt,
            )
            prepared_adapters.append(
                {
                    "include_prompt": bool(include_prompt),
                    "enabled": bool(info.get("enabled", False)),
                    "status": info.get("status"),
                    "cache_hit": info.get("cache_hit"),
                    "cache_path": info.get("cache_path"),
                    "training_prompt_count": info.get("training_prompt_count"),
                    "training_token_count": info.get("training_token_count"),
                    "training_row_cache_hit": info.get("training_row_cache_hit"),
                    "training_row_cache_path": info.get("training_row_cache_path"),
                    "adapter_cache_key_digest": info.get("adapter_cache_key_digest"),
                    "training_rows_cache_key_digest": info.get(
                        "training_rows_cache_key_digest"
                    ),
                    "training_trace_cache_hit_count": info.get(
                        "training_trace_cache_hit_count"
                    ),
                    "training_trace_cache_miss_count": info.get(
                        "training_trace_cache_miss_count"
                    ),
                    "training_trace_cache_hit_rate_percentage": info.get(
                        "training_trace_cache_hit_rate_percentage"
                    ),
                    "training_reconstruction_mse": info.get("training_reconstruction_mse"),
                    "training_mean_cosine_similarity": info.get(
                        "training_mean_cosine_similarity"
                    ),
                }
            )
    prepared_eval_traces = None
    if prepare_eval_traces:
        prepared_eval_traces = _prepare_generated_trajectory_eval_traces(
            dataset_name=dataset_name,
            dataset_split=effective_split,
            limit=limit,
            sample_indices=effective_sample_indices,
            base_cfg=base_cfg,
            variant_cfg=variant_cfg,
            variant_state=variant_state,
            method_names=method_names,
        )
    run_metadata = _run_metadata()
    artifact_manifest = _build_artifact_manifest(
        report_output_path=report_output_path,
        eval_manifest_output_path=eval_manifest_output_path,
        eval_manifest=eval_manifest,
        prepared_adapters=prepared_adapters,
        prepared_eval_traces=prepared_eval_traces,
        run_metadata=run_metadata,
    )
    report_payload = {
        "suite": suite_name,
        "dataset": dataset_name,
        "dataset_split": effective_split,
        "limit": int(limit),
        "sample_indices": effective_sample_indices,
        "sample_content_digest": eval_manifest["sample_content_digest"],
        "eval_manifest": eval_manifest,
        "methods": report_method_names,
        "agent_a_model": str(suite_cfg.agent_a_model),
        "agent_b_model": str(suite_cfg.agent_b_model),
        "answer_only_final": _answer_only_final_enabled(base_cfg),
        "reasoner_max_new_tokens": _reasoner_generation_max_new_tokens(base_cfg),
        "sender_revision": {
            "enabled": _sender_revision_enabled(base_cfg),
            "max_new_tokens": _sender_revision_max_new_tokens(base_cfg),
            "disagreement_verifier_enabled": (
                _sender_revision_disagreement_verifier_enabled(base_cfg)
            ),
            "disagreement_verifier_max_new_tokens": (
                _sender_revision_disagreement_verifier_max_new_tokens(base_cfg)
            ),
        },
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_trajectory_adapter": {
            "enabled": _generated_trajectory_adapter_enabled(base_cfg),
            "train_on_missing": _generated_trajectory_adapter_train_on_missing(base_cfg),
            "train_limit": _generated_trajectory_adapter_train_limit(base_cfg),
            "train_split": _generated_trajectory_adapter_train_split(base_cfg),
            "cache_dir": str(_generated_trajectory_adapter_cache_dir(base_cfg)),
            "training_rows_cache_dir": str(
                _generated_trajectory_adapter_training_rows_cache_dir(base_cfg)
            ),
            "trace_cache_enabled": _generated_trajectory_adapter_trace_cache_enabled(base_cfg),
            "trace_cache_dir": str(_generated_trajectory_adapter_trace_cache_dir(base_cfg)),
            "source_mode": _generated_trajectory_adapter_source_mode(base_cfg),
            "input_space": _generated_trajectory_adapter_input_space(base_cfg),
            "target_mode": _generated_trajectory_adapter_target_mode(base_cfg),
            "target_alignment": _generated_trajectory_adapter_target_alignment(base_cfg),
            "local_residual": {
                "enabled": _generated_trajectory_adapter_local_residual_enabled(base_cfg),
                "top_k": _generated_trajectory_adapter_local_residual_top_k(base_cfg),
                "temperature": _generated_trajectory_adapter_local_residual_temperature(base_cfg),
                "blend": _generated_trajectory_adapter_local_residual_blend(base_cfg),
                "max_memory_rows": _generated_trajectory_adapter_local_residual_max_rows(base_cfg),
            },
            "semantic_memory": {
                "enabled": _generated_trajectory_adapter_semantic_memory_enabled(base_cfg),
                "min_similarity": _generated_trajectory_adapter_semantic_memory_min_similarity(
                    base_cfg
                ),
                "max_entries": _generated_trajectory_adapter_semantic_memory_max_entries(
                    base_cfg
                ),
            },
        },
        "prepared_adapters": prepared_adapters,
        "prepared_eval_traces": prepared_eval_traces,
        "artifact_manifest": artifact_manifest,
        "run_metadata": run_metadata,
    }
    write_json(report_output_path, report_payload)
    return report_payload


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
        choices=("gsm8k", "math", "long_context_handoff"),
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
        "--eval-manifest",
        type=Path,
        default=None,
        help=(
            "Load a locked evaluation manifest and use its suite, dataset, split, "
            "limit, sample indices, methods, model pair, seed, and smoke profile."
        ),
    )
    parser.add_argument(
        "--write-eval-manifest",
        type=Path,
        default=None,
        help="Write the resolved evaluation manifest to this JSON path.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional override for cfg.max_new_tokens to support faster smoke tests.",
    )
    parser.add_argument(
        "--reasoner-max-new-tokens",
        type=int,
        default=None,
        help="Optional override for benchmark.text_hybrid_reasoning_max_new_tokens.",
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
        "--torch-dtype",
        choices=("float32", "float16", "bfloat16"),
        default=None,
        help="Optional override for cfg.torch_dtype.",
    )
    parser.add_argument(
        "--device-map",
        default=None,
        help="Optional override for cfg.device_map, for example auto or none.",
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
        "--enable-generated-trajectory-adapter",
        action="store_true",
        help="Train/load the generated-trajectory adapter for generated latent methods.",
    )
    parser.add_argument(
        "--disable-generated-trajectory-adapter",
        action="store_true",
        help="Disable the generated-trajectory adapter for this benchmark run.",
    )
    parser.add_argument(
        "--generated-trajectory-adapter-train-on-missing",
        action="store_true",
        help="Fit and cache the generated-trajectory adapter if no matching adapter cache exists.",
    )
    parser.add_argument(
        "--generated-trajectory-adapter-no-train-on-missing",
        action="store_true",
        help="Require a prebuilt generated-trajectory adapter cache instead of fitting one.",
    )
    parser.add_argument(
        "--prepare-generated-trajectory-adapter",
        action="store_true",
        help="Fit/load generated-trajectory adapter caches and exit before benchmark decoding.",
    )
    parser.add_argument(
        "--prepare-generated-trajectory-eval-traces",
        action="store_true",
        help="Warm generated sender trace caches for the selected eval samples and exit.",
    )
    parser.add_argument(
        "--generated-trajectory-adapter-train-limit",
        type=int,
        default=None,
        help="Optional override for handoff.generated_trajectory_adapter.train_limit.",
    )
    parser.add_argument(
        "--generated-trajectory-adapter-train-split",
        default=None,
        help="Optional override for handoff.generated_trajectory_adapter.train_split.",
    )
    parser.add_argument(
        "--generated-trajectory-adapter-input-space",
        choices=tuple(sorted(GENERATED_TRAJECTORY_ADAPTER_INPUT_SPACES)),
        default=None,
        help=(
            "Train/apply the generated trajectory adapter from globally aligned sender "
            "states or directly from raw sender states."
        ),
    )
    parser.add_argument(
        "--generated-trajectory-adapter-source-mode",
        choices=("generated_text", "final_answer_tail"),
        default=None,
        help="Optional override for handoff.generated_trajectory_adapter.source_mode.",
    )
    parser.add_argument(
        "--generated-trajectory-adapter-source-tail-tokens",
        type=int,
        default=None,
        help="Optional override for handoff.generated_trajectory_adapter.source_tail_tokens.",
    )
    parser.add_argument(
        "--generated-trajectory-adapter-target-alignment",
        choices=tuple(sorted(GENERATED_TRAJECTORY_ADAPTER_TARGET_ALIGNMENTS)),
        default=None,
        help=(
            "How receiver target embeddings are aligned to sender generated-token "
            "timesteps when fitting the generated trajectory adapter."
        ),
    )
    parser.add_argument(
        "--generated-trajectory-adapter-target-mode",
        choices=("generated_text", "final_answer_line"),
        default=None,
        help="Optional override for handoff.generated_trajectory_adapter.target_mode.",
    )
    parser.add_argument(
        "--enable-generated-trajectory-local-residual",
        action="store_true",
        help="Enable top-k local residual correction for the generated-trajectory adapter.",
    )
    parser.add_argument(
        "--disable-generated-trajectory-local-residual",
        action="store_true",
        help="Disable generated-trajectory local residual correction for this run.",
    )
    parser.add_argument(
        "--generated-trajectory-local-residual-top-k",
        type=int,
        default=None,
        help="Optional override for handoff.generated_trajectory_adapter.local_residual.top_k.",
    )
    parser.add_argument(
        "--generated-trajectory-local-residual-temperature",
        type=float,
        default=None,
        help="Optional override for handoff.generated_trajectory_adapter.local_residual.temperature.",
    )
    parser.add_argument(
        "--generated-trajectory-local-residual-blend",
        type=float,
        default=None,
        help="Optional override for handoff.generated_trajectory_adapter.local_residual.blend.",
    )
    parser.add_argument(
        "--generated-trajectory-local-residual-max-memory-rows",
        type=int,
        default=None,
        help="Optional override for handoff.generated_trajectory_adapter.local_residual.max_memory_rows.",
    )
    parser.add_argument(
        "--enable-generated-trajectory-semantic-memory",
        action="store_true",
        help="Enable nearest-trajectory semantic readout for generated latent methods.",
    )
    parser.add_argument(
        "--disable-generated-trajectory-semantic-memory",
        action="store_true",
        help="Disable generated-trajectory semantic memory readout for this run.",
    )
    parser.add_argument(
        "--generated-trajectory-semantic-memory-min-similarity",
        type=float,
        default=None,
        help="Optional semantic memory cosine threshold for generated latent readout.",
    )
    parser.add_argument(
        "--generated-trajectory-semantic-memory-max-entries",
        type=int,
        default=None,
        help="Optional maximum generated trajectory semantic memory entries.",
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
        "--enable-sender-revision",
        action="store_true",
        help=(
            "Run an opt-in Agent A self-check pass and use it when it produces a "
            "final-answer marker."
        ),
    )
    parser.add_argument(
        "--sender-revision-max-new-tokens",
        type=int,
        default=None,
        help="Optional override for benchmark.sender_revision.max_new_tokens.",
    )
    parser.add_argument(
        "--disable-sender-revision-disagreement-verifier",
        action="store_true",
        help=(
            "Disable the extra verifier pass for missing, disagreeing, or non-scalar "
            "sender revision answers."
        ),
    )
    parser.add_argument(
        "--sender-revision-disagreement-verifier-max-new-tokens",
        type=int,
        default=None,
        help=(
            "Optional override for "
            "benchmark.sender_revision.disagreement_verifier_max_new_tokens."
        ),
    )
    parser.add_argument(
        "--semantic-min-sender-accuracy",
        type=float,
        default=None,
        help="Optional semantic smoke gate for generated sender answer accuracy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional deterministic seed to stamp into benchmark metadata.",
    )
    args = parser.parse_args()
    locked_eval_manifest = None
    if args.eval_manifest is not None:
        locked_eval_manifest = _load_eval_manifest(args.eval_manifest)
        _apply_eval_manifest_to_args(args, locked_eval_manifest)
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
    if (
        args.prepare_generated_trajectory_adapter
        or args.prepare_generated_trajectory_eval_traces
    ) and not args.methods:
        args.methods = "generated_latent_handoff"
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
    selected_method_set = set(method_names or ())
    uses_generated_latent = any(method in GENERATED_LATENT_METHODS for method in selected_method_set)
    if (
        args.prepare_generated_trajectory_adapter
        or args.prepare_generated_trajectory_eval_traces
    ) and uses_generated_latent:
        if not args.disable_generated_trajectory_adapter:
            args.enable_generated_trajectory_adapter = True
        if (
            args.generated_trajectory_adapter_input_space == "raw"
            and not args.disable_generated_trajectory_local_residual
        ):
            args.enable_generated_trajectory_local_residual = True
    if (args.semantic_smoke or args.mvp_smoke or args.hetero_smoke) and uses_generated_latent:
        if not args.disable_generated_trajectory_adapter:
            args.enable_generated_trajectory_adapter = True
        if args.generated_trajectory_adapter_train_limit is None:
            args.generated_trajectory_adapter_train_limit = (
                DEFAULT_HETERO_SMOKE_GENERATED_ADAPTER_TRAIN_LIMIT
                if args.hetero_smoke
                else DEFAULT_SEMANTIC_SMOKE_GENERATED_ADAPTER_TRAIN_LIMIT
            )
        if args.generated_trajectory_adapter_input_space is None:
            args.generated_trajectory_adapter_input_space = "aligned"
        if args.generated_trajectory_adapter_source_mode is None:
            args.generated_trajectory_adapter_source_mode = "generated_text"
        if args.generated_trajectory_adapter_target_mode is None:
            args.generated_trajectory_adapter_target_mode = "generated_text"
        if args.generated_trajectory_adapter_target_alignment is None:
            args.generated_trajectory_adapter_target_alignment = (
                "linear"
                if args.generated_trajectory_adapter_target_mode == "final_answer_line"
                else "character"
            )
        if (
            args.generated_trajectory_adapter_input_space == "raw"
            and not args.disable_generated_trajectory_local_residual
        ):
            args.enable_generated_trajectory_local_residual = True
        if not args.disable_embedding_manifold:
            args.enable_embedding_manifold = True
            if args.embedding_manifold_top_k is None:
                args.embedding_manifold_top_k = 4 if args.hetero_smoke else 1
            if args.embedding_manifold_blend is None:
                args.embedding_manifold_blend = 1.0
    sample_indices = (
        [int(value.strip()) for value in args.sample_indices.split(",") if value.strip()]
        if args.sample_indices
        else None
    )

    if args.prepare_generated_trajectory_adapter or args.prepare_generated_trajectory_eval_traces:
        report_payload = prepare_generated_trajectory_adapter_cache(
            suite_name=args.suite,
            dataset_name=args.dataset,
            dataset_split=args.split,
            limit=args.limit,
            report_output_path=args.report_output,
            eval_manifest_output_path=args.write_eval_manifest,
            max_new_tokens=args.max_new_tokens,
            reasoner_max_new_tokens=args.reasoner_max_new_tokens,
            agent_a_model=args.agent_a_model,
            agent_b_model=args.agent_b_model,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            latent_pooling=args.latent_pooling,
            receiver_context_mode=args.receiver_context_mode,
            receiver_context_latent_position=args.receiver_context_latent_position,
            prompt_calibration_enabled=False if args.disable_prompt_calibration else None,
            prompt_calibration_strength=args.prompt_calibration_strength,
            prompt_calibration_max_norm_ratio=args.prompt_calibration_max_norm_ratio,
            generated_trajectory_adapter_train_on_missing=(
                True
                if args.generated_trajectory_adapter_train_on_missing
                else False
                if args.generated_trajectory_adapter_no_train_on_missing
                else None
            ),
            generated_trajectory_adapter_train_limit=args.generated_trajectory_adapter_train_limit,
            generated_trajectory_adapter_train_split=args.generated_trajectory_adapter_train_split,
            generated_trajectory_adapter_input_space=args.generated_trajectory_adapter_input_space,
            generated_trajectory_adapter_target_alignment=(
                args.generated_trajectory_adapter_target_alignment
            ),
            generated_trajectory_adapter_source_mode=args.generated_trajectory_adapter_source_mode,
            generated_trajectory_adapter_source_tail_tokens=(
                args.generated_trajectory_adapter_source_tail_tokens
            ),
            generated_trajectory_adapter_target_mode=args.generated_trajectory_adapter_target_mode,
            generated_trajectory_adapter_local_residual_enabled=(
                False
                if args.disable_generated_trajectory_local_residual
                else True
                if args.enable_generated_trajectory_local_residual
                else None
            ),
            generated_trajectory_adapter_local_residual_top_k=(
                args.generated_trajectory_local_residual_top_k
            ),
            generated_trajectory_adapter_local_residual_temperature=(
                args.generated_trajectory_local_residual_temperature
            ),
            generated_trajectory_adapter_local_residual_blend=(
                args.generated_trajectory_local_residual_blend
            ),
            generated_trajectory_adapter_local_residual_max_memory_rows=(
                args.generated_trajectory_local_residual_max_memory_rows
            ),
            generated_trajectory_adapter_semantic_memory_enabled=(
                False
                if args.disable_generated_trajectory_semantic_memory
                else True
                if args.enable_generated_trajectory_semantic_memory
                else None
            ),
            generated_trajectory_adapter_semantic_memory_min_similarity=(
                args.generated_trajectory_semantic_memory_min_similarity
            ),
            generated_trajectory_adapter_semantic_memory_max_entries=(
                args.generated_trajectory_semantic_memory_max_entries
            ),
            sender_revision_enabled=True if args.enable_sender_revision else None,
            sender_revision_max_new_tokens=args.sender_revision_max_new_tokens,
            sender_revision_disagreement_verifier_enabled=(
                False if args.disable_sender_revision_disagreement_verifier else None
            ),
            sender_revision_disagreement_verifier_max_new_tokens=(
                args.sender_revision_disagreement_verifier_max_new_tokens
            ),
            seed=args.seed,
            method_names=method_names,
            sample_indices=sample_indices,
            prepare_adapter=args.prepare_generated_trajectory_adapter,
            prepare_eval_traces=args.prepare_generated_trajectory_eval_traces,
            semantic_smoke=args.semantic_smoke,
            mvp_smoke=args.mvp_smoke,
            hetero_smoke=args.hetero_smoke,
            answer_only_final=args.answer_only_final,
            locked_eval_manifest=locked_eval_manifest,
        )
        print(f"Wrote generated trajectory prepare report to {args.report_output}")
        if args.write_eval_manifest is not None:
            print(f"Wrote eval manifest to {args.write_eval_manifest}")
        for prepared in report_payload["prepared_adapters"]:
            print(
                "Prepared generated trajectory adapter "
                f"include_prompt={prepared['include_prompt']} "
                f"status={prepared['status']} "
                f"adapter_cache_hit={prepared['cache_hit']} "
                f"row_cache_hit={prepared['training_row_cache_hit']} "
                f"trace_hit_rate={prepared['training_trace_cache_hit_rate_percentage']}"
            )
        missing_adapters = [
            prepared
            for prepared in report_payload["prepared_adapters"]
            if prepared.get("status") not in ("loaded", "trained")
        ]
        if args.prepare_generated_trajectory_adapter and missing_adapters:
            missing_summary = ", ".join(
                f"include_prompt={prepared.get('include_prompt')} "
                f"status={prepared.get('status')} "
                f"path={prepared.get('cache_path')}"
                for prepared in missing_adapters
            )
            raise SystemExit(
                "Generated trajectory adapter preparation failed: "
                f"{missing_summary}"
            )
        prepared_eval_traces = report_payload.get("prepared_eval_traces")
        if prepared_eval_traces is not None:
            print(
                "Prepared generated trajectory eval traces "
                f"count={prepared_eval_traces['trace_count']} "
                f"cache_hit_rate={prepared_eval_traces['trace_cache_hit_rate_percentage']} "
                f"hits={prepared_eval_traces['trace_cache_hit_count']} "
                f"misses={prepared_eval_traces['trace_cache_miss_count']}"
            )
        return

    _, summary_rows, report_payload = run_benchmark(
        suite_name=args.suite,
        dataset_name=args.dataset,
        dataset_split=args.split,
        limit=args.limit,
        repetitions=args.repetitions,
        samples_output_path=args.samples_output,
        summary_output_path=args.summary_output,
        report_output_path=args.report_output,
        eval_manifest_output_path=args.write_eval_manifest,
        max_new_tokens=args.max_new_tokens,
        reasoner_max_new_tokens=args.reasoner_max_new_tokens,
        latent_steps_values=latent_steps_values,
        agent_a_model=args.agent_a_model,
        agent_b_model=args.agent_b_model,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
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
        generated_trajectory_adapter_enabled=(
            False
            if args.disable_generated_trajectory_adapter
            else True
            if args.enable_generated_trajectory_adapter
            else None
        ),
        generated_trajectory_adapter_train_on_missing=(
            True
            if args.generated_trajectory_adapter_train_on_missing
            else False
            if args.generated_trajectory_adapter_no_train_on_missing
            else None
        ),
        generated_trajectory_adapter_train_limit=args.generated_trajectory_adapter_train_limit,
        generated_trajectory_adapter_train_split=args.generated_trajectory_adapter_train_split,
        generated_trajectory_adapter_input_space=args.generated_trajectory_adapter_input_space,
        generated_trajectory_adapter_target_alignment=args.generated_trajectory_adapter_target_alignment,
        generated_trajectory_adapter_source_mode=args.generated_trajectory_adapter_source_mode,
        generated_trajectory_adapter_source_tail_tokens=(
            args.generated_trajectory_adapter_source_tail_tokens
        ),
        generated_trajectory_adapter_target_mode=args.generated_trajectory_adapter_target_mode,
        generated_trajectory_adapter_local_residual_enabled=(
            False
            if args.disable_generated_trajectory_local_residual
            else True
            if args.enable_generated_trajectory_local_residual
            else None
        ),
        generated_trajectory_adapter_local_residual_top_k=(
            args.generated_trajectory_local_residual_top_k
        ),
        generated_trajectory_adapter_local_residual_temperature=(
            args.generated_trajectory_local_residual_temperature
        ),
        generated_trajectory_adapter_local_residual_blend=(
            args.generated_trajectory_local_residual_blend
        ),
        generated_trajectory_adapter_local_residual_max_memory_rows=(
            args.generated_trajectory_local_residual_max_memory_rows
        ),
        generated_trajectory_adapter_semantic_memory_enabled=(
            False
            if args.disable_generated_trajectory_semantic_memory
            else True
            if args.enable_generated_trajectory_semantic_memory
            else None
        ),
        generated_trajectory_adapter_semantic_memory_min_similarity=(
            args.generated_trajectory_semantic_memory_min_similarity
        ),
        generated_trajectory_adapter_semantic_memory_max_entries=(
            args.generated_trajectory_semantic_memory_max_entries
        ),
        embedding_manifold_enabled=(
            False
            if args.disable_embedding_manifold
            else True
            if args.enable_embedding_manifold
            else None
        ),
        embedding_manifold_top_k=args.embedding_manifold_top_k,
        embedding_manifold_blend=args.embedding_manifold_blend,
        semantic_min_sender_accuracy_percentage=args.semantic_min_sender_accuracy,
        sender_revision_enabled=True if args.enable_sender_revision else None,
        sender_revision_max_new_tokens=args.sender_revision_max_new_tokens,
        sender_revision_disagreement_verifier_enabled=(
            False if args.disable_sender_revision_disagreement_verifier else None
        ),
        sender_revision_disagreement_verifier_max_new_tokens=(
            args.sender_revision_disagreement_verifier_max_new_tokens
        ),
        seed=args.seed,
        method_names=method_names,
        sample_indices=sample_indices,
        semantic_smoke=args.semantic_smoke,
        mvp_smoke=args.mvp_smoke,
        hetero_smoke=args.hetero_smoke,
        answer_only_final=args.answer_only_final,
        locked_eval_manifest=locked_eval_manifest,
    )

    print(f"Wrote per-sample benchmark rows to {args.samples_output}")
    print(f"Wrote benchmark summary rows to {args.summary_output}")
    print(f"Wrote phase-gate report to {args.report_output}")
    if args.write_eval_manifest is not None:
        print(f"Wrote eval manifest to {args.write_eval_manifest}")
    print(f"Phase gate passed: {report_payload['phase_gate_report']['passed']}")
    if report_payload.get("semantic_smoke_report") is not None:
        print(f"Semantic smoke passed: {report_payload['semantic_smoke_report']['passed']}")
    if report_payload.get("transfer_comparison_report") is not None:
        print(
            "Transfer comparison passed: "
            f"{report_payload['transfer_comparison_report']['passed']}"
        )
    if report_payload.get("heterogeneous_transfer_report") is not None:
        print(
            "Heterogeneous transfer readiness passed: "
            f"{report_payload['heterogeneous_transfer_report']['passed']}"
        )
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
