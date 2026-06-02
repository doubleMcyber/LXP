from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import torch

InterventionKind = Literal["ablate", "noise", "replace"]
ReplayFn = Callable[[tuple["LatentPacket", ...]], "RunOutcome"]


def tensor_diagnostics(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.detach().float()
    return {
        "shape": tuple(int(dim) for dim in tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "mean": float(values.mean().cpu().item()) if values.numel() else 0.0,
        "std": float(values.std(unbiased=False).cpu().item()) if values.numel() else 0.0,
        "l2_norm": float(torch.linalg.vector_norm(values.reshape(-1)).cpu().item()),
    }


@dataclass(frozen=True)
class LatentPacket:
    packet_id: str
    sender: str
    receiver: str
    turn: int
    tensor: torch.Tensor
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def detached_cpu(self) -> "LatentPacket":
        return LatentPacket(
            packet_id=self.packet_id,
            sender=self.sender,
            receiver=self.receiver,
            turn=self.turn,
            tensor=self.tensor.detach().cpu().clone(),
            metadata=dict(self.metadata),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "sender": self.sender,
            "receiver": self.receiver,
            "turn": int(self.turn),
            "metadata": dict(self.metadata),
            "tensor": tensor_diagnostics(self.tensor),
        }


@dataclass(frozen=True)
class RunOutcome:
    final_answer: str
    score: float
    success: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LatentRunRecord:
    run_id: str
    packets: tuple[LatentPacket, ...]
    outcome: RunOutcome
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def packet(self, packet_id: str) -> LatentPacket:
        for packet in self.packets:
            if packet.packet_id == packet_id:
                return packet
        raise KeyError(f"Unknown latent packet id: {packet_id}")


@dataclass(frozen=True)
class InterventionResult:
    packet_id: str
    intervention: InterventionKind
    baseline_answer: str
    intervened_answer: str
    baseline_score: float
    intervened_score: float
    score_delta: float
    baseline_success: bool
    intervened_success: bool
    success_changed: bool
    causal_impact: float
    packet_summary: Mapping[str, Any]
    intervention_metadata: Mapping[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "intervention": self.intervention,
            "baseline_answer": self.baseline_answer,
            "intervened_answer": self.intervened_answer,
            "baseline_score": self.baseline_score,
            "intervened_score": self.intervened_score,
            "score_delta": self.score_delta,
            "baseline_success": self.baseline_success,
            "intervened_success": self.intervened_success,
            "success_changed": self.success_changed,
            "causal_impact": self.causal_impact,
            "packet_summary": dict(self.packet_summary),
            "intervention_metadata": dict(self.intervention_metadata),
        }


def record_latent_packet(
    *,
    sender: str,
    receiver: str,
    turn: int,
    tensor: torch.Tensor,
    packet_id: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> LatentPacket:
    resolved_packet_id = packet_id or f"turn-{int(turn)}:{sender}->{receiver}"
    return LatentPacket(
        packet_id=resolved_packet_id,
        sender=sender,
        receiver=receiver,
        turn=int(turn),
        tensor=tensor.detach().cpu().clone(),
        metadata=dict(metadata or {}),
    )


def _replace_packet(
    packets: Sequence[LatentPacket],
    packet_id: str,
    replacement: LatentPacket,
) -> tuple[LatentPacket, ...]:
    replaced = False
    updated_packets: list[LatentPacket] = []
    for packet in packets:
        if packet.packet_id != packet_id:
            updated_packets.append(packet)
            continue
        replaced = True
        updated_packets.append(replacement)
    if not replaced:
        raise KeyError(f"Unknown latent packet id: {packet_id}")
    return tuple(updated_packets)


def apply_packet_intervention(
    packet: LatentPacket,
    *,
    intervention: InterventionKind,
    replacement_packet: Optional[LatentPacket] = None,
    noise_std: float = 0.1,
    generator: Optional[torch.Generator] = None,
) -> LatentPacket:
    if intervention == "replace":
        if replacement_packet is None:
            raise ValueError("replacement_packet is required for replace intervention")
        if tuple(replacement_packet.tensor.shape) != tuple(packet.tensor.shape):
            raise ValueError(
                "replacement packet shape must match target packet shape: "
                f"{tuple(replacement_packet.tensor.shape)} != {tuple(packet.tensor.shape)}"
            )
        return LatentPacket(
            packet_id=packet.packet_id,
            sender=packet.sender,
            receiver=packet.receiver,
            turn=packet.turn,
            tensor=replacement_packet.tensor.detach().cpu().clone(),
            metadata={
                **dict(packet.metadata),
                "intervention": "replace",
                "replacement_packet_id": replacement_packet.packet_id,
            },
        )

    if intervention == "ablate":
        intervened_tensor = torch.zeros_like(packet.tensor)
    elif intervention == "noise":
        if noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        noise = torch.randn(
            packet.tensor.shape,
            generator=generator,
            dtype=packet.tensor.dtype,
            device=packet.tensor.device,
        )
        intervened_tensor = packet.tensor + (noise * float(noise_std))
    else:
        raise ValueError(f"Unsupported packet intervention: {intervention}")

    return LatentPacket(
        packet_id=packet.packet_id,
        sender=packet.sender,
        receiver=packet.receiver,
        turn=packet.turn,
        tensor=intervened_tensor.detach().cpu().clone(),
        metadata={**dict(packet.metadata), "intervention": intervention},
    )


def replay_with_intervention(
    record: LatentRunRecord,
    *,
    replay_fn: ReplayFn,
    packet_id: str,
    intervention: InterventionKind,
    replacement_packet: Optional[LatentPacket] = None,
    noise_std: float = 0.1,
    outcome_flip_weight: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> InterventionResult:
    target_packet = record.packet(packet_id)
    intervened_packet = apply_packet_intervention(
        target_packet,
        intervention=intervention,
        replacement_packet=replacement_packet,
        noise_std=noise_std,
        generator=generator,
    )
    intervened_packets = _replace_packet(record.packets, packet_id, intervened_packet)
    intervened_outcome = replay_fn(intervened_packets)
    score_delta = float(intervened_outcome.score) - float(record.outcome.score)
    success_changed = bool(intervened_outcome.success) != bool(record.outcome.success)
    causal_impact = abs(score_delta) + (float(outcome_flip_weight) if success_changed else 0.0)

    return InterventionResult(
        packet_id=packet_id,
        intervention=intervention,
        baseline_answer=str(record.outcome.final_answer),
        intervened_answer=str(intervened_outcome.final_answer),
        baseline_score=float(record.outcome.score),
        intervened_score=float(intervened_outcome.score),
        score_delta=score_delta,
        baseline_success=bool(record.outcome.success),
        intervened_success=bool(intervened_outcome.success),
        success_changed=success_changed,
        causal_impact=causal_impact,
        packet_summary=target_packet.summary(),
        intervention_metadata={
            "noise_std": float(noise_std) if intervention == "noise" else None,
            "replacement_packet_id": None if replacement_packet is None else replacement_packet.packet_id,
        },
    )


def rank_packet_blame(
    record: LatentRunRecord,
    *,
    replay_fn: ReplayFn,
    interventions: Sequence[InterventionKind] = ("ablate", "noise"),
    replacement_packets: Optional[Mapping[str, LatentPacket]] = None,
    noise_std: float = 0.1,
    outcome_flip_weight: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> list[InterventionResult]:
    replacement_packets = replacement_packets or {}
    results: list[InterventionResult] = []
    for packet in record.packets:
        for intervention in interventions:
            replacement_packet = replacement_packets.get(packet.packet_id)
            if intervention == "replace" and replacement_packet is None:
                continue
            results.append(
                replay_with_intervention(
                    record,
                    replay_fn=replay_fn,
                    packet_id=packet.packet_id,
                    intervention=intervention,
                    replacement_packet=replacement_packet,
                    noise_std=noise_std,
                    outcome_flip_weight=outcome_flip_weight,
                    generator=generator,
                )
            )
    return sorted(
        results,
        key=lambda result: (
            -float(result.causal_impact),
            result.packet_id,
            result.intervention,
        ),
    )


def generate_blame_report(
    record: LatentRunRecord,
    ranked_results: Sequence[InterventionResult],
    *,
    top_k: int = 3,
) -> dict[str, Any]:
    top_results = list(ranked_results[: max(0, int(top_k))])
    suspected = None if not top_results else top_results[0]
    return {
        "run_id": record.run_id,
        "baseline": {
            "final_answer": record.outcome.final_answer,
            "score": float(record.outcome.score),
            "success": bool(record.outcome.success),
            "metadata": dict(record.outcome.metadata),
        },
        "packet_count": len(record.packets),
        "suspected_packet": None if suspected is None else suspected.packet_id,
        "suspected_intervention": None if suspected is None else suspected.intervention,
        "suspected_causal_impact": None if suspected is None else suspected.causal_impact,
        "top_results": [result.summary() for result in top_results],
    }


def _optional_bool(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    return None


def _short_text(value: Any, *, max_chars: int = 220) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)] + "..."


def _unique_non_empty(rows: Sequence[Mapping[str, Any]], key: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        value = str(row.get(key) or "")
        if not value or value in seen:
            continue
        values.append(value)
        seen.add(value)
    return values


def _accuracy_percentage(rows: Sequence[Mapping[str, Any]]) -> Optional[float]:
    values = [
        value for value in (_optional_bool(row.get("correct")) for row in rows)
        if value is not None
    ]
    if not values:
        return None
    return 100.0 * sum(1 for value in values if value) / len(values)


def _row_failure_class(row: Mapping[str, Any], baseline_rows: Sequence[Mapping[str, Any]]) -> str:
    sender_correct = _optional_bool(row.get("sender_answer_matches_target"))
    if sender_correct is False:
        return "sender_wrong"
    if sender_correct is None:
        return "sender_unmeasured"
    if any(
        str(baseline.get("method")) == "sender_answer_text_handoff"
        and _optional_bool(baseline.get("correct")) is True
        for baseline in baseline_rows
    ):
        return "latent_receiver_gap_verified_answer_available"
    if any(_optional_bool(baseline.get("correct")) is True for baseline in baseline_rows):
        return "latent_underperforms_text_context"
    return "latent_receiver_gap_sender_correct"


def build_latent_provenance_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    latent_methods: Sequence[str],
    baseline_methods: Sequence[str] = (),
    max_rows: int = 10,
) -> dict[str, Any]:
    """Summarize static provenance for latent handoff failures.

    This is intentionally separate from intervention replay. It answers: which
    sender trace, adapter cache, receiver context, and post-processing components
    produced the latent row that failed?
    """
    latent_method_set = set(latent_methods)
    baseline_method_set = set(baseline_methods)
    latent_rows = [row for row in rows if row.get("method") in latent_method_set]
    baseline_rows = [row for row in rows if row.get("method") in baseline_method_set]
    baseline_by_sample: dict[str, list[Mapping[str, Any]]] = {}
    for row in baseline_rows:
        baseline_by_sample.setdefault(str(row.get("sample_index")), []).append(row)

    wrong_rows: list[dict[str, Any]] = []
    failure_counts: dict[str, int] = {}
    for row in latent_rows:
        if _optional_bool(row.get("correct")) is not False:
            continue
        sample_baselines = baseline_by_sample.get(str(row.get("sample_index")), [])
        failure_class = _row_failure_class(row, sample_baselines)
        failure_counts[failure_class] = failure_counts.get(failure_class, 0) + 1
        if len(wrong_rows) >= max(0, int(max_rows)):
            continue
        wrong_rows.append(
            {
                "method": row.get("method"),
                "sample_index": row.get("sample_index"),
                "failure_class": failure_class,
                "target_answer": row.get("target_answer"),
                "predicted_answer": row.get("predicted_answer"),
                "sender_predicted_answer": row.get("sender_predicted_answer"),
                "sender_answer_matches_target": row.get("sender_answer_matches_target"),
                "sender_initial_predicted_answer": row.get("sender_initial_predicted_answer"),
                "sender_revision_predicted_answer": row.get("sender_revision_predicted_answer"),
                "sender_revision_decision_predicted_answer": row.get(
                    "sender_revision_decision_predicted_answer"
                ),
                "decoded_preview": _short_text(row.get("decoded_text")),
                "sender_trace_cache_hit": row.get("sender_trace_cache_hit"),
                "sender_trace_cache_path": row.get("sender_trace_cache_path"),
                "handoff_adapter_status": row.get("handoff_adapter_status"),
                "handoff_adapter_cache_path": row.get("handoff_adapter_cache_path"),
                "handoff_adapter_training_row_cache_path": row.get(
                    "handoff_adapter_training_row_cache_path"
                ),
                "receiver_context_status": row.get("receiver_context_status"),
                "receiver_context_latent_position": row.get(
                    "receiver_context_latent_position"
                ),
                "active_kv_cache_status": row.get("active_kv_cache_status"),
                "active_kv_cache_source": row.get("active_kv_cache_source"),
                "post_alignment_cosine_distance": row.get("post_alignment_cosine_distance"),
                "embedding_manifold_applied": row.get("embedding_manifold_applied"),
                "embedding_manifold_unique_token_count": row.get(
                    "embedding_manifold_unique_token_count"
                ),
                "generated_adapter_local_residual_applied": row.get(
                    "generated_adapter_local_residual_applied"
                ),
                "generated_adapter_local_residual_mean_top_similarity": row.get(
                    "generated_adapter_local_residual_mean_top_similarity"
                ),
                "baseline_comparisons": [
                    {
                        "method": baseline.get("method"),
                        "predicted_answer": baseline.get("predicted_answer"),
                        "correct": baseline.get("correct"),
                        "decoded_preview": _short_text(baseline.get("decoded_text")),
                    }
                    for baseline in sample_baselines
                ],
            }
        )

    method_accuracy = {
        method: _accuracy_percentage([row for row in rows if row.get("method") == method])
        for method in [*baseline_methods, *latent_methods]
    }
    return {
        "latent_methods": list(latent_methods),
        "baseline_methods": list(baseline_methods),
        "latent_sample_count": len(latent_rows),
        "baseline_sample_count": len(baseline_rows),
        "method_accuracy_percentage": method_accuracy,
        "failure_counts_by_class": failure_counts,
        "wrong_latent_rows": wrong_rows,
        "cache_paths": {
            "sender_trace": _unique_non_empty(latent_rows, "sender_trace_cache_path"),
            "adapter": _unique_non_empty(latent_rows, "handoff_adapter_cache_path"),
            "adapter_training_rows": _unique_non_empty(
                latent_rows,
                "handoff_adapter_training_row_cache_path",
            ),
        },
        "latent_reader_code_paths": [
            "benchmark_all.py:_collect_sender_generated_consensus_state",
            "benchmark_all.py:_run_generated_latent_variant",
            "benchmark_all.py:_decode_handoff",
            "src.utils.lm_eval.prepare_latent_prefix_state",
            "src.utils.lm_eval.prepare_receiver_context_latent_prefix_state",
            "src.models.handoff_adapter.apply_embedding_manifold_projection",
        ],
    }
