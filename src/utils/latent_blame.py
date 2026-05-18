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
