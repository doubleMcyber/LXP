from __future__ import annotations

import pytest
import torch

from src.utils.latent_blame import (
    LatentRunRecord,
    RunOutcome,
    apply_packet_intervention,
    build_latent_provenance_report,
    generate_blame_report,
    rank_packet_blame,
    record_latent_packet,
)


def _toy_replay(packets) -> RunOutcome:
    score = float(sum(packet.tensor.reshape(-1)[0].item() for packet in packets))
    success = score > 0.5
    return RunOutcome(
        final_answer="correct" if success else "wrong",
        score=score,
        success=success,
    )


def test_rank_packet_blame_identifies_packet_that_flips_failed_run() -> None:
    packets = (
        record_latent_packet(sender="a", receiver="b", turn=0, tensor=torch.tensor([0.1]), packet_id="p0"),
        record_latent_packet(sender="a", receiver="b", turn=1, tensor=torch.tensor([-1.0]), packet_id="p1"),
        record_latent_packet(sender="a", receiver="b", turn=2, tensor=torch.tensor([0.1]), packet_id="p2"),
    )
    record = LatentRunRecord(
        run_id="failed-run",
        packets=packets,
        outcome=_toy_replay(packets),
    )
    replacement = record_latent_packet(
        sender="a",
        receiver="b",
        turn=1,
        tensor=torch.tensor([2.0]),
        packet_id="p1-compatible-replacement",
    )

    ranked = rank_packet_blame(
        record,
        replay_fn=_toy_replay,
        interventions=("replace",),
        replacement_packets={"p1": replacement},
    )
    report = generate_blame_report(record, ranked)

    assert record.outcome.success is False
    assert ranked[0].packet_id == "p1"
    assert ranked[0].success_changed is True
    assert ranked[0].score_delta == pytest.approx(3.0)
    assert report["suspected_packet"] == "p1"
    assert report["suspected_causal_impact"] == pytest.approx(4.0)


def test_ablation_and_noise_interventions_preserve_packet_identity_and_shape() -> None:
    packet = record_latent_packet(
        sender="reasoner",
        receiver="actor",
        turn=3,
        tensor=torch.ones(2, 3),
        packet_id="handoff",
        metadata={"surface": "input_embedding"},
    )

    ablated = apply_packet_intervention(packet, intervention="ablate")
    generator = torch.Generator()
    generator.manual_seed(0)
    noised = apply_packet_intervention(
        packet,
        intervention="noise",
        noise_std=0.1,
        generator=generator,
    )

    assert ablated.packet_id == packet.packet_id
    assert ablated.metadata["surface"] == "input_embedding"
    assert tuple(ablated.tensor.shape) == tuple(packet.tensor.shape)
    assert torch.count_nonzero(ablated.tensor).item() == 0
    assert noised.packet_id == packet.packet_id
    assert tuple(noised.tensor.shape) == tuple(packet.tensor.shape)
    assert not torch.allclose(noised.tensor, packet.tensor)


def test_replacement_intervention_requires_compatible_tensor_shape() -> None:
    packet = record_latent_packet(
        sender="a",
        receiver="b",
        turn=0,
        tensor=torch.ones(2, 3),
        packet_id="p",
    )
    incompatible = record_latent_packet(
        sender="a",
        receiver="b",
        turn=0,
        tensor=torch.ones(4),
        packet_id="replacement",
    )

    with pytest.raises(ValueError, match="shape"):
        apply_packet_intervention(
            packet,
            intervention="replace",
            replacement_packet=incompatible,
        )


def test_static_latent_provenance_classifies_verified_answer_gap() -> None:
    rows = [
        {
            "method": "sender_answer_text_handoff",
            "sample_index": 11,
            "target_answer": "300",
            "predicted_answer": "300",
            "correct": True,
            "decoded_text": "Final answer: 300",
        },
        {
            "method": "generated_context_latent_handoff",
            "sample_index": 11,
            "target_answer": "300",
            "predicted_answer": "175",
            "sender_predicted_answer": "300",
            "sender_answer_matches_target": True,
            "sender_initial_predicted_answer": "175",
            "sender_revision_predicted_answer": "300",
            "sender_revision_decision_predicted_answer": "300",
            "decoded_text": "Final answer: 175",
            "correct": False,
            "sender_trace_cache_hit": True,
            "sender_trace_cache_path": ".cache/generated_trajectory_traces/trace.pt",
            "handoff_adapter_status": "generated_trajectory_loaded_raw",
            "handoff_adapter_cache_path": ".cache/generated_trajectory_adapter/adapter.pt",
            "handoff_adapter_training_row_cache_path": ".cache/generated_trajectory_rows/rows.pt",
            "receiver_context_latent_position": "after_context",
            "post_alignment_cosine_distance": 0.11,
            "embedding_manifold_applied": True,
            "generated_adapter_local_residual_applied": True,
        },
    ]

    report = build_latent_provenance_report(
        rows,
        baseline_methods=("sender_answer_text_handoff",),
        latent_methods=("generated_context_latent_handoff",),
    )

    assert report["method_accuracy_percentage"]["sender_answer_text_handoff"] == 100.0
    assert report["method_accuracy_percentage"]["generated_context_latent_handoff"] == 0.0
    assert report["failure_counts_by_class"] == {
        "latent_receiver_gap_verified_answer_available": 1
    }
    wrong_row = report["wrong_latent_rows"][0]
    assert wrong_row["sample_index"] == 11
    assert wrong_row["baseline_comparisons"][0]["predicted_answer"] == "300"
    assert report["cache_paths"]["adapter"] == [".cache/generated_trajectory_adapter/adapter.pt"]
