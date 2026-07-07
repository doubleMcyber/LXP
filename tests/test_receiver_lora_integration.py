from __future__ import annotations

from argparse import Namespace
from contextlib import contextmanager
import json
import sys
from typing import Any

from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

import benchmark_all
import latent_pipeline


def _write_receiver_lora_artifact(
    path,
    *,
    rank: int = 2,
    alpha: float = 4.0,
    value: float = 1.0,
) -> None:
    torch.save(
        {
            "rank": rank,
            "alpha": alpha,
            "state": {
                "model.layers.0.self_attn.q_proj.A": torch.full((1,), value),
                "model.layers.0.self_attn.q_proj.B": torch.full((1,), value + 1.0),
            },
        },
        path,
    )


def test_receiver_lora_cli_flags_flow_into_cfg(monkeypatch, tmp_path) -> None:
    lora_path = tmp_path / "receiver_lora.pt"
    captured: dict[str, Any] = {}

    def fake_run_benchmark(**kwargs):
        captured.update(kwargs)
        return [], [], {"phase_gate_report": {"passed": True}}

    monkeypatch.setattr(benchmark_all, "run_benchmark", fake_run_benchmark)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_all.py",
            "--limit",
            "1",
            "--receiver-lora-path",
            str(lora_path),
            "--receiver-lora-scope",
            "all",
        ],
    )

    benchmark_all.main()

    assert captured["receiver_lora_path"] == lora_path
    assert captured["receiver_lora_scope"] == "all"

    cfg = benchmark_all._configured_base_cfg(
        receiver_lora_path=lora_path,
        receiver_lora_scope="all",
    )
    assert str(cfg.handoff.receiver_lora.path) == str(lora_path)
    assert cfg.handoff.receiver_lora.scope == "all"


def test_receiver_lora_manifest_round_trip_and_sha_mismatch(tmp_path) -> None:
    lora_path = tmp_path / "receiver_lora.pt"
    _write_receiver_lora_artifact(lora_path)
    cfg = OmegaConf.create(
        {"handoff": {"receiver_lora": {"path": str(lora_path), "scope": "latent_only"}}}
    )
    receiver_lora_identity = benchmark_all._receiver_lora_identity_manifest(cfg)

    manifest = benchmark_all._build_eval_manifest(
        suite_name="standard",
        dataset_name="gsm8k",
        dataset_split="validation",
        limit=1,
        sample_indices=[0],
        methods=("generated_latent_handoff",),
        agent_a_model="agent-a",
        agent_b_model="agent-b",
        seed=0,
        semantic_smoke=False,
        mvp_smoke=False,
        hetero_smoke=False,
        receiver_lora_identity=receiver_lora_identity,
        sample_fingerprints=[],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = benchmark_all._load_eval_manifest(manifest_path)

    assert loaded["receiver_lora"] == receiver_lora_identity

    args = Namespace(receiver_lora_path=None, receiver_lora_scope=None)
    benchmark_all._apply_eval_manifest_to_args(args, loaded)
    assert args.receiver_lora_path == lora_path
    assert args.receiver_lora_scope == "latent_only"

    mismatch_path = tmp_path / "receiver_lora_mismatch.pt"
    _write_receiver_lora_artifact(mismatch_path, value=9.0)
    mismatch_args = Namespace(
        receiver_lora_path=mismatch_path,
        receiver_lora_scope=None,
    )
    with pytest.raises(ValueError, match="Receiver LoRA file_sha256 mismatch"):
        benchmark_all._apply_eval_manifest_to_args(mismatch_args, loaded)


def test_pipeline_state_key_includes_receiver_lora_identity() -> None:
    base = {
        "agent_a_model": "agent-a",
        "agent_b_model": "agent-b",
        "torch_dtype": "bfloat16",
        "device_map": "none",
        "handoff": {"receiver_lora": {"path": "a.pt", "scope": "latent_only"}},
    }
    key_a = latent_pipeline._pipeline_state_key(OmegaConf.create(base))
    key_b = latent_pipeline._pipeline_state_key(
        OmegaConf.create(
            {
                **base,
                "handoff": {"receiver_lora": {"path": "b.pt", "scope": "latent_only"}},
            }
        )
    )
    key_c = latent_pipeline._pipeline_state_key(
        OmegaConf.create(
            {
                **base,
                "handoff": {"receiver_lora": {"path": "a.pt", "scope": "all"}},
            }
        )
    )

    assert key_a != key_b
    assert key_a != key_c
    assert key_a[-2:] == ("a.pt", "latent_only")
    assert key_c[-2:] == ("a.pt", "all")


class _MockAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.probe = nn.Parameter(torch.zeros(()))
        self.embedding = nn.Embedding(8, 2)
        self.events: list[tuple[str, bool]] = []

    def get_input_embeddings(self):
        return self.embedding


def test_receiver_lora_latent_only_scopes_only_generated_decode(monkeypatch) -> None:
    cfg = OmegaConf.create(
        {
            "max_new_tokens": 2,
            "benchmark": {},
            "handoff": {
                "receiver_lora": {"path": "receiver_lora.pt", "scope": "latent_only"},
                "receiver_context": {"latent_answer_suffix": ""},
                "embedding_manifold": {"enabled": False},
                "generated_trajectory_adapter": {
                    "enabled": False,
                    "input_space": "aligned",
                    "source_mode": "generated_text",
                    "target_mode": "generated_text",
                    "target_alignment": "character",
                },
            },
            "alignment": {
                "strategy": "hybrid_affine",
                "prompt_calibration": {"enabled": False},
            },
        }
    )
    agent_b = _MockAgent()
    state = {
        "tokenizer_b": object(),
        "agent_b": agent_b,
        "global_alignment_cache_hit": False,
        "receiver_lora_file_sha256": "sha123",
    }

    @contextmanager
    def fake_receiver_lora_scope(model, enabled):
        model.events.append(("enter", bool(enabled)))
        try:
            yield 1
        finally:
            model.events.append(("exit", False))

    def fake_decode_handoff(**kwargs):
        assert kwargs["agent_b"].events == [("enter", True)]
        return {
            "decoded_text": "1",
            "generated_tokens": 1,
            "receiver_input_token_count": 1,
            "decode_status": "decoded",
            "answer_token_count": 1,
            "answer_nll": 0.0,
            "answer_perplexity": 1.0,
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
            "raw_handoff_entropy": 0.0,
        }

    monkeypatch.setattr(benchmark_all, "receiver_lora_scope", fake_receiver_lora_scope)
    monkeypatch.setattr(
        benchmark_all,
        "_alignment_variant_state",
        lambda cfg, state, **_: (cfg, state),
    )
    monkeypatch.setattr(
        benchmark_all,
        "_collect_sender_generated_consensus_state",
        lambda *_, **__: {
            "consensus_hidden_states": torch.zeros(1, 1, 2),
            "generated_reasoning_text": "Final answer: 1",
            "generated_reasoning_token_count": 1,
            "generated_reasoning_status": "complete",
            "generated_reasoning_final_answer_marker": True,
            "generated_token_ids": [1],
        },
    )
    monkeypatch.setattr(
        benchmark_all,
        "_generated_adapter_alignment_state_from_variant_state",
        lambda _: {"mapping_matrix": torch.eye(2)},
    )
    monkeypatch.setattr(
        benchmark_all,
        "_load_or_train_generated_trajectory_adapter_state",
        lambda *_, **__: {"enabled": False, "status": "disabled", "state": None},
    )
    monkeypatch.setattr(benchmark_all, "apply_alignment", lambda tensor, _: tensor)
    monkeypatch.setattr(
        benchmark_all,
        "apply_handoff_adapter",
        lambda handoff_step, _: (
            handoff_step,
            {"handoff_adapter_applied": False, "handoff_adapter_delta_norm": None},
        ),
    )
    monkeypatch.setattr(
        benchmark_all,
        "_generated_adapter_token_readout",
        lambda *_, **__: {
            "generated_adapter_token_readout_applied": False,
            "generated_adapter_token_readout_mean_similarity": None,
            "generated_adapter_token_readout_token_count": None,
            "generated_adapter_token_readout_text": None,
        },
    )
    monkeypatch.setattr(
        benchmark_all,
        "apply_embedding_manifold_projection",
        lambda handoff_step, *_, **__: (
            handoff_step,
            {
                "embedding_manifold_applied": False,
                "embedding_manifold_delta_norm": None,
                "embedding_manifold_mean_top_similarity": None,
                "embedding_manifold_unique_token_count": None,
            },
        ),
    )
    monkeypatch.setattr(
        benchmark_all,
        "_alignment_distances",
        lambda **_: {
            "pre_alignment_l2_distance": 0.0,
            "pre_alignment_cosine_distance": 0.0,
            "post_alignment_l2_distance": 0.0,
            "post_alignment_cosine_distance": 0.0,
        },
    )
    monkeypatch.setattr(benchmark_all, "_decode_handoff", fake_decode_handoff)

    result = benchmark_all._run_generated_latent_variant(
        "question",
        "1",
        cfg,
        state,
        include_prompt=False,
        method_alignment_mode="generated_latent_handoff",
    )

    assert agent_b.events == [("enter", True), ("exit", False)]
    assert result["receiver_lora_applied"] is True
    assert result["receiver_lora_sha"] == "sha123"

    monkeypatch.setattr(
        benchmark_all,
        "prepare_text_prefix_state",
        lambda **_: {"prefix_seq_len": 1},
    )
    monkeypatch.setattr(
        benchmark_all,
        "greedy_decode_from_prefix",
        lambda **_: {"decoded_text": "1", "generated_tokens": 1},
    )
    monkeypatch.setattr(
        benchmark_all,
        "compute_answer_metrics_from_prefix",
        lambda **_: {
            "answer_token_count": 1,
            "answer_nll": 0.0,
            "answer_perplexity": 1.0,
        },
    )

    agent_b.events.clear()
    benchmark_all.run_pure_text_cot("question", "1", cfg, state)

    assert agent_b.events == []
