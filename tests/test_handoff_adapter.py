from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from src.models.handoff_adapter import (
    HandoffAdapterFitConfig,
    fit_handoff_adapter_state,
    project_to_embedding_manifold,
    resample_sequence,
)
from src.utils.alignment import apply_alignment


class _TinyTokenizer:
    def __call__(self, text: str, **kwargs):
        del kwargs
        token_ids = [((sum(ord(char) for char in piece) % 23) + 2) for piece in text.split()]
        if not token_ids:
            token_ids = [1]
        return {
            "input_ids": torch.tensor([token_ids], dtype=torch.long),
            "attention_mask": torch.ones(1, len(token_ids), dtype=torch.long),
        }


class _TinyBackbone(nn.Module):
    def __init__(self, embedding: nn.Embedding) -> None:
        super().__init__()
        self.embedding = embedding

    def forward(self, input_ids, attention_mask=None, **kwargs):
        del attention_mask, kwargs
        embeddings = self.embedding(input_ids)
        hidden_states = (
            embeddings,
            embeddings + 0.01,
            embeddings + 0.02,
        )
        return SimpleNamespace(hidden_states=hidden_states, last_hidden_state=hidden_states[-1])


class _TinyAgent(nn.Module):
    def __init__(self, embedding: nn.Embedding) -> None:
        super().__init__()
        self.embedding = embedding
        self.model = _TinyBackbone(embedding)

    def get_input_embeddings(self):
        return self.embedding


def test_resample_sequence_changes_step_count_without_changing_dim() -> None:
    sequence = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)

    resized = resample_sequence(sequence, 5)

    assert resized.shape == (1, 5, 4)
    assert torch.allclose(resized[:, :1, :], sequence[:, :1, :])
    assert torch.allclose(resized[:, -1:, :], sequence[:, -1:, :])


def test_fit_handoff_adapter_state_learns_train_split_mapping() -> None:
    torch.manual_seed(0)
    source_embedding = nn.Embedding(32, 4)
    target_embedding = nn.Embedding(32, 4)
    with torch.no_grad():
        source_embedding.weight.copy_(torch.arange(128, dtype=torch.float32).reshape(32, 4) / 100.0)
        target_embedding.weight.copy_(source_embedding.weight * 1.5 + 0.25)
    agent_a = _TinyAgent(source_embedding)
    agent_b = _TinyAgent(target_embedding)
    prompts = ["alpha beta", "gamma delta", "epsilon zeta"]
    identity_alignment = {
        "mapping_matrix": torch.eye(4),
        "mapping_bias": torch.zeros(1, 4),
    }

    adapter = fit_handoff_adapter_state(
        prompts=prompts,
        tokenizer_a=_TinyTokenizer(),
        tokenizer_b=_TinyTokenizer(),
        agent_a=agent_a,
        agent_b=agent_b,
        base_alignment_state=identity_alignment,
        reasoning_layer_indices=(0, 1, 2),
        reasoning_layer_weights=(0.2, 0.3, 0.5),
        fit_config=HandoffAdapterFitConfig(
            regularization=1e-5,
            residual_max_norm_ratio=1.0,
            max_length=16,
            adaptive_projection_strength=0.0,
        ),
    )

    source = torch.stack([source_embedding(torch.tensor([2, 3])).squeeze(0)], dim=0)
    adapted = apply_alignment(source, adapter)

    assert adapter["adapter_type"] == "ridge_sequence_handoff"
    assert adapter["training_prompt_count"] == 3
    assert adapter["training_token_count"] == 6
    assert adapted.shape == source.shape
    assert adapter["training_reconstruction_mse"] < 1e-3


def test_project_to_embedding_manifold_uses_nearest_receiver_embedding() -> None:
    embedding_weight = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    hidden_states = torch.tensor([[[0.9, 0.1], [0.2, 0.8]]])

    projected, metrics = project_to_embedding_manifold(
        hidden_states,
        embedding_weight,
        top_k=1,
        blend=1.0,
    )

    assert torch.allclose(projected, torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))
    assert metrics["embedding_manifold_unique_token_count"] == 2.0
