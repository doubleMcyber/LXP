from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from latent_pipeline import (
    _is_kv_cache_compatible,
    _kv_cache_seq_len,
    _move_kv_cache_to_device,
)
from src.models.dynamics import _kv_cache_compatibility_status
from src.utils.lm_eval import prepare_latent_prefix_state, prepare_receiver_context_latent_prefix_state


class DummyActorModel:
    def __init__(
        self,
        *,
        num_hidden_layers: int,
        num_attention_heads: int,
        hidden_size: int,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
    ) -> None:
        self.config = SimpleNamespace(
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )


def _make_tuple_cache(
    *,
    num_layers: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    batch_size: int = 1,
) -> tuple:
    cache = []
    for _ in range(num_layers):
        k = torch.zeros(batch_size, num_heads, seq_len, head_dim)
        v = torch.zeros(batch_size, num_heads, seq_len, head_dim)
        cache.append((k, v))
    return tuple(cache)


def test_tuple_cache_is_compatible_when_shapes_match() -> None:
    actor = DummyActorModel(
        num_hidden_layers=4,
        num_attention_heads=8,
        hidden_size=512,
        num_key_value_heads=8,
        head_dim=64,
    )
    kv_cache = _make_tuple_cache(num_layers=4, num_heads=8, seq_len=12, head_dim=64)
    assert _is_kv_cache_compatible(kv_cache, actor)
    assert _kv_cache_seq_len(kv_cache) == 12


def test_tuple_cache_is_incompatible_when_layer_count_differs() -> None:
    actor = DummyActorModel(
        num_hidden_layers=6,
        num_attention_heads=8,
        hidden_size=512,
        num_key_value_heads=8,
        head_dim=64,
    )
    kv_cache = _make_tuple_cache(num_layers=4, num_heads=8, seq_len=12, head_dim=64)
    assert not _is_kv_cache_compatible(kv_cache, actor)
    status, reason = _kv_cache_compatibility_status(kv_cache, actor)
    assert status == "unsupported_architecture_mismatch"
    assert "layer_count_mismatch" in reason


def test_tuple_cache_is_incompatible_when_heads_or_head_dim_differ() -> None:
    actor = DummyActorModel(
        num_hidden_layers=4,
        num_attention_heads=8,
        hidden_size=512,
        num_key_value_heads=8,
        head_dim=64,
    )
    bad_heads_cache = _make_tuple_cache(num_layers=4, num_heads=4, seq_len=12, head_dim=64)
    bad_dim_cache = _make_tuple_cache(num_layers=4, num_heads=8, seq_len=12, head_dim=80)
    assert not _is_kv_cache_compatible(bad_heads_cache, actor)
    assert not _is_kv_cache_compatible(bad_dim_cache, actor)
    assert _kv_cache_compatibility_status(bad_heads_cache, actor)[0] == "unsupported_architecture_mismatch"
    assert _kv_cache_compatibility_status(bad_dim_cache, actor)[0] == "unsupported_architecture_mismatch"


def test_tuple_cache_is_incompatible_when_later_layer_shape_differs() -> None:
    actor = DummyActorModel(
        num_hidden_layers=2,
        num_attention_heads=8,
        hidden_size=512,
        num_key_value_heads=8,
        head_dim=64,
    )
    first_layer = _make_tuple_cache(num_layers=1, num_heads=8, seq_len=12, head_dim=64)[0]
    second_layer = _make_tuple_cache(num_layers=1, num_heads=8, seq_len=13, head_dim=64)[0]
    status, reason = _kv_cache_compatibility_status((first_layer, second_layer), actor)

    assert status == "unsupported_architecture_mismatch"
    assert "layer_1_shape_mismatch" in reason


def test_tuple_cache_is_invalid_when_key_value_shapes_differ() -> None:
    actor = DummyActorModel(
        num_hidden_layers=1,
        num_attention_heads=8,
        hidden_size=512,
        num_key_value_heads=8,
        head_dim=64,
    )
    key = torch.zeros(1, 8, 12, 64)
    value = torch.zeros(1, 8, 11, 64)
    status, reason = _kv_cache_compatibility_status(((key, value),), actor)

    assert status == "invalid_cache"
    assert "key_value_shape_mismatch" in reason


class TinyPrefixModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 4)
        self.config = SimpleNamespace(
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=4,
            num_key_value_heads=2,
            head_dim=2,
        )

    def get_input_embeddings(self):
        return self.embedding

    def forward(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        use_cache=True,
        return_dict=True,
    ):
        del position_ids, use_cache, return_dict
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)
        if past_key_values is None:
            seq_len = int(inputs_embeds.shape[1])
            key = torch.zeros(inputs_embeds.shape[0], 2, seq_len, 2, device=inputs_embeds.device)
            value = torch.zeros_like(key)
            past_key_values = ((key, value),)
        logits = torch.zeros(
            inputs_embeds.shape[0],
            inputs_embeds.shape[1],
            self.embedding.num_embeddings,
            device=inputs_embeds.device,
        )
        return SimpleNamespace(
            logits=logits,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
        )


def test_prepare_latent_prefix_state_rejects_wrong_embedding_dimension() -> None:
    model = TinyPrefixModel()

    with pytest.raises(ValueError, match="embedding dimension"):
        prepare_latent_prefix_state(
            model=model,
            handoff_step=torch.zeros(1, 1, 5),
            kv_cache=None,
        )


def test_prepare_latent_prefix_state_drops_batch_mismatched_cache() -> None:
    model = TinyPrefixModel()
    kv_cache = _make_tuple_cache(
        num_layers=1,
        num_heads=2,
        seq_len=3,
        head_dim=2,
        batch_size=2,
    )

    prefix_state = prepare_latent_prefix_state(
        model=model,
        handoff_step=torch.zeros(1, 1, 4),
        kv_cache=kv_cache,
    )

    assert prefix_state["kv_cache_transferred"] is False
    assert prefix_state["kv_cache_status"] == "invalid_cache"
    assert "batch_size_mismatch" in prefix_state["kv_cache_reason"]
    assert prefix_state["attention_mask"].shape == (1, 1)


class TinyTokenizer:
    def __call__(self, text, *, return_tensors=None, add_special_tokens=False):
        del add_special_tokens
        token_count = max(1, len(str(text).split()))
        input_ids = torch.arange(token_count, dtype=torch.long).unsqueeze(0) % 16
        attention_mask = torch.ones_like(input_ids)
        if return_tensors == "pt":
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        return {"input_ids": input_ids.tolist(), "attention_mask": attention_mask.tolist()}


def test_prepare_receiver_context_latent_prefix_state_uses_actor_prompt_cache() -> None:
    model = TinyPrefixModel()

    prefix_state = prepare_receiver_context_latent_prefix_state(
        model=model,
        tokenizer=TinyTokenizer(),
        context_text="What is fifty thousand?",
        handoff_step=torch.zeros(1, 1, 4),
    )

    assert prefix_state["kv_cache_transferred"] is True
    assert prefix_state["kv_cache_status"] == "transferred"
    assert prefix_state["receiver_context_status"] == "used_prompt_prefix"
    assert prefix_state["receiver_context_token_count"] == 4
    assert prefix_state["active_kv_cache_source"] == "receiver_context"
    assert prefix_state["attention_mask"].shape == (1, 5)


def test_prepare_receiver_context_latent_prefix_state_can_append_answer_suffix() -> None:
    model = TinyPrefixModel()

    prefix_state = prepare_receiver_context_latent_prefix_state(
        model=model,
        tokenizer=TinyTokenizer(),
        context_text="What is fifty thousand?",
        handoff_step=torch.zeros(1, 1, 4),
        suffix_text="\n\nFinal answer:",
        decoded_text_prefix="Final answer:",
    )

    assert prefix_state["receiver_context_token_count"] == 4
    assert prefix_state["receiver_context_suffix_token_count"] == 2
    assert prefix_state["decoded_text_prefix"] == "Final answer:"
    assert prefix_state["active_kv_cache_source"] == "receiver_context"
    assert prefix_state["attention_mask"].shape == (1, 7)


def test_prepare_receiver_context_latent_prefix_state_can_place_latent_before_context() -> None:
    model = TinyPrefixModel()

    prefix_state = prepare_receiver_context_latent_prefix_state(
        model=model,
        tokenizer=TinyTokenizer(),
        context_text="What is fifty thousand?",
        handoff_step=torch.zeros(1, 1, 4),
        suffix_text="\n\nFinal answer:",
        decoded_text_prefix="Final answer:",
        latent_position="before_context",
    )

    assert prefix_state["kv_cache_transferred"] is False
    assert prefix_state["kv_cache_status"] == "not_provided"
    assert prefix_state["receiver_context_latent_position"] == "before_context"
    assert prefix_state["receiver_context_token_count"] == 4
    assert prefix_state["receiver_context_suffix_token_count"] == 2
    assert prefix_state["decoded_text_prefix"] == "Final answer:"
    assert prefix_state["active_kv_cache_source"] == "latent_then_receiver_context"
    assert prefix_state["attention_mask"].shape == (1, 7)


def test_dynamic_cache_object_supports_seq_len_and_device_move() -> None:
    class FakeDynamicCache:
        def __init__(self, seq_len: int, device: torch.device) -> None:
            self._seq_len = seq_len
            self.device = device

        def get_seq_length(self) -> int:
            return self._seq_len

        def to(self, device: torch.device) -> "FakeDynamicCache":
            return FakeDynamicCache(self._seq_len, device)

    cache = FakeDynamicCache(seq_len=9, device=torch.device("cpu"))
    moved_cache = _move_kv_cache_to_device(cache, torch.device("cpu"))

    assert _kv_cache_seq_len(cache) == 9
    assert _kv_cache_seq_len(moved_cache) == 9
    assert hasattr(moved_cache, "device")
