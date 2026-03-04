from __future__ import annotations

from types import SimpleNamespace

import torch

from latent_pipeline import (
    _is_kv_cache_compatible,
    _kv_cache_seq_len,
    _move_kv_cache_to_device,
)


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
