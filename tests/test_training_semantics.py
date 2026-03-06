from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from latent_pipeline import _build_alignment_cache_key
from train_compressor import (
    CompressionTrainConfig,
    _tokenize_text_batch,
    resolve_training_alignment_context,
    train_reasoner_stage2,
)


class _TinyTokenizer:
    def __init__(self, *, vocab_size: int = 128, offset: int = 0) -> None:
        self.vocab_size = vocab_size
        self.offset = offset
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def _encode_text(self, text: str) -> list[int]:
        pieces = [piece for piece in text.lower().split() if piece]
        if not pieces:
            return [self.eos_token_id]
        return [
            ((sum(ord(char) for char in piece) + self.offset) % (self.vocab_size - 2)) + 2
            for piece in pieces
        ]

    def __call__(
        self,
        texts,
        *,
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 16,
        return_tensors: str = "pt",
    ):
        del return_tensors
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self._encode_text(text) for text in texts]
        if truncation:
            encoded = [tokens[:max_length] for tokens in encoded]
        target_width = max(len(tokens) for tokens in encoded) if padding else None
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for tokens in encoded:
            if target_width is None:
                input_ids.append(tokens)
                attention_mask.append([1] * len(tokens))
                continue
            pad_width = target_width - len(tokens)
            input_ids.append(tokens + [self.pad_token_id] * pad_width)
            attention_mask.append([1] * len(tokens) + [0] * pad_width)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def _make_tiny_models(hidden_dim: int = 16, vocab_size: int = 128, with_layers: bool = False):
    from torch import nn

    class TinyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_dim)
            self.linear = nn.Linear(hidden_dim, hidden_dim)
            if with_layers:
                self.layers = [object(), object(), object()]

        def forward(
            self,
            input_ids=None,
            inputs_embeds=None,
            attention_mask=None,
            use_cache=False,
            return_dict=True,
        ):
            del attention_mask, use_cache, return_dict
            if inputs_embeds is not None:
                hidden = self.linear(inputs_embeds)
            else:
                hidden = self.linear(self.embedding(input_ids))
            return MagicMock(last_hidden_state=hidden)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = TinyBackbone()
            self.lm_head = nn.Linear(hidden_dim, vocab_size)
            self._embed = self.model.embedding
            self.dtype = torch.float32

        def forward(
            self,
            input_ids=None,
            inputs_embeds=None,
            attention_mask=None,
            use_cache=False,
            return_dict=True,
        ):
            del attention_mask, use_cache, return_dict
            if inputs_embeds is not None:
                hidden = self.model.linear(inputs_embeds)
            else:
                hidden = self.model(input_ids=input_ids).last_hidden_state
            return MagicMock(logits=self.lm_head(hidden))

        def get_input_embeddings(self):
            return self._embed

    return TinyModel(), TinyModel()


def test_train_reasoner_stage2_uses_actor_tokenizer_for_actor_labels() -> None:
    texts = ["alpha beta", "gamma delta"]
    dataloader = DataLoader(
        texts,
        batch_size=2,
        collate_fn=lambda batch: {"texts": list(batch)},
    )
    reasoner, actor = _make_tiny_models()
    reasoner_tokenizer = _TinyTokenizer(offset=0)
    actor_tokenizer = _TinyTokenizer(offset=23)
    config = CompressionTrainConfig(
        compressed_steps=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        num_epochs=1,
        wandb_enabled=False,
        checkpoint_enabled=False,
        reasoner_max_length=8,
        actor_max_length=8,
    )

    class _CapturingLoss:
        captured_actor_labels: torch.Tensor | None = None

        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def __call__(self, *, actor_logits_compressed, actor_logits_full, full_latents, compressed_latents, actor_labels):
            del actor_logits_full, full_latents, compressed_latents
            _CapturingLoss.captured_actor_labels = actor_labels.detach().clone()
            zero = actor_logits_compressed.sum() * 0.0
            return {
                "loss": zero,
                "l_task": zero,
                "l_pref": zero,
                "l_geom": zero,
                "pref_avg_entropy": zero,
                "pref_avg_weight": zero,
                "pref_first_token_entropy": zero,
                "pref_first_token_weight": zero,
                "pref_first_token_kl": zero,
                "pref_avg_top1_probability": zero,
                "pref_first_token_top1_probability": zero,
                "pref_avg_logit_margin": zero,
                "pref_first_token_logit_margin": zero,
                "pref_first_token_weight_ratio": zero,
            }

    with patch("train_compressor.LatentCompressorLoss", _CapturingLoss):
        train_reasoner_stage2(
            reasoner,
            actor,
            dataloader,
            config,
            reasoner_tokenizer=reasoner_tokenizer,
            actor_tokenizer=actor_tokenizer,
        )

    assert _CapturingLoss.captured_actor_labels is not None
    expected_actor_labels = _tokenize_text_batch(
        actor_tokenizer,
        texts,
        device=torch.device("cpu"),
        max_length=config.actor_max_length,
    )["labels"]
    reasoner_labels = _tokenize_text_batch(
        reasoner_tokenizer,
        texts,
        device=torch.device("cpu"),
        max_length=config.reasoner_max_length,
    )["labels"]

    assert torch.equal(_CapturingLoss.captured_actor_labels.cpu(), expected_actor_labels)
    assert not torch.equal(expected_actor_labels, reasoner_labels)


def test_resolve_training_alignment_context_uses_shared_pipeline_cache_key() -> None:
    cfg = OmegaConf.create(
        {
            "agent_a_model": "reasoner-a",
            "agent_b_model": "actor-b",
            "torch_dtype": "bfloat16",
            "alignment": {
                "semantic_anchor_count": 250,
                "cache_dir": ".cache/alignment",
                "reasoning_layer_weights": [0.2, 0.3, 0.5],
            },
        }
    )
    expected_cache_key = _build_alignment_cache_key(
        agent_a_model="reasoner-a",
        agent_b_model="actor-b",
        torch_dtype="bfloat16",
        semantic_anchor_count=250,
        reasoning_layer_indices=(12, 16, 20),
        reasoning_layer_weights=(0.2, 0.3, 0.5),
    )
    reasoner, actor = _make_tiny_models(with_layers=True)
    fake_state = {
        "global_alignment_q": torch.eye(16),
        "alignment_mode": "semantic_anchor_global",
        "global_alignment_cache_key": expected_cache_key,
        "global_alignment_cache_hit": True,
        "global_alignment_cache_path": ".cache/alignment/q.pt",
        "global_reasoning_layer_indices": (12, 16, 20),
        "global_reasoning_layer_weights": (0.2, 0.3, 0.5),
        "semantic_anchor_count": 250,
    }

    with patch("train_compressor.load_or_compute_global_alignment_state", return_value=fake_state):
        context = resolve_training_alignment_context(
            reasoner_model=reasoner,
            actor_model=actor,
            reasoner_tokenizer=_TinyTokenizer(offset=0),
            actor_tokenizer=_TinyTokenizer(offset=31),
            alignment_cfg=cfg,
        )

    assert context["alignment_mode"] == "semantic_anchor_global"
    assert context["global_alignment_cache_key"] == expected_cache_key
    assert context["semantic_anchor_count"] == 250
