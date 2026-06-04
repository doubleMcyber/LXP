from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from src.utils.lm_eval import (
    compute_answer_metrics_from_prefix,
    compute_answer_metrics_from_prefix_embeddings,
    compute_first_token_metrics_from_prefix_embeddings,
    generate_from_prefix_embeddings,
    generate_from_text_prefix,
)


class _SingleTokenTokenizer:
    def __init__(self) -> None:
        self.token_to_id = {
            "1": 1,
            "Final answer: 1": 2,
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [self.token_to_id[str(text)]]


class _StaticLogitModel:
    def __init__(self) -> None:
        self.logits = torch.full((1, 1, 4), -10.0)
        self.logits[:, :, 2] = 10.0

    def __call__(self, **kwargs):
        del kwargs
        return SimpleNamespace(logits=self.logits.clone(), past_key_values=None)


class _EmbeddingTokenizer(_SingleTokenTokenizer):
    pad_token_id = 0
    eos_token_id = 3

    def __call__(self, text: str, **kwargs):
        del kwargs
        return {
            "input_ids": torch.tensor([self.encode(text)], dtype=torch.long),
            "attention_mask": torch.ones(1, 1, dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        values = [int(token_id) for token_id in token_ids]
        if values == [2]:
            return "Final answer: 1"
        return " ".join(str(value) for value in values)


class _EmbeddingLogitModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(4, 3)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, **kwargs):
        inputs_embeds = kwargs["inputs_embeds"]
        logits = torch.full((inputs_embeds.shape[0], inputs_embeds.shape[1], 4), -10.0)
        logits[:, :, 2] = 10.0
        return SimpleNamespace(logits=logits)

    def generate(self, **kwargs):
        if "input_ids" in kwargs:
            return torch.cat(
                [kwargs["input_ids"], torch.tensor([[2]], dtype=torch.long)],
                dim=1,
            )
        return torch.tensor([[2]], dtype=torch.long)


def test_answer_metrics_can_score_final_answer_line_variants() -> None:
    model = _StaticLogitModel()
    prefix_state = {
        "outputs": SimpleNamespace(logits=model.logits.clone(), past_key_values=None),
        "attention_mask": torch.ones(1, 1, dtype=torch.long),
        "prefix_seq_len": 1,
    }

    metrics = compute_answer_metrics_from_prefix(
        model=model,
        tokenizer=_SingleTokenTokenizer(),
        prefix_state=prefix_state,
        answer_text="1",
        answer_variants=("Final answer: 1",),
    )

    assert metrics["answer_token_count"] == 1
    assert metrics["answer_nll"] < 1e-3


def test_embedding_answer_metrics_can_score_final_answer_line_variants() -> None:
    model = _EmbeddingLogitModel()
    prefix_embeds = torch.zeros(1, 1, 3)

    metrics = compute_answer_metrics_from_prefix_embeddings(
        model=model,
        tokenizer=_SingleTokenTokenizer(),
        prefix_embeds=prefix_embeds,
        answer_text="1",
        answer_variants=("Final answer: 1",),
    )

    assert metrics["answer_token_count"] == 1
    assert metrics["answer_nll"] < 1e-3


def test_embedding_first_token_metrics_report_rank_and_top1() -> None:
    model = _EmbeddingLogitModel()

    metrics = compute_first_token_metrics_from_prefix_embeddings(
        model=model,
        tokenizer=_SingleTokenTokenizer(),
        prefix_embeds=torch.zeros(1, 1, 3),
        answer_text="1",
        answer_variants=("Final answer: 1",),
    )

    assert metrics["first_token_rank"] == 1
    assert metrics["first_token_top1"] is True
    assert metrics["first_token_predicted_id"] == 2


def test_native_generate_helpers_decode_text_and_embedding_prefixes() -> None:
    model = _EmbeddingLogitModel()
    tokenizer = _EmbeddingTokenizer()

    text_metrics = generate_from_text_prefix(
        model=model,
        tokenizer=tokenizer,
        prefix_text="1",
        max_new_tokens=1,
    )
    embedding_metrics = generate_from_prefix_embeddings(
        model=model,
        tokenizer=tokenizer,
        prefix_embeds=torch.zeros(1, 1, 3),
        max_new_tokens=1,
    )

    assert text_metrics["decoded_text"] == "Final answer: 1"
    assert embedding_metrics["decoded_text"] == "Final answer: 1"
