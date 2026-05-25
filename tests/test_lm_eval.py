from __future__ import annotations

from types import SimpleNamespace

import torch

from src.utils.lm_eval import compute_answer_metrics_from_prefix


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
