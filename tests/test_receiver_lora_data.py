from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from latent_pipeline import _format_receiver_context_prompt  # noqa: E402
from train_receiver_lora import (  # noqa: E402
    build_context_ids,
    build_parser,
    load_bridge_sample_rows,
    load_frozen_adapter,
    rollout_cache_path,
    rollout_config_digest,
)


class _TinyChatTokenizer:
    chat_template = "tiny"

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ):
        assert tokenize is False
        assert add_generation_prompt is True
        content = messages[0]["content"]
        return f"<user>{content}</user><assistant>"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) % 251 for character in text]


def test_receiver_lora_layout_parity() -> None:
    tokenizer = _TinyChatTokenizer()
    question = (
        "Natalia sold clips to 48 of her friends in April, and then she sold "
        "half as many clips in May. How many clips did Natalia sell altogether?"
    )
    cfg = SimpleNamespace(
        benchmark=SimpleNamespace(sender_reasoning_truncation_fraction=0.5)
    )

    production_text = _format_receiver_context_prompt(question, tokenizer, cfg)
    production_ids = tokenizer.encode(production_text, add_special_tokens=False)

    assert build_context_ids(tokenizer, question) == production_ids


def test_rollout_cache_key_stability(tmp_path) -> None:
    digest = rollout_config_digest(
        receiver_model="Qwen/Qwen3.5-2B",
        adapter_digest="abc123",
        truncation_fraction=0.5,
        max_new_tokens=256,
        temperature=0.7,
        seed=7,
    )
    expected_payload = {
        "receiver_model": "Qwen/Qwen3.5-2B",
        "adapter_digest": "abc123",
        "truncation_fraction": 0.5,
        "max_new_tokens": 256,
        "temperature": 0.7,
        "seed": 7,
    }
    expected_digest = hashlib.sha256(
        json.dumps(
            expected_payload,
            sort_keys=True,
            default=list,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert digest == expected_digest
    assert (
        rollout_cache_path("What is 2+2?", cache_dir=tmp_path).name
        == f"rollout_{hashlib.sha256(b'What is 2+2?').hexdigest()}.pt"
    )
    assert digest == rollout_config_digest(
        receiver_model="Qwen/Qwen3.5-2B",
        adapter_digest="abc123",
        truncation_fraction=0.5,
        max_new_tokens=256,
        temperature=0.7,
        seed=7,
    )


def test_row_range_slicing_uses_train_split_and_sample_index(monkeypatch) -> None:
    calls = []

    def fake_load_bridge_samples(**kwargs):
        calls.append(kwargs)
        return [
            {
                "sample_index": index,
                "question": f"question {index}",
                "answer": str(index),
                "continuation_ids": [index],
                "latents": torch.ones(1, 2),
            }
            for index in range(10)
        ]

    import train_receiver_lora

    monkeypatch.setattr(
        train_receiver_lora,
        "load_bridge_samples",
        fake_load_bridge_samples,
    )
    args = build_parser().parse_args(["--train-rows", "2:5", "--dev-rows", "6:8"])
    tokenizer = object()

    rows = load_bridge_sample_rows(args, tokenizer=tokenizer, include_dev=True)

    assert [sample["sample_index"] for sample in rows["train"]] == [2, 3, 4]
    assert [sample["sample_index"] for sample in rows["dev"]] == [6, 7]
    assert calls == [
        {
            "dataset": "gsm8k",
            "split": "train",
            "limit": 8,
            "model_id": "Qwen/Qwen3.5-2B",
            "torch_dtype": "bfloat16",
            "truncation_fraction": 0.5,
            "tokenizer": tokenizer,
            "max_continuation_tokens": 256,
            "validation_size": 256,
        }
    ]


def test_adapter_digest_mismatch_raises(tmp_path) -> None:
    path = tmp_path / "adapter.pt"
    torch.save(
        {
            "adapter_cache_key_digest": "actual",
            "mapping_matrix": torch.eye(2),
        },
        path,
    )

    with pytest.raises(AssertionError, match="adapter digest mismatch"):
        load_frozen_adapter(path, adapter_digest="expected")
