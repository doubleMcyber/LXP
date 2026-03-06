from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from train_compressor import CompressionTrainConfig, train_reasoner_stage2

_CFG = OmegaConf.load(Path(__file__).resolve().parent.parent / "configs" / "main.yaml")


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
        pad_to = max(len(tokens) for tokens in encoded) if padding else None
        if pad_to is None:
            input_ids = encoded
            attention_mask = [[1] * len(tokens) for tokens in encoded]
        else:
            input_ids = []
            attention_mask = []
            for tokens in encoded:
                pad_width = pad_to - len(tokens)
                input_ids.append(tokens + [self.pad_token_id] * pad_width)
                attention_mask.append([1] * len(tokens) + [0] * pad_width)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def test_wandb_config_fields_parsed_from_yaml() -> None:
    config = CompressionTrainConfig.from_cfg(_CFG)
    assert config.wandb_enabled is True
    assert config.wandb_project == "lxp-stage2"
    assert config.wandb_entity is None
    assert config.reasoner_max_length == 96
    assert config.actor_max_length == 96


def test_wandb_config_custom_values() -> None:
    cfg = OmegaConf.create(
        {
            "training": {
                "compressed_steps": 4,
                "learning_rate": 1e-4,
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
                "num_epochs": 1,
                "lambda_task": 1.0,
                "lambda_pref": 1.0,
                "lambda_geom": 1.0,
                "eps": 1e-8,
                "reasoner_max_length": 48,
                "actor_max_length": 24,
                "wandb": {
                    "enabled": True,
                    "project": "my-project",
                    "entity": "my-team",
                },
            }
        }
    )
    config = CompressionTrainConfig.from_cfg(cfg)
    assert config.wandb_enabled is True
    assert config.wandb_project == "my-project"
    assert config.wandb_entity == "my-team"
    assert config.reasoner_max_length == 48
    assert config.actor_max_length == 24


def test_wandb_config_missing_section_uses_defaults() -> None:
    cfg = OmegaConf.create(
        {
            "training": {
                "compressed_steps": 4,
                "learning_rate": 1e-4,
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
                "num_epochs": 1,
                "lambda_task": 1.0,
                "lambda_pref": 1.0,
                "lambda_geom": 1.0,
                "eps": 1e-8,
            }
        }
    )
    config = CompressionTrainConfig.from_cfg(cfg)
    assert config.wandb_enabled is False
    assert config.wandb_project == "lxp-stage2"
    assert config.wandb_entity is None
    assert config.reasoner_max_length == 128
    assert config.actor_max_length == 128


def _make_tiny_models(hidden_dim: int = 32, vocab_size: int = 128):
    from torch import nn

    class TinyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(hidden_dim, hidden_dim)
            self.embedding = nn.Embedding(vocab_size, hidden_dim)

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

        def parameters(self, recurse: bool = True):
            return super().parameters(recurse=recurse)

        def eval(self):
            return super().eval()

    return TinyModel(), TinyModel()


def _make_text_dataloader(batch_size: int = 2, num_batches: int = 3):
    texts = [f"sample question {index}" for index in range(batch_size * num_batches)]
    return DataLoader(
        texts,
        batch_size=batch_size,
        collate_fn=lambda batch: {"texts": list(batch)},
    )


@patch("train_compressor.wandb")
def test_wandb_init_log_finish_called(mock_wandb) -> None:
    reasoner, actor = _make_tiny_models()
    reasoner_tokenizer = _TinyTokenizer(offset=0)
    actor_tokenizer = _TinyTokenizer(offset=17)
    config = CompressionTrainConfig(
        compressed_steps=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        num_epochs=1,
        wandb_enabled=True,
        wandb_project="test-proj",
        wandb_entity="test-team",
        reasoner_max_length=12,
        actor_max_length=12,
    )

    history = train_reasoner_stage2(
        reasoner,
        actor,
        _make_text_dataloader(),
        config,
        reasoner_tokenizer=reasoner_tokenizer,
        actor_tokenizer=actor_tokenizer,
    )

    mock_wandb.init.assert_called_once_with(
        project="test-proj",
        entity="test-team",
        config=dataclasses.asdict(config),
    )
    assert mock_wandb.log.call_count == len(history)
    for call_args in mock_wandb.log.call_args_list:
        logged = call_args[0][0]
        assert "loss" in logged
        assert "l_task" in logged
        assert "l_pref" in logged
        assert "l_geom" in logged
        assert "pref_avg_entropy" in logged
        assert "pref_avg_weight" in logged
        assert "pref_first_token_entropy" in logged
        assert "pref_first_token_weight" in logged
        assert "pref_first_token_kl" in logged
        assert "pref_avg_top1_probability" in logged
        assert "pref_first_token_top1_probability" in logged
        assert "pref_avg_logit_margin" in logged
        assert "pref_first_token_logit_margin" in logged
        assert "pref_first_token_weight_ratio" in logged
        assert "step" in call_args[1]
    mock_wandb.finish.assert_called_once()


@patch("train_compressor.wandb")
def test_wandb_step_values_are_sequential(mock_wandb) -> None:
    reasoner, actor = _make_tiny_models()
    config = CompressionTrainConfig(
        compressed_steps=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        num_epochs=1,
        wandb_enabled=True,
        wandb_project="test-proj",
        reasoner_max_length=12,
        actor_max_length=12,
    )

    history = train_reasoner_stage2(
        reasoner,
        actor,
        _make_text_dataloader(num_batches=4),
        config,
        reasoner_tokenizer=_TinyTokenizer(offset=0),
        actor_tokenizer=_TinyTokenizer(offset=5),
    )

    steps = [call.kwargs["step"] for call in mock_wandb.log.call_args_list]
    assert steps == list(range(len(history)))


@patch("train_compressor.wandb")
def test_wandb_logs_match_history(mock_wandb) -> None:
    reasoner, actor = _make_tiny_models()
    config = CompressionTrainConfig(
        compressed_steps=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        num_epochs=1,
        wandb_enabled=True,
        wandb_project="test-proj",
        reasoner_max_length=12,
        actor_max_length=12,
    )

    history = train_reasoner_stage2(
        reasoner,
        actor,
        _make_text_dataloader(),
        config,
        reasoner_tokenizer=_TinyTokenizer(offset=0),
        actor_tokenizer=_TinyTokenizer(offset=9),
    )

    for index, call_args in enumerate(mock_wandb.log.call_args_list):
        logged = call_args[0][0]
        for key, value in logged.items():
            assert logged[key] == history[index][key]


@patch("train_compressor.wandb")
def test_no_wandb_calls_when_disabled(mock_wandb) -> None:
    reasoner, actor = _make_tiny_models()
    config = CompressionTrainConfig(
        compressed_steps=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        num_epochs=1,
        wandb_enabled=False,
        reasoner_max_length=12,
        actor_max_length=12,
    )

    history = train_reasoner_stage2(
        reasoner,
        actor,
        _make_text_dataloader(),
        config,
        reasoner_tokenizer=_TinyTokenizer(offset=0),
        actor_tokenizer=_TinyTokenizer(offset=11),
    )

    mock_wandb.init.assert_not_called()
    mock_wandb.log.assert_not_called()
    mock_wandb.finish.assert_not_called()
    assert len(history) > 0


def test_checkpoint_config_parsed_from_yaml() -> None:
    config = CompressionTrainConfig.from_cfg(_CFG)
    assert config.checkpoint_enabled is True
    assert config.checkpoint_dir == "checkpoints"
    assert config.checkpoint_every_n_steps == 0


def test_checkpoint_config_missing_section_uses_defaults() -> None:
    cfg = OmegaConf.create(
        {
            "training": {
                "compressed_steps": 4,
                "learning_rate": 1e-4,
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
                "num_epochs": 1,
                "lambda_task": 1.0,
                "lambda_pref": 1.0,
                "lambda_geom": 1.0,
                "eps": 1e-8,
            }
        }
    )
    config = CompressionTrainConfig.from_cfg(cfg)
    assert config.checkpoint_enabled is True
    assert config.checkpoint_dir == "checkpoints"
    assert config.checkpoint_every_n_steps == 0


@patch("train_compressor.wandb")
def test_epoch_checkpoint_saved(mock_wandb) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        reasoner, actor = _make_tiny_models()
        config = CompressionTrainConfig(
            compressed_steps=4,
            learning_rate=1e-3,
            weight_decay=0.0,
            num_epochs=2,
            wandb_enabled=False,
            checkpoint_enabled=True,
            checkpoint_dir=tmp,
            checkpoint_every_n_steps=0,
            reasoner_max_length=12,
            actor_max_length=12,
        )
        train_reasoner_stage2(
            reasoner,
            actor,
            _make_text_dataloader(),
            config,
            reasoner_tokenizer=_TinyTokenizer(offset=0),
            actor_tokenizer=_TinyTokenizer(offset=13),
        )

        assert (Path(tmp) / "epoch_0.pt").exists()
        assert (Path(tmp) / "epoch_1.pt").exists()
        state = torch.load(Path(tmp) / "epoch_1.pt", weights_only=True)
        assert isinstance(state, dict)
        assert len(state) > 0


@patch("train_compressor.wandb")
def test_step_checkpoint_saved(mock_wandb) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        reasoner, actor = _make_tiny_models()
        config = CompressionTrainConfig(
            compressed_steps=4,
            learning_rate=1e-3,
            weight_decay=0.0,
            num_epochs=1,
            wandb_enabled=False,
            checkpoint_enabled=True,
            checkpoint_dir=tmp,
            checkpoint_every_n_steps=2,
            reasoner_max_length=12,
            actor_max_length=12,
        )
        train_reasoner_stage2(
            reasoner,
            actor,
            _make_text_dataloader(num_batches=5),
            config,
            reasoner_tokenizer=_TinyTokenizer(offset=0),
            actor_tokenizer=_TinyTokenizer(offset=15),
        )

        assert (Path(tmp) / "step_0.pt").exists()
        assert (Path(tmp) / "step_2.pt").exists()
        assert (Path(tmp) / "epoch_0.pt").exists()


@patch("train_compressor.wandb")
def test_no_checkpoints_when_disabled(mock_wandb) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = Path(tmp) / "ckpts"
        reasoner, actor = _make_tiny_models()
        config = CompressionTrainConfig(
            compressed_steps=4,
            learning_rate=1e-3,
            weight_decay=0.0,
            num_epochs=1,
            wandb_enabled=False,
            checkpoint_enabled=False,
            checkpoint_dir=str(ckpt_dir),
            reasoner_max_length=12,
            actor_max_length=12,
        )
        history = train_reasoner_stage2(
            reasoner,
            actor,
            _make_text_dataloader(),
            config,
            reasoner_tokenizer=_TinyTokenizer(offset=0),
            actor_tokenizer=_TinyTokenizer(offset=21),
        )

        assert not ckpt_dir.exists()
        assert len(history) > 0
