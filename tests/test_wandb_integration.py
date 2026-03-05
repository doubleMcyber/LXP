from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from train_compressor import CompressionTrainConfig, train_reasoner_stage2

_CFG = OmegaConf.load(Path(__file__).resolve().parent.parent / "configs" / "main.yaml")


def test_wandb_config_fields_parsed_from_yaml() -> None:
    config = CompressionTrainConfig.from_cfg(_CFG)
    assert config.wandb_enabled is True
    assert config.wandb_project == "lxp-stage2"
    assert config.wandb_entity is None


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
    # Without wandb section, from_cfg falls back to False
    assert config.wandb_enabled is False
    assert config.wandb_project == "lxp-stage2"
    assert config.wandb_entity is None


def _make_tiny_models(hidden_dim: int = 32, vocab_size: int = 64):
    """Build minimal nn.Module stand-ins for reasoner and actor."""
    from torch import nn

    class TinyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, use_cache=False, return_dict=True):
            if inputs_embeds is not None:
                h = self.linear(inputs_embeds)
            else:
                h = self.linear(torch.randn(input_ids.shape[0], input_ids.shape[1], hidden_dim, device=input_ids.device))
            return MagicMock(last_hidden_state=h)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = TinyBackbone()
            self.lm_head = nn.Linear(hidden_dim, vocab_size)
            self._embed = nn.Embedding(vocab_size, hidden_dim)
            self.dtype = torch.float32

        def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, use_cache=False, return_dict=True):
            if inputs_embeds is not None:
                h = self.model.linear(inputs_embeds)
            else:
                h = self.model(input_ids=input_ids).last_hidden_state
            logits = self.lm_head(h)
            return MagicMock(logits=logits)

        def get_input_embeddings(self):
            return self._embed

        def parameters(self, recurse=True):
            return super().parameters(recurse=recurse)

        def eval(self):
            return super().eval()

    return TinyModel(), TinyModel()


def _make_dataloader(batch_size: int = 2, seq_len: int = 8, vocab_size: int = 64, num_batches: int = 3):
    input_ids = torch.randint(0, vocab_size, (batch_size * num_batches, seq_len))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    ds = TensorDataset(input_ids, attention_mask, labels)
    dl = DataLoader(ds, batch_size=batch_size)

    def collate_iter():
        for batch in dl:
            yield {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}

    return collate_iter()


@patch("train_compressor.wandb")
def test_wandb_init_log_finish_called(mock_wandb) -> None:
    reasoner, actor = _make_tiny_models()
    config = CompressionTrainConfig(
        compressed_steps=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        num_epochs=1,
        wandb_enabled=True,
        wandb_project="test-proj",
        wandb_entity="test-team",
    )

    dl = _make_dataloader()
    history = train_reasoner_stage2(reasoner, actor, dl, config)

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
    )

    dl = _make_dataloader(num_batches=4)
    history = train_reasoner_stage2(reasoner, actor, dl, config)

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
    )

    dl = _make_dataloader()
    history = train_reasoner_stage2(reasoner, actor, dl, config)

    for i, call_args in enumerate(mock_wandb.log.call_args_list):
        logged = call_args[0][0]
        assert logged["loss"] == history[i]["loss"]
        assert logged["l_task"] == history[i]["l_task"]
        assert logged["l_pref"] == history[i]["l_pref"]
        assert logged["l_geom"] == history[i]["l_geom"]


@patch("train_compressor.wandb")
def test_no_wandb_calls_when_disabled(mock_wandb) -> None:
    reasoner, actor = _make_tiny_models()
    config = CompressionTrainConfig(
        compressed_steps=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        num_epochs=1,
        wandb_enabled=False,
    )

    dl = _make_dataloader()
    history = train_reasoner_stage2(reasoner, actor, dl, config)

    mock_wandb.init.assert_not_called()
    mock_wandb.log.assert_not_called()
    mock_wandb.finish.assert_not_called()
    assert len(history) > 0


# --- Checkpoint tests ---


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
        )
        dl = _make_dataloader()
        train_reasoner_stage2(reasoner, actor, dl, config)

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
        )
        dl = _make_dataloader(num_batches=5)
        train_reasoner_stage2(reasoner, actor, dl, config)

        # Steps 0, 2 should be saved (global_step 0, 2 match % 2 == 0)
        assert (Path(tmp) / "step_0.pt").exists()
        assert (Path(tmp) / "step_2.pt").exists()
        # Epoch checkpoint always saved
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
        )
        dl = _make_dataloader()
        history = train_reasoner_stage2(reasoner, actor, dl, config)

        assert not ckpt_dir.exists()
        assert len(history) > 0
