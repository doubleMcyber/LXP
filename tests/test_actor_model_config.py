from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf
from transformers import AutoConfig

_CFG = OmegaConf.load(Path(__file__).resolve().parent.parent / "configs" / "main.yaml")


def test_actor_model_config_loads() -> None:
    config = AutoConfig.from_pretrained(_CFG.agent_b_model, trust_remote_code=True)
    assert config is not None
    assert hasattr(config, "num_hidden_layers")
    assert hasattr(config, "num_attention_heads")
