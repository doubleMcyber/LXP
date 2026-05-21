from __future__ import annotations

import os
from pathlib import Path

from omegaconf import OmegaConf
import pytest
from transformers import AutoConfig

_CFG = OmegaConf.load(Path(__file__).resolve().parent.parent / "configs" / "main.yaml")


@pytest.mark.skipif(
    os.environ.get("LXP_RUN_REMOTE_MODEL_TESTS") != "1",
    reason="Remote model config checks are opt-in to keep unit tests offline.",
)
def test_actor_model_config_loads() -> None:
    config = AutoConfig.from_pretrained(_CFG.agent_b_model, trust_remote_code=True)
    text_config = getattr(config, "text_config", config)
    assert config is not None
    assert hasattr(text_config, "num_hidden_layers")
    assert hasattr(text_config, "num_attention_heads")
