from __future__ import annotations

from transformers import AutoConfig

from latent_pipeline import AGENT_B_MODEL_NAME


def test_actor_model_config_loads() -> None:
    config = AutoConfig.from_pretrained(AGENT_B_MODEL_NAME, trust_remote_code=True)
    assert config is not None
    assert hasattr(config, "num_hidden_layers")
    assert hasattr(config, "num_attention_heads")
