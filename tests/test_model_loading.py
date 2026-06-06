from __future__ import annotations

from unittest.mock import patch

import pytest

from latent_pipeline import load_agent


def test_load_agent_fails_before_weight_load_when_mps_unavailable() -> None:
    with (
        patch("latent_pipeline.torch.backends.mps.is_available", return_value=False),
        patch("latent_pipeline.AutoModelForCausalLM.from_pretrained") as from_pretrained,
        pytest.raises(RuntimeError, match="MPS is not available"),
    ):
        load_agent("example/model", torch_dtype="float32", device_map="mps")

    from_pretrained.assert_not_called()
