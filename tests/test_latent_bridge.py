from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from train_latent_bridge import LatentBridge, chat_prefix_ids  # noqa: E402


def test_latent_bridge_shapes_and_identity_projection() -> None:
    bridge = LatentBridge(8, 8, output_steps=4, num_heads=2)
    latents = torch.randn(1, 11, 8)

    out = bridge(latents)
    assert tuple(out.shape) == (1, 4, 8)

    # same-dim init is the identity map, so the projection is lossless pre-training
    assert torch.allclose(bridge.input_proj(latents.float()), latents.float(), atol=1e-6)


def test_latent_bridge_warm_start_matches_pipeline_convention() -> None:
    # pipeline applies hidden @ M; the projection must reproduce that exactly
    bridge = LatentBridge(4, 3, output_steps=2, num_heads=1)
    mapping = torch.randn(4, 3)
    bias = torch.randn(1, 3)
    bridge.warm_start_projection(mapping, bias)

    hidden = torch.randn(5, 4)
    expected = hidden @ mapping + bias
    assert torch.allclose(bridge.input_proj(hidden), expected, atol=1e-5)


def test_latent_bridge_gradients_flow_only_through_bridge() -> None:
    bridge = LatentBridge(6, 6, output_steps=3, num_heads=2)
    latents = torch.randn(1, 9, 6)
    out = bridge(latents)
    out.float().pow(2).mean().backward()
    grads = [p.grad for p in bridge.parameters() if p.grad is not None]
    assert grads, "bridge parameters must receive gradients"
    assert latents.grad is None


class _ChatlessTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [len(word) for word in text.split()]


def test_chat_prefix_ids_falls_back_without_chat_template() -> None:
    ids = chat_prefix_ids(_ChatlessTokenizer(), "What is 2+2?", "Continue.")
    assert ids and all(isinstance(i, int) for i in ids)
    with_body = chat_prefix_ids(_ChatlessTokenizer(), "Q?", "Continue.", body="partial work")
    assert len(with_body) > len(ids) - 2
