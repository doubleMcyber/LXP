from __future__ import annotations

from omegaconf import OmegaConf
import torch

from latent_pipeline import (
    _build_integration_time_space,
    _compute_logits_entropy,
    _confidence_gate_settings,
    _scale_integration_points,
    estimate_problem_complexity,
)


def _make_cfg(math_factor: float = 1.25):
    return OmegaConf.create(
        {
            "dynamics": {
                "complexity_thresholds": {
                    "default": 1.0,
                    "short": 0.75,
                    "long": 1.1,
                    "reasoning": 1.15,
                    "math": math_factor,
                    "code": 1.3,
                    "short_prompt_words": 8,
                    "long_prompt_words": 24,
                }
            }
        }
    )


def test_estimate_problem_complexity_uses_math_threshold_override() -> None:
    prompt = "Solve 12 + 7 = 19."

    default_factor = estimate_problem_complexity(prompt, _make_cfg())
    overridden_factor = estimate_problem_complexity(prompt, _make_cfg(math_factor=1.5))

    assert default_factor == 1.25
    assert overridden_factor == 1.5


def test_estimate_problem_complexity_routes_entropy_prompt_through_math_threshold() -> None:
    prompt = "Explain the concept of entropy"

    default_factor = estimate_problem_complexity(prompt, _make_cfg())
    overridden_factor = estimate_problem_complexity(prompt, _make_cfg(math_factor=1.5))

    assert default_factor == 1.25
    assert overridden_factor == 1.5


def test_estimate_problem_complexity_clamps_to_supported_range() -> None:
    prompt = "Explain how to compare two algorithms."
    cfg = _make_cfg()
    cfg.dynamics.complexity_thresholds.reasoning = 3.0

    complexity_factor = estimate_problem_complexity(prompt, cfg)

    assert complexity_factor == 2.0


def test_scale_integration_points_tracks_complexity_factor() -> None:
    assert _scale_integration_points(10, 0.75) == 8
    assert _scale_integration_points(10, 1.0) == 10
    assert _scale_integration_points(10, 1.3) == 13


def test_build_integration_time_space_keeps_fixed_time_horizon() -> None:
    time_space = _build_integration_time_space(13, torch.device("cpu"), torch.float32)

    assert time_space.shape[0] == 13
    assert float(time_space[0].item()) == 0.0
    assert float(time_space[-1].item()) == 1.0


def test_compute_logits_entropy_is_higher_for_flatter_distribution() -> None:
    peaked_logits = torch.tensor([[8.0, -8.0]], dtype=torch.float32)
    flat_logits = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

    peaked_entropy = _compute_logits_entropy(peaked_logits)
    flat_entropy = _compute_logits_entropy(flat_logits)

    assert flat_entropy.item() > peaked_entropy.item()


def test_confidence_gate_settings_read_cfg_values() -> None:
    cfg = OmegaConf.create(
        {
            "dynamics": {
                "confidence_gate": {
                    "enabled": True,
                    "uncertainty_threshold": 5.5,
                    "extra_discrete_steps": 4,
                }
            }
        }
    )

    enabled, threshold, extra_steps = _confidence_gate_settings(cfg)

    assert enabled is True
    assert threshold == 5.5
    assert extra_steps == 4
