from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.hidden_state import AdaptiveProjection
from src.models.losses import AdaptiveLossBalancer, AdaptiveLossBalancerConfig, LatentCompressorLoss


def test_uncertainty_weighting_emphasizes_low_entropy_teacher_tokens() -> None:
    loss_fn = LatentCompressorLoss(eps=1e-8)
    teacher_logits = torch.tensor(
        [[[8.0, -8.0], [0.0, 0.0]]],
        dtype=torch.float32,
    )
    student_logits = torch.tensor(
        [[[0.0, 0.0], [0.0, 0.0]]],
        dtype=torch.float32,
    )

    l_pref, diagnostics = loss_fn._compute_preference_terms(
        actor_logits_compressed=student_logits,
        actor_logits_full=teacher_logits,
    )

    teacher_probs = F.softmax(teacher_logits, dim=-1)
    teacher_entropy = -(teacher_probs * torch.log(teacher_probs + loss_fn.eps)).sum(dim=-1)
    token_weights = 1.0 / (teacher_entropy + loss_fn.eps)
    kl_per_token = F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        teacher_probs,
        reduction="none",
    ).sum(dim=-1)
    expected_l_pref = (token_weights * kl_per_token).sum() / token_weights.sum()

    assert torch.allclose(l_pref, expected_l_pref, atol=1e-6)
    assert diagnostics["pref_first_token_entropy"] < diagnostics["pref_avg_entropy"]
    assert diagnostics["pref_first_token_weight"] > diagnostics["pref_avg_weight"]
    assert diagnostics["pref_first_token_top1_probability"] > diagnostics["pref_avg_top1_probability"]
    assert diagnostics["pref_first_token_logit_margin"] > diagnostics["pref_avg_logit_margin"]
    assert diagnostics["pref_first_token_weight_ratio"] > 1.0


def test_adaptive_loss_balancer_reweights_large_losses_downward() -> None:
    balancer = AdaptiveLossBalancer(
        {
            "l_task": 1.0,
            "l_pref": 1.0,
        },
        config=AdaptiveLossBalancerConfig(
            enabled=True,
            ema_beta=0.0,
            min_weight=0.25,
            max_weight=4.0,
        ),
    )
    total, weights = balancer.combine(
        {
            "l_task": torch.tensor(10.0),
            "l_pref": torch.tensor(1.0),
        }
    )

    assert float(total.item()) > 0.0
    assert weights["l_task"] < weights["l_pref"]


def test_adaptive_projection_preserves_shape_and_stays_near_identity() -> None:
    projection = AdaptiveProjection(strength=0.15, clip_std_multiplier=4.0)
    hidden_states = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    reference_states = torch.tensor([[[1.1, 2.1], [2.9, 3.9]]])

    projected, diagnostics = projection(hidden_states, reference_states)

    assert projected.shape == hidden_states.shape
    assert diagnostics["projection_scale_mean"] > 0.0
    assert torch.mean(torch.abs(projected - hidden_states)) < 0.5
