from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.losses import LatentCompressorLoss


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
