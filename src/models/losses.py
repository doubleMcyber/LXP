from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _sample_steps(sequence: torch.Tensor, target_steps: int) -> torch.Tensor:
    if sequence.size(1) == target_steps:
        return sequence
    indices = torch.linspace(
        0,
        sequence.size(1) - 1,
        steps=target_steps,
        device=sequence.device,
    ).round().long()
    return sequence.index_select(dim=1, index=indices)


class LatentCompressorLoss(nn.Module):
    def __init__(
        self,
        lambda_task: float = 1.0,
        lambda_pref: float = 1.0,
        lambda_geom: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.lambda_task = lambda_task
        self.lambda_pref = lambda_pref
        self.lambda_geom = lambda_geom
        self.eps = eps

    def _compute_preference_terms(
        self,
        actor_logits_compressed: torch.Tensor,
        actor_logits_full: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        compressed_steps = actor_logits_compressed.size(1)
        teacher_logits = _sample_steps(actor_logits_full.detach(), compressed_steps)

        teacher_probs = F.softmax(teacher_logits, dim=-1)
        student_log_probs = F.log_softmax(actor_logits_compressed, dim=-1)
        kl_per_token = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)

        actor_entropy = -(teacher_probs * torch.log(teacher_probs + self.eps)).sum(dim=-1)
        token_weights = 1.0 / (actor_entropy + self.eps)
        weighted_kl = token_weights * kl_per_token
        l_pref = weighted_kl.sum() / token_weights.sum().clamp_min(self.eps)

        diagnostics = {
            "pref_avg_entropy": actor_entropy.mean(),
            "pref_avg_weight": token_weights.mean(),
            "pref_first_token_entropy": actor_entropy[:, 0].mean(),
            "pref_first_token_weight": token_weights[:, 0].mean(),
            "pref_first_token_kl": kl_per_token[:, 0].mean(),
        }
        return l_pref, diagnostics

    def forward(
        self,
        actor_logits_compressed: torch.Tensor,
        actor_logits_full: torch.Tensor,
        full_latents: torch.Tensor,
        compressed_latents: torch.Tensor,
        labels: torch.LongTensor,
    ) -> dict[str, torch.Tensor]:
        vocab_size = actor_logits_compressed.size(-1)
        compressed_steps = actor_logits_compressed.size(1)

        sampled_labels = _sample_steps(labels, compressed_steps)
        l_task = F.cross_entropy(
            actor_logits_compressed.reshape(-1, vocab_size),
            sampled_labels.reshape(-1),
            ignore_index=-100,
        )

        l_pref, pref_diagnostics = self._compute_preference_terms(
            actor_logits_compressed=actor_logits_compressed,
            actor_logits_full=actor_logits_full,
        )

        full_step_avg = full_latents.mean(dim=1)
        compressed_step_avg = compressed_latents.mean(dim=1)
        l_geom = 1.0 - F.cosine_similarity(compressed_step_avg, full_step_avg.detach(), dim=-1).mean()

        total_loss = (
            self.lambda_task * l_task
            + self.lambda_pref * l_pref
            + self.lambda_geom * l_geom
        )
        return {
            "loss": total_loss,
            "l_task": l_task,
            "l_pref": l_pref,
            "l_geom": l_geom,
            **pref_diagnostics,
        }
