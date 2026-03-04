from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM


@dataclass
class CompressionTrainConfig:
    compressed_steps: int = 16
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_epochs: int = 1
    lambda_task: float = 1.0
    lambda_pref: float = 1.0
    lambda_geom: float = 1.0
    eps: float = 1e-8


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


def compress_latent_trajectory(full_latents: torch.Tensor, compressed_steps: int) -> torch.Tensor:
    # Differentiable temporal compression: [B, T, D] -> [B, K, D]
    pooled = F.adaptive_avg_pool1d(full_latents.transpose(1, 2), compressed_steps)
    return pooled.transpose(1, 2)


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

        teacher_logits = _sample_steps(actor_logits_full.detach(), compressed_steps)
        p_teacher = F.softmax(teacher_logits, dim=-1)
        p_student = F.softmax(actor_logits_compressed, dim=-1)

        # Required numerical safety clamp for KL log probabilities.
        log_p_student = torch.log(p_student + self.eps)
        kl_per_token = F.kl_div(log_p_student, p_teacher, reduction="none").sum(dim=-1)

        entropy_teacher = -(p_teacher * torch.log(p_teacher + self.eps)).sum(dim=-1)
        token_weights = 1.0 / (entropy_teacher + self.eps)
        l_pref = (token_weights * kl_per_token).sum() / token_weights.sum().clamp_min(self.eps)

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
        }


def _model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def freeze_actor(actor_model: AutoModelForCausalLM) -> None:
    actor_model.eval()
    for parameter in actor_model.parameters():
        parameter.requires_grad = False


def train_reasoner_stage2(
    reasoner_model: AutoModelForCausalLM,
    actor_model: AutoModelForCausalLM,
    train_dataloader: Iterable[dict[str, torch.Tensor]],
    config: CompressionTrainConfig,
) -> list[dict[str, float]]:
    """
    Stage II Interlat-style training loop:
    - Freeze Actor (Agent B) completely.
    - Update only Reasoner (Agent A).
    """
    freeze_actor(actor_model)
    reasoner_model.train()

    loss_fn = LatentCompressorLoss(
        lambda_task=config.lambda_task,
        lambda_pref=config.lambda_pref,
        lambda_geom=config.lambda_geom,
        eps=config.eps,
    )
    optimizer = torch.optim.AdamW(
        (p for p in reasoner_model.parameters() if p.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    reasoner_device = _model_device(reasoner_model)
    actor_device = _model_device(actor_model)
    reasoner_backbone = reasoner_model.get_base_model()
    history: list[dict[str, float]] = []

    for epoch in range(config.num_epochs):
        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(reasoner_device)
            attention_mask = batch["attention_mask"].to(reasoner_device)
            labels = batch.get("labels", input_ids).to(reasoner_device)

            reasoner_hidden_outputs = reasoner_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            full_latents = reasoner_hidden_outputs.last_hidden_state

            compressed_latents = compress_latent_trajectory(
                full_latents,
                compressed_steps=config.compressed_steps,
            )

            full_attention = torch.ones(
                (full_latents.size(0), full_latents.size(1)),
                dtype=torch.long,
                device=actor_device,
            )
            compressed_attention = torch.ones(
                (compressed_latents.size(0), compressed_latents.size(1)),
                dtype=torch.long,
                device=actor_device,
            )

            with torch.no_grad():
                actor_logits_full = actor_model(
                    inputs_embeds=full_latents.detach().to(
                        device=actor_device,
                        dtype=actor_model.get_input_embeddings().weight.dtype,
                    ),
                    attention_mask=full_attention,
                    use_cache=False,
                    return_dict=True,
                ).logits

            actor_logits_compressed = actor_model(
                inputs_embeds=compressed_latents.to(
                    device=actor_device,
                    dtype=actor_model.get_input_embeddings().weight.dtype,
                ),
                attention_mask=compressed_attention,
                use_cache=False,
                return_dict=True,
            ).logits

            loss_outputs = loss_fn(
                actor_logits_compressed=actor_logits_compressed.to(reasoner_device),
                actor_logits_full=actor_logits_full.to(reasoner_device),
                full_latents=full_latents,
                compressed_latents=compressed_latents,
                labels=labels,
            )

            optimizer.zero_grad(set_to_none=True)
            loss_outputs["loss"].backward()
            nn.utils.clip_grad_norm_(reasoner_model.parameters(), config.max_grad_norm)
            optimizer.step()

            history.append(
                {
                    "epoch": float(epoch),
                    "step": float(step),
                    "loss": float(loss_outputs["loss"].detach().cpu().item()),
                    "l_task": float(loss_outputs["l_task"].detach().cpu().item()),
                    "l_pref": float(loss_outputs["l_pref"].detach().cpu().item()),
                    "l_geom": float(loss_outputs["l_geom"].detach().cpu().item()),
                }
            )

    return history


if __name__ == "__main__":
    raise SystemExit(
        "This module provides Stage II training utilities. Import and call "
        "`train_reasoner_stage2(...)` with initialized reasoner/actor models and a dataloader."
    )
