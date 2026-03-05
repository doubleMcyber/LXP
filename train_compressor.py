from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import hydra
import torch
import wandb
import torch.nn.functional as F
from omegaconf import DictConfig
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
    wandb_enabled: bool = True
    wandb_project: str = "lxp-stage2"
    wandb_entity: Optional[str] = None
    checkpoint_enabled: bool = True
    checkpoint_dir: str = "checkpoints"
    checkpoint_every_n_steps: int = 0

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "CompressionTrainConfig":
        t = cfg.training
        wandb_cfg = getattr(t, "wandb", None)
        return cls(
            compressed_steps=t.compressed_steps,
            learning_rate=t.learning_rate,
            weight_decay=t.weight_decay,
            max_grad_norm=t.max_grad_norm,
            num_epochs=t.num_epochs,
            lambda_task=t.lambda_task,
            lambda_pref=t.lambda_pref,
            lambda_geom=t.lambda_geom,
            eps=t.eps,
            wandb_enabled=bool(wandb_cfg.enabled) if wandb_cfg else False,
            wandb_project=str(wandb_cfg.project) if wandb_cfg else "lxp-stage2",
            wandb_entity=str(wandb_cfg.entity) if wandb_cfg and wandb_cfg.entity else None,
            checkpoint_enabled=bool(ckpt_cfg.enabled) if (ckpt_cfg := getattr(t, "checkpointing", None)) else True,
            checkpoint_dir=str(ckpt_cfg.dir) if ckpt_cfg else "checkpoints",
            checkpoint_every_n_steps=int(ckpt_cfg.save_every_n_steps) if ckpt_cfg else 0,
        )


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
    device = full_latents.device
    if device.type == "mps":
        # MPS adaptive_avg_pool1d has limitations on non-divisible sizes
        full_latents_cpu = full_latents.cpu()
        pooled = F.adaptive_avg_pool1d(full_latents_cpu.transpose(1, 2), compressed_steps)
        return pooled.transpose(1, 2).to(device)
    
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
    from src.utils.alignment import compute_orthogonal_mapping, apply_orthogonal_mapping

    if config.wandb_enabled:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            config=dataclasses.asdict(config),
        )

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
    
    if hasattr(reasoner_model, "model"):
        reasoner_backbone = reasoner_model.model
    elif hasattr(reasoner_model, "get_base_model"):
        reasoner_backbone = reasoner_model.get_base_model()
    else:
        reasoner_backbone = reasoner_model
        
    history: list[dict[str, float]] = []
    global_step = 0

    if config.checkpoint_enabled:
        ckpt_dir = Path(config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute alignment mapping using the first batch as anchors
    print("Computing initial Procrustes alignment...")
    first_batch = next(iter(train_dataloader))
    with torch.no_grad():
        fb_input_ids = first_batch["input_ids"].to(reasoner_device)
        fb_reasoner_out = reasoner_backbone(input_ids=fb_input_ids, use_cache=False, return_dict=True)
        fb_reasoner_hidden = fb_reasoner_out.last_hidden_state
        
        fb_actor_out = actor_model.model(input_ids=fb_input_ids.to(actor_device), use_cache=False, return_dict=True)
        fb_actor_hidden = fb_actor_out.last_hidden_state
        
        procrustes_q = compute_orthogonal_mapping(fb_reasoner_hidden, fb_actor_hidden)
        procrustes_q = procrustes_q.to(device=reasoner_device, dtype=reasoner_model.dtype)

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

            # Apply alignment mapping to transfer to Actor's space
            full_latents_aligned = apply_orthogonal_mapping(full_latents, procrustes_q)
            compressed_latents_aligned = apply_orthogonal_mapping(compressed_latents, procrustes_q)

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
                    inputs_embeds=full_latents_aligned.detach().to(
                        device=actor_device,
                        dtype=actor_model.get_input_embeddings().weight.dtype,
                    ),
                    attention_mask=full_attention,
                    use_cache=False,
                    return_dict=True,
                ).logits

            actor_logits_compressed = actor_model(
                inputs_embeds=compressed_latents_aligned.to(
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

            metrics = {
                "epoch": float(epoch),
                "step": float(step),
                "loss": float(loss_outputs["loss"].detach().cpu().item()),
                "l_task": float(loss_outputs["l_task"].detach().cpu().item()),
                "l_pref": float(loss_outputs["l_pref"].detach().cpu().item()),
                "l_geom": float(loss_outputs["l_geom"].detach().cpu().item()),
            }
            history.append(metrics)

            if config.wandb_enabled:
                wandb.log(
                    {
                        "loss": metrics["loss"],
                        "l_task": metrics["l_task"],
                        "l_pref": metrics["l_pref"],
                        "l_geom": metrics["l_geom"],
                    },
                    step=global_step,
                )
            if config.checkpoint_enabled and config.checkpoint_every_n_steps > 0:
                if global_step % config.checkpoint_every_n_steps == 0:
                    step_path = ckpt_dir / f"step_{global_step}.pt"
                    torch.save(reasoner_model.state_dict(), step_path)
                    print(f"Saved checkpoint: {step_path}")

            global_step += 1

        if config.checkpoint_enabled:
            epoch_path = ckpt_dir / f"epoch_{epoch}.pt"
            torch.save(reasoner_model.state_dict(), epoch_path)
            print(f"Saved checkpoint: {epoch_path}")

    if config.wandb_enabled:
        wandb.finish()

    return history


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig) -> None:
    config = CompressionTrainConfig.from_cfg(cfg)
    print("Loaded CompressionTrainConfig from Hydra:")
    print(f"  compressed_steps = {config.compressed_steps}")
    print(f"  learning_rate    = {config.learning_rate}")
    print(f"  weight_decay     = {config.weight_decay}")
    print(f"  max_grad_norm    = {config.max_grad_norm}")
    print(f"  num_epochs       = {config.num_epochs}")
    print(f"  lambda_task      = {config.lambda_task}")
    print(f"  lambda_pref      = {config.lambda_pref}")
    print(f"  lambda_geom      = {config.lambda_geom}")
    print(f"  eps              = {config.eps}")
    print(f"  wandb_enabled    = {config.wandb_enabled}")
    print(f"  wandb_project    = {config.wandb_project}")
    print(f"  wandb_entity     = {config.wandb_entity}")
    print(f"  checkpoint_enabled = {config.checkpoint_enabled}")
    print(f"  checkpoint_dir     = {config.checkpoint_dir}")
    print(f"  checkpoint_every_n_steps = {config.checkpoint_every_n_steps}")
    print(
        "\nTo run training, call train_reasoner_stage2(...) with "
        "initialized reasoner/actor models and a dataloader."
    )


if __name__ == "__main__":
    main()
