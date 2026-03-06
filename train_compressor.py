from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import hydra
import torch
import torch.nn.functional as F
import wandb
from latent_pipeline import load_or_compute_global_alignment_state
from omegaconf import DictConfig
from torch import nn
from transformers import AutoModelForCausalLM

from src.models.losses import LatentCompressorLoss
from src.utils.alignment import apply_orthogonal_mapping

EvaluationFn = Callable[[AutoModelForCausalLM, AutoModelForCausalLM, dict[str, Any]], dict[str, float]]


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
    reasoner_max_length: int = 128
    actor_max_length: int = 128

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
            reasoner_max_length=int(getattr(t, "reasoner_max_length", 128)),
            actor_max_length=int(getattr(t, "actor_max_length", 128)),
        )

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


def _model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def _model_backbone(model: AutoModelForCausalLM) -> nn.Module:
    if hasattr(model, "model"):
        return model.model
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def _extract_text_batch(batch: dict[str, Any]) -> list[str]:
    texts = batch.get("texts")
    if texts is None:
        raise ValueError("train_reasoner_stage2 expects batches with a 'texts' field")
    if isinstance(texts, str):
        return [texts]
    return [str(text) for text in texts]


def _ensure_padding_token(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token


def _tokenize_text_batch(
    tokenizer: Any,
    texts: list[str],
    *,
    device: torch.device,
    max_length: int,
) -> dict[str, torch.Tensor]:
    _ensure_padding_token(tokenizer)
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _identity_alignment_mapping(
    reasoner_model: AutoModelForCausalLM,
    actor_model: AutoModelForCausalLM,
) -> torch.Tensor:
    reasoner_dim = int(reasoner_model.get_input_embeddings().weight.shape[-1])
    actor_dim = int(actor_model.get_input_embeddings().weight.shape[-1])
    if reasoner_dim != actor_dim:
        raise ValueError(
            "Global alignment configuration is required when reasoner and actor hidden sizes differ"
        )
    return torch.eye(reasoner_dim, dtype=torch.float32)


def _supports_cached_alignment(model: AutoModelForCausalLM) -> bool:
    return hasattr(model, "model") and hasattr(model.model, "layers")


def resolve_training_alignment_context(
    *,
    reasoner_model: AutoModelForCausalLM,
    actor_model: AutoModelForCausalLM,
    reasoner_tokenizer: Any,
    actor_tokenizer: Any,
    alignment_cfg: Optional[DictConfig],
) -> dict[str, Any]:
    if (
        alignment_cfg is not None
        and reasoner_tokenizer is not None
        and actor_tokenizer is not None
        and _supports_cached_alignment(reasoner_model)
        and _supports_cached_alignment(actor_model)
    ):
        alignment_state = load_or_compute_global_alignment_state(
            alignment_cfg,
            tokenizer_a=reasoner_tokenizer,
            tokenizer_b=actor_tokenizer,
            agent_a=reasoner_model,
            agent_b=actor_model,
        )
        return {
            "alignment_q": alignment_state["global_alignment_q"],
            "alignment_mode": alignment_state["alignment_mode"],
            "global_alignment_cache_key": alignment_state["global_alignment_cache_key"],
            "global_alignment_cache_hit": bool(alignment_state["global_alignment_cache_hit"]),
            "global_alignment_cache_path": str(alignment_state["global_alignment_cache_path"]),
            "reasoning_layer_indices": tuple(alignment_state["global_reasoning_layer_indices"]),
            "reasoning_layer_weights": tuple(alignment_state["global_reasoning_layer_weights"]),
            "semantic_anchor_count": int(alignment_state["semantic_anchor_count"]),
        }

    return {
        "alignment_q": _identity_alignment_mapping(reasoner_model, actor_model),
        "alignment_mode": "identity_fallback",
        "global_alignment_cache_key": None,
        "global_alignment_cache_hit": False,
        "global_alignment_cache_path": "",
        "reasoning_layer_indices": (),
        "reasoning_layer_weights": (),
        "semantic_anchor_count": 0,
    }


def freeze_actor(actor_model: AutoModelForCausalLM) -> None:
    actor_model.eval()
    for parameter in actor_model.parameters():
        parameter.requires_grad = False


def train_reasoner_stage2(
    reasoner_model: AutoModelForCausalLM,
    actor_model: AutoModelForCausalLM,
    train_dataloader: Iterable[dict[str, Any]],
    config: CompressionTrainConfig,
    *,
    reasoner_tokenizer: Any,
    actor_tokenizer: Any,
    alignment_cfg: Optional[DictConfig] = None,
    evaluation_fn: Optional[EvaluationFn] = None,
) -> list[dict[str, float]]:
    """
    Stage II Interlat-style training loop:
    - Freeze Actor (Agent B) completely.
    - Update only Reasoner (Agent A).
    """
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
    reasoner_backbone = _model_backbone(reasoner_model)
    alignment_context = resolve_training_alignment_context(
        reasoner_model=reasoner_model,
        actor_model=actor_model,
        reasoner_tokenizer=reasoner_tokenizer,
        actor_tokenizer=actor_tokenizer,
        alignment_cfg=alignment_cfg,
    )
    procrustes_q = alignment_context["alignment_q"].to(
        device=reasoner_device,
        dtype=reasoner_model.get_input_embeddings().weight.dtype,
    )

    history: list[dict[str, float]] = []
    global_step = 0

    if config.checkpoint_enabled:
        ckpt_dir = Path(config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.num_epochs):
        for step, batch in enumerate(train_dataloader):
            texts = _extract_text_batch(batch)
            reasoner_batch = _tokenize_text_batch(
                reasoner_tokenizer,
                texts,
                device=reasoner_device,
                max_length=config.reasoner_max_length,
            )
            actor_batch = _tokenize_text_batch(
                actor_tokenizer,
                texts,
                device=reasoner_device,
                max_length=config.actor_max_length,
            )
            input_ids = reasoner_batch["input_ids"]
            attention_mask = reasoner_batch["attention_mask"]
            actor_labels = actor_batch["labels"]

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
                actor_labels=actor_labels,
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
                "pref_avg_entropy": float(loss_outputs["pref_avg_entropy"].detach().cpu().item()),
                "pref_avg_weight": float(loss_outputs["pref_avg_weight"].detach().cpu().item()),
                "pref_first_token_entropy": float(
                    loss_outputs["pref_first_token_entropy"].detach().cpu().item()
                ),
                "pref_first_token_weight": float(
                    loss_outputs["pref_first_token_weight"].detach().cpu().item()
                ),
                "pref_first_token_kl": float(
                    loss_outputs["pref_first_token_kl"].detach().cpu().item()
                ),
                "pref_avg_top1_probability": float(
                    loss_outputs["pref_avg_top1_probability"].detach().cpu().item()
                ),
                "pref_first_token_top1_probability": float(
                    loss_outputs["pref_first_token_top1_probability"].detach().cpu().item()
                ),
                "pref_avg_logit_margin": float(
                    loss_outputs["pref_avg_logit_margin"].detach().cpu().item()
                ),
                "pref_first_token_logit_margin": float(
                    loss_outputs["pref_first_token_logit_margin"].detach().cpu().item()
                ),
                "pref_first_token_weight_ratio": float(
                    loss_outputs["pref_first_token_weight_ratio"].detach().cpu().item()
                ),
            }
            history.append(metrics)

            if config.wandb_enabled:
                wandb.log(
                    {
                        "loss": metrics["loss"],
                        "l_task": metrics["l_task"],
                        "l_pref": metrics["l_pref"],
                        "l_geom": metrics["l_geom"],
                        "pref_avg_entropy": metrics["pref_avg_entropy"],
                        "pref_avg_weight": metrics["pref_avg_weight"],
                        "pref_first_token_entropy": metrics["pref_first_token_entropy"],
                        "pref_first_token_weight": metrics["pref_first_token_weight"],
                        "pref_first_token_kl": metrics["pref_first_token_kl"],
                        "pref_avg_top1_probability": metrics["pref_avg_top1_probability"],
                        "pref_first_token_top1_probability": metrics["pref_first_token_top1_probability"],
                        "pref_avg_logit_margin": metrics["pref_avg_logit_margin"],
                        "pref_first_token_logit_margin": metrics["pref_first_token_logit_margin"],
                        "pref_first_token_weight_ratio": metrics["pref_first_token_weight_ratio"],
                    },
                    step=global_step,
                )
            if config.checkpoint_enabled and config.checkpoint_every_n_steps > 0:
                if global_step % config.checkpoint_every_n_steps == 0:
                    step_path = ckpt_dir / f"step_{global_step}.pt"
                    torch.save(reasoner_model.state_dict(), step_path)
                    print(f"Saved checkpoint: {step_path}")

            global_step += 1

        if evaluation_fn is not None:
            evaluation_metrics = {
                key: float(value)
                for key, value in evaluation_fn(reasoner_model, actor_model, alignment_context).items()
            }
            eval_history_entry = {
                "epoch": float(epoch),
                "step": float(global_step),
                **evaluation_metrics,
            }
            history.append(eval_history_entry)
            if config.wandb_enabled:
                wandb.log(evaluation_metrics, step=global_step)

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
    print(f"  reasoner_max_length = {config.reasoner_max_length}")
    print(f"  actor_max_length    = {config.actor_max_length}")
    print(
        "\nTo run training, call train_reasoner_stage2(...) with "
        "initialized reasoner/actor models, tokenizers, and a text-first dataloader."
    )


if __name__ == "__main__":
    main()
