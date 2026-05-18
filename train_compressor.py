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
from omegaconf import DictConfig, OmegaConf
from torch import nn
from transformers import AutoModelForCausalLM

from src.models.hidden_state import AdaptiveProjection, CurriculumStage, HiddenStateProcessor
from src.models.losses import (
    AdaptiveLossBalancer,
    AdaptiveLossBalancerConfig,
    LatentCompressorLoss,
    compute_plan_similarity_loss,
    compute_random_contrast_loss,
)
from src.utils.alignment import apply_alignment

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
    lambda_plan: float = 0.25
    lambda_contrast: float = 0.1
    contrast_temperature: float = 0.1
    curriculum_enabled: bool = True
    curriculum_stages: tuple[str, ...] = ("identity", "orthogonal", "hybrid_affine")
    curriculum_boundaries: tuple[float, ...] = (0.33, 0.66, 1.0)
    adaptive_loss_enabled: bool = True
    adaptive_loss_ema_beta: float = 0.9
    adaptive_loss_min_weight: float = 0.25
    adaptive_loss_max_weight: float = 4.0
    adaptive_projection_enabled: bool = True
    adaptive_projection_strength: float = 0.15
    adaptive_projection_clip_std_multiplier: float = 4.0
    hidden_state_processor_enabled: bool = False
    hidden_state_processor_num_heads: int = 4
    hidden_state_processor_dropout: float = 0.0

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "CompressionTrainConfig":
        t = cfg.training
        wandb_cfg = getattr(t, "wandb", None)
        curriculum_cfg = getattr(t, "curriculum", None)
        adaptive_loss_cfg = getattr(t, "adaptive_loss", None)
        projection_cfg = getattr(t, "adaptive_projection", None)
        processor_cfg = getattr(t, "hidden_state_processor", None)
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
            lambda_plan=float(getattr(t, "lambda_plan", 0.25)),
            lambda_contrast=float(getattr(t, "lambda_contrast", 0.1)),
            contrast_temperature=float(getattr(t, "contrast_temperature", 0.1)),
            curriculum_enabled=bool(getattr(curriculum_cfg, "enabled", True)) if curriculum_cfg else True,
            curriculum_stages=tuple(
                str(stage) for stage in getattr(curriculum_cfg, "stages", ("identity", "orthogonal", "hybrid_affine"))
            ) if curriculum_cfg else ("identity", "orthogonal", "hybrid_affine"),
            curriculum_boundaries=tuple(
                float(value) for value in getattr(curriculum_cfg, "boundaries", (0.33, 0.66, 1.0))
            ) if curriculum_cfg else (0.33, 0.66, 1.0),
            adaptive_loss_enabled=bool(getattr(adaptive_loss_cfg, "enabled", True)) if adaptive_loss_cfg else True,
            adaptive_loss_ema_beta=float(getattr(adaptive_loss_cfg, "ema_beta", 0.9)) if adaptive_loss_cfg else 0.9,
            adaptive_loss_min_weight=float(getattr(adaptive_loss_cfg, "min_weight", 0.25)) if adaptive_loss_cfg else 0.25,
            adaptive_loss_max_weight=float(getattr(adaptive_loss_cfg, "max_weight", 4.0)) if adaptive_loss_cfg else 4.0,
            adaptive_projection_enabled=bool(getattr(projection_cfg, "enabled", True)) if projection_cfg else True,
            adaptive_projection_strength=float(getattr(projection_cfg, "strength", 0.15)) if projection_cfg else 0.15,
            adaptive_projection_clip_std_multiplier=float(
                getattr(projection_cfg, "clip_std_multiplier", 4.0)
            ) if projection_cfg else 4.0,
            hidden_state_processor_enabled=bool(getattr(processor_cfg, "enabled", False)) if processor_cfg else False,
            hidden_state_processor_num_heads=int(getattr(processor_cfg, "num_heads", 4)) if processor_cfg else 4,
            hidden_state_processor_dropout=float(getattr(processor_cfg, "dropout", 0.0)) if processor_cfg else 0.0,
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


def _curriculum_schedule(config: CompressionTrainConfig) -> tuple[CurriculumStage, ...]:
    stage_specs: list[CurriculumStage] = []
    for stage_name, upper_bound in zip(config.curriculum_stages, config.curriculum_boundaries):
        normalized_name = str(stage_name).strip().lower()
        if normalized_name == "identity":
            stage_specs.append(
                CurriculumStage(
                    name="identity",
                    progress_upper_bound=float(upper_bound),
                    alignment_strategy="identity",
                    prompt_calibration_enabled=False,
                )
            )
        elif normalized_name == "orthogonal":
            stage_specs.append(
                CurriculumStage(
                    name="orthogonal",
                    progress_upper_bound=float(upper_bound),
                    alignment_strategy="orthogonal",
                    prompt_calibration_enabled=False,
                )
            )
        elif normalized_name == "hybrid_affine":
            stage_specs.append(
                CurriculumStage(
                    name="hybrid_affine",
                    progress_upper_bound=float(upper_bound),
                    alignment_strategy="hybrid_affine",
                    prompt_calibration_enabled=True,
                )
            )
        else:
            raise ValueError(f"Unsupported curriculum stage {stage_name!r}")
    if not stage_specs:
        raise ValueError("At least one curriculum stage is required")
    return tuple(stage_specs)


def _resolve_curriculum_stage(
    stages: Sequence[CurriculumStage],
    progress_ratio: float,
) -> CurriculumStage:
    for stage in stages:
        if progress_ratio <= stage.progress_upper_bound:
            return stage
    return stages[-1]


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


def _identity_alignment_state(
    reasoner_model: AutoModelForCausalLM,
    actor_model: AutoModelForCausalLM,
) -> dict[str, Any]:
    identity = _identity_alignment_mapping(reasoner_model, actor_model)
    return {
        "mapping_matrix": identity,
        "mapping_bias": torch.zeros((1, identity.shape[-1]), dtype=torch.float32),
        "alignment_strategy": "identity",
    }


def _supports_cached_alignment(model: AutoModelForCausalLM) -> bool:
    return hasattr(model, "model") and hasattr(model.model, "layers")


def resolve_training_alignment_context(
    *,
    reasoner_model: AutoModelForCausalLM,
    actor_model: AutoModelForCausalLM,
    reasoner_tokenizer: Any,
    actor_tokenizer: Any,
    alignment_cfg: Optional[DictConfig],
    strategy_override: Optional[str] = None,
    prompt_calibration_enabled: Optional[bool] = None,
) -> dict[str, Any]:
    effective_cfg = alignment_cfg
    if alignment_cfg is not None and (strategy_override is not None or prompt_calibration_enabled is not None):
        effective_cfg = OmegaConf.create(OmegaConf.to_container(alignment_cfg, resolve=True))
        if strategy_override is not None:
            effective_cfg.alignment.strategy = str(strategy_override)
        if prompt_calibration_enabled is not None:
            effective_cfg.alignment.prompt_calibration.enabled = bool(prompt_calibration_enabled)
    if (
        effective_cfg is not None
        and reasoner_tokenizer is not None
        and actor_tokenizer is not None
        and _supports_cached_alignment(reasoner_model)
        and _supports_cached_alignment(actor_model)
    ):
        alignment_state = load_or_compute_global_alignment_state(
            effective_cfg,
            tokenizer_a=reasoner_tokenizer,
            tokenizer_b=actor_tokenizer,
            agent_a=reasoner_model,
            agent_b=actor_model,
        )
        handoff_mapping = alignment_state.get(
            "handoff_alignment_q",
            alignment_state["global_alignment_q"],
        )
        handoff_bias = alignment_state.get(
            "handoff_alignment_bias",
            alignment_state.get("global_alignment_bias"),
        )
        return {
            "alignment_q": handoff_mapping,
            "alignment_state": {
                "mapping_matrix": handoff_mapping,
                "mapping_bias": handoff_bias,
                "pre_projection_state": alignment_state.get(
                    "handoff_pre_projection_state",
                    alignment_state.get("pre_projection_state"),
                ),
                "post_projection_state": alignment_state.get(
                    "handoff_post_projection_state",
                    alignment_state.get("post_projection_state"),
                ),
                "alignment_strategy": alignment_state.get("alignment_strategy", alignment_state["alignment_mode"]),
                "orthogonal_q": alignment_state.get(
                    "handoff_alignment_backbone_q",
                    alignment_state.get("global_alignment_backbone_q"),
                ),
                "residual_matrix": alignment_state.get("handoff_alignment_residual"),
                "residual_norm_ratio": alignment_state.get(
                    "handoff_residual_norm_ratio",
                    alignment_state.get("residual_norm_ratio"),
                ),
                "bias_norm": alignment_state.get("handoff_bias_norm", alignment_state.get("bias_norm")),
            },
            "diagnostic_alignment_state": {
                "mapping_matrix": alignment_state["global_alignment_q"],
                "mapping_bias": alignment_state.get("global_alignment_bias"),
                "pre_projection_state": alignment_state.get("pre_projection_state"),
                "post_projection_state": alignment_state.get("post_projection_state"),
                "alignment_strategy": alignment_state.get("alignment_strategy", alignment_state["alignment_mode"]),
                "orthogonal_q": alignment_state.get("global_alignment_backbone_q"),
                "residual_matrix": alignment_state.get("global_alignment_residual"),
                "residual_norm_ratio": alignment_state.get("residual_norm_ratio"),
                "bias_norm": alignment_state.get("bias_norm"),
            },
            "alignment_mode": alignment_state["alignment_mode"],
            "handoff_surface": alignment_state.get("handoff_surface", "input_embedding"),
            "diagnostic_surface": alignment_state.get("diagnostic_surface", "hidden_consensus"),
            "global_alignment_cache_key": alignment_state["global_alignment_cache_key"],
            "global_alignment_cache_hit": bool(alignment_state["global_alignment_cache_hit"]),
            "global_alignment_cache_path": str(alignment_state["global_alignment_cache_path"]),
            "reasoning_layer_indices": tuple(alignment_state["global_reasoning_layer_indices"]),
            "reasoning_layer_weights": tuple(alignment_state["global_reasoning_layer_weights"]),
            "semantic_anchor_count": int(alignment_state["semantic_anchor_count"]),
            "selected_anchor_indices": tuple(alignment_state.get("selected_anchor_indices", ())),
            "alignment_strategy": alignment_state.get("alignment_strategy", alignment_state["alignment_mode"]),
            "anchor_reconstruction_mse": alignment_state.get(
                "handoff_anchor_reconstruction_mse",
                alignment_state.get("anchor_reconstruction_mse"),
            ),
            "anchor_pairwise_distance_distortion": alignment_state.get(
                "handoff_anchor_pairwise_distance_distortion",
                alignment_state.get("anchor_pairwise_distance_distortion"),
            ),
        }

    return {
        "alignment_q": _identity_alignment_mapping(reasoner_model, actor_model),
        "alignment_state": _identity_alignment_state(reasoner_model, actor_model),
        "alignment_mode": "identity_fallback",
        "global_alignment_cache_key": None,
        "global_alignment_cache_hit": False,
        "global_alignment_cache_path": "",
        "reasoning_layer_indices": (),
        "reasoning_layer_weights": (),
        "semantic_anchor_count": 0,
        "selected_anchor_indices": (),
        "alignment_strategy": "identity",
        "anchor_reconstruction_mse": None,
        "anchor_pairwise_distance_distortion": None,
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
    adaptive_projection = (
        AdaptiveProjection(
            strength=config.adaptive_projection_strength,
            clip_std_multiplier=config.adaptive_projection_clip_std_multiplier,
        )
        if config.adaptive_projection_enabled
        else None
    )
    actor_hidden_size = int(actor_model.get_input_embeddings().weight.shape[-1])
    hidden_state_processor = (
        HiddenStateProcessor(
            actor_hidden_size,
            num_heads=config.hidden_state_processor_num_heads,
            dropout=config.hidden_state_processor_dropout,
        )
        if config.hidden_state_processor_enabled
        else None
    )
    if hidden_state_processor is not None:
        hidden_state_processor.train()
    optimizer = torch.optim.AdamW(
        (
            list(p for p in reasoner_model.parameters() if p.requires_grad)
            + ([] if hidden_state_processor is None else list(hidden_state_processor.parameters()))
        ),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    reasoner_device = _model_device(reasoner_model)
    actor_device = _model_device(actor_model)
    reasoner_backbone = _model_backbone(reasoner_model)
    if hidden_state_processor is not None:
        hidden_state_processor.to(reasoner_device)
    curriculum_stages = _curriculum_schedule(config) if config.curriculum_enabled else (
        CurriculumStage(
            name="default",
            progress_upper_bound=1.0,
            alignment_strategy=str(getattr(getattr(alignment_cfg, "alignment", None), "strategy", "hybrid_affine")),
            prompt_calibration_enabled=False,
        ),
    )
    alignment_context_cache: dict[str, dict[str, Any]] = {}
    for stage in curriculum_stages:
        if stage.alignment_strategy == "identity":
            try:
                alignment_context_cache[stage.name] = resolve_training_alignment_context(
                    reasoner_model=reasoner_model,
                    actor_model=actor_model,
                    reasoner_tokenizer=reasoner_tokenizer,
                    actor_tokenizer=actor_tokenizer,
                    alignment_cfg=None,
                )
            except ValueError:
                alignment_context_cache[stage.name] = resolve_training_alignment_context(
                    reasoner_model=reasoner_model,
                    actor_model=actor_model,
                    reasoner_tokenizer=reasoner_tokenizer,
                    actor_tokenizer=actor_tokenizer,
                    alignment_cfg=alignment_cfg,
                    strategy_override="orthogonal",
                    prompt_calibration_enabled=False,
                )
            continue
        alignment_context_cache[stage.name] = resolve_training_alignment_context(
            reasoner_model=reasoner_model,
            actor_model=actor_model,
            reasoner_tokenizer=reasoner_tokenizer,
            actor_tokenizer=actor_tokenizer,
            alignment_cfg=alignment_cfg,
            strategy_override=stage.alignment_strategy,
            prompt_calibration_enabled=stage.prompt_calibration_enabled,
        )
    latest_alignment_context = alignment_context_cache[curriculum_stages[-1].name]
    loss_balancer = AdaptiveLossBalancer(
        {
            "l_task": config.lambda_task,
            "l_pref": config.lambda_pref,
            "l_geom": config.lambda_geom,
            "l_plan": config.lambda_plan,
            "l_contrast": config.lambda_contrast,
        },
        config=AdaptiveLossBalancerConfig(
            enabled=config.adaptive_loss_enabled,
            ema_beta=config.adaptive_loss_ema_beta,
            min_weight=config.adaptive_loss_min_weight,
            max_weight=config.adaptive_loss_max_weight,
        ),
    )

    history: list[dict[str, float]] = []
    global_step = 0
    total_training_steps = max(1, int(config.num_epochs) * max(1, len(train_dataloader)))

    if config.checkpoint_enabled:
        ckpt_dir = Path(config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.num_epochs):
        for step, batch in enumerate(train_dataloader):
            progress_ratio = min(
                1.0,
                float(global_step + 1) / float(total_training_steps),
            )
            stage = _resolve_curriculum_stage(curriculum_stages, progress_ratio)
            alignment_context = alignment_context_cache[stage.name]
            latest_alignment_context = alignment_context
            alignment_state = alignment_context["alignment_state"]
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
            full_latents_aligned = apply_alignment(full_latents, alignment_state)
            compressed_latents_aligned = apply_alignment(compressed_latents, alignment_state)
            projection_metrics = {
                "projection_scale_mean": 1.0,
                "projection_scale_std": 0.0,
                "projection_clip_fraction": 0.0,
            }
            if adaptive_projection is not None:
                compressed_latents_aligned, projection_metrics = adaptive_projection(
                    compressed_latents_aligned,
                    full_latents_aligned.detach(),
                )
            if hidden_state_processor is not None:
                compressed_latents_aligned = hidden_state_processor(compressed_latents_aligned)

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
            l_plan = compute_plan_similarity_loss(
                full_latents_aligned.detach(),
                compressed_latents_aligned,
            )
            l_contrast = compute_random_contrast_loss(
                full_latents_aligned.detach(),
                compressed_latents_aligned,
                temperature=config.contrast_temperature,
            )
            total_loss, effective_weights = loss_balancer.combine(
                {
                    "l_task": loss_outputs["l_task"],
                    "l_pref": loss_outputs["l_pref"],
                    "l_geom": loss_outputs["l_geom"],
                    "l_plan": l_plan,
                    "l_contrast": l_contrast,
                }
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            nn.utils.clip_grad_norm_(reasoner_model.parameters(), config.max_grad_norm)
            if hidden_state_processor is not None:
                nn.utils.clip_grad_norm_(hidden_state_processor.parameters(), config.max_grad_norm)
            optimizer.step()

            metrics = {
                "epoch": float(epoch),
                "step": float(step),
                "loss": float(total_loss.detach().cpu().item()),
                "l_task": float(loss_outputs["l_task"].detach().cpu().item()),
                "l_pref": float(loss_outputs["l_pref"].detach().cpu().item()),
                "l_geom": float(loss_outputs["l_geom"].detach().cpu().item()),
                "l_plan": float(l_plan.detach().cpu().item()),
                "l_contrast": float(l_contrast.detach().cpu().item()),
                "effective_weight_task": effective_weights["l_task"],
                "effective_weight_pref": effective_weights["l_pref"],
                "effective_weight_geom": effective_weights["l_geom"],
                "effective_weight_plan": effective_weights["l_plan"],
                "effective_weight_contrast": effective_weights["l_contrast"],
                "curriculum_stage": float(("identity", "orthogonal", "hybrid_affine").index(stage.alignment_strategy) if stage.alignment_strategy in {"identity", "orthogonal", "hybrid_affine"} else 0),
                "projection_scale_mean": projection_metrics["projection_scale_mean"],
                "projection_scale_std": projection_metrics["projection_scale_std"],
                "projection_clip_fraction": projection_metrics["projection_clip_fraction"],
                "alignment_residual_norm_ratio": float(alignment_context.get("alignment_state", {}).get("residual_norm_ratio", 0.0) or 0.0),
                "alignment_anchor_reconstruction_mse": float(alignment_context.get("anchor_reconstruction_mse") or 0.0),
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
                        "l_plan": metrics["l_plan"],
                        "l_contrast": metrics["l_contrast"],
                        "effective_weight_task": metrics["effective_weight_task"],
                        "effective_weight_pref": metrics["effective_weight_pref"],
                        "effective_weight_geom": metrics["effective_weight_geom"],
                        "effective_weight_plan": metrics["effective_weight_plan"],
                        "effective_weight_contrast": metrics["effective_weight_contrast"],
                        "projection_scale_mean": metrics["projection_scale_mean"],
                        "projection_scale_std": metrics["projection_scale_std"],
                        "projection_clip_fraction": metrics["projection_clip_fraction"],
                        "alignment_residual_norm_ratio": metrics["alignment_residual_norm_ratio"],
                        "alignment_anchor_reconstruction_mse": metrics["alignment_anchor_reconstruction_mse"],
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
                    torch.save(
                        {
                            "reasoner_state_dict": reasoner_model.state_dict(),
                            "hidden_state_processor_state_dict": None
                            if hidden_state_processor is None
                            else hidden_state_processor.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "training_config": dataclasses.asdict(config),
                            "alignment_context": alignment_context,
                            "curriculum_stage": stage.name,
                            "global_step": global_step,
                            "history_tail": history[-10:],
                        },
                        step_path,
                    )
                    print(f"Saved checkpoint: {step_path}")

            global_step += 1

        if evaluation_fn is not None:
            evaluation_metrics = {
                key: float(value)
                for key, value in evaluation_fn(reasoner_model, actor_model, latest_alignment_context).items()
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
            torch.save(
                {
                    "reasoner_state_dict": reasoner_model.state_dict(),
                    "hidden_state_processor_state_dict": None
                    if hidden_state_processor is None
                    else hidden_state_processor.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "training_config": dataclasses.asdict(config),
                    "alignment_context": latest_alignment_context,
                    "curriculum_stages": [dataclasses.asdict(stage) for stage in curriculum_stages],
                    "epoch": epoch,
                    "global_step": global_step,
                    "history_tail": history[-25:],
                },
                epoch_path,
            )
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
