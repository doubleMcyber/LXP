from __future__ import annotations

"""
LXP Stage 2 Training Pipeline (Compressor & Reasoner)
-----------------------------------------------------
This module contains the training loop used to explicitly teach the models how to use the continuous latent space.
Because LLMs cannot zero-shot read compressed math trajectories, this stage optimizes the bridge.

Key Losses Optimized Here:
1. Task Loss (L_task): Cross-entropy ensuring the downstream model outputs the correct final answer.
2. Preference Loss (L_pref): KL Divergence ensuring the continuous thought produces the same token distribution as an expert Chain-of-Thought text.
3. Geometric Loss (L_geom): Cosine similarity ensuring the latent vectors don't collapse into unstructured noise.
"""

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import hydra
import torch
import torch.nn.functional as F
import wandb
from latent_pipeline import load_or_compute_global_alignment_state
from omegaconf import DictConfig, OmegaConf
from torch import nn
from transformers import AutoModelForCausalLM

from src.models.hidden_state import (
    AdaptiveProjection,
    CurriculumStage,
    HiddenStateProcessor,
    LatentAnswerProbe,
    LatentHandoffAdapter,
    LatentLogitSteeringHead,
    LatentSequenceDecoderHead,
    LatentSoftPromptDecoder,
    LatentTokenDecoderHead,
    lm_vocabulary_weight,
)
from src.models.losses import (
    AdaptiveLossBalancer,
    AdaptiveLossBalancerConfig,
    LatentCompressorLoss,
    compute_plan_similarity_loss,
    compute_random_contrast_loss,
)
from src.utils.alignment import apply_alignment

EvaluationFn = Callable[[AutoModelForCausalLM, AutoModelForCausalLM, dict[str, Any]], dict[str, Any]]


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
    lambda_answer: float = 4.0
    lambda_answer_first_token: float = 0.0
    lambda_logit_steering: float = 0.0
    answer_suffix_text: str = "\nFinal answer: "
    answer_max_length: int = 32
    answer_first_token_weight: float = 2.0
    answer_first_token_margin: float = 2.0
    logit_steering_margin: float = 2.0
    evaluate_before_training: bool = False
    early_stop_raw_decode_ready: bool = False
    train_reasoner: bool = True
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
    lambda_answer_contrast: float = 0.0
    lambda_answer_probe: float = 0.0
    lambda_latent_sequence_decoder: float = 4.0
    answer_contrast_temperature: float = 1.0
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
    latent_handoff_adapter_enabled: bool = True
    latent_handoff_adapter_rank: int = 64
    latent_handoff_adapter_scale: float = 1.0
    latent_handoff_adapter_dropout: float = 0.0
    latent_answer_probe_enabled: bool = False
    latent_answer_probe_max_candidates: int = 64
    latent_answer_probe_hidden_multiplier: int = 2
    latent_answer_probe_dropout: float = 0.0
    latent_sequence_decoder_enabled: bool = True
    latent_sequence_decoder_max_answer_length: int = 32
    latent_sequence_decoder_vocabulary_mode: str = "tied"
    latent_sequence_decoder_num_heads: int = 4
    latent_sequence_decoder_hidden_multiplier: int = 2
    latent_sequence_decoder_dropout: float = 0.0
    latent_sequence_decoder_scale: float = 1.0
    latent_sequence_decoder_generation_min_accuracy: float = 95.0
    latent_soft_prompt_decoder_enabled: bool = False
    latent_soft_prompt_decoder_output_steps: int = 0
    latent_soft_prompt_decoder_num_heads: int = 4
    latent_soft_prompt_decoder_hidden_multiplier: int = 2
    latent_soft_prompt_decoder_dropout: float = 0.0
    latent_soft_prompt_decoder_residual_scale: float = 1.0
    latent_soft_prompt_decoder_max_delta_norm: float = 0.0
    latent_logit_steering_enabled: bool = False
    latent_logit_steering_rank: int = 64
    latent_logit_steering_vocabulary_mode: str = "tied"
    latent_logit_steering_lr_multiplier: float = 1.0
    latent_logit_steering_output_steps: int = 1
    latent_logit_steering_dropout: float = 0.0
    latent_logit_steering_scale: float = 1.0
    latent_logit_steering_generation_scale: float = 1.0
    latent_logit_steering_pooling: str = "attention"
    latent_logit_steering_max_bias_norm: float = 0.0
    latent_logit_steering_answer_token_weight: float = 1.0
    latent_logit_steering_later_answer_token_weight: float = 1.0
    latent_logit_steering_eos_weight: float = 1.0
    lambda_latent_token_decoder: float = 0.0
    latent_token_decoder_enabled: bool = False
    latent_token_decoder_rank: int = 64
    latent_token_decoder_vocabulary_mode: str = "low_rank"
    latent_token_decoder_lr_multiplier: float = 1.0
    latent_token_decoder_output_steps: int = 8
    latent_token_decoder_dropout: float = 0.0
    latent_token_decoder_scale: float = 1.0
    latent_token_decoder_pooling: str = "attention"
    latent_token_decoder_max_bias_norm: float = 0.0
    latent_token_decoder_candidate_token_mask: bool = False
    latent_token_decoder_require_ready: bool = False
    latent_token_decoder_eos_weight: float = 1.0
    latent_token_decoder_margin: float = 0.0
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
        handoff_adapter_cfg = getattr(t, "latent_handoff_adapter", None)
        answer_probe_cfg = getattr(t, "latent_answer_probe", None)
        sequence_decoder_cfg = getattr(t, "latent_sequence_decoder", None)
        soft_prompt_cfg = getattr(t, "latent_soft_prompt_decoder", None)
        logit_steering_cfg = getattr(t, "latent_logit_steering", None)
        token_decoder_cfg = getattr(t, "latent_token_decoder", None)
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
            lambda_answer=float(getattr(t, "lambda_answer", 4.0)),
            lambda_answer_first_token=float(getattr(t, "lambda_answer_first_token", 0.0)),
            lambda_logit_steering=float(getattr(t, "lambda_logit_steering", 0.0)),
            lambda_latent_token_decoder=float(getattr(t, "lambda_latent_token_decoder", 0.0)),
            answer_suffix_text=str(getattr(t, "answer_suffix_text", "\nFinal answer: ")),
            answer_max_length=int(getattr(t, "answer_max_length", 32)),
            answer_first_token_weight=float(getattr(t, "answer_first_token_weight", 2.0)),
            answer_first_token_margin=float(getattr(t, "answer_first_token_margin", 2.0)),
            logit_steering_margin=float(getattr(t, "logit_steering_margin", 2.0)),
            evaluate_before_training=bool(
                getattr(getattr(t, "evaluation", None), "evaluate_before_training", False)
            ),
            early_stop_raw_decode_ready=bool(
                getattr(getattr(t, "evaluation", None), "early_stop_raw_decode_ready", False)
            ),
            train_reasoner=bool(getattr(t, "train_reasoner", True)),
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
            lambda_answer_contrast=float(getattr(t, "lambda_answer_contrast", 0.0)),
            lambda_answer_probe=float(getattr(t, "lambda_answer_probe", 0.0)),
            lambda_latent_sequence_decoder=float(getattr(t, "lambda_latent_sequence_decoder", 4.0)),
            answer_contrast_temperature=float(getattr(t, "answer_contrast_temperature", 1.0)),
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
            latent_handoff_adapter_enabled=bool(
                getattr(handoff_adapter_cfg, "enabled", True)
            ) if handoff_adapter_cfg else True,
            latent_handoff_adapter_rank=int(
                getattr(handoff_adapter_cfg, "rank", 64)
            ) if handoff_adapter_cfg else 64,
            latent_handoff_adapter_scale=float(
                getattr(handoff_adapter_cfg, "scale", 1.0)
            ) if handoff_adapter_cfg else 1.0,
            latent_handoff_adapter_dropout=float(
                getattr(handoff_adapter_cfg, "dropout", 0.0)
            ) if handoff_adapter_cfg else 0.0,
            latent_answer_probe_enabled=bool(
                getattr(answer_probe_cfg, "enabled", False)
            ) if answer_probe_cfg else False,
            latent_answer_probe_max_candidates=int(
                getattr(answer_probe_cfg, "max_candidates", 64)
            ) if answer_probe_cfg else 64,
            latent_answer_probe_hidden_multiplier=int(
                getattr(answer_probe_cfg, "hidden_multiplier", 2)
            ) if answer_probe_cfg else 2,
            latent_answer_probe_dropout=float(
                getattr(answer_probe_cfg, "dropout", 0.0)
            ) if answer_probe_cfg else 0.0,
            latent_sequence_decoder_enabled=bool(
                getattr(sequence_decoder_cfg, "enabled", True)
            ) if sequence_decoder_cfg else True,
            latent_sequence_decoder_max_answer_length=int(
                getattr(sequence_decoder_cfg, "max_answer_length", getattr(t, "answer_max_length", 32))
            ) if sequence_decoder_cfg else int(getattr(t, "answer_max_length", 32)),
            latent_sequence_decoder_vocabulary_mode=str(
                getattr(sequence_decoder_cfg, "vocabulary_mode", "tied")
            ) if sequence_decoder_cfg else "tied",
            latent_sequence_decoder_num_heads=int(
                getattr(sequence_decoder_cfg, "num_heads", 4)
            ) if sequence_decoder_cfg else 4,
            latent_sequence_decoder_hidden_multiplier=int(
                getattr(sequence_decoder_cfg, "hidden_multiplier", 2)
            ) if sequence_decoder_cfg else 2,
            latent_sequence_decoder_dropout=float(
                getattr(sequence_decoder_cfg, "dropout", 0.0)
            ) if sequence_decoder_cfg else 0.0,
            latent_sequence_decoder_scale=float(
                getattr(sequence_decoder_cfg, "scale", 1.0)
            ) if sequence_decoder_cfg else 1.0,
            latent_sequence_decoder_generation_min_accuracy=float(
                getattr(sequence_decoder_cfg, "generation_min_sequence_accuracy", 95.0)
            ) if sequence_decoder_cfg else 95.0,
            latent_soft_prompt_decoder_enabled=bool(
                getattr(soft_prompt_cfg, "enabled", False)
            ) if soft_prompt_cfg else False,
            latent_soft_prompt_decoder_output_steps=int(
                getattr(soft_prompt_cfg, "output_steps", 0)
            ) if soft_prompt_cfg else 0,
            latent_soft_prompt_decoder_num_heads=int(
                getattr(soft_prompt_cfg, "num_heads", 4)
            ) if soft_prompt_cfg else 4,
            latent_soft_prompt_decoder_hidden_multiplier=int(
                getattr(soft_prompt_cfg, "hidden_multiplier", 2)
            ) if soft_prompt_cfg else 2,
            latent_soft_prompt_decoder_dropout=float(
                getattr(soft_prompt_cfg, "dropout", 0.0)
            ) if soft_prompt_cfg else 0.0,
            latent_soft_prompt_decoder_residual_scale=float(
                getattr(soft_prompt_cfg, "residual_scale", 1.0)
            ) if soft_prompt_cfg else 1.0,
            latent_soft_prompt_decoder_max_delta_norm=float(
                getattr(soft_prompt_cfg, "max_delta_norm", 0.0)
            ) if soft_prompt_cfg else 0.0,
            latent_logit_steering_enabled=bool(
                getattr(logit_steering_cfg, "enabled", False)
            ) if logit_steering_cfg else False,
            latent_logit_steering_rank=int(
                getattr(logit_steering_cfg, "rank", 64)
            ) if logit_steering_cfg else 64,
            latent_logit_steering_vocabulary_mode=str(
                getattr(logit_steering_cfg, "vocabulary_mode", "tied")
            ) if logit_steering_cfg else "tied",
            latent_logit_steering_lr_multiplier=float(
                getattr(logit_steering_cfg, "lr_multiplier", 1.0)
            ) if logit_steering_cfg else 1.0,
            latent_logit_steering_output_steps=int(
                getattr(logit_steering_cfg, "output_steps", 1)
            ) if logit_steering_cfg else 1,
            latent_logit_steering_dropout=float(
                getattr(logit_steering_cfg, "dropout", 0.0)
            ) if logit_steering_cfg else 0.0,
            latent_logit_steering_scale=float(
                getattr(logit_steering_cfg, "scale", 1.0)
            ) if logit_steering_cfg else 1.0,
            latent_logit_steering_generation_scale=float(
                getattr(logit_steering_cfg, "generation_scale", 1.0)
            ) if logit_steering_cfg else 1.0,
            latent_logit_steering_pooling=str(
                getattr(logit_steering_cfg, "pooling", "attention")
            ) if logit_steering_cfg else "attention",
            latent_logit_steering_max_bias_norm=float(
                getattr(logit_steering_cfg, "max_bias_norm", 0.0)
            ) if logit_steering_cfg else 0.0,
            latent_logit_steering_answer_token_weight=float(
                getattr(logit_steering_cfg, "answer_token_weight", 1.0)
            ) if logit_steering_cfg else 1.0,
            latent_logit_steering_later_answer_token_weight=float(
                getattr(logit_steering_cfg, "later_answer_token_weight", 1.0)
            ) if logit_steering_cfg else 1.0,
            latent_logit_steering_eos_weight=float(
                getattr(logit_steering_cfg, "eos_weight", 1.0)
            ) if logit_steering_cfg else 1.0,
            latent_token_decoder_enabled=bool(
                getattr(token_decoder_cfg, "enabled", False)
            ) if token_decoder_cfg else False,
            latent_token_decoder_rank=int(
                getattr(token_decoder_cfg, "rank", 64)
            ) if token_decoder_cfg else 64,
            latent_token_decoder_vocabulary_mode=str(
                getattr(token_decoder_cfg, "vocabulary_mode", "low_rank")
            ) if token_decoder_cfg else "low_rank",
            latent_token_decoder_lr_multiplier=float(
                getattr(token_decoder_cfg, "lr_multiplier", 1.0)
            ) if token_decoder_cfg else 1.0,
            latent_token_decoder_output_steps=int(
                getattr(token_decoder_cfg, "output_steps", 8)
            ) if token_decoder_cfg else 8,
            latent_token_decoder_dropout=float(
                getattr(token_decoder_cfg, "dropout", 0.0)
            ) if token_decoder_cfg else 0.0,
            latent_token_decoder_scale=float(
                getattr(token_decoder_cfg, "scale", 1.0)
            ) if token_decoder_cfg else 1.0,
            latent_token_decoder_pooling=str(
                getattr(token_decoder_cfg, "pooling", "attention")
            ) if token_decoder_cfg else "attention",
            latent_token_decoder_max_bias_norm=float(
                getattr(token_decoder_cfg, "max_bias_norm", 0.0)
            ) if token_decoder_cfg else 0.0,
            latent_token_decoder_candidate_token_mask=bool(
                getattr(token_decoder_cfg, "candidate_token_mask", False)
            ) if token_decoder_cfg else False,
            latent_token_decoder_require_ready=bool(
                getattr(token_decoder_cfg, "require_ready", False)
            ) if token_decoder_cfg else False,
            latent_token_decoder_eos_weight=float(
                getattr(token_decoder_cfg, "eos_weight", 1.0)
            ) if token_decoder_cfg else 1.0,
            latent_token_decoder_margin=float(
                getattr(token_decoder_cfg, "margin", 0.0)
            ) if token_decoder_cfg else 0.0,
            hidden_state_processor_enabled=bool(getattr(processor_cfg, "enabled", False)) if processor_cfg else False,
            hidden_state_processor_num_heads=int(getattr(processor_cfg, "num_heads", 4)) if processor_cfg else 4,
            hidden_state_processor_dropout=float(getattr(processor_cfg, "dropout", 0.0)) if processor_cfg else 0.0,
        )


def _coerce_history_value(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return str(value.detach().cpu().tolist())
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return value


def _numeric_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _raw_decode_ready_for_early_stop(metrics: Mapping[str, Any]) -> bool:
    eval_samples = int(float(metrics.get("heldout_eval_samples", 0.0) or 0.0))
    if eval_samples < 1:
        return False
    raw_accuracy = float(metrics.get("heldout_raw_decode_exact_match_accuracy", 0.0) or 0.0)
    raw_extraction_rate = float(
        metrics.get("heldout_raw_decode_answer_extraction_rate_percentage", 0.0) or 0.0
    )
    unique_count = int(
        float(metrics.get("heldout_raw_decode_unique_predicted_answer_count", 0.0) or 0.0)
    )
    return (
        raw_accuracy >= 100.0
        and raw_extraction_rate >= 100.0
        and (eval_samples <= 1 or unique_count > 1)
        and _enabled_decode_surface_ready(
            metrics,
            enabled_key="heldout_actor_semantic_bridge_decode_enabled",
            accuracy_key="heldout_actor_semantic_bridge_decode_accuracy",
            extraction_key="heldout_actor_semantic_bridge_decode_answer_extraction_rate_percentage",
            unique_key="heldout_actor_semantic_bridge_decode_unique_predicted_answer_count",
            eval_samples=eval_samples,
            require_key=None,
        )
        and _enabled_decode_surface_ready(
            metrics,
            enabled_key="heldout_latent_token_decode_enabled",
            accuracy_key="heldout_latent_token_decode_accuracy",
            extraction_key="heldout_latent_token_decode_answer_extraction_rate_percentage",
            unique_key="heldout_latent_token_decode_unique_predicted_answer_count",
            eval_samples=eval_samples,
            require_key="heldout_latent_token_decode_require_ready",
        )
    )


def _enabled_decode_surface_ready(
    metrics: Mapping[str, Any],
    *,
    enabled_key: str,
    accuracy_key: str,
    extraction_key: str,
    unique_key: str,
    eval_samples: int,
    require_key: str | None,
) -> bool:
    enabled = bool(metrics.get(enabled_key, False))
    required = True if require_key is None else bool(metrics.get(require_key, False))
    if not enabled or not required:
        return True
    accuracy = float(metrics.get(accuracy_key, 0.0) or 0.0)
    extraction_rate = float(metrics.get(extraction_key, 0.0) or 0.0)
    unique_count = int(float(metrics.get(unique_key, 0.0) or 0.0))
    return (
        accuracy >= 100.0
        and extraction_rate >= 100.0
        and (eval_samples <= 1 or unique_count > 1)
    )


def _parameter_count(module: Optional[nn.Module]) -> int:
    if module is None:
        return 0
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _gradient_norm(module: Optional[nn.Module]) -> float:
    if module is None:
        return 0.0
    squared_norm = 0.0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        squared_norm += float(torch.sum(grad * grad).cpu().item())
    return squared_norm ** 0.5


def _module_update_norm(before: dict[str, torch.Tensor], module: Optional[nn.Module]) -> float:
    if module is None or not before:
        return 0.0
    squared_norm = 0.0
    for name, parameter in module.named_parameters():
        previous = before.get(name)
        if previous is None:
            continue
        delta = parameter.detach().cpu().float() - previous
        squared_norm += float(torch.sum(delta * delta).item())
    return squared_norm ** 0.5


def _snapshot_module_parameters(module: Optional[nn.Module]) -> dict[str, torch.Tensor]:
    if module is None:
        return {}
    return {
        name: parameter.detach().cpu().float().clone()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


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


def _extract_answer_batch(batch: dict[str, Any], *, expected_count: int) -> list[str | None]:
    answers = batch.get("answers")
    if answers is None:
        return [None] * expected_count
    if isinstance(answers, str) or answers is None:
        answers = [answers]
    result = [None if answer is None else str(answer) for answer in answers]
    if len(result) != expected_count:
        raise ValueError(
            "train_reasoner_stage2 expected the answers batch to match the texts batch; "
            f"got {len(result)} answers for {expected_count} texts"
        )
    return result


def _extract_prompt_batch(
    batch: dict[str, Any],
    *,
    fallback_texts: Sequence[str],
) -> list[str]:
    prompts = batch.get("prompts")
    if prompts is None:
        return [str(text) for text in fallback_texts]
    if isinstance(prompts, str) or prompts is None:
        prompts = [prompts]
    if len(prompts) != len(fallback_texts):
        raise ValueError(
            "train_reasoner_stage2 expected the prompts batch to match the texts batch; "
            f"got {len(prompts)} prompts for {len(fallback_texts)} texts"
        )
    return [
        str(prompt) if prompt is not None and str(prompt).strip() else str(fallback_text)
        for prompt, fallback_text in zip(prompts, fallback_texts)
    ]


def _normalized_text(value: str | None) -> str | None:
    if value is None:
        return None
    compact = "".join(str(value).strip().casefold().split())
    return compact or None


def _extract_candidate_answers(
    batch: dict[str, Any],
    *,
    answers: Sequence[str | None],
) -> tuple[str, ...]:
    raw_candidates = batch.get("answer_candidates")
    candidates: list[str] = []
    if raw_candidates is None:
        raw_candidates = answers
    if isinstance(raw_candidates, str):
        raw_candidates = [raw_candidates]
    for item in raw_candidates:
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            iterable = item
        else:
            iterable = [item]
        for candidate in iterable:
            if candidate is None:
                continue
            candidate_text = str(candidate).strip()
            if candidate_text:
                candidates.append(candidate_text)
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        normalized = _normalized_text(candidate)
        if normalized is None or normalized in seen:
            continue
        unique.append(candidate)
        seen.add(normalized)
    return tuple(unique)


def _encode_token_ids(tokenizer: Any, text: str, *, max_length: int | None = None) -> list[int]:
    if hasattr(tokenizer, "encode"):
        try:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            token_ids = tokenizer.encode(text)
    else:
        encoded = tokenizer(
            text,
            padding=False,
            truncation=max_length is not None,
            max_length=max_length,
            return_tensors="pt",
        )
        token_ids = encoded["input_ids"][0].tolist()
    token_ids = [int(token_id) for token_id in token_ids]
    if max_length is not None:
        return token_ids[:max_length]
    return token_ids


def _format_answer_continuation(answer: str | None, *, suffix_text: str) -> str | None:
    if answer is None:
        return None
    cleaned = str(answer).strip()
    if not cleaned:
        return None
    separator = "" if suffix_text.endswith((" ", "\n", "\t")) else " "
    return f"{separator}{cleaned}"


def _collect_answer_first_tokens(
    *,
    actor_tokenizer: Any,
    answers: Sequence[str | None],
    suffix_text: str,
    max_answer_length: int,
) -> tuple[list[int], list[int]]:
    valid_indices: list[int] = []
    first_token_ids: list[int] = []
    for index, answer in enumerate(answers):
        continuation = _format_answer_continuation(answer, suffix_text=suffix_text)
        if continuation is None:
            continue
        token_ids = _encode_token_ids(
            actor_tokenizer,
            continuation,
            max_length=max(1, int(max_answer_length)),
        )
        if not token_ids:
            continue
        valid_indices.append(index)
        first_token_ids.append(int(token_ids[0]))
    return valid_indices, first_token_ids


def _compute_latent_answer_loss(
    *,
    actor_model: AutoModelForCausalLM,
    actor_tokenizer: Any,
    latent_prefix: torch.Tensor,
    answers: Sequence[str | None],
    suffix_text: str,
    max_answer_length: int,
    first_token_weight: float,
) -> tuple[torch.Tensor | None, int, float]:
    valid_indices: list[int] = []
    encoded_answers: list[list[int]] = []
    encoded_steering_targets: list[list[int]] = []
    eos_token_id = getattr(actor_tokenizer, "eos_token_id", None)
    for index, answer in enumerate(answers):
        continuation = _format_answer_continuation(answer, suffix_text=suffix_text)
        if continuation is None:
            continue
        token_ids = _encode_token_ids(
            actor_tokenizer,
            continuation,
            max_length=max(1, int(max_answer_length)),
        )
        if not token_ids:
            continue
        valid_indices.append(index)
        encoded_answers.append(token_ids)

    if not encoded_answers:
        return None, 0, 0.0

    actor_device = _model_device(actor_model)
    embedding = actor_model.get_input_embeddings()
    prefix = latent_prefix.index_select(
        0,
        torch.tensor(valid_indices, dtype=torch.long, device=latent_prefix.device),
    ).to(device=actor_device, dtype=embedding.weight.dtype)
    suffix_ids = _encode_token_ids(actor_tokenizer, suffix_text)
    suffix_width = len(suffix_ids)
    answer_width = max(len(token_ids) for token_ids in encoded_answers)
    pad_token_id = int(
        getattr(
            actor_tokenizer,
            "pad_token_id",
            getattr(actor_tokenizer, "eos_token_id", 0) or 0,
        )
        or 0
    )

    suffix_tensor = torch.tensor(
        suffix_ids,
        dtype=torch.long,
        device=actor_device,
    ).unsqueeze(0).expand(prefix.shape[0], suffix_width)
    answer_tensor = torch.full(
        (prefix.shape[0], answer_width),
        pad_token_id,
        dtype=torch.long,
        device=actor_device,
    )
    answer_mask = torch.zeros(
        (prefix.shape[0], answer_width),
        dtype=torch.long,
        device=actor_device,
    )
    token_weights = torch.zeros(
        (prefix.shape[0], answer_width),
        dtype=torch.float32,
        device=actor_device,
    )
    for row_index, token_ids in enumerate(encoded_answers):
        width = len(token_ids)
        answer_tensor[row_index, :width] = torch.tensor(token_ids, dtype=torch.long, device=actor_device)
        answer_mask[row_index, :width] = 1
        token_weights[row_index, :width] = 1.0
        token_weights[row_index, 0] = max(1.0, float(first_token_weight))

    suffix_embeds = embedding(suffix_tensor) if suffix_width else prefix[:, :0, :]
    answer_embeds = embedding(answer_tensor)
    inputs_embeds = torch.cat([prefix, suffix_embeds, answer_embeds], dim=1)
    attention_mask = torch.cat(
        [
            torch.ones(prefix.shape[:2], dtype=torch.long, device=actor_device),
            torch.ones((prefix.shape[0], suffix_width), dtype=torch.long, device=actor_device),
            answer_mask,
        ],
        dim=1,
    )
    outputs = actor_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )

    answer_start = prefix.shape[1] + suffix_width
    flat_logits: list[torch.Tensor] = []
    flat_targets: list[torch.Tensor] = []
    flat_weights: list[torch.Tensor] = []
    for row_index, token_ids in enumerate(encoded_answers):
        width = len(token_ids)
        prediction_positions = torch.arange(
            answer_start - 1,
            answer_start + width - 1,
            dtype=torch.long,
            device=actor_device,
        )
        flat_logits.append(outputs.logits[row_index].index_select(0, prediction_positions))
        flat_targets.append(answer_tensor[row_index, :width])
        flat_weights.append(token_weights[row_index, :width])

    logits = torch.cat(flat_logits, dim=0).float()
    targets = torch.cat(flat_targets, dim=0)
    weights = torch.cat(flat_weights, dim=0)
    token_losses = F.cross_entropy(logits, targets, reduction="none")
    answer_loss = (token_losses * weights).sum() / weights.sum().clamp_min(1.0)
    average_answer_tokens = sum(len(token_ids) for token_ids in encoded_answers) / len(encoded_answers)
    return answer_loss, len(encoded_answers), float(average_answer_tokens)


def _compute_latent_first_token_loss(
    *,
    actor_model: AutoModelForCausalLM,
    actor_tokenizer: Any,
    latent_prefix: torch.Tensor,
    answers: Sequence[str | None],
    suffix_text: str,
    max_answer_length: int,
    margin: float,
) -> tuple[torch.Tensor | None, int, float, float, float]:
    valid_indices, first_token_ids = _collect_answer_first_tokens(
        actor_tokenizer=actor_tokenizer,
        answers=answers,
        suffix_text=suffix_text,
        max_answer_length=max_answer_length,
    )
    if not first_token_ids:
        return None, 0, 0.0, 0.0, 0.0

    actor_device = _model_device(actor_model)
    embedding = actor_model.get_input_embeddings()
    prefix = latent_prefix.index_select(
        0,
        torch.tensor(valid_indices, dtype=torch.long, device=latent_prefix.device),
    ).to(device=actor_device, dtype=embedding.weight.dtype)
    suffix_ids = _encode_token_ids(actor_tokenizer, suffix_text)
    suffix_width = len(suffix_ids)
    if suffix_width:
        suffix_tensor = torch.tensor(
            suffix_ids,
            dtype=torch.long,
            device=actor_device,
        ).unsqueeze(0).expand(prefix.shape[0], suffix_width)
        suffix_embeds = embedding(suffix_tensor)
    else:
        suffix_embeds = prefix[:, :0, :]
    inputs_embeds = torch.cat([prefix, suffix_embeds], dim=1)
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=actor_device)
    outputs = actor_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )

    logits = outputs.logits[:, -1, :].float()
    targets = torch.tensor(first_token_ids, dtype=torch.long, device=actor_device)
    ce_loss = F.cross_entropy(logits, targets)
    target_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    target_mask = torch.zeros_like(logits, dtype=torch.bool)
    target_mask.scatter_(1, targets.unsqueeze(1), True)
    other_logits = logits.masked_fill(target_mask, float("-inf")).max(dim=-1).values
    margin_loss = F.relu(float(margin) - (target_logits - other_logits)).mean()
    first_token_loss = ce_loss + margin_loss
    predictions = logits.argmax(dim=-1)
    accuracy = float((predictions == targets).float().mean().detach().cpu().item() * 100.0)
    ranks = 1 + (logits > target_logits.unsqueeze(1)).sum(dim=-1).float()
    average_rank = float(ranks.mean().detach().cpu().item())
    average_margin = float((target_logits - other_logits).mean().detach().cpu().item())
    return first_token_loss, len(first_token_ids), accuracy, average_rank, average_margin


def _compute_latent_logit_steering_loss(
    *,
    latent_logit_steering: LatentLogitSteeringHead | None,
    actor_model: AutoModelForCausalLM,
    actor_tokenizer: Any,
    latent_prefix: torch.Tensor,
    answers: Sequence[str | None],
    suffix_text: str,
    max_answer_length: int,
    margin: float,
    answer_token_weight: float = 1.0,
    later_answer_token_weight: float = 1.0,
    eos_weight: float = 1.0,
) -> tuple[torch.Tensor | None, int, float, float, float, float]:
    if latent_logit_steering is None:
        return None, 0, 0.0, 0.0, 0.0, 0.0
    valid_indices: list[int] = []
    encoded_answers: list[list[int]] = []
    encoded_steering_targets: list[list[int]] = []
    eos_token_id = getattr(actor_tokenizer, "eos_token_id", None)
    for index, answer in enumerate(answers):
        continuation = _format_answer_continuation(answer, suffix_text=suffix_text)
        if continuation is None:
            continue
        token_ids = _encode_token_ids(
            actor_tokenizer,
            continuation,
            max_length=max(1, int(max_answer_length)),
        )
        if not token_ids:
            continue
        valid_indices.append(index)
        encoded_answers.append(token_ids)
        steering_targets = list(token_ids)
        if eos_token_id is not None:
            steering_targets.append(int(eos_token_id))
        encoded_steering_targets.append(steering_targets)
    if not encoded_answers:
        return None, 0, 0.0, 0.0, 0.0, 0.0

    actor_device = _model_device(actor_model)
    steering_device = next(latent_logit_steering.parameters()).device
    embedding = actor_model.get_input_embeddings()
    row_index_tensor = torch.tensor(valid_indices, dtype=torch.long, device=latent_prefix.device)
    selected_prefix = latent_prefix.index_select(0, row_index_tensor)
    prefix_for_actor = selected_prefix.to(device=actor_device, dtype=embedding.weight.dtype)
    suffix_ids = _encode_token_ids(actor_tokenizer, suffix_text)
    suffix_width = len(suffix_ids)
    answer_width = max(len(token_ids) for token_ids in encoded_answers)
    pad_token_id = int(
        getattr(
            actor_tokenizer,
            "pad_token_id",
            getattr(actor_tokenizer, "eos_token_id", 0) or 0,
        )
        or 0
    )
    if suffix_width:
        suffix_tensor = torch.tensor(
            suffix_ids,
            dtype=torch.long,
            device=actor_device,
        ).unsqueeze(0).expand(prefix_for_actor.shape[0], suffix_width)
        suffix_embeds = embedding(suffix_tensor)
    else:
        suffix_embeds = prefix_for_actor[:, :0, :]
    answer_tensor = torch.full(
        (prefix_for_actor.shape[0], answer_width),
        pad_token_id,
        dtype=torch.long,
        device=actor_device,
    )
    answer_mask = torch.zeros(
        (prefix_for_actor.shape[0], answer_width),
        dtype=torch.long,
        device=actor_device,
    )
    for row_index, token_ids in enumerate(encoded_answers):
        answer_tensor[row_index, : len(token_ids)] = torch.tensor(
            token_ids,
            dtype=torch.long,
            device=actor_device,
        )
        answer_mask[row_index, : len(token_ids)] = 1
    answer_embeds = embedding(answer_tensor)
    inputs_embeds = torch.cat([prefix_for_actor, suffix_embeds, answer_embeds], dim=1)
    attention_mask = torch.cat(
        [
            torch.ones(prefix_for_actor.shape[:2], dtype=torch.long, device=actor_device),
            torch.ones((prefix_for_actor.shape[0], suffix_width), dtype=torch.long, device=actor_device),
            answer_mask,
        ],
        dim=1,
    )
    with torch.no_grad():
        base_logits = actor_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).logits.float()

    vocab_weight = lm_vocabulary_weight(actor_model)
    max_steering_steps = min(
        max(len(token_ids) for token_ids in encoded_steering_targets),
        int(getattr(latent_logit_steering, "output_steps", 1)),
    )
    logits_bias = latent_logit_steering.forward_sequence(
        selected_prefix.to(device=steering_device, dtype=vocab_weight.dtype),
        vocab_weight.to(device=steering_device),
        output_steps=max_steering_steps,
    ).to(device=actor_device, dtype=base_logits.dtype)
    answer_start = prefix_for_actor.shape[1] + suffix_width
    flat_logits: list[torch.Tensor] = []
    flat_targets: list[torch.Tensor] = []
    flat_weights: list[torch.Tensor] = []
    for row_index, token_ids in enumerate(encoded_steering_targets):
        width = min(len(token_ids), max_steering_steps)
        for offset in range(width):
            flat_logits.append(
                base_logits[row_index, answer_start + offset - 1, :] + logits_bias[row_index, offset, :]
            )
            target_id = int(token_ids[offset])
            flat_targets.append(torch.tensor(target_id, dtype=torch.long, device=actor_device))
            is_eos_target = eos_token_id is not None and target_id == int(eos_token_id)
            if is_eos_target:
                target_weight = float(eos_weight)
            elif offset > 0:
                target_weight = float(later_answer_token_weight)
            else:
                target_weight = float(answer_token_weight)
            flat_weights.append(
                torch.tensor(max(0.0, target_weight), dtype=torch.float32, device=actor_device)
            )
    if not flat_logits:
        return None, 0, 0.0, 0.0, 0.0, 0.0
    logits = torch.stack(flat_logits).float()
    targets = torch.stack(flat_targets)
    weights = torch.stack(flat_weights).clamp_min(1.0e-6)
    token_losses = F.cross_entropy(logits, targets, reduction="none")
    ce_loss = (token_losses * weights).sum() / weights.sum().clamp_min(1.0)
    target_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    target_mask = torch.zeros_like(logits, dtype=torch.bool)
    target_mask.scatter_(1, targets.unsqueeze(1), True)
    other_logits = logits.masked_fill(target_mask, float("-inf")).max(dim=-1).values
    margin_values = F.relu(float(margin) - (target_logits - other_logits))
    margin_loss = (margin_values * weights).sum() / weights.sum().clamp_min(1.0)
    steering_loss = ce_loss + margin_loss
    predictions = logits.argmax(dim=-1)
    accuracy = float((predictions == targets).float().mean().detach().cpu().item() * 100.0)
    ranks = 1 + (logits > target_logits.unsqueeze(1)).sum(dim=-1).float()
    average_rank = float(ranks.mean().detach().cpu().item())
    average_margin = float((target_logits - other_logits).mean().detach().cpu().item())
    bias_norm = float(logits_bias.detach().float().norm(dim=-1).mean().cpu().item())
    return steering_loss, len(encoded_answers), accuracy, average_rank, average_margin, bias_norm


def _encode_latent_token_decoder_targets(
    *,
    actor_tokenizer: Any,
    answers: Sequence[str | None],
    max_answer_length: int,
    output_steps: int,
) -> tuple[list[int], list[list[int]], list[int]]:
    valid_indices: list[int] = []
    encoded_targets: list[list[int]] = []
    eos_positions: list[int] = []
    eos_token_id = getattr(actor_tokenizer, "eos_token_id", None)
    max_width = max(1, min(int(max_answer_length), int(output_steps)))
    answer_width = max_width if eos_token_id is None else max(1, max_width - 1)
    for index, answer in enumerate(answers):
        if answer is None:
            continue
        answer_text = str(answer).strip()
        if not answer_text:
            continue
        token_ids = _encode_token_ids(
            actor_tokenizer,
            answer_text,
            max_length=answer_width,
        )
        if not token_ids:
            continue
        if eos_token_id is not None and len(token_ids) < max_width:
            eos_positions.append(len(token_ids))
            token_ids = [*token_ids, int(eos_token_id)]
        else:
            eos_positions.append(-1)
        valid_indices.append(index)
        encoded_targets.append(token_ids[:max_width])
    return valid_indices, encoded_targets, eos_positions


def _latent_token_decoder_allowed_token_ids(
    *,
    actor_tokenizer: Any,
    answers: Sequence[str | None],
    candidate_answers: Sequence[str],
    max_answer_length: int,
    output_steps: int,
) -> set[int]:
    candidates = list(candidate_answers) if candidate_answers else []
    candidates.extend(answer for answer in answers if answer is not None)
    _, encoded_targets, _ = _encode_latent_token_decoder_targets(
        actor_tokenizer=actor_tokenizer,
        answers=candidates,
        max_answer_length=max_answer_length,
        output_steps=output_steps,
    )
    allowed_ids = {
        int(token_id)
        for token_ids in encoded_targets
        for token_id in token_ids
    }
    eos_token_id = getattr(actor_tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        allowed_ids.add(int(eos_token_id))
    return allowed_ids


def _mask_logits_to_allowed_token_ids(
    logits: torch.Tensor,
    allowed_token_ids: set[int],
) -> torch.Tensor:
    if not allowed_token_ids:
        return logits
    allowed = torch.tensor(
        sorted(allowed_token_ids),
        dtype=torch.long,
        device=logits.device,
    )
    mask = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
    mask.scatter_(0, allowed.clamp_min(0).clamp_max(logits.shape[-1] - 1), True)
    return logits.masked_fill(~mask.view(*((1,) * (logits.dim() - 1)), -1), -1.0e9)


def _compute_latent_token_decoder_loss(
    *,
    latent_token_decoder: LatentTokenDecoderHead | None,
    actor_model: AutoModelForCausalLM,
    actor_tokenizer: Any,
    latent_prefix: torch.Tensor,
    answers: Sequence[str | None],
    candidate_answers: Sequence[str],
    max_answer_length: int,
    candidate_token_mask: bool,
    eos_weight: float,
    margin: float,
) -> tuple[torch.Tensor | None, int, float, float, float]:
    if latent_token_decoder is None:
        return None, 0, 0.0, 0.0, 0.0
    output_steps = int(getattr(latent_token_decoder, "output_steps", 1))
    valid_indices, encoded_targets, eos_positions = _encode_latent_token_decoder_targets(
        actor_tokenizer=actor_tokenizer,
        answers=answers,
        max_answer_length=max_answer_length,
        output_steps=output_steps,
    )
    if not encoded_targets:
        return None, 0, 0.0, 0.0, 0.0

    decoder_device = next(latent_token_decoder.parameters()).device
    vocab_weight = lm_vocabulary_weight(actor_model)
    row_index_tensor = torch.tensor(valid_indices, dtype=torch.long, device=latent_prefix.device)
    selected_prefix = latent_prefix.index_select(0, row_index_tensor)
    logits = latent_token_decoder.forward_sequence(
        selected_prefix.to(device=decoder_device, dtype=vocab_weight.dtype),
        vocab_weight.to(device=decoder_device),
        output_steps=output_steps,
    ).float()
    if candidate_token_mask:
        logits = _mask_logits_to_allowed_token_ids(
            logits,
            _latent_token_decoder_allowed_token_ids(
                actor_tokenizer=actor_tokenizer,
                answers=answers,
                candidate_answers=candidate_answers,
                max_answer_length=max_answer_length,
                output_steps=output_steps,
            ),
        )
    pad_target = -100
    targets = torch.full(
        (len(encoded_targets), output_steps),
        pad_target,
        dtype=torch.long,
        device=decoder_device,
    )
    weights = torch.zeros_like(targets, dtype=torch.float32)
    for row_index, token_ids in enumerate(encoded_targets):
        width = min(len(token_ids), output_steps)
        targets[row_index, :width] = torch.tensor(
            token_ids[:width],
            dtype=torch.long,
            device=decoder_device,
        )
        weights[row_index, :width] = 1.0
        eos_position = eos_positions[row_index]
        if 0 <= eos_position < output_steps:
            weights[row_index, eos_position] = max(1.0, float(eos_weight))

    flat_losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=pad_target,
        reduction="none",
    ).reshape(targets.shape)
    token_decoder_loss = (flat_losses * weights).sum() / weights.sum().clamp_min(1.0)
    valid_mask = targets != pad_target
    if float(margin) > 0.0 and bool(valid_mask.any()):
        valid_logits = logits[valid_mask]
        valid_targets = targets[valid_mask]
        target_logits = valid_logits.gather(1, valid_targets.unsqueeze(1)).squeeze(1)
        target_mask = torch.zeros_like(valid_logits, dtype=torch.bool)
        target_mask.scatter_(1, valid_targets.unsqueeze(1), True)
        other_logits = valid_logits.masked_fill(target_mask, float("-inf")).max(dim=-1).values
        token_decoder_loss = token_decoder_loss + F.relu(float(margin) - (target_logits - other_logits)).mean()

    predictions = logits.argmax(dim=-1)
    token_accuracy = float(
        (predictions[valid_mask] == targets[valid_mask]).float().mean().detach().cpu().item() * 100.0
    )
    sequence_matches = []
    for row_index in range(targets.shape[0]):
        row_mask = valid_mask[row_index]
        if not bool(row_mask.any()):
            continue
        sequence_matches.append(
            bool(torch.equal(predictions[row_index][row_mask], targets[row_index][row_mask]))
        )
    sequence_accuracy = (
        100.0 * sum(1 for item in sequence_matches if item) / len(sequence_matches)
        if sequence_matches
        else 0.0
    )
    average_target_tokens = sum(len(token_ids) for token_ids in encoded_targets) / len(encoded_targets)
    return (
        token_decoder_loss,
        len(encoded_targets),
        token_accuracy,
        float(sequence_accuracy),
        float(average_target_tokens),
    )


def _candidate_target_indices(
    answers: Sequence[str | None],
    candidate_answers: Sequence[str],
) -> tuple[list[int], list[int]]:
    if len(candidate_answers) < 2:
        return [], []

    candidate_norms = [_normalized_text(candidate) for candidate in candidate_answers]
    valid_rows: list[int] = []
    target_indices: list[int] = []
    for row_index, answer in enumerate(answers):
        target_norm = _normalized_text(answer)
        if target_norm is None:
            continue
        try:
            target_index = candidate_norms.index(target_norm)
        except ValueError:
            continue
        valid_rows.append(row_index)
        target_indices.append(target_index)
    return valid_rows, target_indices


def _compute_latent_candidate_contrast_loss(
    *,
    actor_model: AutoModelForCausalLM,
    actor_tokenizer: Any,
    latent_prefix: torch.Tensor,
    answers: Sequence[str | None],
    candidate_answers: Sequence[str],
    suffix_text: str,
    max_answer_length: int,
    temperature: float,
) -> tuple[torch.Tensor | None, int, float]:
    valid_rows, target_indices = _candidate_target_indices(answers, candidate_answers)
    if not valid_rows:
        return None, 0, 0.0

    actor_device = _model_device(actor_model)
    embedding = actor_model.get_input_embeddings()
    row_index_tensor = torch.tensor(valid_rows, dtype=torch.long, device=latent_prefix.device)
    selected_prefix = latent_prefix.index_select(0, row_index_tensor)
    pair_prefix = selected_prefix.repeat_interleave(len(candidate_answers), dim=0).to(
        device=actor_device,
        dtype=embedding.weight.dtype,
    )
    pair_answers = [
        candidate
        for _ in valid_rows
        for candidate in candidate_answers
    ]
    encoded_pair_answers: list[list[int]] = []
    for answer in pair_answers:
        continuation = _format_answer_continuation(answer, suffix_text=suffix_text)
        if continuation is None:
            encoded_pair_answers.append([])
            continue
        encoded_pair_answers.append(
            _encode_token_ids(
                actor_tokenizer,
                continuation,
                max_length=max(1, int(max_answer_length)),
            )
        )
    if any(not token_ids for token_ids in encoded_pair_answers):
        return None, 0, 0.0

    suffix_ids = _encode_token_ids(actor_tokenizer, suffix_text)
    suffix_width = len(suffix_ids)
    answer_width = max(len(token_ids) for token_ids in encoded_pair_answers)
    pad_token_id = int(
        getattr(
            actor_tokenizer,
            "pad_token_id",
            getattr(actor_tokenizer, "eos_token_id", 0) or 0,
        )
        or 0
    )
    suffix_tensor = torch.tensor(
        suffix_ids,
        dtype=torch.long,
        device=actor_device,
    ).unsqueeze(0).expand(pair_prefix.shape[0], suffix_width)
    answer_tensor = torch.full(
        (pair_prefix.shape[0], answer_width),
        pad_token_id,
        dtype=torch.long,
        device=actor_device,
    )
    answer_mask = torch.zeros(
        (pair_prefix.shape[0], answer_width),
        dtype=torch.long,
        device=actor_device,
    )
    for pair_index, token_ids in enumerate(encoded_pair_answers):
        answer_tensor[pair_index, : len(token_ids)] = torch.tensor(
            token_ids,
            dtype=torch.long,
            device=actor_device,
        )
        answer_mask[pair_index, : len(token_ids)] = 1

    suffix_embeds = embedding(suffix_tensor) if suffix_width else pair_prefix[:, :0, :]
    answer_embeds = embedding(answer_tensor)
    inputs_embeds = torch.cat([pair_prefix, suffix_embeds, answer_embeds], dim=1)
    attention_mask = torch.cat(
        [
            torch.ones(pair_prefix.shape[:2], dtype=torch.long, device=actor_device),
            torch.ones((pair_prefix.shape[0], suffix_width), dtype=torch.long, device=actor_device),
            answer_mask,
        ],
        dim=1,
    )
    outputs = actor_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )

    answer_start = pair_prefix.shape[1] + suffix_width
    pair_nlls: list[torch.Tensor] = []
    for pair_index, token_ids in enumerate(encoded_pair_answers):
        width = len(token_ids)
        prediction_positions = torch.arange(
            answer_start - 1,
            answer_start + width - 1,
            dtype=torch.long,
            device=actor_device,
        )
        logits = outputs.logits[pair_index].index_select(0, prediction_positions).float()
        targets = answer_tensor[pair_index, :width]
        pair_nlls.append(F.cross_entropy(logits, targets, reduction="mean"))
    nll_matrix = torch.stack(pair_nlls).reshape(len(valid_rows), len(candidate_answers))
    score_matrix = -nll_matrix / max(float(temperature), 1.0e-6)
    target_tensor = torch.tensor(target_indices, dtype=torch.long, device=actor_device)
    contrast_loss = F.cross_entropy(score_matrix, target_tensor)
    predictions = score_matrix.argmax(dim=-1)
    accuracy = float((predictions == target_tensor).float().mean().detach().cpu().item() * 100.0)
    return contrast_loss, len(valid_rows), accuracy


def _compute_latent_answer_probe_loss(
    *,
    latent_answer_probe: LatentAnswerProbe | None,
    latent_prefix: torch.Tensor,
    answers: Sequence[str | None],
    candidate_answers: Sequence[str],
) -> tuple[torch.Tensor | None, int, float]:
    if latent_answer_probe is None:
        return None, 0, 0.0
    if len(candidate_answers) > latent_answer_probe.max_candidates:
        raise ValueError(
            "latent_answer_probe.max_candidates is smaller than the candidate answer set: "
            f"{latent_answer_probe.max_candidates} < {len(candidate_answers)}"
        )

    valid_rows, target_indices = _candidate_target_indices(answers, candidate_answers)
    if not valid_rows:
        return None, 0, 0.0

    row_index_tensor = torch.tensor(valid_rows, dtype=torch.long, device=latent_prefix.device)
    selected_prefix = latent_prefix.index_select(0, row_index_tensor)
    logits = latent_answer_probe(selected_prefix)[:, : len(candidate_answers)]
    targets = torch.tensor(target_indices, dtype=torch.long, device=logits.device)
    probe_loss = F.cross_entropy(logits, targets)
    predictions = logits.argmax(dim=-1)
    accuracy = float((predictions == targets).float().mean().detach().cpu().item() * 100.0)
    return probe_loss, len(valid_rows), accuracy


def _answer_token_variants(
    *,
    actor_tokenizer: Any,
    answer: str | None,
    suffix_text: str,
    max_answer_length: int,
    answer_variants: Sequence[str | None] = (),
) -> list[list[int]]:
    token_variants: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for candidate in (answer, *answer_variants):
        continuation = _format_answer_continuation(candidate, suffix_text=suffix_text)
        if continuation is None:
            continue
        token_ids = _encode_token_ids(
            actor_tokenizer,
            continuation,
            max_length=max(1, int(max_answer_length)),
        )
        if not token_ids:
            continue
        key = tuple(token_ids)
        if key in seen:
            continue
        seen.add(key)
        token_variants.append(token_ids)
    return token_variants


def _target_answer_token_sequences(
    *,
    actor_tokenizer: Any,
    answers: Sequence[str | None],
    suffix_text: str,
    max_answer_length: int,
) -> tuple[list[int], list[list[int]]]:
    valid_indices: list[int] = []
    encoded_answers: list[list[int]] = []
    for index, answer in enumerate(answers):
        token_variants = _answer_token_variants(
            actor_tokenizer=actor_tokenizer,
            answer=answer,
            suffix_text=suffix_text,
            max_answer_length=max_answer_length,
        )
        if not token_variants:
            continue
        valid_indices.append(index)
        encoded_answers.append(token_variants[0])
    return valid_indices, encoded_answers


def _compute_latent_sequence_decoder_loss(
    *,
    latent_sequence_decoder: LatentSequenceDecoderHead | None,
    actor_model: AutoModelForCausalLM,
    actor_tokenizer: Any,
    latent_prefix: torch.Tensor,
    answers: Sequence[str | None],
    suffix_text: str,
    max_answer_length: int,
) -> tuple[torch.Tensor | None, int, float, float, float, float, float]:
    if latent_sequence_decoder is None:
        return None, 0, 0.0, 0.0, 0.0, 0.0, 0.0
    valid_indices, encoded_answers = _target_answer_token_sequences(
        actor_tokenizer=actor_tokenizer,
        answers=answers,
        suffix_text=suffix_text,
        max_answer_length=min(
            int(max_answer_length),
            int(latent_sequence_decoder.max_answer_length),
        ),
    )
    if not encoded_answers:
        return None, 0, 0.0, 0.0, 0.0, 0.0, 0.0

    decoder_device = next(latent_sequence_decoder.parameters()).device
    row_index_tensor = torch.tensor(valid_indices, dtype=torch.long, device=latent_prefix.device)
    selected_prefix = latent_prefix.index_select(0, row_index_tensor).to(decoder_device)
    vocabulary_weight = lm_vocabulary_weight(actor_model).detach().to(decoder_device)
    decoder_outputs = latent_sequence_decoder(selected_prefix, vocabulary_weight)
    token_logits = decoder_outputs["token_logits"].float()
    length_logits = decoder_outputs["length_logits"].float()
    output_steps = int(token_logits.shape[1])
    pad_target = -100
    token_targets = torch.full(
        (len(encoded_answers), output_steps),
        pad_target,
        dtype=torch.long,
        device=decoder_device,
    )
    token_mask = torch.zeros(
        (len(encoded_answers), output_steps),
        dtype=torch.bool,
        device=decoder_device,
    )
    length_targets = torch.empty(len(encoded_answers), dtype=torch.long, device=decoder_device)
    for row_index, token_ids in enumerate(encoded_answers):
        width = min(len(token_ids), output_steps)
        token_targets[row_index, :width] = torch.tensor(
            token_ids[:width],
            dtype=torch.long,
            device=decoder_device,
        )
        token_mask[row_index, :width] = True
        length_targets[row_index] = width

    token_loss = F.cross_entropy(
        token_logits.reshape(-1, token_logits.shape[-1]),
        token_targets.reshape(-1),
        ignore_index=pad_target,
    )
    length_loss = F.cross_entropy(length_logits, length_targets)
    decoder_loss = token_loss + length_loss
    predicted_tokens = token_logits.argmax(dim=-1)
    predicted_lengths = length_logits.argmax(dim=-1).clamp(0, output_steps)
    token_correct = (
        (predicted_tokens == token_targets).masked_select(token_mask).float().sum()
        if token_mask.any()
        else token_logits.sum() * 0.0
    )
    token_count = int(token_mask.sum().detach().cpu().item())
    sequence_correct_count = 0
    for row_index, token_ids in enumerate(encoded_answers):
        target_width = int(length_targets[row_index].detach().cpu().item())
        predicted_width = int(predicted_lengths[row_index].detach().cpu().item())
        if predicted_width != target_width:
            continue
        predicted_sequence = predicted_tokens[row_index, :predicted_width].detach().cpu().tolist()
        if predicted_sequence == token_ids[:target_width]:
            sequence_correct_count += 1
    token_accuracy = 0.0 if token_count == 0 else float(token_correct.detach().cpu().item() * 100.0 / token_count)
    sequence_accuracy = 100.0 * sequence_correct_count / len(encoded_answers)
    length_accuracy = float((predicted_lengths == length_targets).float().mean().detach().cpu().item() * 100.0)
    target_length_mean = sum(len(token_ids) for token_ids in encoded_answers) / len(encoded_answers)
    predicted_length_mean = float(predicted_lengths.float().mean().detach().cpu().item())
    return (
        decoder_loss,
        len(encoded_answers),
        token_accuracy,
        sequence_accuracy,
        length_accuracy,
        float(target_length_mean),
        predicted_length_mean,
    )


def evaluate_latent_sequence_decoder_predictions(
    *,
    latent_sequence_decoder: LatentSequenceDecoderHead | None,
    actor_model: AutoModelForCausalLM,
    actor_tokenizer: Any,
    latent_prefix: torch.Tensor,
    answers: Sequence[str | None],
    suffix_text: str,
    max_answer_length: int,
    answer_variants: Sequence[Sequence[str | None]] | None = None,
) -> dict[str, Any]:
    if latent_sequence_decoder is None:
        return {
            "sample_count": 0,
            "token_count": 0,
            "token_correct_count": 0,
            "sequence_correct_count": 0,
            "length_correct_count": 0,
            "token_accuracy": None,
            "sequence_accuracy": None,
            "length_accuracy": None,
            "predicted_texts": [],
            "predicted_token_ids": [],
            "predicted_lengths": [],
        }

    max_length = min(int(max_answer_length), int(latent_sequence_decoder.max_answer_length))
    target_variants: list[list[list[int]]] = []
    valid_indices: list[int] = []
    for index, answer in enumerate(answers):
        variants = () if answer_variants is None else tuple(answer_variants[index])
        token_variants = _answer_token_variants(
            actor_tokenizer=actor_tokenizer,
            answer=answer,
            suffix_text=suffix_text,
            max_answer_length=max_length,
            answer_variants=variants,
        )
        if not token_variants:
            continue
        valid_indices.append(index)
        target_variants.append(token_variants)
    if not valid_indices:
        return {
            "sample_count": 0,
            "token_count": 0,
            "token_correct_count": 0,
            "sequence_correct_count": 0,
            "length_correct_count": 0,
            "token_accuracy": None,
            "sequence_accuracy": None,
            "length_accuracy": None,
            "predicted_texts": [None for _ in answers],
            "predicted_token_ids": [[] for _ in answers],
            "predicted_lengths": [0 for _ in answers],
        }

    decoder_device = next(latent_sequence_decoder.parameters()).device
    row_index_tensor = torch.tensor(valid_indices, dtype=torch.long, device=latent_prefix.device)
    selected_prefix = latent_prefix.index_select(0, row_index_tensor).to(decoder_device)
    vocabulary_weight = lm_vocabulary_weight(actor_model).detach().to(decoder_device)
    with torch.no_grad():
        decoder_outputs = latent_sequence_decoder(selected_prefix, vocabulary_weight)
        token_logits = decoder_outputs["token_logits"].float()
        length_logits = decoder_outputs["length_logits"].float()
        predicted_tokens = token_logits.argmax(dim=-1)
        predicted_lengths = length_logits.argmax(dim=-1).clamp(0, token_logits.shape[1])

    predicted_texts: list[str | None] = [None for _ in answers]
    predicted_token_ids: list[list[int]] = [[] for _ in answers]
    predicted_length_values: list[int] = [0 for _ in answers]
    token_count = 0
    token_correct_count = 0
    sequence_correct_count = 0
    length_correct_count = 0
    for local_index, source_index in enumerate(valid_indices):
        predicted_length = int(predicted_lengths[local_index].detach().cpu().item())
        token_ids = predicted_tokens[local_index, :predicted_length].detach().cpu().tolist()
        predicted_token_ids[source_index] = [int(token_id) for token_id in token_ids]
        predicted_length_values[source_index] = predicted_length
        predicted_texts[source_index] = (
            actor_tokenizer.decode(token_ids, skip_special_tokens=True)
            if token_ids and hasattr(actor_tokenizer, "decode")
            else ""
        )

        best_token_correct = -1
        best_target_length = len(target_variants[local_index][0])
        sequence_correct = False
        length_correct = False
        for target_ids in target_variants[local_index]:
            target_length = len(target_ids)
            overlap = min(predicted_length, target_length)
            matches = sum(
                1
                for left, right in zip(token_ids[:overlap], target_ids[:overlap])
                if int(left) == int(right)
            )
            exact = predicted_length == target_length and token_ids == target_ids
            if exact:
                sequence_correct = True
            if predicted_length == target_length:
                length_correct = True
            if matches > best_token_correct:
                best_token_correct = matches
                best_target_length = target_length
        token_correct_count += max(0, best_token_correct)
        token_count += best_target_length
        sequence_correct_count += int(sequence_correct)
        length_correct_count += int(length_correct)

    sample_count = len(valid_indices)
    return {
        "sample_count": sample_count,
        "token_count": token_count,
        "token_correct_count": token_correct_count,
        "sequence_correct_count": sequence_correct_count,
        "length_correct_count": length_correct_count,
        "token_accuracy": None if token_count == 0 else 100.0 * token_correct_count / token_count,
        "sequence_accuracy": 100.0 * sequence_correct_count / sample_count,
        "length_accuracy": 100.0 * length_correct_count / sample_count,
        "predicted_texts": predicted_texts,
        "predicted_token_ids": predicted_token_ids,
        "predicted_lengths": predicted_length_values,
    }


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


def freeze_reasoner(reasoner_model: AutoModelForCausalLM) -> None:
    reasoner_model.eval()
    for parameter in reasoner_model.parameters():
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
) -> list[dict[str, Any]]:
    """
    Stage II Interlat-style training loop:
    - Freeze Actor (Agent B) completely.
    - Update only Reasoner (Agent A).
    
    This loop optimizes Agent A to generate continuous ODE thoughts that Agent B can decode zero-shot.
    It passes the trajectory through the Orthogonal Procrustes bridge and updates Agent A's weights 
    using the LatentCompressorLoss (Task Utility + Preference KL Divergence + Geometric Cosine Similarity).
    
    Args:
        reasoner_model: Agent A, whose parameters will be updated.
        actor_model: Agent B, whose parameters are strictly frozen (requires_grad = False).
        train_dataloader: Stream of math/coding problems and expert Chain-of-Thought text targets.
        config: Hyperparameters including loss lambdas (`lambda_task`, `lambda_pref`, `lambda_geom`).
        
    Returns:
        A list of diagnostic dictionaries containing loss curves and metrics for plotting (e.g., WandB).
    """
    if config.wandb_enabled:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            config=dataclasses.asdict(config),
        )

    freeze_actor(actor_model)
    if config.train_reasoner:
        reasoner_model.train()
    else:
        freeze_reasoner(reasoner_model)

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
    latent_handoff_adapter = (
        LatentHandoffAdapter(
            actor_hidden_size,
            rank=config.latent_handoff_adapter_rank,
            scale=config.latent_handoff_adapter_scale,
            dropout=config.latent_handoff_adapter_dropout,
        )
        if config.latent_handoff_adapter_enabled
        else None
    )
    latent_answer_probe = (
        LatentAnswerProbe(
            actor_hidden_size,
            max_candidates=config.latent_answer_probe_max_candidates,
            hidden_multiplier=config.latent_answer_probe_hidden_multiplier,
            dropout=config.latent_answer_probe_dropout,
        )
        if config.latent_answer_probe_enabled
        else None
    )
    latent_sequence_decoder = (
        LatentSequenceDecoderHead(
            actor_hidden_size,
            max_answer_length=config.latent_sequence_decoder_max_answer_length,
            vocabulary_size=int(lm_vocabulary_weight(actor_model).shape[0]),
            vocabulary_mode=config.latent_sequence_decoder_vocabulary_mode,
            num_heads=config.latent_sequence_decoder_num_heads,
            hidden_multiplier=config.latent_sequence_decoder_hidden_multiplier,
            dropout=config.latent_sequence_decoder_dropout,
            scale=config.latent_sequence_decoder_scale,
        )
        if config.latent_sequence_decoder_enabled
        else None
    )
    latent_soft_prompt_decoder = (
        LatentSoftPromptDecoder(
            actor_hidden_size,
            output_steps=config.latent_soft_prompt_decoder_output_steps,
            num_heads=config.latent_soft_prompt_decoder_num_heads,
            hidden_multiplier=config.latent_soft_prompt_decoder_hidden_multiplier,
            dropout=config.latent_soft_prompt_decoder_dropout,
            residual_scale=config.latent_soft_prompt_decoder_residual_scale,
            max_delta_norm=config.latent_soft_prompt_decoder_max_delta_norm,
        )
        if config.latent_soft_prompt_decoder_enabled
        else None
    )
    latent_logit_steering = (
        LatentLogitSteeringHead(
            actor_hidden_size,
            rank=config.latent_logit_steering_rank,
            vocabulary_size=int(lm_vocabulary_weight(actor_model).shape[0]),
            vocabulary_mode=config.latent_logit_steering_vocabulary_mode,
            output_steps=config.latent_logit_steering_output_steps,
            dropout=config.latent_logit_steering_dropout,
            scale=config.latent_logit_steering_scale,
            pooling=config.latent_logit_steering_pooling,
            max_bias_norm=config.latent_logit_steering_max_bias_norm,
        )
        if config.latent_logit_steering_enabled
        else None
    )
    latent_token_decoder = (
        LatentTokenDecoderHead(
            actor_hidden_size,
            rank=config.latent_token_decoder_rank,
            vocabulary_size=int(lm_vocabulary_weight(actor_model).shape[0]),
            vocabulary_mode=config.latent_token_decoder_vocabulary_mode,
            output_steps=config.latent_token_decoder_output_steps,
            dropout=config.latent_token_decoder_dropout,
            scale=config.latent_token_decoder_scale,
            pooling=config.latent_token_decoder_pooling,
            max_bias_norm=config.latent_token_decoder_max_bias_norm,
        )
        if config.latent_token_decoder_enabled
        else None
    )
    if hidden_state_processor is not None:
        hidden_state_processor.train()
    if latent_handoff_adapter is not None:
        latent_handoff_adapter.to(_model_device(reasoner_model))
        latent_handoff_adapter.train()
    if latent_answer_probe is not None:
        latent_answer_probe.to(_model_device(reasoner_model))
        latent_answer_probe.train()
    if latent_sequence_decoder is not None:
        latent_sequence_decoder.to(_model_device(reasoner_model))
        latent_sequence_decoder.train()
    if latent_soft_prompt_decoder is not None:
        latent_soft_prompt_decoder.to(_model_device(reasoner_model))
        latent_soft_prompt_decoder.train()
    if latent_logit_steering is not None:
        latent_logit_steering.to(_model_device(actor_model))
        latent_logit_steering.train()
    if latent_token_decoder is not None:
        latent_token_decoder.to(_model_device(actor_model))
        latent_token_decoder.train()
    base_trainable_parameters = (
        list(p for p in reasoner_model.parameters() if p.requires_grad)
        + ([] if latent_handoff_adapter is None else list(latent_handoff_adapter.parameters()))
        + ([] if latent_answer_probe is None else list(latent_answer_probe.parameters()))
        + ([] if latent_sequence_decoder is None else list(latent_sequence_decoder.parameters()))
        + ([] if latent_soft_prompt_decoder is None else list(latent_soft_prompt_decoder.parameters()))
        + ([] if hidden_state_processor is None else list(hidden_state_processor.parameters()))
    )
    optimizer_param_groups: list[dict[str, Any]] = []
    if base_trainable_parameters:
        optimizer_param_groups.append(
            {
                "params": base_trainable_parameters,
                "lr": config.learning_rate,
            }
        )
    if latent_logit_steering is not None:
        steering_parameters = list(latent_logit_steering.parameters())
        if steering_parameters:
            optimizer_param_groups.append(
                {
                    "params": steering_parameters,
                    "lr": config.learning_rate * config.latent_logit_steering_lr_multiplier,
                }
            )
    if latent_token_decoder is not None:
        token_decoder_parameters = list(latent_token_decoder.parameters())
        if token_decoder_parameters:
            optimizer_param_groups.append(
                {
                    "params": token_decoder_parameters,
                    "lr": config.learning_rate * config.latent_token_decoder_lr_multiplier,
                }
            )
    if not optimizer_param_groups:
        raise ValueError("train_reasoner_stage2 received no trainable parameters")
    optimizer = torch.optim.AdamW(
        optimizer_param_groups,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    reasoner_device = _model_device(reasoner_model)
    actor_device = _model_device(actor_model)
    reasoner_backbone = _model_backbone(reasoner_model)
    if hidden_state_processor is not None:
        hidden_state_processor.to(reasoner_device)
    if latent_handoff_adapter is not None:
        latent_handoff_adapter.to(reasoner_device)
    if latent_answer_probe is not None:
        latent_answer_probe.to(reasoner_device)
    if latent_sequence_decoder is not None:
        latent_sequence_decoder.to(reasoner_device)
    if latent_soft_prompt_decoder is not None:
        latent_soft_prompt_decoder.to(reasoner_device)
    if latent_logit_steering is not None:
        latent_logit_steering.to(actor_device)
    if latent_token_decoder is not None:
        latent_token_decoder.to(actor_device)
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
            "l_answer": config.lambda_answer,
            "l_answer_first_token": config.lambda_answer_first_token,
            "l_logit_steering": config.lambda_logit_steering,
            "l_latent_token_decoder": config.lambda_latent_token_decoder,
            "l_answer_contrast": config.lambda_answer_contrast,
            "l_answer_probe": config.lambda_answer_probe,
            "l_latent_sequence_decoder": config.lambda_latent_sequence_decoder,
        },
        config=AdaptiveLossBalancerConfig(
            enabled=config.adaptive_loss_enabled,
            ema_beta=config.adaptive_loss_ema_beta,
            min_weight=config.adaptive_loss_min_weight,
            max_weight=config.adaptive_loss_max_weight,
        ),
    )

    history: list[dict[str, Any]] = []
    global_step = 0
    latest_candidate_answers: tuple[str, ...] = ()
    total_training_steps = max(1, int(config.num_epochs) * max(1, len(train_dataloader)))

    def record_evaluation(event: str, epoch: float) -> dict[str, Any] | None:
        nonlocal global_step
        if evaluation_fn is None:
            return None
        eval_context = dict(latest_alignment_context)
        if latent_handoff_adapter is not None:
            eval_context["stage2_latent_handoff_adapter"] = latent_handoff_adapter
        if hidden_state_processor is not None:
            eval_context["stage2_hidden_state_processor"] = hidden_state_processor
        if latent_answer_probe is not None:
            eval_context["stage2_latent_answer_probe"] = latent_answer_probe
            eval_context["stage2_latent_answer_candidates"] = latest_candidate_answers
        if latent_sequence_decoder is not None:
            eval_context["stage2_latent_sequence_decoder"] = latent_sequence_decoder
            eval_context["stage2_latent_sequence_decoder_generation_min_accuracy"] = (
                config.latent_sequence_decoder_generation_min_accuracy
            )
        if latent_soft_prompt_decoder is not None:
            eval_context["stage2_latent_soft_prompt_decoder"] = latent_soft_prompt_decoder
        if latent_logit_steering is not None:
            eval_context["stage2_latent_logit_steering"] = latent_logit_steering
            eval_context["stage2_latent_logit_steering_generation_scale"] = (
                config.latent_logit_steering_generation_scale
            )
        if latent_token_decoder is not None:
            eval_context["stage2_latent_token_decoder"] = latent_token_decoder
            eval_context["stage2_latent_token_decoder_candidate_token_mask"] = (
                config.latent_token_decoder_candidate_token_mask
            )
            eval_context["stage2_latent_token_decoder_require_ready"] = (
                config.latent_token_decoder_require_ready
            )
        module_training_states = [
            (module, module.training)
            for module in (
                latent_handoff_adapter,
                hidden_state_processor,
                latent_answer_probe,
                latent_sequence_decoder,
                latent_soft_prompt_decoder,
                latent_logit_steering,
                latent_token_decoder,
            )
            if module is not None
        ]
        for module, _was_training in module_training_states:
            module.eval()
        try:
            evaluation_metrics = {
                key: _coerce_history_value(value)
                for key, value in evaluation_fn(
                    reasoner_model,
                    actor_model,
                    eval_context,
                ).items()
            }
        finally:
            for module, was_training in module_training_states:
                module.train(was_training)
        eval_history_entry = {
            "event": event,
            "epoch": float(epoch),
            "step": float(global_step),
            **evaluation_metrics,
        }
        history.append(eval_history_entry)
        if config.wandb_enabled:
            wandb.log(_numeric_metrics(evaluation_metrics), step=global_step)
        return eval_history_entry

    if config.checkpoint_enabled:
        ckpt_dir = Path(config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    if config.evaluate_before_training:
        record_evaluation("pretrain_eval", -1.0)

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
            reasoner_texts = _extract_prompt_batch(batch, fallback_texts=texts)
            answers = _extract_answer_batch(batch, expected_count=len(texts))
            candidate_answers = _extract_candidate_answers(batch, answers=answers)
            if candidate_answers:
                latest_candidate_answers = candidate_answers
            reasoner_batch = _tokenize_text_batch(
                reasoner_tokenizer,
                reasoner_texts,
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
            if latent_handoff_adapter is not None:
                compressed_latents_aligned = latent_handoff_adapter(compressed_latents_aligned)
            if hidden_state_processor is not None:
                compressed_latents_aligned = hidden_state_processor(compressed_latents_aligned)
            actor_prefix_latents = compressed_latents_aligned
            soft_prompt_delta_norm = 0.0
            if latent_soft_prompt_decoder is not None:
                actor_prefix_latents = latent_soft_prompt_decoder(compressed_latents_aligned)
                if actor_prefix_latents.shape == compressed_latents_aligned.shape:
                    soft_prompt_delta_norm = float(
                        (actor_prefix_latents.detach().float() - compressed_latents_aligned.detach().float())
                        .norm(dim=-1)
                        .mean()
                        .cpu()
                        .item()
                    )
                else:
                    soft_prompt_delta_norm = float(
                        actor_prefix_latents.detach().float().norm(dim=-1).mean().cpu().item()
                    )

            compute_distillation_losses = any(
                weight > 0.0
                for weight in (
                    config.lambda_task,
                    config.lambda_pref,
                    config.lambda_geom,
                )
            )
            if compute_distillation_losses:
                full_attention = torch.ones(
                    (full_latents.size(0), full_latents.size(1)),
                    dtype=torch.long,
                    device=actor_device,
                )
                compressed_attention = torch.ones(
                    (actor_prefix_latents.size(0), actor_prefix_latents.size(1)),
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
                    inputs_embeds=actor_prefix_latents.to(
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
            else:
                zero = actor_prefix_latents.sum() * 0.0
                loss_outputs = {
                    "loss": zero,
                    "l_task": zero,
                    "l_pref": zero,
                    "l_geom": zero,
                    "pref_avg_entropy": zero,
                    "pref_avg_weight": zero,
                    "pref_first_token_entropy": zero,
                    "pref_first_token_weight": zero,
                    "pref_first_token_kl": zero,
                    "pref_avg_top1_probability": zero,
                    "pref_first_token_top1_probability": zero,
                    "pref_avg_logit_margin": zero,
                    "pref_first_token_logit_margin": zero,
                    "pref_first_token_weight_ratio": zero,
                }
            l_plan = compute_plan_similarity_loss(
                full_latents_aligned.detach(),
                actor_prefix_latents,
            )
            l_contrast = compute_random_contrast_loss(
                full_latents_aligned.detach(),
                actor_prefix_latents,
                temperature=config.contrast_temperature,
            )
            if config.lambda_answer > 0.0:
                answer_loss, answer_loss_sample_count, answer_token_count_mean = _compute_latent_answer_loss(
                    actor_model=actor_model,
                    actor_tokenizer=actor_tokenizer,
                    latent_prefix=actor_prefix_latents,
                    answers=answers,
                    suffix_text=config.answer_suffix_text,
                    max_answer_length=config.answer_max_length,
                    first_token_weight=config.answer_first_token_weight,
                )
            else:
                answer_loss, answer_loss_sample_count, answer_token_count_mean = None, 0, 0.0
            if config.lambda_answer_first_token > 0.0:
                (
                    answer_first_token_loss,
                    answer_first_token_sample_count,
                    answer_first_token_accuracy,
                    answer_first_token_rank_mean,
                    answer_first_token_margin_mean,
                ) = _compute_latent_first_token_loss(
                    actor_model=actor_model,
                    actor_tokenizer=actor_tokenizer,
                    latent_prefix=actor_prefix_latents,
                    answers=answers,
                    suffix_text=config.answer_suffix_text,
                    max_answer_length=config.answer_max_length,
                    margin=config.answer_first_token_margin,
                )
            else:
                (
                    answer_first_token_loss,
                    answer_first_token_sample_count,
                    answer_first_token_accuracy,
                    answer_first_token_rank_mean,
                    answer_first_token_margin_mean,
                ) = (None, 0, 0.0, 0.0, 0.0)
            if config.lambda_logit_steering > 0.0:
                (
                    logit_steering_loss,
                    logit_steering_sample_count,
                    logit_steering_accuracy,
                    logit_steering_rank_mean,
                    logit_steering_margin_mean,
                    logit_steering_bias_norm,
                ) = _compute_latent_logit_steering_loss(
                    latent_logit_steering=latent_logit_steering,
                    actor_model=actor_model,
                    actor_tokenizer=actor_tokenizer,
                    latent_prefix=actor_prefix_latents,
                    answers=answers,
                    suffix_text=config.answer_suffix_text,
                    max_answer_length=config.answer_max_length,
                    margin=config.logit_steering_margin,
                    answer_token_weight=config.latent_logit_steering_answer_token_weight,
                    later_answer_token_weight=(
                        config.latent_logit_steering_later_answer_token_weight
                    ),
                    eos_weight=config.latent_logit_steering_eos_weight,
                )
            else:
                (
                    logit_steering_loss,
                    logit_steering_sample_count,
                    logit_steering_accuracy,
                    logit_steering_rank_mean,
                    logit_steering_margin_mean,
                    logit_steering_bias_norm,
                ) = (None, 0, 0.0, 0.0, 0.0, 0.0)
            if config.lambda_answer_contrast > 0.0:
                answer_contrast_loss, answer_contrast_sample_count, answer_contrast_accuracy = (
                    _compute_latent_candidate_contrast_loss(
                        actor_model=actor_model,
                        actor_tokenizer=actor_tokenizer,
                        latent_prefix=actor_prefix_latents,
                        answers=answers,
                        candidate_answers=candidate_answers,
                        suffix_text=config.answer_suffix_text,
                        max_answer_length=config.answer_max_length,
                        temperature=config.answer_contrast_temperature,
                    )
                )
            else:
                answer_contrast_loss, answer_contrast_sample_count, answer_contrast_accuracy = None, 0, 0.0
            if config.lambda_answer_probe > 0.0:
                answer_probe_loss, answer_probe_sample_count, answer_probe_accuracy = (
                    _compute_latent_answer_probe_loss(
                        latent_answer_probe=latent_answer_probe,
                        latent_prefix=compressed_latents_aligned,
                        answers=answers,
                        candidate_answers=candidate_answers,
                    )
                )
            else:
                answer_probe_loss, answer_probe_sample_count, answer_probe_accuracy = None, 0, 0.0
            (
                latent_token_decoder_loss,
                latent_token_decoder_sample_count,
                latent_token_decoder_token_accuracy,
                latent_token_decoder_sequence_accuracy,
                latent_token_decoder_token_count_mean,
            ) = (
                _compute_latent_token_decoder_loss(
                    latent_token_decoder=latent_token_decoder,
                    actor_model=actor_model,
                    actor_tokenizer=actor_tokenizer,
                    latent_prefix=compressed_latents_aligned,
                    answers=answers,
                    candidate_answers=candidate_answers,
                    max_answer_length=config.answer_max_length,
                    candidate_token_mask=config.latent_token_decoder_candidate_token_mask,
                    eos_weight=config.latent_token_decoder_eos_weight,
                    margin=config.latent_token_decoder_margin,
                )
                if config.lambda_latent_token_decoder > 0.0
                else (None, 0, 0.0, 0.0, 0.0)
            )
            (
                latent_sequence_decoder_loss,
                latent_sequence_decoder_sample_count,
                latent_sequence_decoder_token_accuracy,
                latent_sequence_decoder_sequence_accuracy,
                latent_sequence_decoder_length_accuracy,
                latent_sequence_decoder_target_length_mean,
                latent_sequence_decoder_predicted_length_mean,
            ) = _compute_latent_sequence_decoder_loss(
                latent_sequence_decoder=latent_sequence_decoder,
                actor_model=actor_model,
                actor_tokenizer=actor_tokenizer,
                latent_prefix=compressed_latents_aligned,
                answers=answers,
                suffix_text=config.answer_suffix_text,
                max_answer_length=config.latent_sequence_decoder_max_answer_length,
            )
            loss_terms = {
                "l_task": loss_outputs["l_task"],
                "l_pref": loss_outputs["l_pref"],
                "l_geom": loss_outputs["l_geom"],
                "l_plan": l_plan,
                "l_contrast": l_contrast,
            }
            if answer_loss is not None:
                loss_terms["l_answer"] = answer_loss.to(reasoner_device)
            if answer_first_token_loss is not None:
                loss_terms["l_answer_first_token"] = answer_first_token_loss.to(reasoner_device)
            if logit_steering_loss is not None:
                loss_terms["l_logit_steering"] = logit_steering_loss.to(reasoner_device)
            if latent_token_decoder_loss is not None:
                loss_terms["l_latent_token_decoder"] = latent_token_decoder_loss.to(reasoner_device)
            if answer_contrast_loss is not None:
                loss_terms["l_answer_contrast"] = answer_contrast_loss.to(reasoner_device)
            if answer_probe_loss is not None:
                loss_terms["l_answer_probe"] = answer_probe_loss.to(reasoner_device)
            if latent_sequence_decoder_loss is not None:
                loss_terms["l_latent_sequence_decoder"] = latent_sequence_decoder_loss.to(reasoner_device)
            total_loss, effective_weights = loss_balancer.combine(loss_terms)

            optimizer.zero_grad(set_to_none=True)
            adapter_before = _snapshot_module_parameters(latent_handoff_adapter)
            answer_probe_before = _snapshot_module_parameters(latent_answer_probe)
            sequence_decoder_before = _snapshot_module_parameters(latent_sequence_decoder)
            soft_prompt_decoder_before = _snapshot_module_parameters(latent_soft_prompt_decoder)
            logit_steering_before = _snapshot_module_parameters(latent_logit_steering)
            latent_token_decoder_before = _snapshot_module_parameters(latent_token_decoder)
            total_loss.backward()
            reasoner_grad_norm = _gradient_norm(reasoner_model)
            handoff_adapter_grad_norm = _gradient_norm(latent_handoff_adapter)
            latent_answer_probe_grad_norm = _gradient_norm(latent_answer_probe)
            latent_sequence_decoder_grad_norm = _gradient_norm(latent_sequence_decoder)
            latent_soft_prompt_decoder_grad_norm = _gradient_norm(latent_soft_prompt_decoder)
            latent_logit_steering_grad_norm = _gradient_norm(latent_logit_steering)
            latent_token_decoder_grad_norm = _gradient_norm(latent_token_decoder)
            hidden_processor_grad_norm = _gradient_norm(hidden_state_processor)
            nn.utils.clip_grad_norm_(reasoner_model.parameters(), config.max_grad_norm)
            if latent_handoff_adapter is not None:
                nn.utils.clip_grad_norm_(latent_handoff_adapter.parameters(), config.max_grad_norm)
            if latent_answer_probe is not None:
                nn.utils.clip_grad_norm_(latent_answer_probe.parameters(), config.max_grad_norm)
            if latent_sequence_decoder is not None:
                nn.utils.clip_grad_norm_(latent_sequence_decoder.parameters(), config.max_grad_norm)
            if latent_soft_prompt_decoder is not None:
                nn.utils.clip_grad_norm_(latent_soft_prompt_decoder.parameters(), config.max_grad_norm)
            if latent_logit_steering is not None:
                nn.utils.clip_grad_norm_(latent_logit_steering.parameters(), config.max_grad_norm)
            if latent_token_decoder is not None:
                nn.utils.clip_grad_norm_(latent_token_decoder.parameters(), config.max_grad_norm)
            if hidden_state_processor is not None:
                nn.utils.clip_grad_norm_(hidden_state_processor.parameters(), config.max_grad_norm)
            optimizer.step()
            handoff_adapter_update_norm = _module_update_norm(adapter_before, latent_handoff_adapter)
            latent_answer_probe_update_norm = _module_update_norm(answer_probe_before, latent_answer_probe)
            latent_sequence_decoder_update_norm = _module_update_norm(
                sequence_decoder_before,
                latent_sequence_decoder,
            )
            latent_soft_prompt_decoder_update_norm = _module_update_norm(
                soft_prompt_decoder_before,
                latent_soft_prompt_decoder,
            )
            latent_logit_steering_update_norm = _module_update_norm(
                logit_steering_before,
                latent_logit_steering,
            )
            latent_token_decoder_update_norm = _module_update_norm(
                latent_token_decoder_before,
                latent_token_decoder,
            )

            metrics = {
                "epoch": float(epoch),
                "step": float(step),
                "loss": float(total_loss.detach().cpu().item()),
                "l_task": float(loss_outputs["l_task"].detach().cpu().item()),
                "l_pref": float(loss_outputs["l_pref"].detach().cpu().item()),
                "l_geom": float(loss_outputs["l_geom"].detach().cpu().item()),
                "l_plan": float(l_plan.detach().cpu().item()),
                "l_contrast": float(l_contrast.detach().cpu().item()),
                "l_answer": 0.0 if answer_loss is None else float(answer_loss.detach().cpu().item()),
                "l_answer_first_token": 0.0
                if answer_first_token_loss is None
                else float(answer_first_token_loss.detach().cpu().item()),
                "l_logit_steering": 0.0
                if logit_steering_loss is None
                else float(logit_steering_loss.detach().cpu().item()),
                "l_latent_token_decoder": 0.0
                if latent_token_decoder_loss is None
                else float(latent_token_decoder_loss.detach().cpu().item()),
                "l_answer_contrast": 0.0
                if answer_contrast_loss is None
                else float(answer_contrast_loss.detach().cpu().item()),
                "l_answer_probe": 0.0
                if answer_probe_loss is None
                else float(answer_probe_loss.detach().cpu().item()),
                "l_latent_sequence_decoder": 0.0
                if latent_sequence_decoder_loss is None
                else float(latent_sequence_decoder_loss.detach().cpu().item()),
                "answer_loss_sample_count": float(answer_loss_sample_count),
                "answer_loss_token_count_mean": float(answer_token_count_mean),
                "answer_first_token_sample_count": float(answer_first_token_sample_count),
                "answer_first_token_accuracy": float(answer_first_token_accuracy),
                "answer_first_token_rank_mean": float(answer_first_token_rank_mean),
                "answer_first_token_margin_mean": float(answer_first_token_margin_mean),
                "logit_steering_sample_count": float(logit_steering_sample_count),
                "logit_steering_accuracy": float(logit_steering_accuracy),
                "logit_steering_rank_mean": float(logit_steering_rank_mean),
                "logit_steering_margin_mean": float(logit_steering_margin_mean),
                "logit_steering_bias_norm": float(logit_steering_bias_norm),
                "latent_token_decoder_sample_count": float(latent_token_decoder_sample_count),
                "latent_token_decoder_token_accuracy": float(latent_token_decoder_token_accuracy),
                "latent_token_decoder_sequence_accuracy": float(latent_token_decoder_sequence_accuracy),
                "latent_token_decoder_token_count_mean": float(latent_token_decoder_token_count_mean),
                "answer_contrast_sample_count": float(answer_contrast_sample_count),
                "answer_contrast_accuracy": float(answer_contrast_accuracy),
                "answer_probe_sample_count": float(answer_probe_sample_count),
                "answer_probe_accuracy": float(answer_probe_accuracy),
                "latent_sequence_decoder_sample_count": float(latent_sequence_decoder_sample_count),
                "latent_sequence_decoder_token_accuracy": float(latent_sequence_decoder_token_accuracy),
                "latent_sequence_decoder_sequence_accuracy": float(latent_sequence_decoder_sequence_accuracy),
                "latent_sequence_decoder_length_accuracy": float(latent_sequence_decoder_length_accuracy),
                "latent_sequence_decoder_target_length_mean": float(latent_sequence_decoder_target_length_mean),
                "latent_sequence_decoder_predicted_length_mean": float(
                    latent_sequence_decoder_predicted_length_mean
                ),
                "answer_candidate_count": float(len(candidate_answers)),
                "effective_weight_task": effective_weights["l_task"],
                "effective_weight_pref": effective_weights["l_pref"],
                "effective_weight_geom": effective_weights["l_geom"],
                "effective_weight_plan": effective_weights["l_plan"],
                "effective_weight_contrast": effective_weights["l_contrast"],
                "effective_weight_answer": effective_weights.get("l_answer", 0.0),
                "effective_weight_answer_first_token": effective_weights.get("l_answer_first_token", 0.0),
                "effective_weight_logit_steering": effective_weights.get("l_logit_steering", 0.0),
                "effective_weight_latent_token_decoder": effective_weights.get(
                    "l_latent_token_decoder",
                    0.0,
                ),
                "effective_weight_answer_contrast": effective_weights.get("l_answer_contrast", 0.0),
                "effective_weight_answer_probe": effective_weights.get("l_answer_probe", 0.0),
                "effective_weight_latent_sequence_decoder": effective_weights.get(
                    "l_latent_sequence_decoder",
                    0.0,
                ),
                "reasoner_grad_norm": reasoner_grad_norm,
                "handoff_adapter_grad_norm": handoff_adapter_grad_norm,
                "latent_answer_probe_grad_norm": latent_answer_probe_grad_norm,
                "latent_sequence_decoder_grad_norm": latent_sequence_decoder_grad_norm,
                "latent_soft_prompt_decoder_grad_norm": latent_soft_prompt_decoder_grad_norm,
                "latent_logit_steering_grad_norm": latent_logit_steering_grad_norm,
                "latent_token_decoder_grad_norm": latent_token_decoder_grad_norm,
                "hidden_processor_grad_norm": hidden_processor_grad_norm,
                "handoff_adapter_update_norm": handoff_adapter_update_norm,
                "latent_answer_probe_update_norm": latent_answer_probe_update_norm,
                "latent_sequence_decoder_update_norm": latent_sequence_decoder_update_norm,
                "latent_soft_prompt_decoder_update_norm": latent_soft_prompt_decoder_update_norm,
                "latent_logit_steering_update_norm": latent_logit_steering_update_norm,
                "latent_token_decoder_update_norm": latent_token_decoder_update_norm,
                "latent_soft_prompt_decoder_delta_norm": soft_prompt_delta_norm,
                "trainable_reasoner_parameter_count": float(_parameter_count(reasoner_model)),
                "trainable_handoff_adapter_parameter_count": float(_parameter_count(latent_handoff_adapter)),
                "trainable_latent_answer_probe_parameter_count": float(_parameter_count(latent_answer_probe)),
                "trainable_latent_sequence_decoder_parameter_count": float(
                    _parameter_count(latent_sequence_decoder)
                ),
                "trainable_latent_soft_prompt_decoder_parameter_count": float(
                    _parameter_count(latent_soft_prompt_decoder)
                ),
                "trainable_latent_logit_steering_parameter_count": float(
                    _parameter_count(latent_logit_steering)
                ),
                "trainable_latent_token_decoder_parameter_count": float(
                    _parameter_count(latent_token_decoder)
                ),
                "trainable_hidden_processor_parameter_count": float(_parameter_count(hidden_state_processor)),
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
                        "l_answer": metrics["l_answer"],
                        "l_answer_first_token": metrics["l_answer_first_token"],
                        "l_logit_steering": metrics["l_logit_steering"],
                        "l_latent_token_decoder": metrics["l_latent_token_decoder"],
                        "l_answer_contrast": metrics["l_answer_contrast"],
                        "l_answer_probe": metrics["l_answer_probe"],
                        "l_latent_sequence_decoder": metrics["l_latent_sequence_decoder"],
                        "answer_loss_sample_count": metrics["answer_loss_sample_count"],
                        "answer_loss_token_count_mean": metrics["answer_loss_token_count_mean"],
                        "answer_first_token_sample_count": metrics["answer_first_token_sample_count"],
                        "answer_first_token_accuracy": metrics["answer_first_token_accuracy"],
                        "answer_first_token_rank_mean": metrics["answer_first_token_rank_mean"],
                        "answer_first_token_margin_mean": metrics["answer_first_token_margin_mean"],
                        "logit_steering_sample_count": metrics["logit_steering_sample_count"],
                        "logit_steering_accuracy": metrics["logit_steering_accuracy"],
                        "logit_steering_rank_mean": metrics["logit_steering_rank_mean"],
                        "logit_steering_margin_mean": metrics["logit_steering_margin_mean"],
                        "logit_steering_bias_norm": metrics["logit_steering_bias_norm"],
                        "latent_token_decoder_sample_count": metrics[
                            "latent_token_decoder_sample_count"
                        ],
                        "latent_token_decoder_token_accuracy": metrics[
                            "latent_token_decoder_token_accuracy"
                        ],
                        "latent_token_decoder_sequence_accuracy": metrics[
                            "latent_token_decoder_sequence_accuracy"
                        ],
                        "latent_token_decoder_token_count_mean": metrics[
                            "latent_token_decoder_token_count_mean"
                        ],
                        "answer_contrast_sample_count": metrics["answer_contrast_sample_count"],
                        "answer_contrast_accuracy": metrics["answer_contrast_accuracy"],
                        "answer_probe_sample_count": metrics["answer_probe_sample_count"],
                        "answer_probe_accuracy": metrics["answer_probe_accuracy"],
                        "latent_sequence_decoder_sample_count": metrics[
                            "latent_sequence_decoder_sample_count"
                        ],
                        "latent_sequence_decoder_token_accuracy": metrics[
                            "latent_sequence_decoder_token_accuracy"
                        ],
                        "latent_sequence_decoder_sequence_accuracy": metrics[
                            "latent_sequence_decoder_sequence_accuracy"
                        ],
                        "latent_sequence_decoder_length_accuracy": metrics[
                            "latent_sequence_decoder_length_accuracy"
                        ],
                        "latent_sequence_decoder_target_length_mean": metrics[
                            "latent_sequence_decoder_target_length_mean"
                        ],
                        "latent_sequence_decoder_predicted_length_mean": metrics[
                            "latent_sequence_decoder_predicted_length_mean"
                        ],
                        "answer_candidate_count": metrics["answer_candidate_count"],
                        "effective_weight_task": metrics["effective_weight_task"],
                        "effective_weight_pref": metrics["effective_weight_pref"],
                        "effective_weight_geom": metrics["effective_weight_geom"],
                        "effective_weight_plan": metrics["effective_weight_plan"],
                        "effective_weight_contrast": metrics["effective_weight_contrast"],
                        "effective_weight_answer": metrics["effective_weight_answer"],
                        "effective_weight_answer_first_token": metrics[
                            "effective_weight_answer_first_token"
                        ],
                        "effective_weight_logit_steering": metrics[
                            "effective_weight_logit_steering"
                        ],
                        "effective_weight_latent_token_decoder": metrics[
                            "effective_weight_latent_token_decoder"
                        ],
                        "effective_weight_answer_contrast": metrics["effective_weight_answer_contrast"],
                        "effective_weight_answer_probe": metrics["effective_weight_answer_probe"],
                        "effective_weight_latent_sequence_decoder": metrics[
                            "effective_weight_latent_sequence_decoder"
                        ],
                        "reasoner_grad_norm": metrics["reasoner_grad_norm"],
                        "handoff_adapter_grad_norm": metrics["handoff_adapter_grad_norm"],
                        "latent_answer_probe_grad_norm": metrics["latent_answer_probe_grad_norm"],
                        "latent_sequence_decoder_grad_norm": metrics["latent_sequence_decoder_grad_norm"],
                        "latent_soft_prompt_decoder_grad_norm": metrics[
                            "latent_soft_prompt_decoder_grad_norm"
                        ],
                        "latent_logit_steering_grad_norm": metrics[
                            "latent_logit_steering_grad_norm"
                        ],
                        "latent_token_decoder_grad_norm": metrics[
                            "latent_token_decoder_grad_norm"
                        ],
                        "hidden_processor_grad_norm": metrics["hidden_processor_grad_norm"],
                        "handoff_adapter_update_norm": metrics["handoff_adapter_update_norm"],
                        "latent_answer_probe_update_norm": metrics["latent_answer_probe_update_norm"],
                        "latent_sequence_decoder_update_norm": metrics[
                            "latent_sequence_decoder_update_norm"
                        ],
                        "latent_soft_prompt_decoder_update_norm": metrics[
                            "latent_soft_prompt_decoder_update_norm"
                        ],
                        "latent_logit_steering_update_norm": metrics[
                            "latent_logit_steering_update_norm"
                        ],
                        "latent_token_decoder_update_norm": metrics[
                            "latent_token_decoder_update_norm"
                        ],
                        "latent_soft_prompt_decoder_delta_norm": metrics[
                            "latent_soft_prompt_decoder_delta_norm"
                        ],
                        "trainable_reasoner_parameter_count": metrics["trainable_reasoner_parameter_count"],
                        "trainable_handoff_adapter_parameter_count": metrics[
                            "trainable_handoff_adapter_parameter_count"
                        ],
                        "trainable_latent_answer_probe_parameter_count": metrics[
                            "trainable_latent_answer_probe_parameter_count"
                        ],
                        "trainable_latent_sequence_decoder_parameter_count": metrics[
                            "trainable_latent_sequence_decoder_parameter_count"
                        ],
                        "trainable_latent_soft_prompt_decoder_parameter_count": metrics[
                            "trainable_latent_soft_prompt_decoder_parameter_count"
                        ],
                        "trainable_latent_logit_steering_parameter_count": metrics[
                            "trainable_latent_logit_steering_parameter_count"
                        ],
                        "trainable_latent_token_decoder_parameter_count": metrics[
                            "trainable_latent_token_decoder_parameter_count"
                        ],
                        "trainable_hidden_processor_parameter_count": metrics[
                            "trainable_hidden_processor_parameter_count"
                        ],
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
                            "latent_handoff_adapter_state_dict": None
                            if latent_handoff_adapter is None
                            else latent_handoff_adapter.state_dict(),
                            "latent_answer_probe_state_dict": None
                            if latent_answer_probe is None
                            else latent_answer_probe.state_dict(),
                            "latent_sequence_decoder_state_dict": None
                            if latent_sequence_decoder is None
                            else latent_sequence_decoder.state_dict(),
                            "latent_soft_prompt_decoder_state_dict": None
                            if latent_soft_prompt_decoder is None
                            else latent_soft_prompt_decoder.state_dict(),
                            "latent_logit_steering_state_dict": None
                            if latent_logit_steering is None
                            else latent_logit_steering.state_dict(),
                            "latent_token_decoder_state_dict": None
                            if latent_token_decoder is None
                            else latent_token_decoder.state_dict(),
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

        eval_history_entry = record_evaluation("epoch_eval", float(epoch))
        if (
            config.early_stop_raw_decode_ready
            and eval_history_entry is not None
            and _raw_decode_ready_for_early_stop(eval_history_entry)
        ):
            eval_history_entry["early_stop_triggered"] = True
            eval_history_entry["early_stop_reason"] = "raw_actor_free_decode_ready"
            print("Early stopping: raw actor free decode reached readiness gate.")
            break

        if config.checkpoint_enabled:
            epoch_path = ckpt_dir / f"epoch_{epoch}.pt"
            torch.save(
                {
                    "reasoner_state_dict": reasoner_model.state_dict(),
                    "hidden_state_processor_state_dict": None
                    if hidden_state_processor is None
                    else hidden_state_processor.state_dict(),
                    "latent_handoff_adapter_state_dict": None
                    if latent_handoff_adapter is None
                    else latent_handoff_adapter.state_dict(),
                    "latent_answer_probe_state_dict": None
                    if latent_answer_probe is None
                    else latent_answer_probe.state_dict(),
                    "latent_sequence_decoder_state_dict": None
                    if latent_sequence_decoder is None
                    else latent_sequence_decoder.state_dict(),
                    "latent_soft_prompt_decoder_state_dict": None
                    if latent_soft_prompt_decoder is None
                    else latent_soft_prompt_decoder.state_dict(),
                    "latent_logit_steering_state_dict": None
                    if latent_logit_steering is None
                    else latent_logit_steering.state_dict(),
                    "latent_token_decoder_state_dict": None
                    if latent_token_decoder is None
                    else latent_token_decoder.state_dict(),
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
