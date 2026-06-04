from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import hydra
import torch
import torch.nn.functional as F
import wandb
from latent_pipeline import load_or_compute_global_alignment_state
from omegaconf import DictConfig, OmegaConf
from torch import nn
from transformers import AutoModelForCausalLM

from src.models.hidden_state import AdaptiveProjection, CurriculumStage, HiddenStateProcessor, LatentHandoffAdapter
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
    answer_suffix_text: str = "\nFinal answer:"
    answer_max_length: int = 32
    answer_first_token_weight: float = 2.0
    evaluate_before_training: bool = False
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
            answer_suffix_text=str(getattr(t, "answer_suffix_text", "\nFinal answer:")),
            answer_max_length=int(getattr(t, "answer_max_length", 32)),
            answer_first_token_weight=float(getattr(t, "answer_first_token_weight", 2.0)),
            evaluate_before_training=bool(
                getattr(getattr(t, "evaluation", None), "evaluate_before_training", False)
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
    if hidden_state_processor is not None:
        hidden_state_processor.train()
    if latent_handoff_adapter is not None:
        latent_handoff_adapter.to(_model_device(reasoner_model))
        latent_handoff_adapter.train()
    optimizer = torch.optim.AdamW(
        (
            list(p for p in reasoner_model.parameters() if p.requires_grad)
            + ([] if latent_handoff_adapter is None else list(latent_handoff_adapter.parameters()))
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
    if latent_handoff_adapter is not None:
        latent_handoff_adapter.to(reasoner_device)
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
    total_training_steps = max(1, int(config.num_epochs) * max(1, len(train_dataloader)))

    def record_evaluation(event: str, epoch: float) -> None:
        nonlocal global_step
        if evaluation_fn is None:
            return
        eval_context = dict(latest_alignment_context)
        if latent_handoff_adapter is not None:
            eval_context["stage2_latent_handoff_adapter"] = latent_handoff_adapter
        if hidden_state_processor is not None:
            eval_context["stage2_hidden_state_processor"] = hidden_state_processor
        evaluation_metrics = {
            key: _coerce_history_value(value)
            for key, value in evaluation_fn(
                reasoner_model,
                actor_model,
                eval_context,
            ).items()
        }
        eval_history_entry = {
            "event": event,
            "epoch": float(epoch),
            "step": float(global_step),
            **evaluation_metrics,
        }
        history.append(eval_history_entry)
        if config.wandb_enabled:
            wandb.log(_numeric_metrics(evaluation_metrics), step=global_step)

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
            answer_loss, answer_loss_sample_count, answer_token_count_mean = _compute_latent_answer_loss(
                actor_model=actor_model,
                actor_tokenizer=actor_tokenizer,
                latent_prefix=compressed_latents_aligned,
                answers=answers,
                suffix_text=config.answer_suffix_text,
                max_answer_length=config.answer_max_length,
                first_token_weight=config.answer_first_token_weight,
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
            total_loss, effective_weights = loss_balancer.combine(loss_terms)

            optimizer.zero_grad(set_to_none=True)
            adapter_before = _snapshot_module_parameters(latent_handoff_adapter)
            total_loss.backward()
            reasoner_grad_norm = _gradient_norm(reasoner_model)
            handoff_adapter_grad_norm = _gradient_norm(latent_handoff_adapter)
            hidden_processor_grad_norm = _gradient_norm(hidden_state_processor)
            nn.utils.clip_grad_norm_(reasoner_model.parameters(), config.max_grad_norm)
            if latent_handoff_adapter is not None:
                nn.utils.clip_grad_norm_(latent_handoff_adapter.parameters(), config.max_grad_norm)
            if hidden_state_processor is not None:
                nn.utils.clip_grad_norm_(hidden_state_processor.parameters(), config.max_grad_norm)
            optimizer.step()
            handoff_adapter_update_norm = _module_update_norm(adapter_before, latent_handoff_adapter)

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
                "answer_loss_sample_count": float(answer_loss_sample_count),
                "answer_loss_token_count_mean": float(answer_token_count_mean),
                "effective_weight_task": effective_weights["l_task"],
                "effective_weight_pref": effective_weights["l_pref"],
                "effective_weight_geom": effective_weights["l_geom"],
                "effective_weight_plan": effective_weights["l_plan"],
                "effective_weight_contrast": effective_weights["l_contrast"],
                "effective_weight_answer": effective_weights.get("l_answer", 0.0),
                "reasoner_grad_norm": reasoner_grad_norm,
                "handoff_adapter_grad_norm": handoff_adapter_grad_norm,
                "hidden_processor_grad_norm": hidden_processor_grad_norm,
                "handoff_adapter_update_norm": handoff_adapter_update_norm,
                "trainable_reasoner_parameter_count": float(_parameter_count(reasoner_model)),
                "trainable_handoff_adapter_parameter_count": float(_parameter_count(latent_handoff_adapter)),
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
                        "answer_loss_sample_count": metrics["answer_loss_sample_count"],
                        "answer_loss_token_count_mean": metrics["answer_loss_token_count_mean"],
                        "effective_weight_task": metrics["effective_weight_task"],
                        "effective_weight_pref": metrics["effective_weight_pref"],
                        "effective_weight_geom": metrics["effective_weight_geom"],
                        "effective_weight_plan": metrics["effective_weight_plan"],
                        "effective_weight_contrast": metrics["effective_weight_contrast"],
                        "effective_weight_answer": metrics["effective_weight_answer"],
                        "reasoner_grad_norm": metrics["reasoner_grad_norm"],
                        "handoff_adapter_grad_norm": metrics["handoff_adapter_grad_norm"],
                        "hidden_processor_grad_norm": metrics["hidden_processor_grad_norm"],
                        "handoff_adapter_update_norm": metrics["handoff_adapter_update_norm"],
                        "trainable_reasoner_parameter_count": metrics["trainable_reasoner_parameter_count"],
                        "trainable_handoff_adapter_parameter_count": metrics[
                            "trainable_handoff_adapter_parameter_count"
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

        record_evaluation("epoch_eval", float(epoch))

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
