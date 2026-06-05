from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


class AdaptiveProjection(nn.Module):
    """Near-identity range adaptation using reference activation statistics."""

    def __init__(
        self,
        *,
        strength: float = 0.15,
        clip_std_multiplier: float = 4.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.strength = float(strength)
        self.clip_std_multiplier = float(clip_std_multiplier)
        self.eps = float(eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        reference_states: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        source = hidden_states.float()
        reference = reference_states.float().to(source.device)
        source_mean = source.mean(dim=1, keepdim=True)
        source_std = source.std(dim=1, unbiased=False, keepdim=True).clamp_min(self.eps)
        reference_mean = reference.mean(dim=1, keepdim=True)
        reference_std = reference.std(dim=1, unbiased=False, keepdim=True).clamp_min(self.eps)
        raw_ratio = (reference_std / source_std).clamp(0.5, 2.0)
        scale = 1.0 + (self.strength * (raw_ratio - 1.0))
        projected = (source - source_mean) * scale + source_mean
        clip_low = reference_mean - (self.clip_std_multiplier * reference_std)
        clip_high = reference_mean + (self.clip_std_multiplier * reference_std)
        clipped = torch.maximum(torch.minimum(projected, clip_high), clip_low)
        diagnostics = {
            "projection_scale_mean": float(scale.mean().detach().cpu().item()),
            "projection_scale_std": float(scale.std(unbiased=False).detach().cpu().item()),
            "projection_clip_fraction": float(
                ((projected < clip_low) | (projected > clip_high)).float().mean().detach().cpu().item()
            ),
        }
        return clipped.to(dtype=hidden_states.dtype), diagnostics


class HiddenStateProcessor(nn.Module):
    """Optional shallow residual processor for aligned latent prefixes."""

    def __init__(
        self,
        hidden_size: int,
        *,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(hidden_states)
        attended, _ = self.attn(normalized, normalized, normalized, need_weights=False)
        hidden_states = hidden_states + attended
        return hidden_states + self.mlp(hidden_states)


class LatentHandoffAdapter(nn.Module):
    """Identity-initialized residual adapter for Stage-II latent prefixes."""

    def __init__(
        self,
        hidden_size: int,
        *,
        rank: int = 64,
        scale: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        adapter_rank = max(1, min(int(rank), int(hidden_size)))
        self.scale = float(scale)
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, adapter_rank, bias=False)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.up = nn.Linear(adapter_rank, hidden_size, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.up(self.dropout(self.activation(self.down(self.norm(hidden_states)))))
        return hidden_states + (self.scale * residual.to(dtype=hidden_states.dtype))


class LatentAnswerProbe(nn.Module):
    """Lightweight candidate readout for inspecting whether latent prefixes encode answers."""

    def __init__(
        self,
        hidden_size: int,
        *,
        max_candidates: int = 64,
        hidden_multiplier: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        probe_hidden = max(int(hidden_size), int(hidden_size) * max(1, int(hidden_multiplier)))
        self.max_candidates = int(max_candidates)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, probe_hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(probe_hidden, self.max_candidates),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled = hidden_states.mean(dim=1)
        return self.net(pooled.float())


class LatentSoftPromptDecoder(nn.Module):
    """Condition latent trajectories into virtual actor prompt embeddings."""

    def __init__(
        self,
        hidden_size: int,
        *,
        output_steps: int = 0,
        num_heads: int = 4,
        hidden_multiplier: int = 2,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
        max_delta_norm: float = 0.0,
    ) -> None:
        super().__init__()
        self.output_steps = max(0, int(output_steps))
        self.residual_scale = float(residual_scale)
        self.max_delta_norm = max(0.0, float(max_delta_norm))
        prompt_hidden = max(int(hidden_size), int(hidden_size) * max(1, int(hidden_multiplier)))
        attn_heads = max(1, min(int(num_heads), int(hidden_size)))
        while hidden_size % attn_heads != 0 and attn_heads > 1:
            attn_heads -= 1

        self.source_norm = nn.LayerNorm(hidden_size)
        self.prompt_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attn_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, prompt_hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(prompt_hidden, hidden_size),
        )
        self.query_tokens = (
            nn.Parameter(torch.empty(self.output_steps, hidden_size))
            if self.output_steps > 0
            else None
        )
        self.summary_to_prompt = (
            nn.Linear(hidden_size, self.output_steps * hidden_size)
            if self.output_steps > 0
            else None
        )
        if self.query_tokens is not None:
            nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        if self.summary_to_prompt is not None:
            nn.init.normal_(self.summary_to_prompt.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.summary_to_prompt.bias)
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.out_proj.bias)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        source = self.source_norm(hidden_states.float())
        if self.output_steps > 0:
            if self.query_tokens is None or self.summary_to_prompt is None:
                raise RuntimeError("LatentSoftPromptDecoder output projection is not initialized")
            batch_size = hidden_states.shape[0]
            summary = source.mean(dim=1)
            prompt = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
            prompt = prompt + self.summary_to_prompt(summary).view(
                batch_size,
                self.output_steps,
                hidden_states.shape[-1],
            )
            base_prompt = prompt
        else:
            prompt = hidden_states.float()
            base_prompt = prompt

        attended, _ = self.cross_attn(
            self.prompt_norm(prompt),
            source,
            source,
            need_weights=False,
        )
        decoded = prompt + (self.residual_scale * attended)
        decoded = decoded + (self.residual_scale * self.mlp(decoded))
        if self.max_delta_norm > 0.0:
            delta = decoded - base_prompt
            delta_norm = delta.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
            scale = (self.max_delta_norm / delta_norm).clamp_max(1.0)
            decoded = base_prompt + (delta * scale)
        return decoded.to(dtype=hidden_states.dtype)


class LatentLogitSteeringHead(nn.Module):
    """Predict a small token-tied vocabulary bias from a latent prefix."""

    def __init__(
        self,
        hidden_size: int,
        *,
        rank: int = 64,
        vocabulary_size: int | None = None,
        vocabulary_mode: str = "tied",
        output_steps: int = 1,
        dropout: float = 0.0,
        scale: float = 1.0,
        pooling: str = "attention",
        max_bias_norm: float = 0.0,
    ) -> None:
        super().__init__()
        steering_rank = max(1, min(int(rank), int(hidden_size)))
        self.output_steps = max(1, int(output_steps))
        self.scale = float(scale)
        self.pooling = str(pooling).strip().lower()
        self.vocabulary_mode = str(vocabulary_mode).strip().lower()
        self.max_bias_norm = max(0.0, float(max_bias_norm))
        self.norm = nn.LayerNorm(hidden_size)
        self.pool_query = nn.Parameter(torch.zeros(hidden_size))
        self.down = nn.Linear(hidden_size, steering_rank, bias=False)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.step_embeddings = nn.Parameter(torch.zeros(self.output_steps, steering_rank))
        self.up = (
            nn.Linear(steering_rank, hidden_size, bias=False)
            if self.vocabulary_mode == "tied"
            else None
        )
        if self.vocabulary_mode == "low_rank":
            if vocabulary_size is None or int(vocabulary_size) < 1:
                raise ValueError("vocabulary_size is required for low_rank vocabulary steering")
            self.vocab_out = nn.Linear(steering_rank, int(vocabulary_size), bias=True)
        else:
            self.vocab_out = None
        if self.vocabulary_mode not in {"tied", "low_rank"}:
            raise ValueError("vocabulary_mode must be one of: tied, low_rank")
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        if self.up is not None:
            nn.init.zeros_(self.up.weight)
        if self.vocab_out is not None:
            nn.init.normal_(self.vocab_out.weight, mean=0.0, std=0.001)
            nn.init.zeros_(self.vocab_out.bias)
        nn.init.normal_(self.step_embeddings, mean=0.0, std=0.01)

    def _pool(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(hidden_states.float())
        if self.pooling == "mean":
            return normalized.mean(dim=1)
        if self.pooling == "last":
            return normalized[:, -1, :]
        if self.pooling == "mean_last":
            return 0.5 * (normalized.mean(dim=1) + normalized[:, -1, :])
        if self.pooling != "attention":
            raise ValueError(
                "LatentLogitSteeringHead pooling must be one of: "
                "attention, mean, last, mean_last"
            )
        query = self.pool_query.float()
        scores = torch.matmul(normalized, query) / math.sqrt(max(1, normalized.shape[-1]))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (normalized * weights).sum(dim=1)

    def _logits_from_features(
        self,
        steering_features: torch.Tensor,
        vocabulary_weight: torch.Tensor,
    ) -> torch.Tensor:
        leading_shape = steering_features.shape[:-1]
        flat_features = steering_features.reshape(-1, steering_features.shape[-1])
        if self.vocabulary_mode == "low_rank":
            if self.vocab_out is None:
                raise RuntimeError("low_rank vocabulary steering output is not initialized")
            flat_logits = self.vocab_out(flat_features).float()
            if int(flat_logits.shape[-1]) != int(vocabulary_weight.shape[0]):
                raise ValueError(
                    "low_rank vocabulary steering output size does not match vocabulary weight: "
                    f"{flat_logits.shape[-1]} != {vocabulary_weight.shape[0]}"
                )
        else:
            if self.up is None:
                raise RuntimeError("tied vocabulary steering projection is not initialized")
            flat_hidden = self.up(flat_features)
            flat_logits = torch.matmul(flat_hidden.float(), vocabulary_weight.float().transpose(0, 1))
        logits_bias = flat_logits.reshape(*leading_shape, flat_logits.shape[-1])
        logits_bias = logits_bias - logits_bias.mean(dim=-1, keepdim=True)
        logits_bias = logits_bias * self.scale
        if self.max_bias_norm > 0.0:
            bias_norm = logits_bias.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
            logits_bias = logits_bias * (self.max_bias_norm / bias_norm).clamp_max(1.0)
        return logits_bias

    def forward_sequence(
        self,
        hidden_states: torch.Tensor,
        vocabulary_weight: torch.Tensor,
        *,
        output_steps: int | None = None,
    ) -> torch.Tensor:
        if hidden_states.dim() != 3:
            raise ValueError(
                "hidden_states must have shape [batch, steps, hidden_size]; "
                f"received {tuple(hidden_states.shape)}"
            )
        if vocabulary_weight.dim() != 2:
            raise ValueError(
                "vocabulary_weight must have shape [vocab_size, hidden_size]; "
                f"received {tuple(vocabulary_weight.shape)}"
            )
        if int(hidden_states.shape[-1]) != int(vocabulary_weight.shape[-1]):
            raise ValueError(
                "hidden size mismatch between latent prefix and vocabulary weight: "
                f"{hidden_states.shape[-1]} != {vocabulary_weight.shape[-1]}"
            )
        step_count = max(1, min(int(output_steps or self.output_steps), self.output_steps))
        pooled = self._pool(hidden_states)
        base_features = self.dropout(self.activation(self.down(pooled)))
        step_features = base_features.unsqueeze(1) + self.step_embeddings[:step_count].unsqueeze(0)
        return self._logits_from_features(step_features, vocabulary_weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        vocabulary_weight: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_sequence(hidden_states, vocabulary_weight, output_steps=1)[:, 0, :]


class LatentTokenDecoderHead(LatentLogitSteeringHead):
    """Direct answer-token decoder over the actor tokenizer vocabulary.

    This shares the low-rank vocabulary projection machinery used by logit steering,
    but its logits are interpreted as the decoded answer sequence rather than as a
    bias added to the actor's autoregressive logits.
    """


def lm_vocabulary_weight(model: Any) -> torch.Tensor:
    """Return the output vocabulary projection weight, falling back to input embeddings."""

    output_embeddings = None
    if hasattr(model, "get_output_embeddings"):
        output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None and hasattr(output_embeddings, "weight"):
        return output_embeddings.weight
    return model.get_input_embeddings().weight


def build_plan_summary(hidden_states: torch.Tensor) -> torch.Tensor:
    start = hidden_states[:, :1, :]
    middle = hidden_states.mean(dim=1, keepdim=True)
    end = hidden_states[:, -1:, :]
    return torch.cat([start, middle, end], dim=1).mean(dim=1)


@dataclass
class CurriculumStage:
    name: str
    progress_upper_bound: float
    alignment_strategy: str
    prompt_calibration_enabled: bool
