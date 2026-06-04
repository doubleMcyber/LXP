from __future__ import annotations

from dataclasses import dataclass

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
