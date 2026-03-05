from __future__ import annotations

import torch


def _flatten_hidden_states(hidden_states: torch.Tensor) -> torch.Tensor:
    if hidden_states.dim() < 2:
        raise ValueError("hidden_states must have at least 2 dimensions")
    if hidden_states.dim() == 2:
        return hidden_states
    return hidden_states.reshape(-1, hidden_states.shape[-1])


def _pair_anchor_states(
    sender_hidden_states: torch.Tensor, receiver_hidden_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    sender = _flatten_hidden_states(sender_hidden_states).to(torch.float32)
    receiver = _flatten_hidden_states(receiver_hidden_states).to(torch.float32)
    receiver = receiver.to(sender.device)

    pair_count = min(sender.shape[0], receiver.shape[0])
    if pair_count == 0:
        raise ValueError("sender/receiver hidden states are empty")

    return sender[:pair_count], receiver[:pair_count]


def compute_cross_covariance(
    sender_hidden_states: torch.Tensor, receiver_hidden_states: torch.Tensor
) -> torch.Tensor:
    sender, receiver = _pair_anchor_states(sender_hidden_states, receiver_hidden_states)
    return sender.transpose(0, 1) @ receiver


def compute_orthogonal_mapping(
    sender_hidden_states: torch.Tensor, receiver_hidden_states: torch.Tensor
) -> torch.Tensor:
    cross_covariance = compute_cross_covariance(sender_hidden_states, receiver_hidden_states)
    u, _, vh = torch.linalg.svd(cross_covariance, full_matrices=False)
    return u @ vh


def apply_orthogonal_mapping(hidden_states: torch.Tensor, mapping_q: torch.Tensor) -> torch.Tensor:
    return hidden_states @ mapping_q.to(device=hidden_states.device, dtype=hidden_states.dtype)
