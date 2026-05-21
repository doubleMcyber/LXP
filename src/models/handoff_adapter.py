from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from src.utils.alignment import apply_alignment, compute_alignment_state


@dataclass(frozen=True)
class HandoffAdapterFitConfig:
    strategy: str = "hybrid_affine"
    regularization: float = 1e-3
    residual_alpha: float = 1.0
    residual_max_norm_ratio: float = 0.5
    center: bool = True
    use_bias: bool = True
    max_length: int = 96
    adaptive_projection_strength: float = 0.15
    adaptive_projection_clip_std_multiplier: float = 4.0


def build_position_ids(attention_mask: torch.Tensor) -> torch.LongTensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    return position_ids.clamp_min_(0)


def resample_sequence(sequence: torch.Tensor, target_steps: int) -> torch.Tensor:
    if sequence.dim() != 3:
        raise ValueError("sequence must have shape [batch, steps, dim]")
    if target_steps <= 0:
        raise ValueError("target_steps must be positive")
    if int(sequence.shape[1]) == int(target_steps):
        return sequence
    if int(target_steps) == 1:
        return sequence[:, -1:, :]
    resized = F.interpolate(
        sequence.float().transpose(1, 2),
        size=int(target_steps),
        mode="linear",
        align_corners=True,
    )
    return resized.transpose(1, 2).to(dtype=sequence.dtype)


def select_hidden_layers(
    hidden_states: Sequence[torch.Tensor],
    layer_indices: Sequence[int],
) -> tuple[torch.Tensor, ...]:
    if not hidden_states:
        raise ValueError("hidden_states must contain at least one layer")
    layers: list[torch.Tensor] = []
    layer_count = len(hidden_states)
    for index in layer_indices:
        resolved = int(index)
        if resolved < 0:
            resolved = layer_count + resolved
        if resolved < 0 or resolved >= layer_count:
            raise IndexError(f"layer index {index} is out of range for {layer_count} layers")
        layers.append(hidden_states[resolved])
    if not layers:
        raise ValueError("layer_indices must select at least one hidden layer")
    return tuple(layers)


def aggregate_hidden_layers(
    layers: Sequence[torch.Tensor],
    weights: Sequence[float],
) -> torch.Tensor:
    if len(layers) != len(weights):
        raise ValueError("layers and weights must have the same length")
    if not layers:
        raise ValueError("at least one layer is required")
    normalized = torch.as_tensor(weights, dtype=torch.float32, device=layers[0].device)
    normalized = normalized / normalized.sum().clamp_min(1e-8)
    output = torch.zeros_like(layers[0], dtype=torch.float32)
    for layer, weight in zip(layers, normalized):
        output = output + layer.float() * weight
    return output.to(dtype=layers[0].dtype)


def _tokenize_prompt(tokenizer: Any, prompt: str, *, max_length: int, device: torch.device) -> dict[str, torch.Tensor]:
    kwargs: dict[str, Any] = {"return_tensors": "pt"}
    if max_length > 0:
        kwargs.update({"truncation": True, "max_length": int(max_length)})
    encoded = tokenizer(prompt, **kwargs)
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
    }


def collect_handoff_adapter_rows(
    *,
    prompts: Iterable[str],
    tokenizer_a: Any,
    tokenizer_b: Any,
    agent_a: Any,
    agent_b: Any,
    base_alignment_state: dict[str, Any],
    reasoning_layer_indices: Sequence[int],
    reasoning_layer_weights: Sequence[float],
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    source_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    prompt_count = 0
    token_count = 0

    agent_a_device = next(agent_a.parameters()).device
    agent_b_device = next(agent_b.parameters()).device

    for prompt in prompts:
        if not str(prompt).strip():
            continue
        prompt_count += 1
        encoded_a = _tokenize_prompt(
            tokenizer_a,
            str(prompt),
            max_length=max_length,
            device=agent_a_device,
        )
        position_ids_a = build_position_ids(encoded_a["attention_mask"])
        with torch.no_grad():
            outputs_a = agent_a.model(
                input_ids=encoded_a["input_ids"],
                attention_mask=encoded_a["attention_mask"],
                position_ids=position_ids_a,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        if outputs_a.hidden_states is None:
            raise ValueError("Agent A did not return hidden states for handoff adapter training")

        sender_layers = select_hidden_layers(outputs_a.hidden_states, reasoning_layer_indices)
        sender_consensus = aggregate_hidden_layers(sender_layers, reasoning_layer_weights)
        valid_a = encoded_a["attention_mask"][0].to(dtype=torch.bool)
        sender_sequence = sender_consensus[:, valid_a, :]
        if int(sender_sequence.shape[1]) == 0:
            continue
        aligned_sender = apply_alignment(sender_sequence, base_alignment_state)

        encoded_b = _tokenize_prompt(
            tokenizer_b,
            str(prompt),
            max_length=max_length,
            device=agent_b_device,
        )
        valid_b = encoded_b["attention_mask"][0].to(dtype=torch.bool)
        with torch.no_grad():
            receiver_embeddings = agent_b.get_input_embeddings()(encoded_b["input_ids"])
        receiver_sequence = receiver_embeddings[:, valid_b, :]
        if int(receiver_sequence.shape[1]) == 0:
            continue

        target_sequence = resample_sequence(receiver_sequence, int(aligned_sender.shape[1]))
        source_rows.append(aligned_sender.detach().float().cpu().reshape(-1, aligned_sender.shape[-1]))
        target_rows.append(target_sequence.detach().float().cpu().reshape(-1, target_sequence.shape[-1]))
        token_count += int(aligned_sender.shape[1])

    if not source_rows or not target_rows:
        raise ValueError("No usable prompts were available for handoff adapter fitting")

    sources = torch.cat(source_rows, dim=0)
    targets = torch.cat(target_rows, dim=0)
    metrics = {
        "training_prompt_count": float(prompt_count),
        "training_token_count": float(token_count),
    }
    return sources, targets, metrics


def fit_handoff_adapter_state(
    *,
    prompts: Iterable[str],
    tokenizer_a: Any,
    tokenizer_b: Any,
    agent_a: Any,
    agent_b: Any,
    base_alignment_state: dict[str, Any],
    reasoning_layer_indices: Sequence[int],
    reasoning_layer_weights: Sequence[float],
    fit_config: HandoffAdapterFitConfig,
) -> dict[str, Any]:
    source_rows, target_rows, row_metrics = collect_handoff_adapter_rows(
        prompts=prompts,
        tokenizer_a=tokenizer_a,
        tokenizer_b=tokenizer_b,
        agent_a=agent_a,
        agent_b=agent_b,
        base_alignment_state=base_alignment_state,
        reasoning_layer_indices=reasoning_layer_indices,
        reasoning_layer_weights=reasoning_layer_weights,
        max_length=fit_config.max_length,
    )
    adapter_state = compute_alignment_state(
        source_rows,
        target_rows,
        strategy=fit_config.strategy,
        center=fit_config.center,
        use_bias=fit_config.use_bias,
        regularization=fit_config.regularization,
        residual_alpha=fit_config.residual_alpha,
        residual_max_norm_ratio=fit_config.residual_max_norm_ratio,
        adaptive_projection_strength=fit_config.adaptive_projection_strength,
        adaptive_projection_clip_std_multiplier=fit_config.adaptive_projection_clip_std_multiplier,
    )
    fitted_rows = apply_alignment(source_rows, adapter_state).float()
    mse = torch.mean((fitted_rows - target_rows.float()) ** 2)
    cosine = F.cosine_similarity(fitted_rows, target_rows.float(), dim=-1).mean()
    return {
        **adapter_state,
        "adapter_type": "ridge_sequence_handoff",
        "training_prompt_count": int(row_metrics["training_prompt_count"]),
        "training_token_count": int(row_metrics["training_token_count"]),
        "training_reconstruction_mse": float(mse.item()),
        "training_mean_cosine_similarity": float(cosine.item()),
    }


def project_to_embedding_manifold(
    hidden_states: torch.Tensor,
    embedding_weight: torch.Tensor,
    *,
    top_k: int = 1,
    temperature: float = 0.05,
    blend: float = 1.0,
    normalize: bool = True,
    chunk_size: int = 32,
) -> tuple[torch.Tensor, dict[str, float]]:
    if hidden_states.dim() != 3:
        raise ValueError("hidden_states must have shape [batch, steps, dim]")
    if embedding_weight.dim() != 2:
        raise ValueError("embedding_weight must have shape [vocab, dim]")
    if int(hidden_states.shape[-1]) != int(embedding_weight.shape[-1]):
        raise ValueError(
            "hidden state dimension must match receiver embedding dimension: "
            f"{hidden_states.shape[-1]} != {embedding_weight.shape[-1]}"
        )
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    blend = max(0.0, min(1.0, float(blend)))
    if blend == 0.0:
        return hidden_states, {
            "embedding_manifold_mean_top_similarity": 0.0,
            "embedding_manifold_unique_token_count": 0.0,
        }

    original_shape = hidden_states.shape
    rows = hidden_states.reshape(-1, hidden_states.shape[-1]).float()
    embeddings = embedding_weight.detach().to(device=hidden_states.device).float()
    if normalize:
        query_rows = F.normalize(rows, dim=-1)
        candidate_embeddings = F.normalize(embeddings, dim=-1)
    else:
        query_rows = rows
        candidate_embeddings = embeddings

    projected_chunks: list[torch.Tensor] = []
    top_similarity_chunks: list[torch.Tensor] = []
    token_chunks: list[torch.Tensor] = []
    k = min(int(top_k), int(embeddings.shape[0]))

    for start in range(0, int(rows.shape[0]), int(chunk_size)):
        end = min(start + int(chunk_size), int(rows.shape[0]))
        scores = query_rows[start:end] @ candidate_embeddings.transpose(0, 1)
        top_values, top_indices = torch.topk(scores, k=k, dim=-1)
        selected = embeddings.index_select(0, top_indices.reshape(-1)).reshape(
            top_indices.shape[0],
            k,
            embeddings.shape[-1],
        )
        if k == 1:
            projected = selected[:, 0, :]
        else:
            weights = torch.softmax(top_values / float(temperature), dim=-1)
            projected = torch.sum(selected * weights.unsqueeze(-1), dim=1)
        projected_chunks.append(projected)
        top_similarity_chunks.append(top_values[:, 0].detach().float().cpu())
        token_chunks.append(top_indices[:, 0].detach().cpu())

    projected_rows = torch.cat(projected_chunks, dim=0)
    blended = (rows * (1.0 - blend)) + (projected_rows * blend)
    top_similarities = torch.cat(top_similarity_chunks, dim=0)
    top_tokens = torch.cat(token_chunks, dim=0)
    metrics = {
        "embedding_manifold_mean_top_similarity": float(top_similarities.mean().item()),
        "embedding_manifold_unique_token_count": float(torch.unique(top_tokens).numel()),
    }
    return blended.reshape(original_shape).to(dtype=hidden_states.dtype), metrics
