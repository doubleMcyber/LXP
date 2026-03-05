"""Automated comparative benchmark for three experiment modes.

Modes:
  1. pure_text_cot   — Agent B generates directly from text prompt.
  2. vanilla_latent  — Agent A encode → Procrustes → Agent B decode (no ODE).
  3. hybrid_hl_mas   — Full hybrid pipeline with ODE integration.

Usage:
  python3 benchmark_all.py --dataset math --limit 10
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

import torch
from omegaconf import OmegaConf

from latent_pipeline import (
    _build_position_ids,
    _get_pipeline_state,
    run_hybrid_pipeline,
)
from src.data.loader import get_dataloader, pick_field
from src.models.dynamics import (
    _is_kv_cache_compatible,
    _kv_cache_seq_len,
    _move_kv_cache_to_device,
    _normalize_kv_cache,
)
from src.utils.alignment import apply_orthogonal_mapping, compute_orthogonal_mapping
from src.utils.metrics import (
    EvalSampleResult,
    calculate_latency_stats,
    extract_boxed_text,
    normalize_answer,
)


def _load_cfg() -> Any:
    return OmegaConf.load(Path(__file__).resolve().parent / "configs" / "main.yaml")


# ---------------------------------------------------------------------------
# Runner functions: each takes (prompt, cfg, state) -> decoded_text
# ---------------------------------------------------------------------------


def run_pure_text_cot(prompt: str, cfg: Any, state: dict[str, Any]) -> str:
    """Agent B generates directly from the text prompt via model.generate()."""
    tokenizer_b = state["tokenizer_b"]
    agent_b = state["agent_b"]
    device = next(agent_b.parameters()).device

    inputs = tokenizer_b(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = agent_b.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
        )
    generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer_b.decode(generated_ids, skip_special_tokens=True)


def run_vanilla_latent(prompt: str, cfg: Any, state: dict[str, Any]) -> str:
    """Agent A encode → Procrustes alignment → Agent B greedy decode. No ODE."""
    tokenizer_a = state["tokenizer_a"]
    tokenizer_b = state["tokenizer_b"]
    agent_a = state["agent_a"]
    agent_b = state["agent_b"]

    # --- Agent A forward ---
    encoded_a = tokenizer_a(prompt, return_tensors="pt")
    agent_a_device = next(agent_a.parameters()).device
    input_ids_a = encoded_a["input_ids"].to(agent_a_device)
    attention_mask_a = encoded_a["attention_mask"].to(agent_a_device)
    position_ids_a = _build_position_ids(attention_mask_a)

    with torch.no_grad():
        hidden_states, kv_cache_a = agent_a(
            input_ids=input_ids_a,
            attention_mask=attention_mask_a,
            position_ids=position_ids_a,
            use_cache=True,
        )

    # --- Agent B anchor hidden states for Procrustes ---
    agent_b_device = next(agent_b.parameters()).device
    encoded_b = tokenizer_b(prompt, return_tensors="pt")
    input_ids_b = encoded_b["input_ids"].to(agent_b_device)
    attention_mask_b_anchor = encoded_b["attention_mask"].to(agent_b_device)
    position_ids_b_anchor = _build_position_ids(attention_mask_b_anchor)

    with torch.no_grad():
        agent_b_anchor_hidden = agent_b.model(
            input_ids=input_ids_b,
            attention_mask=attention_mask_b_anchor,
            position_ids=position_ids_b_anchor,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state

    # --- Procrustes alignment ---
    procrustes_q = compute_orthogonal_mapping(hidden_states, agent_b_anchor_hidden)
    procrustes_q = procrustes_q.to(
        device=hidden_states.device, dtype=hidden_states.dtype
    )

    # Take last hidden state directly (no ODE integration)
    current_latent_step = hidden_states[:, -1:, :]

    # --- Hand off to Agent B ---
    agent_b_embed_dtype = agent_b.get_input_embeddings().weight.dtype
    handoff_step = apply_orthogonal_mapping(current_latent_step, procrustes_q)
    handoff_step = handoff_step.to(device=agent_b_device, dtype=agent_b_embed_dtype)

    kv_cache_b_candidate = _move_kv_cache_to_device(kv_cache_a, agent_b_device)
    kv_cache_transferred = _is_kv_cache_compatible(kv_cache_b_candidate, agent_b)
    kv_cache_b = kv_cache_b_candidate if kv_cache_transferred else None

    attention_mask_b = torch.ones(
        (handoff_step.shape[0], _kv_cache_seq_len(kv_cache_b) + 1),
        dtype=torch.long,
        device=agent_b_device,
    )
    position_ids_b = _build_position_ids(attention_mask_b)[:, -1:]

    with torch.no_grad():
        outputs_b = agent_b(
            inputs_embeds=handoff_step,
            past_key_values=kv_cache_b,
            attention_mask=attention_mask_b,
            position_ids=position_ids_b,
            use_cache=True,
            return_dict=True,
        )

    # --- Greedy decode loop ---
    generated_token_ids: list[int] = []
    eos_token_id = tokenizer_b.eos_token_id

    for _ in range(cfg.max_new_tokens):
        next_token = torch.argmax(outputs_b.logits[:, -1, :], dim=-1)
        next_token_id = int(next_token.item())
        if eos_token_id is not None and next_token_id == eos_token_id:
            break
        generated_token_ids.append(next_token_id)

        kv_cache_b = _normalize_kv_cache(outputs_b.past_key_values)
        attention_mask_b = torch.cat(
            [
                attention_mask_b,
                torch.ones(
                    (attention_mask_b.shape[0], 1),
                    dtype=attention_mask_b.dtype,
                    device=attention_mask_b.device,
                ),
            ],
            dim=1,
        )
        position_ids_b = _build_position_ids(attention_mask_b)[:, -1:]

        with torch.no_grad():
            outputs_b = agent_b(
                input_ids=next_token.unsqueeze(-1),
                past_key_values=kv_cache_b,
                attention_mask=attention_mask_b,
                position_ids=position_ids_b,
                use_cache=True,
                return_dict=True,
            )

    return tokenizer_b.decode(generated_token_ids, skip_special_tokens=True)


def run_hybrid(prompt: str, cfg: Any, state: dict[str, Any]) -> str:
    """Full hybrid HL-MAS pipeline with ODE integration."""
    del state  # run_hybrid_pipeline uses _get_pipeline_state internally
    result = run_hybrid_pipeline(cfg, prompt=prompt)
    return result["decoded_text"]


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

_METHODS: list[tuple[str, Any]] = [
    ("pure_text_cot", run_pure_text_cot),
    ("vanilla_latent", run_vanilla_latent),
    ("hybrid_hl_mas", run_hybrid),
]


def _evaluate_method(
    method_name: str,
    runner: Any,
    samples: Any,
    cfg: Any,
    state: dict[str, Any],
    tokenizer_b: Any,
) -> dict[str, Any]:
    """Run one method across all samples and return aggregate stats."""
    per_sample: list[EvalSampleResult] = []
    total_correct = 0

    for idx, row in enumerate(samples):
        problem = pick_field(row, ("problem", "question"))
        target_boxed = extract_boxed_text(pick_field(row, ("solution", "answer")))

        error: Optional[str] = None
        decoded_text = ""
        predicted_boxed: Optional[str] = None
        generated_tokens = 0

        start = time.perf_counter()
        try:
            decoded_text = runner(problem, cfg, state)
            predicted_boxed = extract_boxed_text(decoded_text)
            generated_tokens = len(
                tokenizer_b.encode(decoded_text, add_special_tokens=False)
            )
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        latency = time.perf_counter() - start

        correct = normalize_answer(predicted_boxed) == normalize_answer(target_boxed)
        total_correct += int(correct)

        per_sample.append(
            EvalSampleResult(
                index=idx,
                latency_seconds=latency,
                generated_tokens=generated_tokens,
                predicted_boxed=predicted_boxed,
                target_boxed=target_boxed,
                correct=correct,
                error=error,
            )
        )

    sample_count = len(per_sample)
    accuracy_pct = (100.0 * total_correct / sample_count) if sample_count else 0.0
    latency_stats = calculate_latency_stats(per_sample)

    return {
        "method": method_name,
        "accuracy_pct": accuracy_pct,
        "avg_latency_s": latency_stats["average_latency_seconds"],
        "tokens_per_second": latency_stats["tokens_per_second"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comparative benchmark: pure-text vs vanilla-latent vs hybrid HL-MAS"
    )
    parser.add_argument(
        "--dataset", default="math", help="Dataset name (default: math)"
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="Number of samples (default: 10)"
    )
    args = parser.parse_args()

    cfg = _load_cfg()
    samples = get_dataloader(args.dataset, limit=args.limit)
    state = _get_pipeline_state(cfg)
    tokenizer_b = state["tokenizer_b"]

    experiments: list[dict[str, Any]] = []
    for method_name, runner in _METHODS:
        print(f"\n--- Running: {method_name} ---")
        result = _evaluate_method(
            method_name, runner, samples, cfg, state, tokenizer_b
        )
        experiments.append(result)
        print(
            f"  accuracy: {result['accuracy_pct']:.1f}%  "
            f"avg_latency: {result['avg_latency_s']:.3f}s  "
            f"tok/s: {result['tokens_per_second']:.1f}"
        )

    comparison = {
        "dataset": args.dataset,
        "limit": args.limit,
        "experiments": experiments,
    }

    output_path = Path("comparison.json")
    output_path.write_text(json.dumps(comparison, indent=2))
    print(f"\nWrote {output_path}")

    # Summary table
    print(f"\n{'Method':<20} {'Accuracy':>10} {'Avg Latency':>13} {'Tok/s':>8}")
    print("-" * 55)
    for exp in experiments:
        print(
            f"{exp['method']:<20} {exp['accuracy_pct']:>9.1f}% "
            f"{exp['avg_latency_s']:>12.3f}s {exp['tokens_per_second']:>8.1f}"
        )


if __name__ == "__main__":
    main()
