"""Chunk 2 receiver-LoRA trainer data path.

This file intentionally stops before the optimizer loop. It builds the same
context/latent/target samples the eventual receiver-LoRA training step will
consume, and it can prepare verified rollout continuations for objectives A/B.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from benchmark_all import _answers_match, _predicted_answer  # noqa: E402
from scripts.train_latent_bridge import (  # noqa: E402
    CONTINUATION_INSTRUCTION,
    chat_prefix_ids,
    load_bridge_samples,
)
from src.models.receiver_lora import RECEIVER_LORA_TARGET_SUFFIXES  # noqa: E402
from src.utils.alignment import apply_alignment  # noqa: E402
from src.utils.lm_eval import (  # noqa: E402
    _normalize_kv_cache,
    _reset_kv_cache_to_prefix,
    build_position_ids,
    greedy_decode_from_prefix,
    prepare_receiver_context_latent_prefix_state,
)

DEFAULT_ADAPTER_CACHE_PATH = (
    ".cache/generated_trajectory_adapter/"
    "generated_trajectory_adapter_8959603bbf6825e4dc4df3f373ca600ccd7e61425e0aa9c66d7f0adb1070b207.pt"
)
DEFAULT_ADAPTER_DIGEST = (
    "5c041b5521a4663a1a469410c7114df8753762df45f7b6fad66bb9ca2f03c02b"
)
DEFAULT_OUTPUT_DIR = "outputs/receiver_lora/objective_a"
DEFAULT_ROLLOUT_CACHE_DIR = REPO_ROOT / ".cache" / "receiver_lora_rollouts"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receiver-model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--sender-model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--trace-dtype", default="bfloat16")
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--train-rows", default="0:224")
    parser.add_argument("--dev-rows", default="224:256")
    parser.add_argument("--truncation-fraction", type=float, default=0.5)
    parser.add_argument("--validation-size", type=int, default=256)
    parser.add_argument("--adapter-cache-path", default=DEFAULT_ADAPTER_CACHE_PATH)
    parser.add_argument("--adapter-digest", default=DEFAULT_ADAPTER_DIGEST)
    parser.add_argument("--objective", choices=("A", "B", "C"), default="A")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-continuation-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--rollout-max-tries", type=int, default=4)
    parser.add_argument("--rollout-temperature", type=float, default=0.7)
    parser.add_argument("--rollout-top-p", type=float, default=0.95)
    parser.add_argument("--checkpoint-every", type=int, default=32)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prepare-rollouts", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    return parser


def parse_row_range(row_range: str) -> tuple[int, int]:
    try:
        start_text, stop_text = str(row_range).split(":", 1)
        start = int(start_text)
        stop = int(stop_text)
    except ValueError as error:
        raise ValueError(f"row range must be START:STOP, got {row_range!r}") from error
    if start < 0 or stop < start:
        raise ValueError(f"row range must satisfy 0 <= START <= STOP, got {row_range!r}")
    return start, stop


def prompt_sha(prompt: str) -> str:
    return hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()


def context_text(tokenizer: Any, question: str) -> str:
    user_message = f"{question}\n\n{CONTINUATION_INSTRUCTION}"
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:  # noqa: BLE001
        return user_message


def build_context_ids(tokenizer: Any, question: str) -> list[int]:
    return chat_prefix_ids(tokenizer, question, CONTINUATION_INSTRUCTION)


def load_frozen_adapter(
    adapter_cache_path: str | Path,
    *,
    adapter_digest: str = DEFAULT_ADAPTER_DIGEST,
) -> dict[str, Any]:
    adapter_payload = torch.load(
        Path(adapter_cache_path),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(adapter_payload, dict):
        raise ValueError("adapter payload must be a dict")
    actual_digest = adapter_payload.get("adapter_cache_key_digest")
    assert actual_digest == adapter_digest, (
        "adapter digest mismatch: "
        f"expected {adapter_digest}, found {actual_digest}"
    )
    return adapter_payload


def project_sample_latents(sample: dict[str, Any], adapter_payload: dict[str, Any]) -> torch.Tensor:
    latents = sample["latents"]
    if not torch.is_tensor(latents):
        raise TypeError("sample latents must be a torch.Tensor")
    projected = apply_alignment(latents.unsqueeze(0).float(), adapter_payload)
    return projected.to(torch.bfloat16)


def assemble_training_sample(
    sample: dict[str, Any],
    *,
    tokenizer: Any,
    adapter_payload: dict[str, Any],
) -> dict[str, Any]:
    target_ids = [int(token_id) for token_id in sample["continuation_ids"]]
    return {
        "question": sample["question"],
        "answer": sample["answer"],
        "sample_index": int(sample["sample_index"]),
        "truncated_text": sample.get("truncated_text", ""),
        "context_ids": build_context_ids(tokenizer, str(sample["question"])),
        "context_text": context_text(tokenizer, str(sample["question"])),
        "latents": project_sample_latents(sample, adapter_payload),
        "target_ids": target_ids,
        "target_continuation_ids": target_ids,
    }


def _select_sample_index_range(
    samples: list[dict[str, Any]],
    row_range: tuple[int, int],
) -> list[dict[str, Any]]:
    start, stop = row_range
    return [
        sample
        for sample in samples
        if start <= int(sample["sample_index"]) < stop
    ]


def load_bridge_sample_rows(
    args: argparse.Namespace,
    *,
    tokenizer: Any,
    include_dev: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    train_range = parse_row_range(args.train_rows)
    dev_range = parse_row_range(args.dev_rows)
    load_limit = max(train_range[1], dev_range[1] if include_dev else train_range[1])
    samples = load_bridge_samples(
        dataset=args.dataset,
        split="train",
        limit=load_limit,
        model_id=args.sender_model,
        torch_dtype=args.trace_dtype,
        truncation_fraction=args.truncation_fraction,
        tokenizer=tokenizer,
        max_continuation_tokens=args.max_continuation_tokens,
        validation_size=args.validation_size,
    )
    rows = {"train": _select_sample_index_range(samples, train_range)}
    if include_dev:
        rows["dev"] = _select_sample_index_range(samples, dev_range)
    return rows


def assemble_training_samples(
    samples: list[dict[str, Any]],
    *,
    tokenizer: Any,
    adapter_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        assemble_training_sample(sample, tokenizer=tokenizer, adapter_payload=adapter_payload)
        for sample in samples
    ]


def _stable_json_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=list, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def rollout_config_digest(
    *,
    receiver_model: str,
    adapter_digest: str,
    truncation_fraction: float,
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> str:
    return _stable_json_digest(
        {
            "receiver_model": str(receiver_model),
            "adapter_digest": str(adapter_digest),
            "truncation_fraction": float(truncation_fraction),
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "seed": int(seed),
        }
    )


def rollout_cache_path(
    question: str,
    *,
    cache_dir: str | Path = DEFAULT_ROLLOUT_CACHE_DIR,
) -> Path:
    return Path(cache_dir) / f"rollout_{prompt_sha(question)}.pt"


def _top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if top_p >= 1.0:
        return probs
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove_mask = cumulative > float(top_p)
    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
    remove_mask[..., 0] = False
    filtered_sorted = sorted_probs.masked_fill(remove_mask, 0.0)
    filtered = torch.zeros_like(probs).scatter(-1, sorted_indices, filtered_sorted)
    normalizer = filtered.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(filtered.dtype).tiny)
    return filtered / normalizer


def sampled_decode_from_prefix(
    *,
    model: Any,
    tokenizer: Any,
    prefix_state: dict[str, Any],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop_regex: Optional[Any] = None,
) -> dict[str, Any]:
    if temperature <= 0:
        raise ValueError("temperature must be positive for sampled decoding")
    outputs = prefix_state["outputs"]
    attention_mask = prefix_state["attention_mask"]
    prefix_seq_len = int(prefix_state.get("prefix_seq_len", attention_mask.shape[1]))
    _reset_kv_cache_to_prefix(outputs.past_key_values, prefix_seq_len)
    generated_token_ids: list[int] = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    last_position_ids = build_position_ids(attention_mask)[:, -1:]

    for _ in range(max_new_tokens):
        logits = outputs.logits[:, -1, :].float() / float(temperature)
        probs = _top_p_filter(torch.softmax(logits, dim=-1), top_p)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        next_token_id = int(next_token.item())
        if eos_token_id is not None and next_token_id == eos_token_id:
            break
        generated_token_ids.append(next_token_id)
        if stop_regex is not None:
            decoded_text = str(prefix_state.get("decoded_text_prefix", "")) + tokenizer.decode(
                generated_token_ids,
                skip_special_tokens=True,
            )
            if stop_regex.search(decoded_text) is not None:
                break
        kv_cache = _normalize_kv_cache(outputs.past_key_values)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ],
            dim=1,
        )
        last_position_ids = last_position_ids + 1
        with torch.no_grad():
            outputs = model(
                input_ids=next_token.unsqueeze(-1),
                past_key_values=kv_cache,
                attention_mask=attention_mask,
                position_ids=last_position_ids,
                use_cache=True,
                return_dict=True,
            )

    decoded_text = str(prefix_state.get("decoded_text_prefix", "")) + tokenizer.decode(
        generated_token_ids,
        skip_special_tokens=True,
    )
    return {
        "decoded_text": decoded_text,
        "generated_tokens": len(generated_token_ids),
        "generated_token_ids": generated_token_ids,
    }


def _encoded_rollout_ids(tokenizer: Any, decoded_text: str, max_new_tokens: int) -> list[int]:
    try:
        token_ids = tokenizer.encode(decoded_text, add_special_tokens=False)
    except Exception:  # noqa: BLE001
        token_ids = []
    return [int(token_id) for token_id in token_ids[:max_new_tokens]]


def _rollout_verified(dataset: str, decoded_text: str, target_answer: Any) -> bool:
    predicted = _predicted_answer(dataset, decoded_text)
    return bool(
        predicted is not None and _answers_match(dataset, predicted, target_answer)
    )


def _current_rollout_config_digest(args: argparse.Namespace) -> str:
    return rollout_config_digest(
        receiver_model=args.receiver_model,
        adapter_digest=args.adapter_digest,
        truncation_fraction=args.truncation_fraction,
        max_new_tokens=args.max_new_tokens,
        temperature=args.rollout_temperature,
        seed=args.seed,
    )


def _cached_rollout_current(path: Path, config_digest: str) -> bool:
    if not path.exists():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001
        return False
    return isinstance(payload, dict) and payload.get("config_digest") == config_digest


def prepare_rollouts(
    *,
    args: argparse.Namespace,
    receiver: Any,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    cache_dir: str | Path = DEFAULT_ROLLOUT_CACHE_DIR,
) -> dict[str, int]:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    config_digest = _current_rollout_config_digest(args)
    stats = {"accepted": 0, "skipped": 0, "failed": 0, "written": 0}

    for sample in samples:
        path = rollout_cache_path(str(sample["question"]), cache_dir=cache_root)
        if _cached_rollout_current(path, config_digest):
            stats["skipped"] += 1
            continue

        prefix_state = prepare_receiver_context_latent_prefix_state(
            model=receiver,
            tokenizer=tokenizer,
            context_text=str(sample["context_text"]),
            handoff_step=sample["latents"],
            kv_cache=None,
            suffix_text="",
            latent_position="after_context",
        )
        greedy = greedy_decode_from_prefix(
            model=receiver,
            tokenizer=tokenizer,
            prefix_state=prefix_state,
            max_new_tokens=args.max_new_tokens,
            stop_regex=None,
        )
        decoded_text = str(greedy["decoded_text"])
        greedy_correct = _rollout_verified(args.dataset, decoded_text, sample["answer"])
        attempt = 0
        token_ids = _encoded_rollout_ids(tokenizer, decoded_text, args.max_new_tokens)
        verified = greedy_correct

        if not verified:
            max_sampled_retries = min(4, int(args.rollout_max_tries))
            row_seed = int(args.seed) * 100_003 + int(sample["sample_index"])
            torch.manual_seed(row_seed)
            random.seed(row_seed)
            for retry_index in range(1, max_sampled_retries + 1):
                sampled = sampled_decode_from_prefix(
                    model=receiver,
                    tokenizer=tokenizer,
                    prefix_state=prefix_state,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.rollout_temperature,
                    top_p=args.rollout_top_p,
                    stop_regex=None,
                )
                decoded_text = str(sampled["decoded_text"])
                token_ids = [
                    int(token_id)
                    for token_id in sampled.get("generated_token_ids", [])[: args.max_new_tokens]
                ]
                attempt = retry_index
                verified = _rollout_verified(args.dataset, decoded_text, sample["answer"])
                if verified:
                    break

        payload = {
            "prompt_sha": prompt_sha(str(sample["question"])),
            "sample_index": int(sample["sample_index"]),
            "split": "train",
            "token_ids": token_ids[: args.max_new_tokens],
            "decoded_text": decoded_text,
            "greedy_correct": bool(greedy_correct),
            "attempt": int(attempt),
            "verified": bool(verified),
            "config_digest": config_digest,
        }
        torch.save(payload, path)
        stats["written"] += 1
        if verified:
            stats["accepted"] += 1
        else:
            stats["failed"] += 1

    return stats


def train(args: argparse.Namespace) -> None:
    del args
    _ = RECEIVER_LORA_TARGET_SUFFIXES
    raise SystemExit("chunk 3")


def _torch_dtype(trace_dtype: str) -> torch.dtype:
    return torch.bfloat16 if str(trace_dtype) == "bfloat16" else torch.float32


def _resolve_workspace_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def run_prepare_rollouts(args: argparse.Namespace) -> dict[str, int]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    receiver_tokenizer = AutoTokenizer.from_pretrained(
        args.receiver_model,
        trust_remote_code=True,
    )
    sender_tokenizer = (
        receiver_tokenizer
        if args.sender_model == args.receiver_model
        else AutoTokenizer.from_pretrained(args.sender_model, trust_remote_code=True)
    )
    adapter_payload = load_frozen_adapter(
        _resolve_workspace_path(args.adapter_cache_path),
        adapter_digest=args.adapter_digest,
    )
    rows = load_bridge_sample_rows(args, tokenizer=sender_tokenizer, include_dev=False)
    train_samples = assemble_training_samples(
        rows["train"],
        tokenizer=receiver_tokenizer,
        adapter_payload=adapter_payload,
    )
    if not train_samples:
        raise SystemExit("No cached train traces matched the requested row range.")

    model_dtype = _torch_dtype(args.trace_dtype)
    receiver = AutoModelForCausalLM.from_pretrained(
        args.receiver_model,
        dtype=model_dtype,
        trust_remote_code=True,
    ).to(torch.device(args.device))
    receiver.eval()
    for parameter in receiver.parameters():
        parameter.requires_grad = False

    stats = prepare_rollouts(
        args=args,
        receiver=receiver,
        tokenizer=receiver_tokenizer,
        samples=train_samples,
    )
    print(
        "rollouts: "
        f"accepted={stats['accepted']} failed={stats['failed']} "
        f"skipped={stats['skipped']} written={stats['written']}",
        flush=True,
    )
    return stats


def main() -> None:
    args = build_parser().parse_args()
    if args.prepare_rollouts:
        run_prepare_rollouts(args)
        return
    train(args)


if __name__ == "__main__":
    main()
