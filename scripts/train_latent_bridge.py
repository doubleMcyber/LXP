"""Phase 0 of the latent-consumption plan: train a behavioral latent bridge.

Trains a small bridge (input projection + LatentSoftPromptDecoder) that turns a
sender's truncated mid-reasoning latents into K virtual receiver tokens, with the
receiver's own NLL of the sender's post-truncation continuation as the loss —
backpropagated through the FROZEN receiver into the bridge only.

Everything heavy is already on disk: sender traces come from the benchmark trace
cache (hidden states are causal, so truncated latents are prefix slices), so the
sender model is never loaded. The whole budget goes to the frozen receiver.

Go/no-go (from docs/latent_consumption_training_plan.md): the bridge must beat
the receiver-alone floor on held-out continuation; the truncated-text baseline is
the target. Both baselines are re-measured in-script under the identical prompt
layout and decode budget so the triplet is directly comparable.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from benchmark_all import (  # noqa: E402
    _predicted_answer,
    _answers_match,
    _target_answer,
    _truncate_reasoning_token_ids,
)
from src.data.loader import get_dataloader, pick_field  # noqa: E402
from src.models.hidden_state import LatentSoftPromptDecoder  # noqa: E402
from src.utils.lm_eval import generate_from_prefix_embeddings  # noqa: E402

TRACE_CACHE_DIR = REPO_ROOT / ".cache" / "generated_trajectory_traces"
ADAPTER_CACHE_DIR = REPO_ROOT / ".cache" / "generated_trajectory_adapter"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "latent_bridge"

CONTINUATION_INSTRUCTION = (
    "An assistant started solving this problem but stopped mid-reasoning. "
    "Its partial work is handed to you as a latent summary after this message. "
    "Continue the reasoning step by step and finish with exactly one line: "
    "Final answer: <answer>."
)
TEXT_INSTRUCTION = (
    "An assistant started solving this problem but stopped mid-reasoning. "
    "Its unfinished work follows. Continue the reasoning step by step and finish "
    "with exactly one line: Final answer: <answer>."
)
ALONE_INSTRUCTION = "Think step by step, then give the final answer."


class LatentBridge(nn.Module):
    """sender consensus latents -> K virtual receiver-embedding tokens."""

    def __init__(
        self,
        sender_dim: int,
        receiver_dim: int,
        *,
        output_steps: int = 48,
        num_heads: int = 8,
        hidden_multiplier: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(sender_dim, receiver_dim, bias=True)
        self.decoder = LatentSoftPromptDecoder(
            receiver_dim,
            output_steps=output_steps,
            num_heads=num_heads,
            hidden_multiplier=hidden_multiplier,
            dropout=dropout,
        )
        nn.init.zeros_(self.input_proj.bias)
        if sender_dim == receiver_dim:
            with torch.no_grad():
                self.input_proj.weight.copy_(torch.eye(receiver_dim))

    def warm_start_projection(self, mapping_matrix: torch.Tensor, bias: Optional[torch.Tensor]) -> None:
        # pipeline convention is hidden @ M; nn.Linear computes x @ W.T -> W = M.T
        with torch.no_grad():
            self.input_proj.weight.copy_(mapping_matrix.float().T)
            if bias is not None:
                self.input_proj.bias.copy_(bias.float().reshape(-1))

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.input_proj(latents.float()))


def _prompt_sha(prompt: str) -> str:
    return hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()


def index_cached_traces(model_id: str, torch_dtype: str) -> dict[str, Path]:
    """Map prompt sha256 -> trace path for one (sender model, dtype)."""
    by_prompt: dict[str, Path] = {}
    for path in glob.glob(str(TRACE_CACHE_DIR / "*.pt")):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001
            continue
        key = payload.get("cache_key")
        if not isinstance(key, list) or len(key) < 4:
            continue
        if key[1] != model_id or key[2] != torch_dtype:
            continue
        if bool(payload.get("generated_latent_includes_prompt", False)):
            continue
        by_prompt[str(key[-1])] = Path(path)
    return by_prompt


def load_bridge_samples(
    *,
    dataset: str,
    split: str,
    limit: int,
    model_id: str,
    torch_dtype: str,
    truncation_fraction: float,
    tokenizer: Any,
    max_continuation_tokens: int,
    validation_size: int = 256,
) -> list[dict[str, Any]]:
    """Build (truncated latents, continuation ids, question, answer) samples.

    Only rows whose sender trace is already cached are used — Phase 0 never runs
    the sender. Rows whose sender failed to produce a complete final answer are
    skipped (their continuation carries no answer signal). ``validation_size``
    must match the benchmark config (configs/main.yaml datasets.gsm8k) so train
    and validation rows line up with the cached traces.
    """
    trace_index = index_cached_traces(model_id, torch_dtype)
    rows = get_dataloader(dataset, limit=limit, split=split, validation_size=validation_size)
    samples: list[dict[str, Any]] = []
    for row_index in range(len(rows)):
        row = rows[row_index]
        question = str(pick_field(row, ("question", "problem")))
        trace_path = trace_index.get(_prompt_sha(question))
        if trace_path is None:
            continue
        payload = torch.load(trace_path, map_location="cpu", weights_only=False)
        if payload.get("generated_reasoning_status") != "complete":
            continue
        full_ids = [int(t) for t in payload["generated_token_ids"]]
        truncated_ids = _truncate_reasoning_token_ids(tokenizer, full_ids, truncation_fraction)
        cut = len(truncated_ids)
        if cut < 4 or cut >= len(full_ids):
            continue
        continuation_ids = full_ids[cut : cut + max_continuation_tokens]
        latents = payload["consensus_hidden_states"][:, :cut, :].float().clone()
        samples.append(
            {
                "question": question,
                # the gsm8k answer field is the full rationale; score against the
                # scalar exactly as the benchmark does
                "answer": _target_answer(dataset, row),
                "latents": latents.squeeze(0),
                "continuation_ids": continuation_ids,
                "truncated_text": tokenizer.decode(truncated_ids, skip_special_tokens=True),
                "sample_index": row_index,
            }
        )
    return samples


def find_ridge_warm_start(model_id: str, truncation_fraction: float) -> Optional[dict[str, Any]]:
    """Locate a cached ridge transcoder for this pair/truncation to initialize from."""
    best: Optional[dict[str, Any]] = None
    for path in glob.glob(str(ADAPTER_CACHE_DIR / "*.pt")):
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001
            continue
        key = state.get("adapter_cache_key")
        if not isinstance(key, list):
            continue
        if model_id not in str(key):
            continue
        if ["reasoning_truncation", truncation_fraction] not in [k for k in key if isinstance(k, list)]:
            continue
        mapping = state.get("mapping_matrix")
        if isinstance(mapping, torch.Tensor) and mapping.dim() == 2:
            best = {
                "mapping_matrix": mapping,
                "mapping_bias": state.get("mapping_bias"),
                "path": path,
            }
    return best


def chat_prefix_ids(tokenizer: Any, question: str, instruction: str, body: str = "") -> list[int]:
    user_message = f"{question}\n\n{instruction}"
    if body:
        user_message = f"{question}\n\nUnfinished reasoning:\n{body}\n\n{instruction}"
    try:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:  # noqa: BLE001
        text = user_message
    return tokenizer.encode(text, add_special_tokens=False)


def training_step_loss(
    *,
    receiver: Any,
    tokenizer: Any,
    bridge: LatentBridge,
    sample: dict[str, Any],
    device: torch.device,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    embedding = receiver.get_input_embeddings()
    prompt_ids = chat_prefix_ids(tokenizer, sample["question"], CONTINUATION_INSTRUCTION)
    prompt_embeds = embedding(torch.tensor([prompt_ids], device=device))
    bridge_out = bridge(sample["latents"].unsqueeze(0).to(device)).to(dtype=model_dtype)
    continuation = torch.tensor([sample["continuation_ids"]], device=device)
    continuation_embeds = embedding(continuation)
    inputs_embeds = torch.cat([prompt_embeds.to(model_dtype), bridge_out, continuation_embeds.to(model_dtype)], dim=1)
    prefix_len = prompt_embeds.shape[1] + bridge_out.shape[1]
    labels = torch.cat(
        [
            torch.full((1, prefix_len), -100, dtype=torch.long, device=device),
            continuation,
        ],
        dim=1,
    )
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
    outputs = receiver(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
        return_dict=True,
    )
    return outputs.loss


@torch.no_grad()
def evaluate_triplet(
    *,
    receiver: Any,
    tokenizer: Any,
    bridge: Optional[LatentBridge],
    samples: list[dict[str, Any]],
    device: torch.device,
    model_dtype: torch.dtype,
    max_new_tokens: int,
    dataset: str,
) -> dict[str, Any]:
    embedding = receiver.get_input_embeddings()
    results: dict[str, Any] = {"rows": []}
    counts = {"bridge": 0, "alone": 0, "text": 0, "instruction_only": 0}
    for sample in samples:
        row: dict[str, Any] = {"sample_index": sample["sample_index"], "answer": sample["answer"]}
        variants: dict[str, torch.Tensor] = {}
        alone_ids = chat_prefix_ids(tokenizer, sample["question"], ALONE_INSTRUCTION)
        variants["alone"] = embedding(torch.tensor([alone_ids], device=device)).to(model_dtype)
        text_ids = chat_prefix_ids(
            tokenizer, sample["question"], TEXT_INSTRUCTION, body=sample["truncated_text"]
        )
        variants["text"] = embedding(torch.tensor([text_ids], device=device)).to(model_dtype)
        # control for instruction wording: the continuation instruction with NO
        # latents attributes any bridge-vs-alone gap to the latents themselves
        instr_ids = chat_prefix_ids(tokenizer, sample["question"], CONTINUATION_INSTRUCTION)
        variants["instruction_only"] = embedding(
            torch.tensor([instr_ids], device=device)
        ).to(model_dtype)
        if bridge is not None:
            prompt_embeds = variants["instruction_only"]
            bridge_out = bridge(sample["latents"].unsqueeze(0).to(device)).to(dtype=model_dtype)
            variants["bridge"] = torch.cat([prompt_embeds, bridge_out], dim=1)
        for name, prefix in variants.items():
            decoded = generate_from_prefix_embeddings(
                model=receiver,
                tokenizer=tokenizer,
                prefix_embeds=prefix,
                max_new_tokens=max_new_tokens,
            )["decoded_text"]
            predicted = _predicted_answer(dataset, decoded)
            correct = bool(
                predicted is not None and _answers_match(dataset, predicted, sample["answer"])
            )
            counts[name] += int(correct)
            row[name] = {"predicted": predicted, "correct": correct, "decoded": decoded[:200]}
        results["rows"].append(row)
    total = max(1, len(samples))
    results["accuracy"] = {
        name: round(100.0 * counts[name] / total, 2)
        for name in counts
        if name != "bridge" or bridge is not None
    }
    results["sample_count"] = len(samples)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receiver-model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--sender-model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--trace-dtype", default="bfloat16")
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--train-limit", type=int, default=128)
    parser.add_argument("--eval-limit", type=int, default=8)
    parser.add_argument("--truncation-fraction", type=float, default=0.5)
    parser.add_argument("--validation-size", type=int, default=256)
    parser.add_argument(
        "--output-steps",
        type=int,
        default=0,
        help=(
            "0 = refinement mode: cross-attention refines the full-length projected "
            "sequence (identity-initialized, so training starts from the ridge "
            "transcoder). K>0 compresses to K virtual tokens but adds a K*d^2 "
            "projection that overfits small sample counts."
        ),
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-continuation-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0, help="debug cap; 0 = unlimited")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "bridge_checkpoint.pt"
    report_path = output_dir / "bridge_report.json"
    device = torch.device(args.device)
    model_dtype = torch.bfloat16 if args.trace_dtype == "bfloat16" else torch.float32

    print(f"Loading frozen receiver {args.receiver_model} ({model_dtype}) ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.receiver_model, trust_remote_code=True)
    receiver = AutoModelForCausalLM.from_pretrained(
        args.receiver_model, dtype=model_dtype, trust_remote_code=True
    ).to(device)
    receiver.eval()
    for parameter in receiver.parameters():
        parameter.requires_grad = False

    sender_tokenizer = (
        tokenizer
        if args.sender_model == args.receiver_model
        else AutoTokenizer.from_pretrained(args.sender_model, trust_remote_code=True)
    )

    print("Indexing cached traces and building samples ...", flush=True)
    train_samples = load_bridge_samples(
        dataset=args.dataset,
        split=args.train_split,
        limit=args.train_limit,
        model_id=args.sender_model,
        torch_dtype=args.trace_dtype,
        truncation_fraction=args.truncation_fraction,
        tokenizer=sender_tokenizer,
        max_continuation_tokens=args.max_continuation_tokens,
        validation_size=args.validation_size,
    )
    eval_samples = load_bridge_samples(
        dataset=args.dataset,
        split=args.eval_split,
        limit=args.eval_limit,
        model_id=args.sender_model,
        torch_dtype=args.trace_dtype,
        truncation_fraction=args.truncation_fraction,
        tokenizer=sender_tokenizer,
        max_continuation_tokens=args.max_continuation_tokens,
        validation_size=args.validation_size,
    )
    print(
        f"train samples: {len(train_samples)} | eval samples: {len(eval_samples)}",
        flush=True,
    )
    if not train_samples or not eval_samples:
        raise SystemExit("No cached traces matched — run the benchmark trace warmers first.")

    sender_dim = int(train_samples[0]["latents"].shape[-1])
    receiver_dim = int(receiver.get_input_embeddings().weight.shape[-1])
    bridge = LatentBridge(sender_dim, receiver_dim, output_steps=args.output_steps).to(device)
    warm = find_ridge_warm_start(args.sender_model, args.truncation_fraction)
    if warm is not None:
        bridge.warm_start_projection(warm["mapping_matrix"], warm.get("mapping_bias"))
        print(f"Warm-started input projection from {warm['path']}", flush=True)
    else:
        print("No ridge warm start found; using identity/random init", flush=True)

    optimizer = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(1, args.epochs * len(train_samples))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    start_epoch, global_step, best_accuracy = 0, 0, -1.0
    if checkpoint_path.exists():
        snapshot = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        bridge.load_state_dict(snapshot["bridge"])
        optimizer.load_state_dict(snapshot["optimizer"])
        scheduler.load_state_dict(snapshot["scheduler"])
        start_epoch = int(snapshot["epoch"])
        global_step = int(snapshot["global_step"])
        best_accuracy = float(snapshot.get("best_accuracy", -1.0))
        print(f"Resumed from step {global_step} (epoch {start_epoch})", flush=True)

    history: list[dict[str, Any]] = []
    if report_path.exists():
        try:
            history = json.loads(report_path.read_text()).get("history", [])
        except Exception:  # noqa: BLE001
            history = []

    def run_eval(tag: str) -> dict[str, Any]:
        bridge.eval()
        outcome = evaluate_triplet(
            receiver=receiver,
            tokenizer=tokenizer,
            bridge=bridge,
            samples=eval_samples,
            device=device,
            model_dtype=model_dtype,
            max_new_tokens=args.max_new_tokens,
            dataset=args.dataset,
        )
        outcome["tag"] = tag
        outcome["global_step"] = global_step
        history.append({k: outcome[k] for k in ("tag", "global_step", "accuracy")})
        report_path.write_text(
            json.dumps({"history": history, "latest": outcome}, indent=2), encoding="utf-8"
        )
        print(f"[eval:{tag}] accuracy={outcome['accuracy']}", flush=True)
        bridge.train()
        return outcome

    if args.eval_only:
        run_eval("eval_only")
        return

    print(
        f"Training bridge: {sum(p.numel() for p in bridge.parameters())/1e6:.1f}M params, "
        f"{args.epochs} epochs x {len(train_samples)} samples",
        flush=True,
    )
    run_eval("baseline")
    bridge.train()
    order = list(range(len(train_samples)))
    for epoch in range(start_epoch, args.epochs):
        random.Random(args.seed + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for position, sample_index in enumerate(order):
            if args.max_steps and global_step >= args.max_steps:
                break
            loss = training_step_loss(
                receiver=receiver,
                tokenizer=tokenizer,
                bridge=bridge,
                sample=train_samples[sample_index],
                device=device,
                model_dtype=model_dtype,
            )
            (loss / args.grad_accum).backward()
            running += float(loss.detach().cpu().item())
            global_step += 1
            if global_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if global_step % 10 == 0:
                print(
                    f"epoch {epoch} step {global_step}/{total_steps} "
                    f"loss={running / max(1, (position + 1)):.4f}",
                    flush=True,
                )
            if global_step % args.checkpoint_every == 0:
                torch.save(
                    {
                        "bridge": bridge.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_accuracy": best_accuracy,
                        "args": vars(args),
                    },
                    checkpoint_path,
                )
                if device.type == "mps" and hasattr(torch, "mps"):
                    torch.mps.empty_cache()
        outcome = run_eval(f"epoch_{epoch}")
        bridge_accuracy = float(outcome["accuracy"].get("bridge", -1.0))
        if bridge_accuracy > best_accuracy:
            best_accuracy = bridge_accuracy
            torch.save(
                {"bridge": bridge.state_dict(), "accuracy": outcome["accuracy"], "args": vars(args)},
                output_dir / "bridge_best.pt",
            )
        torch.save(
            {
                "bridge": bridge.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
                "global_step": global_step,
                "best_accuracy": best_accuracy,
                "args": vars(args),
            },
            checkpoint_path,
        )
    print(f"DONE: best bridge accuracy {best_accuracy} (report: {report_path})", flush=True)


if __name__ == "__main__":
    main()
