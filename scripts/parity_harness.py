"""Phase 1 parity harness: why does the benchmark latent path score 0% on the
same truncated latents the Phase 0 trainer continues from at 70%?

Isolates the three suspects independently, on identical inputs:
  A. adapter identity — which ridge mapping each path applies, and how far apart
     their outputs are;
  B. precision — the benchmark applies the mapping in bf16 (activation dtype),
     the trainer projects in fp32;
  C. forward mechanics — the benchmark prefills the context then feeds latents
     with past_key_values (two chunks through the hybrid SSM cache); the trainer
     runs one concatenated forward.

Each stage prints a verdict the next debugging step can act on.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from src.utils.alignment import apply_alignment  # noqa: E402
from src.utils.lm_eval import (  # noqa: E402
    build_position_ids,
    generate_from_prefix_embeddings,
)
from train_latent_bridge import (  # noqa: E402
    CONTINUATION_INSTRUCTION,
    chat_prefix_ids,
    load_bridge_samples,
)

ADAPTER_CACHE_DIR = REPO_ROOT / ".cache" / "generated_trajectory_adapter"


def list_candidate_adapters(model_id: str, truncation: float) -> list[dict]:
    candidates = []
    for path in sorted(glob.glob(str(ADAPTER_CACHE_DIR / "*.pt"))):
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001
            continue
        key = state.get("adapter_cache_key")
        if not isinstance(key, list) or model_id not in str(key):
            continue
        if ["reasoning_truncation", truncation] not in [k for k in key if isinstance(k, list)]:
            continue
        mapping = state.get("mapping_matrix")
        if not isinstance(mapping, torch.Tensor):
            continue
        candidates.append(
            {
                "path": Path(path).name,
                "state": state,
                "key": key,
                "shape": tuple(mapping.shape),
            }
        )
    return candidates


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receiver-model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--truncation-fraction", type=float, default=0.5)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--decode-tokens", type=int, default=48)
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.receiver_model, trust_remote_code=True)
    receiver = AutoModelForCausalLM.from_pretrained(
        args.receiver_model, dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    receiver.eval()
    embedding = receiver.get_input_embeddings()
    model_dtype = embedding.weight.dtype

    samples = load_bridge_samples(
        dataset="gsm8k",
        split="validation",
        limit=args.samples + 2,
        model_id=args.receiver_model,
        torch_dtype="bfloat16",
        truncation_fraction=args.truncation_fraction,
        tokenizer=tokenizer,
        max_continuation_tokens=256,
    )[: args.samples]

    print("\n================ STAGE A: adapter identity ================")
    candidates = list_candidate_adapters(args.receiver_model, args.truncation_fraction)
    for cand in candidates:
        meta = [k for k in cand["key"] if isinstance(k, str)]
        print(f"  {cand['path'][:60]} shape={cand['shape']} key_strs={meta[:8]}")
    if not candidates:
        raise SystemExit("no candidate adapters found")
    # the trainer's find_ridge_warm_start keeps the LAST matching candidate
    trainer_adapter = candidates[-1]["state"]
    # the benchmark with strategy=ridge/source=generated_text selects by exact key;
    # emulate by preferring a candidate whose key mentions generated_text + ridge
    bench_adapter = next(
        (
            c["state"]
            for c in candidates
            if "generated_text" in str(c["key"]) and "ridge" in str(c["key"])
        ),
        candidates[0]["state"],
    )
    same_file = trainer_adapter is bench_adapter
    print(f"  trainer picks: {candidates[-1]['path'][:60]}")
    print(f"  same adapter object: {same_file}")

    sample = samples[0]
    latents = sample["latents"].unsqueeze(0)

    def trainer_projection(x: torch.Tensor) -> torch.Tensor:
        mapping = trainer_adapter["mapping_matrix"].float()
        bias = trainer_adapter.get("mapping_bias")
        out = x.float() @ mapping
        if bias is not None:
            out = out + bias.float().reshape(1, 1, -1)
        return out

    proj_fp32 = trainer_projection(latents)
    bench_fp32 = apply_alignment(latents.float(), bench_adapter)
    bench_bf16 = apply_alignment(latents.to(torch.bfloat16), bench_adapter).float()
    cos_ab = torch.nn.functional.cosine_similarity(proj_fp32, bench_fp32, dim=-1).mean()
    cos_dtype = torch.nn.functional.cosine_similarity(bench_fp32, bench_bf16, dim=-1).mean()
    print(f"  cos(trainer_proj, bench_fp32) = {float(cos_ab):.4f}")
    print(f"  cos(bench_fp32, bench_bf16)  = {float(cos_dtype):.4f}  (STAGE B precision)")
    print(
        "  note: trainer projection skips the adapter's adaptive projection states; "
        "apply_alignment includes them"
    )

    print("\n================ STAGE C: chunked vs single forward ================")
    prompt_ids = chat_prefix_ids(tokenizer, sample["question"], CONTINUATION_INSTRUCTION)
    prompt_tensor = torch.tensor([prompt_ids], device=device)
    prompt_embeds = embedding(prompt_tensor).to(model_dtype)
    bridged = proj_fp32.to(device=device, dtype=model_dtype)

    # single concatenated forward (trainer mechanics)
    full = torch.cat([prompt_embeds, bridged], dim=1)
    full_mask = torch.ones(full.shape[:2], dtype=torch.long, device=device)
    single = receiver(
        inputs_embeds=full,
        attention_mask=full_mask,
        position_ids=build_position_ids(full_mask),
        use_cache=True,
        return_dict=True,
    )
    single_logits = single.logits[:, -1, :].float()

    # chunked forward (benchmark mechanics): context prefill, then latents with past
    ctx_mask = torch.ones(prompt_embeds.shape[:2], dtype=torch.long, device=device)
    ctx = receiver(
        inputs_embeds=prompt_embeds,
        attention_mask=ctx_mask,
        position_ids=build_position_ids(ctx_mask),
        use_cache=True,
        return_dict=True,
    )
    chunk_mask = torch.ones(
        (1, prompt_embeds.shape[1] + bridged.shape[1]), dtype=torch.long, device=device
    )
    chunk_positions = build_position_ids(chunk_mask)[:, -bridged.shape[1] :]
    chunked = receiver(
        inputs_embeds=bridged,
        past_key_values=ctx.past_key_values,
        attention_mask=chunk_mask,
        position_ids=chunk_positions,
        use_cache=True,
        return_dict=True,
    )
    chunked_logits = chunked.logits[:, -1, :].float()

    max_diff = float((single_logits - chunked_logits).abs().max())
    top_single = int(single_logits.argmax())
    top_chunked = int(chunked_logits.argmax())
    print(f"  max |logit diff| single vs chunked: {max_diff:.4f}")
    print(
        f"  top-1 single={top_single} ({tokenizer.decode([top_single])!r}) | "
        f"chunked={top_chunked} ({tokenizer.decode([top_chunked])!r}) | "
        f"match={top_single == top_chunked}"
    )

    print("\n================ STAGE D: end-to-end decode comparison ================")
    for name, mapped in (
        ("trainer_proj_fp32", proj_fp32),
        ("bench_apply_bf16", bench_bf16),
    ):
        prefix = torch.cat([prompt_embeds, mapped.to(device=device, dtype=model_dtype)], dim=1)
        decoded = generate_from_prefix_embeddings(
            model=receiver,
            tokenizer=tokenizer,
            prefix_embeds=prefix,
            max_new_tokens=args.decode_tokens,
        )["decoded_text"]
        print(f"  [{name}] {decoded[:160]!r}")
    print(f"\n  gold answer: {sample['answer']}")


if __name__ == "__main__":
    main()
