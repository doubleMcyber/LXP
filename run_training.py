"""Stage II training runner with WandB logging."""
from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from train_compressor import CompressionTrainConfig, train_reasoner_stage2

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def build_dataloader(
    tokenizer: AutoTokenizer,
    num_samples: int = 32,
    seq_len: int = 64,
    batch_size: int = 2,
) -> DataLoader:
    """Build a small synthetic dataloader for verification."""
    prompts = [
        "Explain the concept of entropy in thermodynamics.",
        "What is the derivative of x squared?",
        "Describe the Pythagorean theorem and its proof.",
        "How does gradient descent work in machine learning?",
        "What are eigenvalues and eigenvectors?",
        "Explain the central limit theorem.",
        "What is the difference between supervised and unsupervised learning?",
        "Describe how a neural network learns through backpropagation.",
    ]
    # Cycle prompts to fill num_samples
    texts = [prompts[i % len(prompts)] for i in range(num_samples)]

    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    labels = input_ids.clone()
    # Mask padding tokens in labels
    labels[attention_mask == 0] = -100

    ds = TensorDataset(input_ids, attention_mask, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def dict_dataloader(dl: DataLoader):
    """Wrap a TensorDataset DataLoader to yield dicts."""
    for batch in dl:
        yield {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2],
        }


def main() -> None:
    cfg = OmegaConf.load(Path(__file__).resolve().parent / "configs" / "main.yaml")
    config = CompressionTrainConfig.from_cfg(cfg)

    torch_dtype = _DTYPE_MAP.get(cfg.torch_dtype, torch.bfloat16)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Dtype: {torch_dtype}")
    print(f"Reasoner (Agent A): {cfg.agent_a_model}")
    print(f"Actor (Agent B): {cfg.agent_b_model}")
    print(f"WandB enabled: {config.wandb_enabled}")
    print(f"WandB project: {config.wandb_project}")
    print()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.agent_a_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading reasoner model...")
    reasoner_cfg = AutoConfig.from_pretrained(cfg.agent_a_model, trust_remote_code=True)
    # Qwen3.5 uses a composite config; the CausalLM needs the text sub-config.
    if hasattr(reasoner_cfg, "text_config"):
        reasoner_cfg = reasoner_cfg.text_config
    reasoner = AutoModelForCausalLM.from_pretrained(
        cfg.agent_a_model,
        config=reasoner_cfg,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )

    print("Loading actor model...")
    actor_cfg = AutoConfig.from_pretrained(cfg.agent_b_model, trust_remote_code=True)
    if hasattr(actor_cfg, "text_config"):
        actor_cfg = actor_cfg.text_config
    actor = AutoModelForCausalLM.from_pretrained(
        cfg.agent_b_model,
        config=actor_cfg,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )

    print("Building dataloader...")
    dl = build_dataloader(tokenizer, num_samples=16, seq_len=32, batch_size=2)

    print("Starting Stage II training...\n")
    history = train_reasoner_stage2(
        reasoner_model=reasoner,
        actor_model=actor,
        train_dataloader=dict_dataloader(dl),
        config=config,
    )

    print(f"\nTraining complete — {len(history)} steps")
    for entry in history:
        print(
            f"  epoch={int(entry['epoch'])} step={int(entry['step'])} "
            f"loss={entry['loss']:.4f} l_task={entry['l_task']:.4f} "
            f"l_pref={entry['l_pref']:.4f} l_geom={entry['l_geom']:.4f}"
        )


if __name__ == "__main__":
    main()
