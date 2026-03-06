"""Stage II training runner with text-first batching and held-out evaluation."""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Iterable

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.data.loader import get_dataset_split, pick_field
from src.utils.alignment import apply_orthogonal_mapping
from src.utils.benchmarking import build_training_phase2_report, write_json
from src.utils.metrics import extract_boxed_text, normalize_answer
from train_compressor import (
    CompressionTrainConfig,
    compress_latent_trajectory,
    resolve_training_alignment_context,
    train_reasoner_stage2,
)

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
_GSM8K_FINAL_ANSWER_REGEX = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")
_NUMERIC_ANSWER_REGEX = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_SMOKE_TRAIN_EXAMPLES: tuple[dict[str, str], ...] = (
    {"prompt": "What is 2 + 2?", "answer": "4"},
    {"prompt": "What is 7 * 6?", "answer": "42"},
    {"prompt": "What is 9 - 4?", "answer": "5"},
    {"prompt": "What is 12 / 3?", "answer": "4"},
    {"prompt": "What is the derivative of x squared?", "answer": "2x"},
    {"prompt": "What is the square root of 81?", "answer": "9"},
    {"prompt": "What is 15 + 10?", "answer": "25"},
    {"prompt": "What is 3 cubed?", "answer": "27"},
)
_SMOKE_EVAL_EXAMPLES: tuple[dict[str, str], ...] = (
    {"prompt": "What is 8 + 5?", "answer": "13"},
    {"prompt": "What is 6 * 7?", "answer": "42"},
    {"prompt": "What is the derivative of x cubed?", "answer": "3x^2"},
)


def _load_cfg() -> Any:
    cfg_path = Path(__file__).resolve().parent / "configs" / "main.yaml"
    return OmegaConf.merge(OmegaConf.load(cfg_path), OmegaConf.from_cli())


def _reporting_cfg(cfg: Any) -> Any:
    return getattr(getattr(cfg, "reporting", None), "training", None)


def _phase2_gate_cfg(cfg: Any) -> Any:
    return getattr(getattr(getattr(cfg, "reporting", None), "phase_gates", None), "phase2", None)


def _ensure_padding_token(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token


def _build_position_ids(attention_mask: torch.Tensor) -> torch.LongTensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    return position_ids.clamp_min_(0)


def _model_backbone(model: AutoModelForCausalLM) -> Any:
    if hasattr(model, "model"):
        return model.model
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def _load_model(model_name: str, *, torch_dtype: torch.dtype, device_map: str) -> AutoModelForCausalLM:
    model_cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if hasattr(model_cfg, "text_config"):
        model_cfg = model_cfg.text_config
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        config=model_cfg,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )


def _collate_text_batch(batch: list[str]) -> dict[str, list[str]]:
    return {"texts": [str(text) for text in batch]}


def _build_text_dataloader(
    texts: Iterable[str],
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        list(texts),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_text_batch,
    )


def _smoke_supervision_texts(num_samples: int) -> list[str]:
    examples = [
        f"Question: {example['prompt']}\nAnswer: {example['answer']}"
        for example in _SMOKE_TRAIN_EXAMPLES
    ]
    return [examples[index % len(examples)] for index in range(num_samples)]


def _extract_gsm8k_target_answer(text: str) -> str | None:
    match = _GSM8K_FINAL_ANSWER_REGEX.search(text)
    return None if match is None else match.group(1)


def _extract_gsm8k_predicted_answer(text: str) -> str | None:
    matches = _NUMERIC_ANSWER_REGEX.findall(text)
    if not matches:
        return None
    return matches[-1]


def _normalize_numeric_answer(answer: str | None) -> str | None:
    normalized = normalize_answer(answer)
    if normalized is None:
        return None
    return normalized.replace(",", "")


def _target_answer(dataset_name: str, row: Any) -> str | None:
    if dataset_name == "gsm8k":
        return _extract_gsm8k_target_answer(pick_field(row, ("answer", "solution")))
    return extract_boxed_text(pick_field(row, ("solution", "answer")))


def _predicted_answer(dataset_name: str, decoded_text: str) -> str | None:
    if dataset_name == "gsm8k":
        return _extract_gsm8k_predicted_answer(decoded_text)
    if dataset_name == "smoke":
        return normalize_answer(decoded_text)
    return extract_boxed_text(decoded_text)


def _answers_match(dataset_name: str, predicted_answer: str | None, target_answer: str | None) -> bool:
    if dataset_name == "gsm8k":
        return _normalize_numeric_answer(predicted_answer) == _normalize_numeric_answer(target_answer)
    return normalize_answer(predicted_answer) == normalize_answer(target_answer)


def _build_real_examples(dataset_name: str, split: str, limit: int) -> list[dict[str, str | None]]:
    rows = get_dataset_split(dataset_name, split, limit=limit)
    examples: list[dict[str, str | None]] = []
    for row in rows:
        prompt = pick_field(row, ("question", "problem"))
        solution = pick_field(row, ("answer", "solution"))
        examples.append(
            {
                "prompt": prompt,
                "answer": _target_answer(dataset_name, row),
                "supervision_text": f"Question: {prompt}\nAnswer: {solution}",
            }
        )
    return examples


def _build_training_payloads(cfg: Any) -> tuple[DataLoader, list[dict[str, str | None]], str]:
    data_cfg = getattr(cfg.training, "data", None)
    mode = str(getattr(data_cfg, "mode", "smoke")).lower()
    batch_size = int(getattr(data_cfg, "batch_size", 2))

    if mode == "smoke":
        num_samples = int(getattr(data_cfg, "smoke_num_samples", 16))
        train_loader = _build_text_dataloader(
            _smoke_supervision_texts(num_samples),
            batch_size=batch_size,
            shuffle=True,
        )
        return train_loader, list(_SMOKE_EVAL_EXAMPLES), "smoke"

    if mode == "real":
        dataset_name = str(getattr(data_cfg, "dataset_name", "gsm8k")).lower()
        train_limit = int(getattr(data_cfg, "train_limit", 32))
        eval_limit = int(getattr(data_cfg, "eval_limit", 16))
        train_examples = _build_real_examples(dataset_name, "train", train_limit)
        eval_examples = _build_real_examples(dataset_name, "test", eval_limit)
        train_loader = _build_text_dataloader(
            [str(example["supervision_text"]) for example in train_examples],
            batch_size=batch_size,
            shuffle=True,
        )
        return train_loader, eval_examples, dataset_name

    raise ValueError("training.data.mode must be either 'smoke' or 'real'")


def _tokenize_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    device: torch.device,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        prompt,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


def _decode_actor_response(
    actor_model: AutoModelForCausalLM,
    actor_tokenizer: Any,
    aligned_latents: torch.Tensor,
    *,
    max_new_tokens: int,
) -> str:
    actor_device = next(actor_model.parameters()).device
    embed_dtype = actor_model.get_input_embeddings().weight.dtype
    handoff = aligned_latents.to(device=actor_device, dtype=embed_dtype)
    attention_mask = torch.ones(
        (handoff.shape[0], handoff.shape[1]),
        dtype=torch.long,
        device=actor_device,
    )
    position_ids = _build_position_ids(attention_mask)
    generated_token_ids: list[int] = []
    eos_token_id = getattr(actor_tokenizer, "eos_token_id", None)

    with torch.no_grad():
        outputs = actor_model(
            inputs_embeds=handoff,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )

    for _ in range(max_new_tokens):
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        next_token_id = int(next_token.item())
        if eos_token_id is not None and next_token_id == eos_token_id:
            break
        generated_token_ids.append(next_token_id)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device),
            ],
            dim=1,
        )
        position_ids = _build_position_ids(attention_mask)[:, -1:]
        with torch.no_grad():
            outputs = actor_model(
                input_ids=next_token.unsqueeze(-1),
                past_key_values=outputs.past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
            )

    return actor_tokenizer.decode(generated_token_ids, skip_special_tokens=True)


def _build_evaluation_callback(
    *,
    eval_examples: list[dict[str, str | None]],
    dataset_name: str,
    reasoner_tokenizer: Any,
    actor_tokenizer: Any,
    config: CompressionTrainConfig,
    max_new_tokens: int,
) -> Any:
    def evaluate(
        reasoner_model: AutoModelForCausalLM,
        actor_model: AutoModelForCausalLM,
        alignment_context: dict[str, Any],
    ) -> dict[str, float]:
        if not eval_examples:
            return {
                "heldout_exact_match_accuracy": 0.0,
                "heldout_eval_samples": 0.0,
            }

        reasoner_device = next(reasoner_model.parameters()).device
        reasoner_backbone = _model_backbone(reasoner_model)
        alignment_q = alignment_context["alignment_q"].to(
            device=reasoner_device,
            dtype=reasoner_model.get_input_embeddings().weight.dtype,
        )
        correct_count = 0

        for example in eval_examples:
            prompt = str(example["prompt"])
            target_answer = example["answer"]
            input_ids, attention_mask = _tokenize_prompt(
                reasoner_tokenizer,
                prompt,
                device=reasoner_device,
                max_length=config.reasoner_max_length,
            )
            with torch.no_grad():
                outputs = reasoner_backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
            compressed_latents = compress_latent_trajectory(
                outputs.last_hidden_state,
                compressed_steps=config.compressed_steps,
            )
            aligned_latents = apply_orthogonal_mapping(compressed_latents, alignment_q)
            decoded_text = _decode_actor_response(
                actor_model,
                actor_tokenizer,
                aligned_latents,
                max_new_tokens=max_new_tokens,
            )
            predicted_answer = _predicted_answer(dataset_name, decoded_text)
            if dataset_name == "smoke":
                predicted_answer = normalize_answer(decoded_text)
            if _answers_match(dataset_name, predicted_answer, target_answer):
                correct_count += 1

        return {
            "heldout_exact_match_accuracy": 100.0 * correct_count / len(eval_examples),
            "heldout_eval_samples": float(len(eval_examples)),
        }

    return evaluate


def _write_history_csv(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    fieldnames = sorted({key for entry in history for key in entry.keys()})
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    cfg = _load_cfg()
    config = CompressionTrainConfig.from_cfg(cfg)
    torch_dtype = _DTYPE_MAP.get(cfg.torch_dtype, torch.bfloat16)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = getattr(cfg.training, "data", None)
    training_mode = str(getattr(data_cfg, "mode", "smoke")).lower()
    reporting_cfg = _reporting_cfg(cfg)
    history_output_path = Path(str(getattr(reporting_cfg, "history_output", "training_history.csv")))
    report_output_path = Path(str(getattr(reporting_cfg, "report_output", "training_report.json")))
    phase2_cfg = _phase2_gate_cfg(cfg)
    baseline_accuracy_percentage = getattr(
        getattr(cfg.training, "evaluation", None),
        "baseline_accuracy_percentage",
        None,
    )
    seed_count = int(getattr(getattr(cfg.training, "evaluation", None), "seed_count", 1))

    print(f"Device: {device}")
    print(f"Dtype: {torch_dtype}")
    print(f"Reasoner (Agent A): {cfg.agent_a_model}")
    print(f"Actor (Agent B): {cfg.agent_b_model}")
    print(f"Training mode: {training_mode}")
    print(f"WandB enabled: {config.wandb_enabled}")
    print(f"WandB project: {config.wandb_project}")
    print()

    print("Loading tokenizers...")
    reasoner_tokenizer = AutoTokenizer.from_pretrained(cfg.agent_a_model, trust_remote_code=True)
    actor_tokenizer = AutoTokenizer.from_pretrained(cfg.agent_b_model, trust_remote_code=True)
    _ensure_padding_token(reasoner_tokenizer)
    _ensure_padding_token(actor_tokenizer)

    print("Loading reasoner model...")
    reasoner = _load_model(cfg.agent_a_model, torch_dtype=torch_dtype, device_map=device)

    print("Loading actor model...")
    actor = _load_model(cfg.agent_b_model, torch_dtype=torch_dtype, device_map=device)

    print("Building text-first dataloaders...")
    train_loader, eval_examples, dataset_name = _build_training_payloads(cfg)
    evaluation_max_new_tokens = int(getattr(getattr(cfg.training, "evaluation", None), "max_new_tokens", 16))
    evaluation_fn = _build_evaluation_callback(
        eval_examples=eval_examples,
        dataset_name=dataset_name,
        reasoner_tokenizer=reasoner_tokenizer,
        actor_tokenizer=actor_tokenizer,
        config=config,
        max_new_tokens=evaluation_max_new_tokens,
    )
    alignment_context = resolve_training_alignment_context(
        reasoner_model=reasoner,
        actor_model=actor,
        reasoner_tokenizer=reasoner_tokenizer,
        actor_tokenizer=actor_tokenizer,
        alignment_cfg=cfg,
    )

    print("Starting Stage II training...\n")
    history = train_reasoner_stage2(
        reasoner_model=reasoner,
        actor_model=actor,
        train_dataloader=train_loader,
        config=config,
        reasoner_tokenizer=reasoner_tokenizer,
        actor_tokenizer=actor_tokenizer,
        alignment_cfg=cfg,
        evaluation_fn=evaluation_fn,
    )
    _write_history_csv(history_output_path, history)
    training_report = build_training_phase2_report(
        history=history,
        cfg=cfg,
        alignment_context=alignment_context,
        dataset_name=dataset_name,
        training_mode=training_mode,
        seed_count=seed_count,
        required_seed_count=int(getattr(phase2_cfg, "required_seed_count", 3)),
        min_accuracy_retention_ratio=float(
            getattr(phase2_cfg, "min_accuracy_retention_ratio", 0.85)
        ),
        baseline_accuracy_percentage=None
        if baseline_accuracy_percentage is None
        else float(baseline_accuracy_percentage),
    )
    write_json(report_output_path, training_report)

    print(f"\nTraining complete — {len(history)} logged entries")
    print(
        "Alignment mode: "
        f"{alignment_context['alignment_mode']} | "
        f"anchors={alignment_context['semantic_anchor_count']} | "
        f"cache_hit={alignment_context['global_alignment_cache_hit']}"
    )
    for entry in history:
        if "loss" in entry:
            print(
                f"  epoch={int(entry['epoch'])} step={int(entry['step'])} "
                f"loss={entry['loss']:.4f} l_task={entry['l_task']:.4f} "
                f"l_pref={entry['l_pref']:.4f} l_geom={entry['l_geom']:.4f} "
                f"heldout_exact_match_accuracy={entry.get('heldout_exact_match_accuracy', float('nan')):.2f}"
            )
            continue
        print(
            f"  epoch={int(entry['epoch'])} step={int(entry['step'])} "
            f"heldout_exact_match_accuracy={entry['heldout_exact_match_accuracy']:.2f} "
            f"heldout_eval_samples={entry['heldout_eval_samples']:.0f}"
        )

    print(f"Wrote training history to {history_output_path}")
    print(f"Wrote phase-2 training report to {report_output_path}")
    print(f"Phase 2 gate passed: {training_report['passed']}")


if __name__ == "__main__":
    main()
