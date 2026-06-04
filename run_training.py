"""Stage II training runner with text-first batching and held-out evaluation."""
from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.data.loader import get_dataset_split, pick_field
from src.utils.alignment import apply_alignment
from src.utils.benchmarking import (
    build_training_phase2_report,
    build_training_smoke_report,
    write_json,
)
from src.utils.lm_eval import (
    compute_answer_metrics_from_prefix_embeddings,
    compute_first_token_metrics_from_prefix_embeddings,
    generate_from_prefix_embeddings,
    generate_from_text_prefix,
)
from src.utils.metrics import extract_boxed_text, normalize_answer
from src.models.hidden_state import lm_vocabulary_weight
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
_DTYPE_NAME_BY_VALUE = {value: key for key, value in _DTYPE_MAP.items()}
_GSM8K_FINAL_ANSWER_REGEX = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")
_NUMERIC_ANSWER_REGEX = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_FINAL_ANSWER_MARKER_REGEX = re.compile(
    r"final\s+answer\s*[:=]\s*\$?(.+?)(?=(?:\s+final\s+answer\s*[:=])|[.\n]|$)",
    re.IGNORECASE,
)
_FINAL_ANSWER_STOP_REGEX = re.compile(
    r"final\s+answer\s*[:=]\s*\$?(?:-?\d[\d,]*(?:\.\d+)?|[a-z0-9^*/+\-]+)(?:[\s.]|$)",
    re.IGNORECASE,
)
_ANSWER_SUFFIX_TEXT = "\nFinal answer: "
_ANSWER_DECODED_PREFIX = "Final answer: "
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


def _dataset_validation_size(cfg: Any, dataset_name: str) -> int | None:
    dataset_cfg = getattr(getattr(cfg, "datasets", None), dataset_name, None)
    if dataset_cfg is None:
        return None
    raw_value = getattr(dataset_cfg, "validation_size", None)
    return None if raw_value is None else int(raw_value)


def _ensure_padding_token(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token


def _model_backbone(model: AutoModelForCausalLM) -> Any:
    if hasattr(model, "model"):
        return model.model
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def _runtime_cfg(cfg: Any) -> Any:
    return getattr(cfg, "runtime", None)


def _runtime_device_request(cfg: Any) -> str:
    return str(getattr(_runtime_cfg(cfg), "device", "auto")).lower()


def _mps_runtime_cfg(cfg: Any) -> Any:
    return getattr(_runtime_cfg(cfg), "mps", None)


def _mps_fallback_to_cpu(cfg: Any) -> bool:
    return bool(getattr(_mps_runtime_cfg(cfg), "fallback_to_cpu", True))


def _resolve_training_device(cfg: Any) -> torch.device:
    requested = _runtime_device_request(cfg)
    if requested == "auto":
        if bool(getattr(_mps_runtime_cfg(cfg), "enabled", True)) and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if _mps_fallback_to_cpu(cfg):
            return torch.device("cpu")
        raise RuntimeError("runtime.device=mps was requested, but torch.backends.mps is unavailable")
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("runtime.device=cuda was requested, but torch.cuda is unavailable")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("runtime.device must be one of: auto, mps, cuda, cpu")


def _resolve_training_torch_dtype(cfg: Any, device: torch.device) -> torch.dtype:
    configured_dtype = _DTYPE_MAP.get(str(getattr(cfg, "torch_dtype", "bfloat16")), torch.bfloat16)
    if device.type != "mps":
        return configured_dtype
    mps_dtype_name = getattr(_mps_runtime_cfg(cfg), "torch_dtype", None)
    if mps_dtype_name is not None:
        return _DTYPE_MAP.get(str(mps_dtype_name), torch.float32)
    if configured_dtype == torch.bfloat16:
        return torch.float32
    return configured_dtype


def _training_device_map(cfg: Any, device: torch.device) -> str | None:
    configured = str(getattr(cfg, "device_map", "auto"))
    if configured.lower() in {"none", "null", ""}:
        return None
    if device.type in {"mps", "cpu"}:
        return None
    return configured


def _load_model(
    model_name: str,
    *,
    torch_dtype: torch.dtype,
    device: torch.device,
    device_map: str | None,
) -> AutoModelForCausalLM:
    model_cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if hasattr(model_cfg, "text_config"):
        model_cfg = model_cfg.text_config
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=model_cfg,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    if device_map is None:
        model = model.to(device)
    return model


def _collate_training_batch(batch: list[Any]) -> dict[str, list[str | None]]:
    texts: list[str] = []
    prompts: list[str | None] = []
    answers: list[str | None] = []
    answer_candidates: list[list[str]] = []
    for item in batch:
        if isinstance(item, dict):
            prompt = item.get("prompt")
            answer = item.get("answer")
            text = item.get("supervision_text") or item.get("text") or prompt
            candidates = item.get("answer_candidates") or ()
            texts.append(str(text))
            prompts.append(None if prompt is None else str(prompt))
            answers.append(None if answer is None else str(answer))
            answer_candidates.append([str(candidate) for candidate in candidates if candidate is not None])
            continue
        texts.append(str(item))
        prompts.append(None)
        answers.append(None)
        answer_candidates.append([])
    return {
        "texts": texts,
        "prompts": prompts,
        "answers": answers,
        "answer_candidates": answer_candidates,
    }


def _build_text_dataloader(
    rows: Iterable[Any],
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        list(rows),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_training_batch,
    )


def _smoke_training_examples(num_samples: int) -> list[dict[str, Any]]:
    examples = []
    candidate_answers = list(dict.fromkeys(example["answer"] for example in _SMOKE_TRAIN_EXAMPLES))
    for index in range(num_samples):
        example = _SMOKE_TRAIN_EXAMPLES[index % len(_SMOKE_TRAIN_EXAMPLES)]
        examples.append(
            {
                "prompt": example["prompt"],
                "answer": example["answer"],
                "answer_candidates": candidate_answers,
                "supervision_text": f"Question: {example['prompt']}\nAnswer: {example['answer']}",
            }
        )
    return examples


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


def _is_numeric_answer(answer: str | None) -> bool:
    if answer is None:
        return False
    return _NUMERIC_ANSWER_REGEX.fullmatch(str(answer).strip()) is not None


def _final_answer_marker_value(decoded_text: str) -> str | None:
    match = _FINAL_ANSWER_MARKER_REGEX.search(decoded_text)
    if match is None:
        return None
    return match.group(1).strip()


def _predicted_answer_for_target(
    dataset_name: str,
    decoded_text: str,
    target_answer: str | None,
) -> str | None:
    if dataset_name != "smoke":
        return _predicted_answer(dataset_name, decoded_text)
    marked_answer = _final_answer_marker_value(decoded_text)
    if marked_answer is not None:
        return marked_answer
    if _is_numeric_answer(target_answer):
        return _extract_gsm8k_predicted_answer(decoded_text)
    return normalize_answer(decoded_text)


def _answers_match(dataset_name: str, predicted_answer: str | None, target_answer: str | None) -> bool:
    if dataset_name == "gsm8k":
        return _normalize_numeric_answer(predicted_answer) == _normalize_numeric_answer(target_answer)
    return normalize_answer(predicted_answer) == normalize_answer(target_answer)


def _unique_candidate_answers(eval_examples: list[dict[str, str | None]]) -> tuple[str, ...]:
    answers: list[str] = []
    seen: set[str] = set()
    for example in eval_examples:
        answer = normalize_answer(example.get("answer"))
        if answer is None or answer in seen:
            continue
        answers.append(answer)
        seen.add(answer)
    return tuple(answers)


def _select_candidate_answer_by_nll(
    *,
    model: AutoModelForCausalLM,
    tokenizer: Any,
    prefix_embeds: torch.Tensor,
    candidate_answers: Iterable[str],
) -> tuple[str | None, float | None]:
    best_answer: str | None = None
    best_nll: float | None = None
    for answer in candidate_answers:
        metrics = compute_answer_metrics_from_prefix_embeddings(
            model=model,
            tokenizer=tokenizer,
            prefix_embeds=prefix_embeds,
            answer_text=answer,
            answer_variants=_answer_metric_variants(answer),
        )
        answer_nll = metrics.get("answer_nll")
        if answer_nll is None:
            continue
        score = float(answer_nll)
        if best_nll is None or score < best_nll:
            best_answer = answer
            best_nll = score
    return best_answer, best_nll


def _text_prefix_embeddings(
    *,
    model: AutoModelForCausalLM,
    tokenizer: Any,
    prefix_text: str,
) -> torch.Tensor:
    model_device = next(model.parameters()).device
    encoded = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(model_device)
    with torch.no_grad():
        return model.get_input_embeddings()(input_ids)


def _append_text_embeddings(
    *,
    model: AutoModelForCausalLM,
    tokenizer: Any,
    prefix_embeds: torch.Tensor,
    suffix_text: str,
) -> torch.Tensor:
    model_device = next(model.parameters()).device
    encoded = tokenizer(suffix_text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(model_device)
    if input_ids.numel() == 0:
        return prefix_embeds
    with torch.no_grad():
        suffix_embeds = model.get_input_embeddings()(input_ids)
    suffix_embeds = suffix_embeds.expand(prefix_embeds.shape[0], -1, -1)
    return torch.cat([prefix_embeds, suffix_embeds.to(prefix_embeds.dtype)], dim=1)


def _answer_metric_variants(answer: str | None) -> tuple[str, ...]:
    if answer is None:
        return ()
    cleaned = str(answer).strip()
    if not cleaned:
        return ()
    return (f" {cleaned}",)


def _unique_normalized_answer_count(answers: Iterable[str | None]) -> int:
    normalized_answers = {
        normalized
        for normalized in (normalize_answer(answer) for answer in answers)
        if normalized
    }
    return len(normalized_answers)


def _format_actor_answer_prompt(
    prompt: str,
    *,
    baseline_examples: list[dict[str, str | None]],
) -> str:
    lines: list[str] = [
        "Answer each question with exactly one final-answer line.",
    ]
    current_prompt = normalize_answer(prompt)
    for example in baseline_examples:
        example_prompt = example.get("prompt")
        example_answer = example.get("answer")
        if example_prompt is None or example_answer is None:
            continue
        if normalize_answer(str(example_prompt)) == current_prompt:
            continue
        lines.extend(
            [
                "",
                f"Question: {str(example_prompt).strip()}",
                f"Final answer: {str(example_answer).strip()}",
            ]
        )
    lines.extend(["", f"Question: {prompt.strip()}"])
    return "\n".join(lines)


def _build_real_examples(
    cfg: Any,
    dataset_name: str,
    split: str,
    limit: int,
) -> list[dict[str, str | None]]:
    rows = get_dataset_split(
        dataset_name,
        split,
        limit=limit,
        validation_size=_dataset_validation_size(cfg, dataset_name),
    )
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


def _training_evaluation_cfg(cfg: Any) -> Any:
    return getattr(getattr(cfg, "training", None), "evaluation", None)


def _baseline_few_shot_count(cfg: Any) -> int:
    return int(getattr(_training_evaluation_cfg(cfg), "baseline_few_shot_examples", 6))


def _smoke_eval_set(cfg: Any) -> str:
    return str(getattr(_training_evaluation_cfg(cfg), "smoke_eval_set", "heldout")).lower()


def _build_training_payloads(
    cfg: Any,
) -> tuple[DataLoader, list[dict[str, str | None]], list[dict[str, str | None]], str]:
    data_cfg = getattr(cfg.training, "data", None)
    mode = str(getattr(data_cfg, "mode", "smoke")).lower()
    batch_size = int(getattr(data_cfg, "batch_size", 2))

    if mode == "smoke":
        num_samples = int(getattr(data_cfg, "smoke_num_samples", 16))
        train_examples = _smoke_training_examples(num_samples)
        train_loader = _build_text_dataloader(
            train_examples,
            batch_size=batch_size,
            shuffle=True,
        )
        if _smoke_eval_set(cfg) in {"train", "train_overfit", "overfit"}:
            eval_examples = [
                {"prompt": example["prompt"], "answer": example["answer"]}
                for example in train_examples[: max(1, min(len(train_examples), 3))]
            ]
        else:
            eval_examples = list(_SMOKE_EVAL_EXAMPLES)
        baseline_examples = [
            {"prompt": example["prompt"], "answer": example["answer"]}
            for example in train_examples[: _baseline_few_shot_count(cfg)]
        ]
        return train_loader, eval_examples, baseline_examples, "smoke"

    if mode == "real":
        dataset_name = str(getattr(data_cfg, "dataset_name", "gsm8k")).lower()
        train_limit = int(getattr(data_cfg, "train_limit", 32))
        eval_limit = int(getattr(data_cfg, "eval_limit", 16))
        eval_split = str(getattr(data_cfg, "eval_split", "validation")).lower()
        train_examples = _build_real_examples(cfg, dataset_name, "train", train_limit)
        eval_examples = _build_real_examples(cfg, dataset_name, eval_split, eval_limit)
        train_loader = _build_text_dataloader(
            train_examples,
            batch_size=batch_size,
            shuffle=True,
        )
        baseline_examples = [
            {"prompt": example["prompt"], "answer": example["answer"]}
            for example in train_examples[: _baseline_few_shot_count(cfg)]
        ]
        return train_loader, eval_examples, baseline_examples, dataset_name

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


def _build_evaluation_callback(
    *,
    eval_examples: list[dict[str, str | None]],
    baseline_examples: list[dict[str, str | None]],
    dataset_name: str,
    reasoner_tokenizer: Any,
    actor_tokenizer: Any,
    config: CompressionTrainConfig,
    max_new_tokens: int,
) -> Any:
    candidate_answers = _unique_candidate_answers(eval_examples) if dataset_name == "smoke" else ()

    def evaluate(
        reasoner_model: AutoModelForCausalLM,
        actor_model: AutoModelForCausalLM,
        alignment_context: dict[str, Any],
    ) -> dict[str, Any]:
        if not eval_examples:
            return {
                "heldout_exact_match_accuracy": 0.0,
                "heldout_eval_samples": 0.0,
            }

        reasoner_device = next(reasoner_model.parameters()).device
        reasoner_backbone = _model_backbone(reasoner_model)
        alignment_state = alignment_context["alignment_state"]
        correct_count = 0
        decode_extracted_answer_count = 0
        extracted_answer_count = 0
        candidate_fallback_count = 0
        extraction_failure_count = 0
        latent_candidate_correct_count = 0
        latent_probe_correct_count = 0
        latent_first_token_correct_count = 0
        latent_first_token_rank_total = 0.0
        latent_first_token_rank_count = 0
        latent_logit_steering_bias_norm_total = 0.0
        latent_logit_steering_bias_norm_count = 0
        baseline_correct_count = 0
        baseline_extracted_answer_count = 0
        baseline_candidate_correct_count = 0
        baseline_first_token_correct_count = 0
        baseline_first_token_rank_total = 0.0
        baseline_first_token_rank_count = 0
        total_answer_tokens = 0
        total_answer_nll = 0.0
        predicted_answers: list[str | None] = []
        latent_candidate_predicted_answers: list[str | None] = []
        latent_probe_predicted_answers: list[str | None] = []
        baseline_predicted_answers: list[str | None] = []
        baseline_candidate_predicted_answers: list[str | None] = []
        diagnostic_rows: list[str] = []

        for example in eval_examples:
            prompt = str(example["prompt"])
            target_answer = example["answer"]
            actor_answer_prefix_text = (
                _format_actor_answer_prompt(
                    prompt,
                    baseline_examples=baseline_examples,
                )
                + _ANSWER_SUFFIX_TEXT
            )
            baseline_decode_metrics = generate_from_text_prefix(
                model=actor_model,
                tokenizer=actor_tokenizer,
                prefix_text=actor_answer_prefix_text,
                max_new_tokens=max_new_tokens,
                decoded_text_prefix=_ANSWER_DECODED_PREFIX,
                stop_regex=_FINAL_ANSWER_STOP_REGEX,
            )
            baseline_decoded_text = str(baseline_decode_metrics["decoded_text"])
            baseline_predicted_answer = _predicted_answer_for_target(
                dataset_name,
                baseline_decoded_text,
                target_answer,
            )
            baseline_predicted_answers.append(baseline_predicted_answer)
            if baseline_predicted_answer is not None and str(baseline_predicted_answer).strip():
                baseline_extracted_answer_count += 1
            if _answers_match(dataset_name, baseline_predicted_answer, target_answer):
                baseline_correct_count += 1
            baseline_candidate_predicted_answer = None
            if candidate_answers:
                actor_text_prefix_embeds = _text_prefix_embeddings(
                    model=actor_model,
                    tokenizer=actor_tokenizer,
                    prefix_text=actor_answer_prefix_text,
                )
                baseline_first_token_metrics = compute_first_token_metrics_from_prefix_embeddings(
                    model=actor_model,
                    tokenizer=actor_tokenizer,
                    prefix_embeds=actor_text_prefix_embeds,
                    answer_text=target_answer,
                    answer_variants=_answer_metric_variants(target_answer),
                )
                if baseline_first_token_metrics.get("first_token_rank") is not None:
                    baseline_first_token_rank_count += 1
                    baseline_first_token_rank_total += float(
                        baseline_first_token_metrics["first_token_rank"] or 0.0
                    )
                if bool(baseline_first_token_metrics.get("first_token_top1")):
                    baseline_first_token_correct_count += 1
                baseline_candidate_predicted_answer, _ = _select_candidate_answer_by_nll(
                    model=actor_model,
                    tokenizer=actor_tokenizer,
                    prefix_embeds=actor_text_prefix_embeds,
                    candidate_answers=candidate_answers,
                )
                baseline_candidate_predicted_answers.append(baseline_candidate_predicted_answer)
                if _answers_match(dataset_name, baseline_candidate_predicted_answer, target_answer):
                    baseline_candidate_correct_count += 1

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
            aligned_latents = apply_alignment(compressed_latents, alignment_state)
            latent_adapter = alignment_context.get("stage2_latent_handoff_adapter")
            if latent_adapter is not None:
                with torch.no_grad():
                    aligned_latents = latent_adapter(aligned_latents)
            hidden_processor = alignment_context.get("stage2_hidden_state_processor")
            if hidden_processor is not None:
                with torch.no_grad():
                    aligned_latents = hidden_processor(aligned_latents)
            latent_probe_predicted_answer = None
            latent_probe = alignment_context.get("stage2_latent_answer_probe")
            latent_probe_candidates = tuple(
                alignment_context.get("stage2_latent_answer_candidates") or candidate_answers
            )
            if latent_probe is not None and latent_probe_candidates:
                if len(latent_probe_candidates) <= int(
                    getattr(latent_probe, "max_candidates", len(latent_probe_candidates))
                ):
                    with torch.no_grad():
                        probe_device = next(latent_probe.parameters()).device
                        probe_logits = latent_probe(aligned_latents.to(probe_device))[
                            :, : len(latent_probe_candidates)
                        ]
                        probe_index = int(probe_logits.argmax(dim=-1)[0].detach().cpu().item())
                    latent_probe_predicted_answer = latent_probe_candidates[probe_index]
                    latent_probe_predicted_answers.append(latent_probe_predicted_answer)
                    if _answers_match(dataset_name, latent_probe_predicted_answer, target_answer):
                        latent_probe_correct_count += 1
            latent_soft_prompt_decoder = alignment_context.get("stage2_latent_soft_prompt_decoder")
            if latent_soft_prompt_decoder is not None:
                with torch.no_grad():
                    decoder_device = next(latent_soft_prompt_decoder.parameters()).device
                    aligned_latents = latent_soft_prompt_decoder(aligned_latents.to(decoder_device)).to(
                        reasoner_device
                    )
            step_logits_bias = None
            first_step_logits_bias = None
            step_logits_bias_scale = 1.0
            latent_logit_steering = alignment_context.get("stage2_latent_logit_steering")
            if latent_logit_steering is not None:
                with torch.no_grad():
                    steering_device = next(latent_logit_steering.parameters()).device
                    vocabulary_weight = lm_vocabulary_weight(actor_model)
                    step_logits_bias = latent_logit_steering.forward_sequence(
                        aligned_latents.to(device=steering_device, dtype=vocabulary_weight.dtype),
                        vocabulary_weight.to(device=steering_device),
                    )
                    first_step_logits_bias = step_logits_bias[:, 0, :]
                    latent_logit_steering_bias_norm_total += float(
                        step_logits_bias.detach().float().norm(dim=-1).mean().cpu().item()
                    )
                    latent_logit_steering_bias_norm_count += 1
                    step_logits_bias_scale = float(
                        alignment_context.get("stage2_latent_logit_steering_generation_scale", 1.0)
                    )
            answer_prefix_embeds = _append_text_embeddings(
                model=actor_model,
                tokenizer=actor_tokenizer,
                prefix_embeds=aligned_latents,
                suffix_text=_ANSWER_SUFFIX_TEXT,
            )
            decode_metrics = generate_from_prefix_embeddings(
                model=actor_model,
                tokenizer=actor_tokenizer,
                prefix_embeds=answer_prefix_embeds,
                max_new_tokens=max_new_tokens,
                decoded_text_prefix=_ANSWER_DECODED_PREFIX,
                stop_regex=_FINAL_ANSWER_STOP_REGEX,
                step_logits_bias=step_logits_bias,
                step_logits_bias_scale=step_logits_bias_scale,
            )
            decoded_text = str(decode_metrics["decoded_text"])
            answer_metrics = compute_answer_metrics_from_prefix_embeddings(
                model=actor_model,
                tokenizer=actor_tokenizer,
                prefix_embeds=answer_prefix_embeds,
                answer_text=target_answer,
                answer_variants=_answer_metric_variants(target_answer),
                step_logits_bias=step_logits_bias,
                step_logits_bias_scale=step_logits_bias_scale,
            )
            latent_first_token_metrics = compute_first_token_metrics_from_prefix_embeddings(
                model=actor_model,
                tokenizer=actor_tokenizer,
                prefix_embeds=answer_prefix_embeds,
                answer_text=target_answer,
                answer_variants=_answer_metric_variants(target_answer),
                first_step_logits_bias=first_step_logits_bias,
                first_step_logits_bias_scale=step_logits_bias_scale,
            )
            if latent_first_token_metrics.get("first_token_rank") is not None:
                latent_first_token_rank_count += 1
                latent_first_token_rank_total += float(
                    latent_first_token_metrics["first_token_rank"] or 0.0
                )
            if bool(latent_first_token_metrics.get("first_token_top1")):
                latent_first_token_correct_count += 1
            if answer_metrics["answer_nll"] is not None:
                token_count = int(answer_metrics["answer_token_count"] or 0)
                total_answer_tokens += token_count
                total_answer_nll += float(answer_metrics["answer_nll"]) * token_count
            predicted_answer = _predicted_answer_for_target(
                dataset_name,
                decoded_text,
                target_answer,
            )
            latent_candidate_predicted_answer = None
            if candidate_answers:
                latent_candidate_predicted_answer, _ = _select_candidate_answer_by_nll(
                    model=actor_model,
                    tokenizer=actor_tokenizer,
                    prefix_embeds=answer_prefix_embeds,
                    candidate_answers=candidate_answers,
                )
                latent_candidate_predicted_answers.append(latent_candidate_predicted_answer)
                if _answers_match(dataset_name, latent_candidate_predicted_answer, target_answer):
                    latent_candidate_correct_count += 1
            extraction_source = "decode"
            if predicted_answer is not None and str(predicted_answer).strip():
                decode_extracted_answer_count += 1
                extracted_answer_count += 1
            elif candidate_answers:
                predicted_answer = latent_candidate_predicted_answer
                if predicted_answer is not None:
                    extraction_source = "candidate_nll"
                    candidate_fallback_count += 1
                    extracted_answer_count += 1
                else:
                    extraction_source = "missing"
                    extraction_failure_count += 1
            else:
                extraction_source = "missing"
                extraction_failure_count += 1
            predicted_answers.append(predicted_answer)
            if _answers_match(dataset_name, predicted_answer, target_answer):
                correct_count += 1
            if len(diagnostic_rows) < 5:
                decoded_preview = re.sub(r"\s+", " ", decoded_text).strip()[:120]
                baseline_preview = re.sub(r"\s+", " ", baseline_decoded_text).strip()[:120]
                diagnostic_rows.append(
                    " | ".join(
                        (
                            f"target={target_answer}",
                            f"predicted={predicted_answer}",
                            f"candidate_predicted={latent_candidate_predicted_answer}",
                            f"probe_predicted={latent_probe_predicted_answer}",
                            f"latent_first_token_rank={latent_first_token_metrics.get('first_token_rank')}",
                            f"latent_first_token_predicted={latent_first_token_metrics.get('first_token_predicted_text')}",
                            f"generated_first_token={decode_metrics.get('first_generated_token_text')}",
                            f"source={extraction_source}",
                            f"baseline_predicted={baseline_predicted_answer}",
                            f"baseline_candidate_predicted={baseline_candidate_predicted_answer}",
                            f"decoded={decoded_preview}",
                            f"baseline_decoded={baseline_preview}",
                        )
                    )
                )

        heldout_answer_nll = (
            total_answer_nll / total_answer_tokens if total_answer_tokens > 0 else 0.0
        )
        return {
            "heldout_exact_match_accuracy": 100.0 * correct_count / len(eval_examples),
            "heldout_eval_samples": float(len(eval_examples)),
            "heldout_answer_extraction_rate_percentage": (
                100.0 * extracted_answer_count / len(eval_examples)
            ),
            "heldout_decode_answer_extraction_rate_percentage": (
                100.0 * decode_extracted_answer_count / len(eval_examples)
            ),
            "heldout_candidate_fallback_rate_percentage": (
                100.0 * candidate_fallback_count / len(eval_examples)
            ),
            "heldout_latent_candidate_accuracy": (
                100.0 * latent_candidate_correct_count / len(eval_examples)
                if candidate_answers
                else None
            ),
            "heldout_latent_candidate_unique_predicted_answer_count": (
                float(_unique_normalized_answer_count(latent_candidate_predicted_answers))
                if candidate_answers
                else None
            ),
            "heldout_latent_probe_accuracy": (
                100.0 * latent_probe_correct_count / len(eval_examples)
                if latent_probe is not None and latent_probe_candidates
                else None
            ),
            "heldout_latent_probe_unique_predicted_answer_count": (
                float(_unique_normalized_answer_count(latent_probe_predicted_answers))
                if latent_probe is not None and latent_probe_candidates
                else None
            ),
            "heldout_latent_first_token_accuracy": (
                100.0 * latent_first_token_correct_count / latent_first_token_rank_count
                if latent_first_token_rank_count
                else None
            ),
            "heldout_latent_first_token_rank_mean": (
                latent_first_token_rank_total / latent_first_token_rank_count
                if latent_first_token_rank_count
                else None
            ),
            "heldout_latent_logit_steering_enabled": (
                latent_logit_steering_bias_norm_count > 0
            ),
            "heldout_latent_logit_steering_bias_norm_mean": (
                latent_logit_steering_bias_norm_total / latent_logit_steering_bias_norm_count
                if latent_logit_steering_bias_norm_count
                else None
            ),
            "heldout_extraction_failure_count": float(extraction_failure_count),
            "heldout_unique_predicted_answer_count": float(
                _unique_normalized_answer_count(predicted_answers)
            ),
            "heldout_actor_text_baseline_accuracy": (
                100.0 * baseline_correct_count / len(eval_examples)
            ),
            "heldout_actor_text_baseline_answer_extraction_rate_percentage": (
                100.0 * baseline_extracted_answer_count / len(eval_examples)
            ),
            "heldout_actor_text_baseline_unique_predicted_answer_count": float(
                _unique_normalized_answer_count(baseline_predicted_answers)
            ),
            "heldout_actor_text_baseline_candidate_accuracy": (
                100.0 * baseline_candidate_correct_count / len(eval_examples)
                if candidate_answers
                else None
            ),
            "heldout_actor_text_baseline_candidate_unique_predicted_answer_count": (
                float(_unique_normalized_answer_count(baseline_candidate_predicted_answers))
                if candidate_answers
                else None
            ),
            "heldout_actor_text_baseline_first_token_accuracy": (
                100.0 * baseline_first_token_correct_count / baseline_first_token_rank_count
                if baseline_first_token_rank_count
                else None
            ),
            "heldout_actor_text_baseline_first_token_rank_mean": (
                baseline_first_token_rank_total / baseline_first_token_rank_count
                if baseline_first_token_rank_count
                else None
            ),
            "heldout_correct_count": float(correct_count),
            "heldout_answer_nll": heldout_answer_nll,
            "heldout_answer_perplexity": float(torch.exp(torch.tensor(heldout_answer_nll)).item()),
            "heldout_eval_diagnostics": "\n".join(diagnostic_rows),
        }

    return evaluate


def _write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    fieldnames = sorted({key for entry in history for key in entry.keys()})
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    cfg = _load_cfg()
    torch.manual_seed(int(getattr(cfg, "seed", 0)))
    config = CompressionTrainConfig.from_cfg(cfg)
    device = _resolve_training_device(cfg)
    torch_dtype = _resolve_training_torch_dtype(cfg, device)
    device_map = _training_device_map(cfg, device)
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

    print(f"Requested device: {_runtime_device_request(cfg)}")
    print(f"Effective device: {device}")
    print(f"Device map: {device_map}")
    print(f"Dtype: {_DTYPE_NAME_BY_VALUE.get(torch_dtype, str(torch_dtype))}")
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
    reasoner = _load_model(
        cfg.agent_a_model,
        torch_dtype=torch_dtype,
        device=device,
        device_map=device_map,
    )

    print("Loading actor model...")
    actor = _load_model(
        cfg.agent_b_model,
        torch_dtype=torch_dtype,
        device=device,
        device_map=device_map,
    )

    print("Building text-first dataloaders...")
    train_loader, eval_examples, baseline_examples, dataset_name = _build_training_payloads(cfg)
    evaluation_max_new_tokens = int(getattr(getattr(cfg.training, "evaluation", None), "max_new_tokens", 16))
    evaluation_fn = _build_evaluation_callback(
        eval_examples=eval_examples,
        baseline_examples=baseline_examples,
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
    history_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_history_csv(history_output_path, history)
    runtime_metadata = {
        "requested_device": _runtime_device_request(cfg),
        "effective_device": str(device),
        "device_map": "none" if device_map is None else str(device_map),
        "configured_torch_dtype": str(getattr(cfg, "torch_dtype", "")),
        "effective_torch_dtype": _DTYPE_NAME_BY_VALUE.get(torch_dtype, str(torch_dtype)),
        "mps_available": torch.backends.mps.is_available(),
        "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
    }
    training_smoke_report = build_training_smoke_report(history)
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
        runtime_metadata=runtime_metadata,
    )
    training_report["training_smoke_report"] = training_smoke_report
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
                f"l_answer={entry.get('l_answer', 0.0):.4f} "
                f"first_token_acc={entry.get('answer_first_token_accuracy', 0.0):.2f} "
                f"first_token_rank={entry.get('answer_first_token_rank_mean', 0.0):.2f} "
                f"l_answer_contrast={entry.get('l_answer_contrast', 0.0):.4f} "
                f"answer_contrast_accuracy={entry.get('answer_contrast_accuracy', 0.0):.2f} "
                f"l_answer_probe={entry.get('l_answer_probe', 0.0):.4f} "
                f"answer_probe_accuracy={entry.get('answer_probe_accuracy', 0.0):.2f} "
                f"adapter_grad={entry.get('handoff_adapter_grad_norm', 0.0):.4f} "
                f"adapter_update={entry.get('handoff_adapter_update_norm', 0.0):.6f} "
                f"probe_grad={entry.get('latent_answer_probe_grad_norm', 0.0):.4f} "
                f"probe_update={entry.get('latent_answer_probe_update_norm', 0.0):.6f} "
                f"soft_prompt_grad={entry.get('latent_soft_prompt_decoder_grad_norm', 0.0):.4f} "
                f"soft_prompt_update={entry.get('latent_soft_prompt_decoder_update_norm', 0.0):.6f} "
            )
            continue
        print(
            f"  event={entry.get('event', 'eval')} epoch={int(entry['epoch'])} step={int(entry['step'])} "
            f"heldout_exact_match_accuracy={entry['heldout_exact_match_accuracy']:.2f} "
            f"heldout_eval_samples={entry['heldout_eval_samples']:.0f} "
            f"heldout_answer_extraction_rate={entry.get('heldout_answer_extraction_rate_percentage', 0.0):.2f} "
            f"decode_extraction_rate={entry.get('heldout_decode_answer_extraction_rate_percentage', 0.0):.2f} "
            f"candidate_fallback_rate={entry.get('heldout_candidate_fallback_rate_percentage', 0.0):.2f} "
            f"latent_probe_accuracy={entry.get('heldout_latent_probe_accuracy', 0.0) or 0.0:.2f} "
            f"latent_first_token_accuracy={entry.get('heldout_latent_first_token_accuracy', 0.0) or 0.0:.2f} "
            f"unique_predictions={entry.get('heldout_unique_predicted_answer_count', 0.0):.0f} "
            f"actor_text_baseline_accuracy={entry.get('heldout_actor_text_baseline_accuracy', 0.0):.2f}"
        )

    print(f"Wrote training history to {history_output_path}")
    print(f"Wrote phase-2 training report to {report_output_path}")
    print(f"Training smoke passed: {training_smoke_report['passed']}")
    print(f"Phase 2 gate passed: {training_report['passed']}")


if __name__ == "__main__":
    main()
