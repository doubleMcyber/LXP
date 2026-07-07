"""Dataset loading abstraction for multiple benchmarks."""
from __future__ import annotations

from typing import Any, Optional, Sequence

from datasets import Dataset, concatenate_datasets, load_dataset

# EleutherAI mirror of the Hendrycks MATH dataset. Split across per-subject
# configs; each config exposes "train"/"test" splits with fields
# problem/level/type/solution (same schema as the legacy competition_math
# script dataset). Order is fixed so seeded splits are reproducible.
_HENDRYCKS_MATH_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def pick_field(row: dict, keys: tuple[str, ...]) -> str:
    """Return the first non-None value from *row* for the given *keys*."""
    for key in keys:
        value = row.get(key)
        if value is not None:
            return str(value)
    return ""


def _select_rows(
    dataset: Any,
    limit: Optional[int],
    *,
    sample_indices: Optional[Sequence[int]] = None,
) -> Any:
    if sample_indices is not None:
        indices = [int(index) for index in sample_indices]
        if not indices:
            return dataset.select([])
        out_of_bounds = [index for index in indices if index < 0 or index >= len(dataset)]
        if out_of_bounds:
            raise IndexError(
                "sample_indices out of range for dataset of size "
                f"{len(dataset)}: {out_of_bounds}"
            )
        dataset = dataset.select(indices)
    if limit is None:
        return dataset
    return dataset.select(range(min(int(limit), len(dataset))))


def _resolve_validation_window(dataset: Any, validation_size: int) -> tuple[int, int]:
    if validation_size <= 0:
        raise ValueError("validation_size must be positive")
    if validation_size >= len(dataset):
        raise ValueError(
            f"validation_size={validation_size} must be smaller than dataset size {len(dataset)}"
        )
    validation_start = len(dataset) - validation_size
    return validation_start, len(dataset)


def _apply_train_validation_split(dataset: Any, split: str, validation_size: Optional[int]) -> Any:
    if split not in {"train", "validation"}:
        return dataset
    if validation_size is None:
        raise ValueError(
            "validation_size is required when requesting a derived train/validation split"
        )

    validation_start, validation_end = _resolve_validation_window(dataset, validation_size)
    if split == "validation":
        return dataset.select(range(validation_start, validation_end))
    return dataset.select(range(0, validation_start))


def _finalize_math_level5(
    dataset: Any,
    split: str,
    limit: Optional[int],
    validation_size: Optional[int],
    sample_indices: Optional[Sequence[int]],
) -> Any:
    if "level" in dataset.column_names:
        level5 = dataset.filter(lambda row: "Level 5" in str(row.get("level", "")))
    else:
        level5 = dataset
    if split in {"train", "validation"}:
        level5 = _apply_train_validation_split(level5, split, validation_size)
    return _select_rows(level5, limit, sample_indices=sample_indices)


def load_math_level5(
    limit: int = 100,
    split: str = "test",
    *,
    validation_size: Optional[int] = None,
    sample_indices: Optional[Sequence[int]] = None,
):
    """Load MATH Level 5 problems from Hugging Face."""
    # Validation is derived from the train tail, so it reads the train split.
    dataset_split = "train" if split == "validation" else split
    last_error: Optional[Exception] = None

    # Preferred source: EleutherAI/hendrycks_math, split across per-subject
    # configs. Load each config in the fixed subject order and concatenate so
    # the row order is deterministic and seeded splits are reproducible.
    try:
        parts = [
            load_dataset("EleutherAI/hendrycks_math", subject, split=dataset_split)
            for subject in _HENDRYCKS_MATH_SUBJECTS
        ]
        dataset = concatenate_datasets(parts)
        return _finalize_math_level5(
            dataset, split, limit, validation_size, sample_indices
        )
    except Exception as exc:  # noqa: BLE001
        last_error = exc

    # Fallbacks: legacy script datasets (only load if someone has them cached;
    # modern `datasets` can no longer fetch them fresh).
    for dataset_name in ("hendrycks/competition_math", "competition_math"):
        try:
            dataset = load_dataset(dataset_name, split=dataset_split)
            return _finalize_math_level5(
                dataset, split, limit, validation_size, sample_indices
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise RuntimeError("Failed to load a MATH dataset candidate") from last_error


def load_gsm8k(
    limit: int = 100,
    split: str = "test",
    *,
    validation_size: Optional[int] = None,
    sample_indices: Optional[Sequence[int]] = None,
):
    """Load GSM8K problems from Hugging Face."""
    dataset_candidates = [
        ("openai/gsm8k", "main", "train" if split == "validation" else split),
        ("gsm8k", "main", "train" if split == "validation" else split),
    ]
    last_error: Optional[Exception] = None

    for args in dataset_candidates:
        dataset_name, config, dataset_split = args
        try:
            dataset = load_dataset(dataset_name, config, split=dataset_split)
            if split in {"train", "validation"}:
                dataset = _apply_train_validation_split(dataset, split, validation_size)
            return _select_rows(dataset, limit, sample_indices=sample_indices)
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise RuntimeError("Failed to load a GSM8K dataset candidate") from last_error


def _long_context_reasoning(sample_id: int, answer: int, *, horizon_steps: int) -> str:
    lines = [
        f"Scratch step {step:04d}: sample {sample_id} keeps candidate answer {answer}; "
        "ignore distractors and preserve the exact final scalar."
        for step in range(1, horizon_steps + 1)
    ]
    lines.append(f"Verification: every retained candidate is {answer}.")
    lines.append(f"Final answer: {answer}.")
    return "\n".join(lines)


def load_long_context_handoff(
    limit: int = 100,
    split: str = "validation",
    *,
    validation_size: Optional[int] = None,
    sample_indices: Optional[Sequence[int]] = None,
):
    """Build a deterministic local long-horizon handoff benchmark.

    The prompt is intentionally short while ``sender_reasoning_text`` is long.
    This isolates receiver-side handoff payload length: token-context baselines
    must prefill the full upstream trace, while latent handoff methods transfer
    the sender hidden trajectory and a compact receiver prefix.
    """
    del validation_size
    split_offsets = {"train": 0, "validation": 10_000, "test": 20_000}
    offset = split_offsets.get(split, 30_000)
    row_count = max(int(limit or 0), max(sample_indices or [0]) + 1 if sample_indices else 0, 256)
    rows = []
    for local_index in range(row_count):
        sample_id = offset + local_index
        answer = 1000 + ((sample_id * 37) % 9000)
        horizon_steps = 96 + (local_index % 5) * 32
        rows.append(
            {
                "question": (
                    f"Long-horizon handoff sample {sample_id}. "
                    "Use the upstream sender trace and return only the final scalar answer."
                ),
                "answer": f"#### {answer}",
                "sender_reasoning_text": _long_context_reasoning(
                    sample_id,
                    answer,
                    horizon_steps=horizon_steps,
                ),
                "horizon_steps": horizon_steps,
            }
        )
    return _select_rows(Dataset.from_list(rows), limit, sample_indices=sample_indices)


_LOADERS = {
    "math": load_math_level5,
    "gsm8k": load_gsm8k,
    "long_context_handoff": load_long_context_handoff,
}


def get_dataset_split(
    dataset_name: str,
    split: str,
    limit: int = 100,
    *,
    validation_size: Optional[int] = None,
    sample_indices: Optional[Sequence[int]] = None,
):
    """Load a benchmark dataset split by name."""
    dataset_name = dataset_name.lower()
    if dataset_name == "math":
        return load_math_level5(
            limit=limit,
            split=split,
            validation_size=validation_size,
            sample_indices=sample_indices,
        )
    if dataset_name == "gsm8k":
        return load_gsm8k(
            limit=limit,
            split=split,
            validation_size=validation_size,
            sample_indices=sample_indices,
        )
    if dataset_name == "long_context_handoff":
        return load_long_context_handoff(
            limit=limit,
            split=split,
            validation_size=validation_size,
            sample_indices=sample_indices,
        )
    raise ValueError(
        f"Unknown dataset {dataset_name!r}. "
        f"Supported: {', '.join(sorted(_LOADERS))}"
    )


def get_dataloader(
    dataset_name: str,
    limit: int = 100,
    *,
    split: str = "test",
    validation_size: Optional[int] = None,
    sample_indices: Optional[Sequence[int]] = None,
):
    """Load a benchmark dataset by name.

    Supported names: "math", "gsm8k", "long_context_handoff".
    """
    return get_dataset_split(
        dataset_name,
        split=split,
        limit=limit,
        validation_size=validation_size,
        sample_indices=sample_indices,
    )
