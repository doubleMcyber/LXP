"""Dataset loading abstraction for multiple benchmarks."""
from __future__ import annotations

from typing import Any, Optional

from datasets import load_dataset


def pick_field(row: dict, keys: tuple[str, ...]) -> str:
    """Return the first non-None value from *row* for the given *keys*."""
    for key in keys:
        value = row.get(key)
        if value is not None:
            return str(value)
    return ""


def _select_rows(dataset: Any, limit: Optional[int]) -> Any:
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


def load_math_level5(
    limit: int = 100,
    split: str = "test",
    *,
    validation_size: Optional[int] = None,
):
    """Load MATH Level 5 problems from Hugging Face."""
    dataset_candidates = [
        ("hendrycks/competition_math", "train" if split == "validation" else split),
        ("competition_math", "train" if split == "validation" else split),
    ]
    last_error: Optional[Exception] = None

    for dataset_name, dataset_split in dataset_candidates:
        try:
            dataset = load_dataset(dataset_name, split=dataset_split)
            if "level" in dataset.column_names:
                level5 = dataset.filter(lambda row: "Level 5" in str(row.get("level", "")))
            else:
                level5 = dataset
            if split in {"train", "validation"}:
                level5 = _apply_train_validation_split(level5, split, validation_size)
            return _select_rows(level5, limit)
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise RuntimeError("Failed to load a MATH dataset candidate") from last_error


def load_gsm8k(
    limit: int = 100,
    split: str = "test",
    *,
    validation_size: Optional[int] = None,
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
            return _select_rows(dataset, limit)
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise RuntimeError("Failed to load a GSM8K dataset candidate") from last_error


_LOADERS = {
    "math": load_math_level5,
    "gsm8k": load_gsm8k,
}


def get_dataset_split(
    dataset_name: str,
    split: str,
    limit: int = 100,
    *,
    validation_size: Optional[int] = None,
):
    """Load a benchmark dataset split by name."""
    dataset_name = dataset_name.lower()
    if dataset_name == "math":
        return load_math_level5(limit=limit, split=split, validation_size=validation_size)
    if dataset_name == "gsm8k":
        return load_gsm8k(limit=limit, split=split, validation_size=validation_size)
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
):
    """Load a benchmark dataset by name.

    Supported names: "math", "gsm8k".
    """
    return get_dataset_split(
        dataset_name,
        split=split,
        limit=limit,
        validation_size=validation_size,
    )
