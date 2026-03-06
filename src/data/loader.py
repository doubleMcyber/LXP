"""Dataset loading abstraction for multiple benchmarks."""
from __future__ import annotations

from typing import Optional

from datasets import load_dataset


def pick_field(row: dict, keys: tuple[str, ...]) -> str:
    """Return the first non-None value from *row* for the given *keys*."""
    for key in keys:
        value = row.get(key)
        if value is not None:
            return str(value)
    return ""


def load_math_level5(limit: int = 100, split: str = "test"):
    """Load MATH Level 5 problems from Hugging Face."""
    dataset_candidates = [
        ("hendrycks/competition_math", split),
        ("competition_math", split),
    ]
    last_error: Optional[Exception] = None

    for dataset_name, split in dataset_candidates:
        try:
            dataset = load_dataset(dataset_name, split=split)
            if "level" in dataset.column_names:
                level5 = dataset.filter(lambda row: "Level 5" in str(row.get("level", "")))
            else:
                level5 = dataset
            return level5.select(range(min(limit, len(level5))))
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise RuntimeError("Failed to load a MATH dataset candidate") from last_error


def load_gsm8k(limit: int = 100, split: str = "test"):
    """Load GSM8K problems from Hugging Face."""
    dataset_candidates = [
        ("openai/gsm8k", "main", split),
        ("gsm8k", "main", split),
    ]
    last_error: Optional[Exception] = None

    for args in dataset_candidates:
        dataset_name, config, split = args
        try:
            dataset = load_dataset(dataset_name, config, split=split)
            return dataset.select(range(min(limit, len(dataset))))
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise RuntimeError("Failed to load a GSM8K dataset candidate") from last_error


_LOADERS = {
    "math": load_math_level5,
    "gsm8k": load_gsm8k,
}


def get_dataset_split(dataset_name: str, split: str, limit: int = 100):
    """Load a benchmark dataset split by name."""
    dataset_name = dataset_name.lower()
    if dataset_name == "math":
        return load_math_level5(limit=limit, split=split)
    if dataset_name == "gsm8k":
        return load_gsm8k(limit=limit, split=split)
    raise ValueError(
        f"Unknown dataset {dataset_name!r}. "
        f"Supported: {', '.join(sorted(_LOADERS))}"
    )


def get_dataloader(dataset_name: str, limit: int = 100):
    """Load a benchmark dataset by name.

    Supported names: "math", "gsm8k".
    """
    loader = _LOADERS.get(dataset_name.lower())
    if loader is None:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}. "
            f"Supported: {', '.join(sorted(_LOADERS))}"
        )
    return loader(limit=limit)
