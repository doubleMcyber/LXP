from __future__ import annotations

from unittest.mock import patch

from datasets import Dataset

from src.data.loader import get_dataset_split


def test_gsm8k_validation_split_uses_deterministic_train_tail() -> None:
    dataset = Dataset.from_dict(
        {
            "question": [f"q{i}" for i in range(10)],
            "answer": [f"a{i}" for i in range(10)],
        }
    )

    with patch("src.data.loader.load_dataset", return_value=dataset):
        train_split = get_dataset_split("gsm8k", "train", limit=10, validation_size=3)
        validation_split = get_dataset_split("gsm8k", "validation", limit=10, validation_size=3)

    assert len(train_split) == 7
    assert len(validation_split) == 3
    assert train_split[-1]["question"] == "q6"
    assert validation_split[0]["question"] == "q7"
    assert validation_split[-1]["question"] == "q9"


def test_gsm8k_split_can_select_explicit_sample_indices_after_split() -> None:
    dataset = Dataset.from_dict(
        {
            "question": [f"q{i}" for i in range(10)],
            "answer": [f"a{i}" for i in range(10)],
        }
    )

    with patch("src.data.loader.load_dataset", return_value=dataset):
        validation_split = get_dataset_split(
            "gsm8k",
            "validation",
            limit=2,
            validation_size=5,
            sample_indices=[0, 3, 4],
        )

    assert [row["question"] for row in validation_split] == ["q5", "q8"]
