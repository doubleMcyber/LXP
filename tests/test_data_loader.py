from __future__ import annotations

from unittest.mock import patch

from datasets import Dataset

from src.data.loader import _HENDRYCKS_MATH_SUBJECTS, get_dataset_split


def _make_math_subject_dataset(subject: str, levels: list[str]) -> Dataset:
    return Dataset.from_dict(
        {
            "problem": [f"{subject}-p{i}" for i in range(len(levels))],
            "level": levels,
            "type": [subject] * len(levels),
            "solution": [f"{subject}-s{i}" for i in range(len(levels))],
        }
    )


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


def test_math_level5_uses_eleuther_source_and_filters_in_config_order() -> None:
    # Each subject has one Level 5 row and one non-Level-5 row that must be dropped.
    parts = [
        _make_math_subject_dataset(subject, ["Level 5", "Level 4"])
        for subject in _HENDRYCKS_MATH_SUBJECTS
    ]

    with patch("src.data.loader.load_dataset", side_effect=parts) as mock_load:
        result = get_dataset_split("math", "test", limit=100)

    # One Level 5 row survives per subject, concatenated in fixed config order.
    assert [row["problem"] for row in result] == [
        f"{subject}-p0" for subject in _HENDRYCKS_MATH_SUBJECTS
    ]
    assert [row["solution"] for row in result] == [
        f"{subject}-s0" for subject in _HENDRYCKS_MATH_SUBJECTS
    ]

    # The EleutherAI source is tried first: one call per subject config, test split.
    assert [call.args[0] for call in mock_load.call_args_list] == [
        "EleutherAI/hendrycks_math"
    ] * len(_HENDRYCKS_MATH_SUBJECTS)
    assert [call.args[1] for call in mock_load.call_args_list] == list(
        _HENDRYCKS_MATH_SUBJECTS
    )
    assert all(call.kwargs["split"] == "test" for call in mock_load.call_args_list)


def test_math_validation_split_derives_from_train_tail_over_configs() -> None:
    parts = [
        _make_math_subject_dataset(subject, ["Level 5"])
        for subject in _HENDRYCKS_MATH_SUBJECTS
    ]

    with patch("src.data.loader.load_dataset", side_effect=parts) as mock_load:
        validation = get_dataset_split(
            "math", "validation", limit=10, validation_size=2
        )

    # Validation reads the train split (tail is derived from it), and takes the
    # last validation_size rows of the deterministically concatenated dataset.
    assert all(call.kwargs["split"] == "train" for call in mock_load.call_args_list)
    assert [row["problem"] for row in validation] == [
        f"{_HENDRYCKS_MATH_SUBJECTS[-2]}-p0",
        f"{_HENDRYCKS_MATH_SUBJECTS[-1]}-p0",
    ]


def test_long_context_handoff_is_local_and_has_frozen_sender_trace() -> None:
    rows = get_dataset_split(
        "long_context_handoff",
        "validation",
        limit=2,
        sample_indices=[0, 1],
    )

    assert len(rows) == 2
    assert rows[0]["answer"].startswith("#### ")
    assert "Final answer:" in rows[0]["sender_reasoning_text"]
    assert rows[0]["horizon_steps"] >= 96
    assert len(rows[0]["sender_reasoning_text"]) > len(rows[0]["question"]) * 10
