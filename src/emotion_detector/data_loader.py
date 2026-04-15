"""Dataset loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = ("text", "label")
EXPECTED_LABELS = ("joy", "sadness", "anger", "fear", "surprise", "neutral")


def validate_required_columns(
    data_frame: pd.DataFrame,
    text_column: str,
    label_column: str,
) -> None:
    """Validate that the dataset contains the required text and label columns."""
    missing_columns = [
        column for column in (text_column, label_column) if column not in data_frame.columns
    ]
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(missing_columns)
            + ". Expected at least: "
            + ", ".join((text_column, label_column))
        )


def standardize_dataset_columns(
    data_frame: pd.DataFrame,
    text_column: str,
    label_column: str,
    rename_to_standard: bool = False,
) -> pd.DataFrame:
    """
    Optionally rename dataset columns to the standard names: text and label.

    This is useful when adapting another dataset that uses different column names.
    """
    validate_required_columns(data_frame, text_column=text_column, label_column=label_column)

    if not rename_to_standard:
        return data_frame.copy()

    return data_frame.rename(
        columns={
            text_column: REQUIRED_COLUMNS[0],
            label_column: REQUIRED_COLUMNS[1],
        }
    )


def drop_invalid_rows(
    data_frame: pd.DataFrame,
    text_column: str,
    label_column: str,
) -> pd.DataFrame:
    """Drop rows with missing or blank text/label values."""
    cleaned_frame = data_frame.dropna(subset=[text_column, label_column]).copy()
    cleaned_frame[text_column] = cleaned_frame[text_column].astype(str).str.strip()
    cleaned_frame[label_column] = cleaned_frame[label_column].astype(str).str.strip().str.lower()
    cleaned_frame = cleaned_frame[
        cleaned_frame[text_column].ne("") & cleaned_frame[label_column].ne("")
    ].copy()
    return cleaned_frame.reset_index(drop=True)


def validate_label_values(
    data_frame: pd.DataFrame,
    label_column: str,
    allowed_labels: Iterable[str] | None = None,
) -> None:
    """Validate that dataset labels match the expected label set."""
    labels_to_check = list(allowed_labels or EXPECTED_LABELS)
    invalid_labels = sorted(set(data_frame[label_column].unique()) - set(labels_to_check))
    if invalid_labels:
        raise ValueError(
            "Dataset contains unsupported labels: "
            + ", ".join(invalid_labels)
            + ". Expected labels are: "
            + ", ".join(labels_to_check)
        )


def get_class_distribution(
    data_frame: pd.DataFrame,
    label_column: str,
) -> pd.DataFrame:
    """Return class counts and percentages for a label column."""
    if label_column not in data_frame.columns:
        raise ValueError(f"Label column '{label_column}' was not found in the dataset.")

    counts = data_frame[label_column].value_counts().sort_index()
    distribution = counts.rename_axis(label_column).reset_index(name="count")
    distribution["percentage"] = (distribution["count"] / len(data_frame) * 100).round(2)
    return distribution


def format_class_distribution(
    data_frame: pd.DataFrame,
    label_column: str,
    title: str,
) -> str:
    """Return a readable multi-line string for class distribution logging."""
    distribution = get_class_distribution(data_frame, label_column=label_column)
    lines = [title]
    for row in distribution.itertuples(index=False):
        lines.append(f"  - {row[0]}: {row[1]} ({row[2]:.2f}%)")
    return "\n".join(lines)


def split_dataset(
    data_frame: pd.DataFrame,
    text_column: str,
    label_column: str,
    test_size: float,
    validation_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a labeled dataset into train, validation, and test DataFrames.

    Stratified splitting is used to preserve label balance where possible.
    """
    validate_required_columns(data_frame, text_column=text_column, label_column=label_column)

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0 <= validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")
    if test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be less than 1.")

    minimum_examples_per_label = 3 if validation_size > 0 else 2
    label_counts = data_frame[label_column].value_counts()
    underrepresented_labels = label_counts[label_counts < minimum_examples_per_label]
    if not underrepresented_labels.empty:
        raise ValueError(
            "Each label must have at least "
            f"{minimum_examples_per_label} examples for the requested split. "
            "Labels with too few examples: "
            + ", ".join(
                f"{label} ({count})" for label, count in underrepresented_labels.items()
            )
        )

    train_validation_frame, test_frame = train_test_split(
        data_frame,
        test_size=test_size,
        random_state=random_state,
        stratify=data_frame[label_column],
    )

    if validation_size == 0:
        return (
            train_validation_frame.reset_index(drop=True),
            pd.DataFrame(columns=data_frame.columns),
            test_frame.reset_index(drop=True),
        )

    remaining_size = 1 - test_size
    validation_ratio = validation_size / remaining_size
    train_frame, validation_frame = train_test_split(
        train_validation_frame,
        test_size=validation_ratio,
        random_state=random_state,
        stratify=train_validation_frame[label_column],
    )

    return (
        train_frame.reset_index(drop=True),
        validation_frame.reset_index(drop=True),
        test_frame.reset_index(drop=True),
    )


def load_dataset(
    dataset_path: Path,
    text_column: str,
    label_column: str,
    allowed_labels: list[str] | None = None,
    remove_duplicates: bool = False,
    rename_to_standard: bool = False,
) -> pd.DataFrame:
    """
    Load a labeled CSV dataset for emotion classification.

    Expected label format:
    joy, sadness, anger, fear, surprise, neutral
    """
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{dataset_path}'. Add your CSV file before training."
        )

    if dataset_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV file, got '{dataset_path.suffix}'.")

    data_frame = pd.read_csv(dataset_path)
    if data_frame.empty:
        raise ValueError("Dataset is empty. Add at least a few labeled rows before training.")

    standardized_frame = standardize_dataset_columns(
        data_frame=data_frame,
        text_column=text_column,
        label_column=label_column,
        rename_to_standard=rename_to_standard,
    )
    active_text_column = REQUIRED_COLUMNS[0] if rename_to_standard else text_column
    active_label_column = REQUIRED_COLUMNS[1] if rename_to_standard else label_column
    cleaned_frame = drop_invalid_rows(
        standardized_frame,
        text_column=active_text_column,
        label_column=active_label_column,
    )

    if cleaned_frame.empty:
        raise ValueError("Dataset does not contain any non-empty text/label rows.")

    if remove_duplicates:
        cleaned_frame = cleaned_frame.drop_duplicates(
            subset=[active_text_column, active_label_column]
        ).reset_index(drop=True)

    validate_label_values(
        cleaned_frame,
        label_column=active_label_column,
        allowed_labels=allowed_labels,
    )

    label_counts = cleaned_frame[active_label_column].value_counts().sort_index()
    if label_counts.size < 2:
        raise ValueError("Dataset must contain at least two distinct labels.")

    return cleaned_frame
