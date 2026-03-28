"""Dataset loading helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("text", "label")


def load_dataset(
    dataset_path: Path,
    text_column: str,
    label_column: str,
    allowed_labels: list[str] | None = None,
    remove_duplicates: bool = False,
) -> pd.DataFrame:
    """Load a CSV dataset and validate the required columns."""
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{dataset_path}'. Add your CSV file before training."
        )

    if dataset_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV file, got '{dataset_path.suffix}'.")

    data_frame = pd.read_csv(dataset_path)
    if data_frame.empty:
        raise ValueError("Dataset is empty. Add at least a few labeled rows before training.")

    missing_columns = [
        column for column in (text_column, label_column) if column not in data_frame.columns
    ]
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_columns)
        )

    cleaned_frame = data_frame[[text_column, label_column]].dropna().copy()
    cleaned_frame[text_column] = cleaned_frame[text_column].astype(str).str.strip()
    cleaned_frame[label_column] = cleaned_frame[label_column].astype(str).str.strip().str.lower()
    cleaned_frame = cleaned_frame[
        cleaned_frame[text_column].ne("") & cleaned_frame[label_column].ne("")
    ].copy()

    if cleaned_frame.empty:
        raise ValueError("Dataset does not contain any non-empty text/label rows.")

    if remove_duplicates:
        cleaned_frame = cleaned_frame.drop_duplicates(
            subset=[text_column, label_column]
        ).reset_index(drop=True)

    if allowed_labels is not None:
        invalid_labels = sorted(
            set(cleaned_frame[label_column].unique()) - set(allowed_labels)
        )
        if invalid_labels:
            raise ValueError(
                "Dataset contains unsupported labels: "
                + ", ".join(invalid_labels)
                + ". Allowed labels are: "
                + ", ".join(allowed_labels)
            )

    label_counts = cleaned_frame[label_column].value_counts().sort_index()
    if label_counts.size < 2:
        raise ValueError("Dataset must contain at least two distinct labels.")

    return cleaned_frame
