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
) -> pd.DataFrame:
    """Load a CSV dataset and validate the required columns."""
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{dataset_path}'. Add your CSV file before training."
        )

    data_frame = pd.read_csv(dataset_path)

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

    return cleaned_frame
