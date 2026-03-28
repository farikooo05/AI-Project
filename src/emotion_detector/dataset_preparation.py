"""Helpers for preparing raw emotion datasets for the training pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from emotion_detector.data_loader import (
    EXPECTED_LABELS,
    drop_invalid_rows,
    standardize_dataset_columns,
    validate_label_values,
)


def load_label_mapping(mapping_path: Path) -> dict[str, str]:
    """
    Load a JSON label-mapping file.

    The JSON file should map source labels to the final project labels.
    """
    if not mapping_path.exists():
        raise FileNotFoundError(f"Label mapping file not found at '{mapping_path}'.")
    if mapping_path.suffix.lower() != ".json":
        raise ValueError("Label mapping file must be a JSON file.")

    with mapping_path.open("r", encoding="utf-8") as file:
        mapping = json.load(file)

    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Label mapping file must contain a non-empty JSON object.")

    normalized_mapping: dict[str, str] = {}
    for source_label, target_label in mapping.items():
        if not isinstance(source_label, str) or not isinstance(target_label, str):
            raise ValueError("All mapping keys and values must be strings.")
        normalized_mapping[source_label.strip().lower()] = target_label.strip().lower()

    invalid_targets = sorted(set(normalized_mapping.values()) - set(EXPECTED_LABELS))
    if invalid_targets:
        raise ValueError(
            "Label mapping contains unsupported target labels: "
            + ", ".join(invalid_targets)
            + ". Expected labels are: "
            + ", ".join(EXPECTED_LABELS)
        )

    return normalized_mapping


def load_raw_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load a raw CSV dataset before column selection and label mapping."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at '{dataset_path}'.")
    if dataset_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV dataset, got '{dataset_path.suffix}'.")

    data_frame = pd.read_csv(dataset_path)
    if data_frame.empty:
        raise ValueError("Raw dataset is empty. Provide a CSV with labeled rows.")

    return data_frame


def map_labels(
    data_frame: pd.DataFrame,
    label_column: str,
    label_mapping: dict[str, str],
) -> pd.DataFrame:
    """
    Map source labels to the final project labels.

    Rows with labels that are not present in the mapping are dropped.
    """
    result = data_frame.copy()
    result[label_column] = result[label_column].astype(str).str.strip().str.lower()
    result[label_column] = result[label_column].map(label_mapping)
    result = result.dropna(subset=[label_column]).reset_index(drop=True)
    return result


def prepare_dataset(
    input_path: Path,
    output_path: Path,
    text_column: str,
    label_column: str,
    label_mapping: dict[str, str],
    remove_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Prepare a raw emotion dataset and save it as a clean training-ready CSV.

    The saved CSV uses the standard project columns: text, label.
    """
    raw_frame = load_raw_dataset(input_path)
    standardized_frame = standardize_dataset_columns(
        data_frame=raw_frame,
        text_column=text_column,
        label_column=label_column,
        rename_to_standard=True,
    )

    mapped_frame = map_labels(
        standardized_frame,
        label_column="label",
        label_mapping=label_mapping,
    )
    cleaned_frame = drop_invalid_rows(
        mapped_frame,
        text_column="text",
        label_column="label",
    )

    if remove_duplicates:
        cleaned_frame = cleaned_frame.drop_duplicates(subset=["text", "label"]).reset_index(
            drop=True
        )

    if cleaned_frame.empty:
        raise ValueError(
            "No rows remain after label mapping and cleanup. Check the label mapping and input CSV."
        )

    validate_label_values(cleaned_frame, label_column="label", allowed_labels=EXPECTED_LABELS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_frame.to_csv(output_path, index=False)
    return cleaned_frame
