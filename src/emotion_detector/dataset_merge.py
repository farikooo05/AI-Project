"""Helpers for safely merging the cleaned main dataset with curated augmentation data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from emotion_detector.data_loader import EXPECTED_LABELS, load_dataset
from emotion_detector.utils.io import save_json


def _load_with_source(
    dataset_path: Path,
    data_source: str,
    text_column: str = "text",
    label_column: str = "label",
) -> pd.DataFrame:
    """Load a labeled dataset and attach a fixed provenance column."""
    frame = load_dataset(
        dataset_path=dataset_path,
        text_column=text_column,
        label_column=label_column,
        allowed_labels=list(EXPECTED_LABELS),
        remove_duplicates=True,
    )
    frame = frame.copy()
    frame["data_source"] = data_source
    return frame


def merge_training_datasets(
    main_dataset_path: Path,
    additional_dataset_path: Path,
    output_path: Path,
    reports_dir: Path,
    random_state: int = 42,
    text_column: str = "text",
    label_column: str = "label",
) -> tuple[pd.DataFrame, dict[str, Path]]:
    """
    Merge the cleaned main dataset with the curated additional dataset.

    The merged CSV preserves the original label names and adds a `data_source`
    column so later review can separate GoEmotions-derived rows from curated rows.
    """
    main_frame = _load_with_source(
        dataset_path=main_dataset_path,
        data_source="goemotions_cleaned",
        text_column=text_column,
        label_column=label_column,
    )
    additional_frame = _load_with_source(
        dataset_path=additional_dataset_path,
        data_source="curated_additional",
        text_column=text_column,
        label_column=label_column,
    )

    merged = pd.concat([main_frame, additional_frame], ignore_index=True)
    before_dedup = len(merged)
    merged = merged.drop_duplicates(subset=[text_column, label_column]).reset_index(drop=True)
    duplicate_rows_removed = before_dedup - len(merged)
    merged = merged.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8")

    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "merge_summary.json"
    per_label_path = reports_dir / "label_distribution.csv"
    per_source_path = reports_dir / "data_source_distribution.csv"
    source_label_path = reports_dir / "source_label_distribution.csv"

    save_json(
        {
            "main_dataset_path": str(main_dataset_path),
            "additional_dataset_path": str(additional_dataset_path),
            "merged_output_path": str(output_path),
            "total_rows": int(len(merged)),
            "duplicate_rows_removed": int(duplicate_rows_removed),
            "label_counts": {
                label: int((merged[label_column] == label).sum()) for label in EXPECTED_LABELS
            },
            "data_source_counts": {
                source: int(count)
                for source, count in merged["data_source"].value_counts().sort_index().items()
            },
        },
        summary_path,
    )

    merged[label_column].value_counts().reindex(EXPECTED_LABELS, fill_value=0).rename_axis(
        label_column
    ).reset_index(name="count").to_csv(per_label_path, index=False)
    merged["data_source"].value_counts().sort_index().rename_axis("data_source").reset_index(
        name="count"
    ).to_csv(per_source_path, index=False)
    merged.groupby(["data_source", label_column]).size().reset_index(name="count").to_csv(
        source_label_path,
        index=False,
    )

    return merged, {
        "summary": summary_path,
        "label_distribution": per_label_path,
        "data_source_distribution": per_source_path,
        "source_label_distribution": source_label_path,
    }
