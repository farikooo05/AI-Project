"""Helpers for preparing raw emotion datasets for the training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from emotion_detector.data_loader import (
    EXPECTED_LABELS,
    standardize_dataset_columns,
    validate_label_values,
)


@dataclass(frozen=True)
class LabelMappingConfig:
    """Explicit source-label mapping rules for dataset preparation."""

    accepted_mappings: dict[str, str]
    dropped_labels: tuple[str, ...] = ()
    excluded_source_labels: tuple[str, ...] = ()


def _normalize_mapping_values(
    accepted_mappings: dict[str, str],
    dropped_labels: list[str] | tuple[str, ...],
    excluded_source_labels: list[str] | tuple[str, ...] = (),
) -> LabelMappingConfig:
    """Validate and normalize mapping configuration values."""
    normalized_mapping: dict[str, str] = {}
    for source_label, target_label in accepted_mappings.items():
        if not isinstance(source_label, str) or not isinstance(target_label, str):
            raise ValueError("All accepted mapping keys and values must be strings.")
        normalized_mapping[source_label.strip().lower()] = target_label.strip().lower()

    invalid_targets = sorted(set(normalized_mapping.values()) - set(EXPECTED_LABELS))
    if invalid_targets:
        raise ValueError(
            "Label mapping contains unsupported target labels: "
            + ", ".join(invalid_targets)
            + ". Expected labels are: "
            + ", ".join(EXPECTED_LABELS)
        )

    normalized_dropped_labels: list[str] = []
    for label in dropped_labels:
        if not isinstance(label, str):
            raise ValueError("Dropped labels must be strings.")
        normalized_dropped_labels.append(label.strip().lower())

    normalized_excluded_labels: list[str] = []
    for label in excluded_source_labels:
        if not isinstance(label, str):
            raise ValueError("Excluded source labels must be strings.")
        normalized_excluded_labels.append(label.strip().lower())

    overlap = sorted(set(normalized_mapping) & set(normalized_dropped_labels))
    if overlap:
        raise ValueError(
            "A source label cannot be both mapped and dropped: " + ", ".join(overlap)
        )

    active_mapping = {
        source_label: target_label
        for source_label, target_label in normalized_mapping.items()
        if source_label not in set(normalized_excluded_labels)
    }
    effective_dropped_labels = tuple(
        sorted(set(normalized_dropped_labels) | set(normalized_excluded_labels))
    )

    return LabelMappingConfig(
        accepted_mappings=active_mapping,
        dropped_labels=effective_dropped_labels,
        excluded_source_labels=tuple(sorted(set(normalized_excluded_labels))),
    )


def load_label_mapping(
    mapping_path: Path,
    excluded_source_labels: list[str] | tuple[str, ...] | None = None,
) -> LabelMappingConfig:
    """
    Load a JSON label-mapping file.

    Supported formats:
    - legacy flat mapping: {"joy": "joy", "fear": "fear", ...}
    - structured mapping:
      {
        "accepted_mappings": {...},
        "dropped_labels": [...]
      }
    """
    if not mapping_path.exists():
        raise FileNotFoundError(f"Label mapping file not found at '{mapping_path}'.")
    if mapping_path.suffix.lower() != ".json":
        raise ValueError("Label mapping file must be a JSON file.")

    with mapping_path.open("r", encoding="utf-8") as file:
        mapping = json.load(file)

    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Label mapping file must contain a non-empty JSON object.")

    if "accepted_mappings" in mapping:
        accepted_mappings = mapping.get("accepted_mappings")
        dropped_labels = mapping.get("dropped_labels", [])
        if not isinstance(accepted_mappings, dict):
            raise ValueError("'accepted_mappings' must be a JSON object.")
        if not isinstance(dropped_labels, list):
            raise ValueError("'dropped_labels' must be a JSON array.")
        return _normalize_mapping_values(
            accepted_mappings,
            dropped_labels,
            excluded_source_labels or [],
        )

    return _normalize_mapping_values(mapping, [], excluded_source_labels or [])


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
    label_mapping: LabelMappingConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Map source labels to the final project labels.

    Rows with dropped or unknown labels are removed. The function returns the
    mapped DataFrame together with an audit report describing what happened.
    """
    result = data_frame.copy()
    result["source_label"] = result[label_column].astype(str).str.strip().str.lower()

    original_counts = result["source_label"].value_counts().sort_index()
    accepted_mapping = label_mapping.accepted_mappings
    dropped_set = set(label_mapping.dropped_labels)

    result["mapped_label"] = result["source_label"].map(accepted_mapping)
    result["drop_reason"] = pd.NA
    result.loc[result["source_label"].isin(dropped_set), "drop_reason"] = "configured_drop"
    result.loc[
        result["mapped_label"].isna() & result["drop_reason"].isna(),
        "drop_reason",
    ] = "unknown_label"

    kept_frame = result[result["mapped_label"].notna()].copy()
    kept_frame[label_column] = kept_frame["mapped_label"]
    kept_frame = kept_frame.drop(columns=["mapped_label", "drop_reason"]).reset_index(drop=True)

    dropped_counts = (
        result[result["drop_reason"].notna()]["source_label"].value_counts().sort_index().to_dict()
    )
    unknown_labels = sorted(
        result.loc[result["drop_reason"] == "unknown_label", "source_label"].unique().tolist()
    )

    source_labels_by_target: dict[str, list[str]] = {}
    for source_label, target_label in sorted(accepted_mapping.items()):
        source_labels_by_target.setdefault(target_label, []).append(source_label)

    audit: dict[str, Any] = {
        "input_rows": int(len(result)),
        "mapped_rows": int(len(kept_frame)),
        "dropped_rows": int(result["drop_reason"].notna().sum()),
        "dropped_percentage": round(
            float(result["drop_reason"].notna().mean() * 100),
            2,
        ),
        "original_label_counts": {label: int(count) for label, count in original_counts.items()},
        "mapped_label_counts": {
            label: int(count)
            for label, count in kept_frame[label_column].value_counts().sort_index().items()
        },
        "dropped_label_counts": {label: int(count) for label, count in dropped_counts.items()},
        "configured_dropped_labels": list(label_mapping.dropped_labels),
        "runtime_excluded_labels": list(label_mapping.excluded_source_labels),
        "unknown_labels": unknown_labels,
        "source_labels_by_target": source_labels_by_target,
    }

    return kept_frame, audit


def _build_class_distribution_frame(
    data_frame: pd.DataFrame,
    label_column: str,
) -> pd.DataFrame:
    """Create a count and percentage summary for the final target classes."""
    counts = data_frame[label_column].value_counts().reindex(EXPECTED_LABELS, fill_value=0)
    total = max(len(data_frame), 1)
    return pd.DataFrame(
        {
            "label": counts.index,
            "count": counts.values,
            "percentage": [round((count / total) * 100, 2) for count in counts.values],
        }
    )


def _build_class_weight_frame(
    data_frame: pd.DataFrame,
    label_column: str,
) -> pd.DataFrame:
    """Create a simple inverse-frequency class-weight table."""
    counts = data_frame[label_column].value_counts().reindex(EXPECTED_LABELS, fill_value=0)
    total = len(data_frame)
    class_count = len(EXPECTED_LABELS)

    return pd.DataFrame(
        {
            "label": counts.index,
            "count": counts.values,
            "weight": [
                round((total / (class_count * count)), 6) if count else 0.0
                for count in counts.values
            ],
        }
    )


def _build_warnings(
    class_distribution: pd.DataFrame,
    source_labels_by_target: dict[str, list[str]],
) -> list[str]:
    """Create human-readable warnings for suspicious dataset conditions."""
    warnings: list[str] = []

    if not class_distribution.empty:
        dominant_row = class_distribution.sort_values("percentage", ascending=False).iloc[0]
        if float(dominant_row["percentage"]) >= 40.0:
            warnings.append(
                f"Class imbalance warning: '{dominant_row['label']}' represents "
                f"{dominant_row['percentage']:.2f}% of the final dataset."
            )

    for target_label, source_labels in sorted(source_labels_by_target.items()):
        if len(source_labels) > 4:
            warnings.append(
                f"Mapping breadth warning: '{target_label}' contains "
                f"{len(source_labels)} source labels ({', '.join(source_labels)})."
            )

    return warnings


def save_preparation_reports(
    prepared_frame: pd.DataFrame,
    audit: dict[str, Any],
    reports_dir: Path,
    sample_examples_per_class: int = 3,
    focus_classes: tuple[str, ...] = ("joy", "fear", "anger", "neutral"),
    focus_samples_per_class: int = 8,
    source_group_example_count: int = 3,
) -> dict[str, Path]:
    """Save mapping and dataset-quality diagnostics for preparation."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    class_distribution = _build_class_distribution_frame(prepared_frame, label_column="label")
    class_weights = _build_class_weight_frame(prepared_frame, label_column="label")
    warnings = _build_warnings(class_distribution, audit["source_labels_by_target"])
    audit["warnings"] = warnings

    mapping_audit_path = reports_dir / "mapping_audit.json"
    class_distribution_path = reports_dir / "class_distribution.csv"
    class_weights_path = reports_dir / "class_weights.csv"
    sample_examples_path = reports_dir / "sample_examples_per_class.txt"
    focused_examples_path = reports_dir / "focused_examples_per_class.txt"
    source_groups_path = reports_dir / "source_label_groups.json"
    warnings_path = reports_dir / "warnings.txt"

    with mapping_audit_path.open("w", encoding="utf-8") as file:
        json.dump(audit, file, indent=2, ensure_ascii=False)

    class_distribution.to_csv(class_distribution_path, index=False)
    class_weights.to_csv(class_weights_path, index=False)
    source_group_report: dict[str, dict[str, Any]] = {}
    for target_label, source_labels in audit["source_labels_by_target"].items():
        source_group_report[target_label] = {}
        for source_label in source_labels:
            source_rows = prepared_frame[prepared_frame["source_label"] == source_label]
            source_group_report[target_label][source_label] = {
                "count": int(len(source_rows)),
                "examples": source_rows["text"].head(source_group_example_count).tolist(),
            }

    with source_groups_path.open("w", encoding="utf-8") as file:
        json.dump(source_group_report, file, indent=2, ensure_ascii=False)

    lines: list[str] = []
    for label in EXPECTED_LABELS:
        examples = (
            prepared_frame[prepared_frame["label"] == label]["text"]
            .head(sample_examples_per_class)
            .tolist()
        )
        lines.append(f"[{label}]")
        if not examples:
            lines.append("  (no examples available)")
        else:
            for index, text in enumerate(examples, start=1):
                lines.append(f"  {index}. {text}")
        lines.append("")

    sample_examples_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    focused_lines: list[str] = []
    for label in focus_classes:
        examples = (
            prepared_frame[prepared_frame["label"] == label]["text"]
            .head(focus_samples_per_class)
            .tolist()
        )
        focused_lines.append(f"[{label}]")
        if not examples:
            focused_lines.append("  (no examples available)")
        else:
            for index, text in enumerate(examples, start=1):
                focused_lines.append(f"  {index}. {text}")
        focused_lines.append("")
    focused_examples_path.write_text(
        "\n".join(focused_lines).strip() + "\n",
        encoding="utf-8",
    )

    warnings_path.write_text(
        "\n".join(warnings) + ("\n" if warnings else ""),
        encoding="utf-8",
    )

    return {
        "mapping_audit": mapping_audit_path,
        "class_distribution": class_distribution_path,
        "class_weights": class_weights_path,
        "sample_examples_per_class": sample_examples_path,
        "focused_examples_per_class": focused_examples_path,
        "source_label_groups": source_groups_path,
        "warnings": warnings_path,
    }


def prepare_dataset(
    input_path: Path,
    output_path: Path,
    text_column: str,
    label_column: str,
    label_mapping: LabelMappingConfig,
    remove_duplicates: bool = True,
    reports_dir: Path | None = None,
    sample_examples_per_class: int = 3,
    focus_classes: tuple[str, ...] = ("joy", "fear", "anger", "neutral"),
    focus_samples_per_class: int = 8,
    source_group_example_count: int = 3,
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

    mapped_frame, audit = map_labels(
        standardized_frame,
        label_column="label",
        label_mapping=label_mapping,
    )
    cleaned_frame = mapped_frame.dropna(subset=["text", "label"]).copy()
    cleaned_frame["text"] = cleaned_frame["text"].astype(str).str.strip()
    cleaned_frame["label"] = cleaned_frame["label"].astype(str).str.strip().str.lower()
    cleaned_frame = cleaned_frame[
        cleaned_frame["text"].ne("") & cleaned_frame["label"].ne("")
    ].reset_index(drop=True)

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

    if reports_dir is not None:
        report_paths = save_preparation_reports(
            cleaned_frame,
            audit,
            reports_dir=reports_dir,
            sample_examples_per_class=sample_examples_per_class,
            focus_classes=focus_classes,
            focus_samples_per_class=focus_samples_per_class,
            source_group_example_count=source_group_example_count,
        )
        cleaned_frame.attrs["report_paths"] = report_paths
        cleaned_frame.attrs["mapping_audit"] = audit

    return cleaned_frame
