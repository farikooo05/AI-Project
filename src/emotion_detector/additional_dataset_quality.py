"""Quality-control helpers for the curated additional emotion dataset."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
import json
import re
from pathlib import Path

import pandas as pd

from emotion_detector.data_loader import EXPECTED_LABELS
from emotion_detector.utils.io import save_json


WORD_PATTERN = re.compile(r"[a-zA-Z']+")


@dataclass(frozen=True)
class QualityControlConfig:
    """Quality-control settings for the curated additional dataset."""

    min_words: int = 5
    near_duplicate_threshold: float = 0.9
    target_examples_per_class: int = 240


def normalize_text(text: str) -> str:
    """Normalize text for duplicate and diversity checks."""
    words = WORD_PATTERN.findall(text.lower())
    return " ".join(words)


def apply_quality_filters(
    data_frame: pd.DataFrame,
    config: QualityControlConfig,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove weak rows, exact duplicates, and near-duplicates within each class."""
    if data_frame.empty:
        raise ValueError("The generated dataset is empty.")

    frame = data_frame.copy()
    frame["normalized_text"] = frame["text"].map(normalize_text)
    original_count = len(frame)

    min_length_mask = frame["normalized_text"].str.split().map(len) >= config.min_words
    removed_short = int((~min_length_mask).sum())
    frame = frame[min_length_mask].copy()

    before_exact = len(frame)
    frame = frame.drop_duplicates(subset=["label", "normalized_text"]).reset_index(drop=True)
    removed_exact = before_exact - len(frame)

    kept_rows: list[dict[str, str]] = []
    removed_near = 0
    for label in EXPECTED_LABELS:
        label_rows = frame[frame["label"] == label].copy()
        accepted_normalized: list[str] = []
        for row in label_rows.itertuples(index=False):
            normalized = row.normalized_text
            if any(
                SequenceMatcher(a=normalized, b=previous).ratio()
                >= config.near_duplicate_threshold
                for previous in accepted_normalized
            ):
                removed_near += 1
                continue
            accepted_normalized.append(normalized)
            kept_rows.append(
                {
                    "text": row.text,
                    "label": row.label,
                    "source": row.source,
                    "quality_tag": row.quality_tag,
                    "normalized_text": normalized,
                }
            )

    filtered = pd.DataFrame(kept_rows)
    if filtered.empty:
        raise ValueError("Quality control removed all generated examples.")

    filtered = filtered.groupby("label", group_keys=False).head(
        config.target_examples_per_class
    ).reset_index(drop=True)
    final_count = len(filtered)

    return filtered.drop(columns=["normalized_text"]), {
        "input_rows": original_count,
        "removed_short_rows": removed_short,
        "removed_exact_duplicates": removed_exact,
        "removed_near_duplicates": removed_near,
        "final_rows": final_count,
    }


def build_summary(
    data_frame: pd.DataFrame,
    qc_stats: dict[str, int],
) -> dict[str, object]:
    """Create a compact JSON-serializable summary for manual review."""
    counts = data_frame["label"].value_counts().reindex(EXPECTED_LABELS, fill_value=0)
    normalized_texts = [normalize_text(text) for text in data_frame["text"].tolist()]
    vocabulary = Counter(word for text in normalized_texts for word in text.split())

    return {
        **qc_stats,
        "label_counts": {label: int(count) for label, count in counts.items()},
        "vocabulary_size": len(vocabulary),
        "average_words_per_example": round(
            sum(len(text.split()) for text in normalized_texts) / max(len(normalized_texts), 1),
            2,
        ),
    }


def write_examples_report(data_frame: pd.DataFrame, output_path: Path, samples_per_class: int = 10) -> None:
    """Write a readable text file with sample examples for each label."""
    lines: list[str] = []
    for label in EXPECTED_LABELS:
        lines.append(f"[{label}]")
        examples = data_frame[data_frame["label"] == label]["text"].head(samples_per_class).tolist()
        for index, text in enumerate(examples, start=1):
            lines.append(f"  {index}. {text}")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def write_guidelines_report(guidelines: dict[str, dict[str, list[str]]], output_path: Path) -> None:
    """Save strict label guidelines for later manual review."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(guidelines, file, indent=2, ensure_ascii=False)


def save_quality_reports(
    data_frame: pd.DataFrame,
    qc_stats: dict[str, int],
    guidelines: dict[str, dict[str, list[str]]],
    reports_dir: Path,
) -> dict[str, Path]:
    """Save summary and examples reports for the curated additional dataset."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "summary.json"
    examples_path = reports_dir / "examples_per_class.txt"
    guidelines_path = reports_dir / "label_guidelines.json"

    save_json(build_summary(data_frame, qc_stats), summary_path)
    write_examples_report(data_frame, examples_path)
    write_guidelines_report(guidelines, guidelines_path)

    return {
        "summary": summary_path,
        "examples_per_class": examples_path,
        "label_guidelines": guidelines_path,
    }
