"""Create a high-precision additional emotion dataset for later transformer retraining."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from emotion_detector.additional_dataset_builder import (
    STRICT_LABEL_GUIDELINES,
    build_curated_examples,
)
from emotion_detector.additional_dataset_quality import (
    QualityControlConfig,
    apply_quality_filters,
    save_quality_reports,
)
from emotion_detector.data_loader import format_class_distribution


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Generate a balanced, high-precision additional emotion dataset."
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=240,
        help="Final number of examples to keep per class after quality control.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic candidate shuffling.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/emotion_additional_curated.csv",
        help="Path for the generated additional dataset CSV.",
    )
    parser.add_argument(
        "--reports-dir",
        default="artifacts/reports/additional_dataset",
        help="Directory for summary and manual-review reports.",
    )
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=0.9,
        help="Similarity threshold used to remove near-duplicate rows within each class.",
    )
    return parser


def main() -> None:
    """Generate the additional curated dataset and save quality reports."""
    parser = build_parser()
    args = parser.parse_args()

    raw_candidates = build_curated_examples(per_class=args.per_class, random_seed=args.seed)
    filtered_frame, qc_stats = apply_quality_filters(
        raw_candidates,
        QualityControlConfig(
            min_words=5,
            near_duplicate_threshold=args.near_duplicate_threshold,
            target_examples_per_class=args.per_class,
        ),
    )

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_frame.to_csv(output_path, index=False, encoding="utf-8")

    report_paths = save_quality_reports(
        data_frame=filtered_frame,
        qc_stats=qc_stats,
        guidelines=STRICT_LABEL_GUIDELINES,
        reports_dir=PROJECT_ROOT / args.reports_dir,
    )

    print("Additional curated dataset created successfully.")
    print(f"Saved dataset CSV to: {output_path}")
    print(f"Saved summary report to: {report_paths['summary']}")
    print(f"Saved example report to: {report_paths['examples_per_class']}")
    print(f"Saved label guidelines to: {report_paths['label_guidelines']}")
    print(
        format_class_distribution(
            filtered_frame,
            label_column="label",
            title="Additional dataset class distribution:",
        )
    )


if __name__ == "__main__":
    main()
