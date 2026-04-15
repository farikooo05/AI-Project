"""Prepare a raw emotion dataset for baseline training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from emotion_detector.dataset_preparation import load_label_mapping, prepare_dataset
from emotion_detector.data_loader import format_class_distribution


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Prepare a raw emotion dataset and save a clean training CSV."
    )
    parser.add_argument(
        "--input",
        default="data/raw/goemotions_raw.csv",
        help="Path to the raw input CSV file.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/goemotions_6class.csv",
        help="Path for the cleaned output CSV file.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Name of the source text column.",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Name of the source label column.",
    )
    parser.add_argument(
        "--mapping-file",
        default="configs/goemotions_label_mapping.json",
        help="Path to a JSON file that maps source labels to the 6 final labels.",
    )
    parser.add_argument(
        "--reports-dir",
        default="artifacts/reports/dataset_preparation",
        help="Directory where mapping audit and class distribution reports will be saved.",
    )
    parser.add_argument(
        "--exclude-source-labels",
        nargs="*",
        default=(),
        help=(
            "Optional source labels to exclude for a stricter experiment, "
            "for example: --exclude-source-labels gratitude relief"
        ),
    )
    parser.add_argument(
        "--examples-per-class",
        type=int,
        default=3,
        help="How many sample examples to save for each class in the general report.",
    )
    parser.add_argument(
        "--focus-classes",
        nargs="*",
        default=("joy", "fear", "anger", "neutral"),
        help="Classes that should receive a larger focused inspection report.",
    )
    parser.add_argument(
        "--focus-samples",
        type=int,
        default=8,
        help="How many examples to save for each focused class.",
    )
    parser.add_argument(
        "--source-group-examples",
        type=int,
        default=3,
        help="How many example texts to save per source-label group.",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep duplicate rows instead of removing them.",
    )
    return parser


def main() -> None:
    """Run dataset preparation from the command line."""
    parser = build_parser()
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / args.output
    mapping_path = PROJECT_ROOT / args.mapping_file
    reports_dir = PROJECT_ROOT / args.reports_dir

    label_mapping = load_label_mapping(
        mapping_path,
        excluded_source_labels=args.exclude_source_labels,
    )
    prepared_frame = prepare_dataset(
        input_path=input_path,
        output_path=output_path,
        text_column=args.text_column,
        label_column=args.label_column,
        label_mapping=label_mapping,
        remove_duplicates=not args.keep_duplicates,
        reports_dir=reports_dir,
        sample_examples_per_class=args.examples_per_class,
        focus_classes=tuple(args.focus_classes),
        focus_samples_per_class=args.focus_samples,
        source_group_example_count=args.source_group_examples,
    )

    print("Dataset preparation completed successfully.")
    print(f"Prepared rows: {len(prepared_frame)}")
    print(f"Saved cleaned dataset to: {output_path}")
    print(f"Saved preparation reports to: {reports_dir}")
    report_paths = prepared_frame.attrs.get("report_paths", {})
    if report_paths:
        print(f"Mapping audit JSON: {report_paths['mapping_audit']}")
        print(f"Class distribution CSV: {report_paths['class_distribution']}")
        print(f"Class weights CSV: {report_paths['class_weights']}")
        print(f"Sample examples TXT: {report_paths['sample_examples_per_class']}")
        print(f"Focused examples TXT: {report_paths['focused_examples_per_class']}")
        print(f"Source label groups JSON: {report_paths['source_label_groups']}")
        print(f"Warnings TXT: {report_paths['warnings']}")
    print(
        format_class_distribution(
            prepared_frame,
            label_column="label",
            title="Prepared dataset class distribution:",
        )
    )


if __name__ == "__main__":
    main()
