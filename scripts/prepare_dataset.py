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
    parser.add_argument("--input", required=True, help="Path to the raw input CSV file.")
    parser.add_argument("--output", required=True, help="Path for the cleaned output CSV file.")
    parser.add_argument("--text-column", required=True, help="Name of the source text column.")
    parser.add_argument("--label-column", required=True, help="Name of the source label column.")
    parser.add_argument(
        "--mapping-file",
        required=True,
        help="Path to a JSON file that maps source labels to the 6 final labels.",
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

    label_mapping = load_label_mapping(mapping_path)
    prepared_frame = prepare_dataset(
        input_path=input_path,
        output_path=output_path,
        text_column=args.text_column,
        label_column=args.label_column,
        label_mapping=label_mapping,
        remove_duplicates=not args.keep_duplicates,
    )

    print("Dataset preparation completed successfully.")
    print(f"Prepared rows: {len(prepared_frame)}")
    print(f"Saved cleaned dataset to: {output_path}")
    print(format_class_distribution(prepared_frame, label_column="label", title="Prepared dataset class distribution:"))


if __name__ == "__main__":
    main()
