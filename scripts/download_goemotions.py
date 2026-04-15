"""Download GoEmotions and save a simple raw CSV for this project."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from emotion_detector.public_datasets import download_and_prepare_goemotions


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Download GoEmotions and save a raw text,label CSV in data/raw/."
    )
    parser.add_argument(
        "--output",
        default="data/raw/goemotions_raw.csv",
        help="Path to the output CSV file relative to the project root.",
    )
    return parser


def main() -> None:
    """Download and convert GoEmotions."""
    parser = build_parser()
    args = parser.parse_args()

    output_path = PROJECT_ROOT / args.output
    prepared_frame = download_and_prepare_goemotions(output_path)

    print("GoEmotions download completed successfully.")
    print(f"Saved CSV: {output_path}")
    print(f"Rows saved: {len(prepared_frame)}")
    print("Columns: text, label")


if __name__ == "__main__":
    main()
