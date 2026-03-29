"""Generate prediction-inspection CSV files for the trained baseline model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from emotion_detector.prediction_analysis import run_prediction_analysis


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate prediction analysis artifacts for the baseline model."
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of correct and wrong examples to save.",
    )
    return parser


def main() -> None:
    """Run prediction analysis and print the saved artifact paths."""
    parser = build_parser()
    args = parser.parse_args()

    output_paths = run_prediction_analysis(
        PROJECT_ROOT / "configs" / "baseline_config.json",
        top_n=args.top_n,
    )

    print("Prediction analysis completed successfully.")
    print(f"Full prediction report: {output_paths['full_report']}")
    print(f"Correct example CSV: {output_paths['correct_examples']}")
    print(f"Wrong example CSV: {output_paths['wrong_examples']}")
    print(f"Misclassified example CSV: {output_paths['misclassified_examples']}")
    print(f"Confusion matrix CSV: {output_paths['confusion_matrix']}")
    print(f"Top confusions JSON: {output_paths['top_confusions']}")

    correct_examples = pd.read_csv(output_paths["correct_examples"])
    wrong_examples = pd.read_csv(output_paths["wrong_examples"])

    print("\nA few correct predictions:")
    if correct_examples.empty:
        print("  No correct examples were found.")
    else:
        print(
            correct_examples[["text", "true_label", "predicted_label", "confidence"]]
            .head(5)
            .to_string(index=False)
        )

    print("\nA few wrong predictions:")
    if wrong_examples.empty:
        print("  No wrong examples were found.")
    else:
        print(
            wrong_examples[["text", "true_label", "predicted_label", "confidence"]]
            .head(5)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
