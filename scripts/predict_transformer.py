"""Run local inference using a transformer model exported from Google Colab."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from emotion_detector.transformer_inference import TransformerPredictor


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser for transformer inference."""
    parser = argparse.ArgumentParser(
        description="Run local inference with a Colab-trained exported transformer model."
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Input text to classify.",
    )
    parser.add_argument(
        "--model-dir",
        default="artifacts/models/transformer",
        help="Directory containing exported transformer artifacts.",
    )
    return parser


def main() -> None:
    """Load the local transformer artifacts and print a prediction."""
    parser = build_parser()
    args = parser.parse_args()

    predictor = TransformerPredictor(model_dir=args.model_dir)
    probabilities = predictor.predict_proba(args.text)
    prediction = predictor.predict(args.text)

    print("Transformer inference completed successfully.")
    print(f"Predicted label: {prediction['predicted_label']}")
    print("Probabilities:")
    for label, score in sorted(probabilities.items(), key=lambda item: item[1], reverse=True):
        print(f"  - {label}: {score:.4f}")


if __name__ == "__main__":
    main()
