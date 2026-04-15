"""Train the transformer model on the prepared 6-class dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from emotion_detector.transformer_training import TransformerTrainer, TransformerTrainingConfig


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser for transformer fine-tuning."""
    parser = argparse.ArgumentParser(
        description=(
            "Train a Hugging Face transformer on the prepared 6-class dataset. "
            "This command is intended for Colab or another GPU-backed environment."
        )
    )
    parser.add_argument(
        "--dataset-path",
        default="data/processed/goemotions_6class.csv",
        help="Path to the cleaned 6-class CSV dataset.",
    )
    parser.add_argument(
        "--model-name",
        default="distilbert-base-uncased",
        help="Pretrained Hugging Face model name.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/models/transformer",
        help="Directory where model and tokenizer artifacts will be saved.",
    )
    parser.add_argument(
        "--reports-dir",
        default="artifacts/reports/transformer_training",
        help="Directory where training reports will be saved.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of fine-tuning epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate for training.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum tokenizer sequence length.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=16,
        help="Training batch size per device.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=16,
        help="Evaluation batch size per device.",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable weighted cross-entropy and train with unweighted loss.",
    )
    return parser


def main() -> None:
    """Run transformer training and print the generated artifact paths."""
    args = build_parser().parse_args()

    trainer = TransformerTrainer(
        TransformerTrainingConfig(
            dataset_path=str(PROJECT_ROOT / args.dataset_path),
            model_name=args.model_name,
            output_dir=str(PROJECT_ROOT / args.output_dir),
            reports_dir=str(PROJECT_ROOT / args.reports_dir),
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            use_class_weights=not args.no_class_weights,
        )
    )

    output_paths = trainer.train()

    print("Transformer training completed successfully.")
    for name, path in output_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
