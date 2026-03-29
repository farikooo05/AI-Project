"""Merge the cleaned main dataset with the curated dataset and retrain the transformer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from emotion_detector.dataset_merge import merge_training_datasets
from emotion_detector.transformer_training import TransformerTrainer, TransformerTrainingConfig
from emotion_detector.utils.io import load_json, save_json


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for merge-plus-train workflow."""
    parser = argparse.ArgumentParser(
        description=(
            "Merge the cleaned main dataset with the curated additional dataset, "
            "then retrain the transformer and save before/after comparison reports."
        )
    )
    parser.add_argument(
        "--main-dataset",
        default="data/processed/goemotions_6class.csv",
        help="Path to the cleaned main dataset.",
    )
    parser.add_argument(
        "--additional-dataset",
        default="data/processed/emotion_additional_curated.csv",
        help="Path to the curated additional dataset.",
    )
    parser.add_argument(
        "--merged-output",
        default="data/processed/emotion_training_merged.csv",
        help="Path where the merged training CSV will be saved.",
    )
    parser.add_argument(
        "--merge-reports-dir",
        default="artifacts/reports/merged_dataset",
        help="Directory for dataset merge summaries.",
    )
    parser.add_argument(
        "--before-metrics",
        default="artifacts/reports/transformer_training/evaluation_report.json",
        help="Path to the transformer metrics JSON from before augmentation.",
    )
    parser.add_argument(
        "--model-name",
        default="distilbert-base-uncased",
        help="Pretrained Hugging Face model name.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/models/transformer_augmented",
        help="Directory where augmented-model artifacts will be saved.",
    )
    parser.add_argument(
        "--reports-dir",
        default="artifacts/reports/transformer_training_augmented",
        help="Directory where augmented training reports will be saved.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of transformer fine-tuning epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate for transformer fine-tuning.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum sequence length for tokenization.",
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
        help="Disable weighted cross-entropy for the augmented run.",
    )
    return parser


def main() -> None:
    """Merge datasets, retrain the transformer, and save comparison reports."""
    args = build_parser().parse_args()

    merged_frame, merge_reports = merge_training_datasets(
        main_dataset_path=PROJECT_ROOT / args.main_dataset,
        additional_dataset_path=PROJECT_ROOT / args.additional_dataset,
        output_path=PROJECT_ROOT / args.merged_output,
        reports_dir=PROJECT_ROOT / args.merge_reports_dir,
    )

    trainer = TransformerTrainer(
        TransformerTrainingConfig(
            dataset_path=str(PROJECT_ROOT / args.merged_output),
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

    comparison_path = PROJECT_ROOT / args.reports_dir / "before_after_comparison.json"
    before_metrics_path = PROJECT_ROOT / args.before_metrics
    comparison_payload: dict[str, object] = {
        "merged_dataset_path": str(PROJECT_ROOT / args.merged_output),
        "merged_rows": int(len(merged_frame)),
        "before_metrics_path": str(before_metrics_path),
        "after_metrics_path": str(output_paths["evaluation_report"]),
    }

    if before_metrics_path.exists():
        before_metrics = load_json(before_metrics_path)
        after_metrics = load_json(output_paths["evaluation_report"])
        comparison_payload["before"] = {
            "accuracy": before_metrics.get("accuracy"),
            "macro_f1": before_metrics.get("macro_f1"),
            "weighted_f1": before_metrics.get("weighted_f1"),
        }
        comparison_payload["after"] = {
            "accuracy": after_metrics.get("accuracy"),
            "macro_f1": after_metrics.get("macro_f1"),
            "weighted_f1": after_metrics.get("weighted_f1"),
        }
        before_report = before_metrics.get("classification_report", {})
        after_report = after_metrics.get("classification_report", {})
        comparison_payload["per_class_f1_delta"] = {
            label: round(
                float(after_report.get(label, {}).get("f1-score", 0.0))
                - float(before_report.get(label, {}).get("f1-score", 0.0)),
                4,
            )
            for label in ("joy", "sadness", "anger", "fear", "surprise", "neutral")
        }
    else:
        comparison_payload["before_metrics_missing"] = True

    save_json(comparison_payload, comparison_path)

    print("Merged dataset and transformer retraining completed.")
    print(f"Merged dataset CSV: {PROJECT_ROOT / args.merged_output}")
    print(f"Merge summary: {merge_reports['summary']}")
    print(f"Augmented evaluation report: {output_paths['evaluation_report']}")
    print(f"Augmented confusion matrix: {output_paths['confusion_matrix']}")
    print(f"Augmented top confusions: {output_paths['top_confusions']}")
    print(f"Before/after comparison: {comparison_path}")


if __name__ == "__main__":
    main()
