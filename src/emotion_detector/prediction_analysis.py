"""Prediction inspection helpers for baseline error analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from emotion_detector.config import load_config
from emotion_detector.data_loader import load_dataset, split_dataset
from emotion_detector.utils.io import load_joblib


def build_prediction_report(
    model: object,
    test_frame: pd.DataFrame,
    text_column: str,
    label_column: str,
) -> pd.DataFrame:
    """
    Run the trained model on the test set and return a prediction report DataFrame.

    The report includes the original text, true label, predicted label, confidence,
    and whether the prediction was correct.
    """
    texts = test_frame[text_column].tolist()
    true_labels = test_frame[label_column].tolist()

    predicted_labels = model.predict(texts)
    probabilities = model.predict_proba(texts)
    confidences = probabilities.max(axis=1)

    report_frame = test_frame[[text_column, label_column]].copy()
    report_frame = report_frame.rename(
        columns={
            text_column: "text",
            label_column: "true_label",
        }
    )
    report_frame["predicted_label"] = predicted_labels
    report_frame["confidence"] = confidences.round(4)
    report_frame["is_correct"] = report_frame["true_label"] == report_frame["predicted_label"]
    return report_frame


def save_prediction_analysis(
    report_frame: pd.DataFrame,
    output_dir: Path,
    top_n: int = 20,
) -> dict[str, Path]:
    """Save full predictions plus example subsets for correct and wrong predictions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    full_report_path = output_dir / "test_predictions.csv"
    correct_examples_path = output_dir / "correct_examples.csv"
    wrong_examples_path = output_dir / "wrong_examples.csv"

    report_frame.to_csv(full_report_path, index=False)

    correct_examples = report_frame[report_frame["is_correct"]].sort_values(
        by="confidence",
        ascending=False,
    )
    wrong_examples = report_frame[~report_frame["is_correct"]].sort_values(
        by="confidence",
        ascending=False,
    )

    correct_examples.head(top_n).to_csv(correct_examples_path, index=False)
    wrong_examples.head(top_n).to_csv(wrong_examples_path, index=False)

    return {
        "full_report": full_report_path,
        "correct_examples": correct_examples_path,
        "wrong_examples": wrong_examples_path,
    }


def run_prediction_analysis(config_path: Path, top_n: int = 20) -> dict[str, Path]:
    """Rebuild the test split from config, run the trained model, and save CSV artifacts."""
    config = load_config(config_path)
    project_root = config_path.parents[1]

    dataset = load_dataset(
        dataset_path=project_root / config.dataset_path,
        text_column=config.text_column,
        label_column=config.label_column,
        allowed_labels=config.labels,
        remove_duplicates=config.remove_duplicates,
    )
    _, _, test_frame = split_dataset(
        data_frame=dataset,
        text_column=config.text_column,
        label_column=config.label_column,
        test_size=config.test_size,
        validation_size=config.validation_size,
        random_state=config.random_state,
    )

    model_path = project_root / config.model_output_path
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. Run training first."
        )

    model = load_joblib(model_path)
    report_frame = build_prediction_report(
        model=model,
        test_frame=test_frame,
        text_column=config.text_column,
        label_column=config.label_column,
    )

    analysis_dir = (project_root / config.metrics_output_path).parent / "prediction_analysis"
    return save_prediction_analysis(report_frame, output_dir=analysis_dir, top_n=top_n)
