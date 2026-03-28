"""Training workflow for the baseline TF-IDF + Logistic Regression model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from emotion_detector.config import load_config
from emotion_detector.data_loader import load_dataset
from emotion_detector.evaluation import (
    build_metrics_report,
    save_confusion_matrix,
    save_metrics,
)
from emotion_detector.preprocessing import clean_text
from emotion_detector.utils.io import save_joblib, save_json


def split_dataset(
    data_frame: pd.DataFrame,
    text_column: str,
    label_column: str,
    test_size: float,
    validation_size: float,
    random_state: int,
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
    """Split the dataset into train, validation, and test sets."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0 <= validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")
    if test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be less than 1.")

    texts = data_frame[text_column].tolist()
    labels = data_frame[label_column].tolist()
    x_temp, x_test, y_temp, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    if validation_size == 0:
        return x_temp, [], x_test, y_temp, [], y_test

    remaining_size = 1 - test_size
    validation_ratio = validation_size / remaining_size

    x_train, x_val, y_train, y_val = train_test_split(
        x_temp,
        y_temp,
        test_size=validation_ratio,
        random_state=random_state,
        stratify=y_temp,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def build_baseline_pipeline(
    max_features: int,
    ngram_range: tuple[int, int],
    min_df: int,
    max_iter: int,
    random_state: int,
) -> Pipeline:
    """Create the baseline sklearn pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    max_features=max_features,
                    ngram_range=ngram_range,
                    min_df=min_df,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=max_iter,
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )


def run_training(config_path: Path) -> None:
    """Execute the full baseline training pipeline."""
    config = load_config(config_path)
    project_root = config_path.parents[1]

    dataset = load_dataset(
        dataset_path=project_root / config.dataset_path,
        text_column=config.text_column,
        label_column=config.label_column,
        allowed_labels=config.labels,
        remove_duplicates=config.remove_duplicates,
    )

    minimum_examples_per_label = 3 if config.validation_size > 0 else 2
    label_counts = dataset[config.label_column].value_counts()
    underrepresented_labels = label_counts[label_counts < minimum_examples_per_label]
    if not underrepresented_labels.empty:
        raise ValueError(
            "Each label must have at least "
            f"{minimum_examples_per_label} examples for the requested split. "
            "Labels with too few examples: "
            + ", ".join(
                f"{label} ({count})" for label, count in underrepresented_labels.items()
            )
        )

    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(
        data_frame=dataset,
        text_column=config.text_column,
        label_column=config.label_column,
        test_size=config.test_size,
        validation_size=config.validation_size,
        random_state=config.random_state,
    )

    pipeline = build_baseline_pipeline(
        max_features=config.max_features,
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        max_iter=config.max_iter,
        random_state=config.random_state,
    )
    pipeline.fit(x_train, y_train)

    label_names = pipeline.classes_.tolist()
    validation_metrics = None
    if x_val:
        validation_predictions = pipeline.predict(x_val)
        validation_metrics = build_metrics_report(
            y_val,
            validation_predictions.tolist(),
            label_names,
        )

    predictions = pipeline.predict(x_test)
    metrics = build_metrics_report(y_test, predictions.tolist(), label_names)
    if validation_metrics is not None:
        metrics["validation"] = validation_metrics

    save_joblib(pipeline, project_root / config.model_output_path)
    save_json(label_names, project_root / config.labels_output_path)
    save_metrics(metrics, project_root / config.metrics_output_path)
    save_confusion_matrix(
        y_true=y_test,
        y_pred=predictions.tolist(),
        labels=label_names,
        output_path=project_root / config.confusion_matrix_output_path,
    )

    print("Training completed successfully.")
    print(f"Dataset size: {len(dataset)} rows")
    print(f"Train split: {len(x_train)} rows")
    print(f"Validation split: {len(x_val)} rows")
    print(f"Test split: {len(x_test)} rows")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1-score: {metrics['macro_f1']:.4f}")
    if validation_metrics is not None:
        print(f"Validation Macro F1-score: {validation_metrics['macro_f1']:.4f}")
    print(f"Model saved to: {project_root / config.model_output_path}")
    print(f"Metrics saved to: {project_root / config.metrics_output_path}")
    print(f"Confusion matrix saved to: {project_root / config.confusion_matrix_output_path}")
