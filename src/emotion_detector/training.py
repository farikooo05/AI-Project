"""Training workflow for the baseline TF-IDF + Logistic Regression model."""

from __future__ import annotations

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from emotion_detector.config import load_config
from emotion_detector.data_loader import (
    format_class_distribution,
    load_dataset,
    split_dataset,
)
from emotion_detector.evaluation import (
    build_metrics_report,
    save_confusion_matrix,
    save_metrics,
)
from emotion_detector.preprocessing import clean_text
from emotion_detector.utils.io import save_joblib, save_json


def build_baseline_pipeline(
    max_features: int,
    ngram_range: tuple[int, int],
    min_df: int,
    max_iter: int,
    random_state: int,
    use_balanced_class_weight: bool,
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
                    class_weight="balanced" if use_balanced_class_weight else None,
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

    train_frame, validation_frame, test_frame = split_dataset(
        data_frame=dataset,
        text_column=config.text_column,
        label_column=config.label_column,
        test_size=config.test_size,
        validation_size=config.validation_size,
        random_state=config.random_state,
    )
    x_train = train_frame[config.text_column].tolist()
    y_train = train_frame[config.label_column].tolist()
    x_val = validation_frame[config.text_column].tolist()
    y_val = validation_frame[config.label_column].tolist()
    x_test = test_frame[config.text_column].tolist()
    y_test = test_frame[config.label_column].tolist()

    pipeline = build_baseline_pipeline(
        max_features=config.max_features,
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        max_iter=config.max_iter,
        random_state=config.random_state,
        use_balanced_class_weight=config.use_balanced_class_weight,
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
    print(
        format_class_distribution(
            dataset,
            label_column=config.label_column,
            title="Prepared dataset class distribution:",
        )
    )
    print(f"Train split: {len(x_train)} rows")
    print(
        format_class_distribution(
            train_frame,
            label_column=config.label_column,
            title="Train split class distribution:",
        )
    )
    print(f"Validation split: {len(x_val)} rows")
    if not validation_frame.empty:
        print(
            format_class_distribution(
                validation_frame,
                label_column=config.label_column,
                title="Validation split class distribution:",
            )
        )
    print(f"Test split: {len(x_test)} rows")
    print(
        format_class_distribution(
            test_frame,
            label_column=config.label_column,
            title="Test split class distribution:",
        )
    )
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1-score: {metrics['macro_f1']:.4f}")
    if validation_metrics is not None:
        print(f"Validation Macro F1-score: {validation_metrics['macro_f1']:.4f}")
    print(f"Model saved to: {project_root / config.model_output_path}")
    print(f"Metrics saved to: {project_root / config.metrics_output_path}")
    print(f"Confusion matrix saved to: {project_root / config.confusion_matrix_output_path}")
