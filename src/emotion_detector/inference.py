"""Inference helpers for trained emotion classification models."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from emotion_detector.config import load_config
from emotion_detector.explainability import (
    explain_prediction,
    format_feature_contributions,
)
from emotion_detector.utils.io import load_joblib, load_json

EXIT_COMMANDS = {"exit", "quit", "q"}


def predict_text(model: object, labels: list[str], text: str) -> tuple[str, list[tuple[str, float]]]:
    """Predict the emotion of a single text and return sorted probabilities."""
    probabilities = model.predict_proba([text])[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_label = labels[predicted_index]

    scored_labels = sorted(
        zip(labels, probabilities, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    return predicted_label, [(label, float(score)) for label, score in scored_labels]


def format_top_predictions(
    scored_labels: list[tuple[str, float]],
    top_n: int = 3,
) -> str:
    """Format the top predicted emotions for console presentation."""
    top_predictions = scored_labels[:top_n]
    lines = []
    for index, (label, score) in enumerate(top_predictions, start=1):
        prefix = "->" if index == 1 else "  "
        lines.append(f"{prefix} Top {index}: {label} ({score * 100:.2f}%)")
    return "\n".join(lines)


def format_all_probabilities(scored_labels: list[tuple[str, float]]) -> str:
    """Format all class probabilities in a compact presentation-friendly style."""
    return " | ".join(
        f"{label}: {score * 100:.2f}%"
        for label, score in scored_labels
    )


def run_console_inference(config_path: Path) -> None:
    """Run an interactive console loop for demo-friendly prediction."""
    config = load_config(config_path)
    project_root = config_path.parents[1]

    model_path = project_root / config.model_output_path
    labels_path = project_root / config.labels_output_path

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. Run training first."
        )

    model = load_joblib(model_path)
    labels = load_json(labels_path)

    print("=" * 48)
    print("Emotion Detection Demo")
    print("Baseline: TF-IDF + Logistic Regression")
    print("Type a comment and press Enter.")
    print("Type 'exit', 'quit', or 'q' to close the demo.")
    print("=" * 48)

    while True:
        user_input = input("\nComment: ").strip()
        if user_input.lower() in EXIT_COMMANDS:
            print("Exiting console demo.")
            break

        if not user_input:
            print("Please enter a non-empty comment.")
            continue

        predicted_label, scored_labels = predict_text(model, labels, user_input)
        print(f"\nPredicted emotion: {predicted_label.upper()}")
        print("Top predictions:")
        print(format_top_predictions(scored_labels, top_n=3))
        print("All class probabilities:")
        print(format_all_probabilities(scored_labels))
        _, contributions = explain_prediction(model, user_input, top_n=5)
        print("Top contributing words/features:")
        print(format_feature_contributions(contributions))
