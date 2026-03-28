"""Inference helpers for trained emotion classification models."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from emotion_detector.config import load_config
from emotion_detector.utils.io import load_joblib, load_json


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


def run_console_inference(config_path: Path) -> None:
    """Run an interactive console loop for manual testing."""
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

    print("Emotion detector console demo")
    print("Type a comment and press Enter. Type 'exit' to quit.")

    while True:
        user_input = input("\nComment: ").strip()
        if user_input.lower() == "exit":
            print("Exiting console demo.")
            break

        if not user_input:
            print("Please enter a non-empty comment.")
            continue

        predicted_label, scored_labels = predict_text(model, labels, user_input)
        print(f"Predicted emotion: {predicted_label}")
        print("Probabilities:")
        for label, score in scored_labels:
            print(f"  - {label}: {score:.4f}")
