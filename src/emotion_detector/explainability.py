"""Lightweight explainability helpers for the TF-IDF + Logistic Regression baseline."""

from __future__ import annotations

from typing import Any

import numpy as np


def _get_pipeline_components(model: Any) -> tuple[Any, Any]:
    """Return the TF-IDF vectorizer and Logistic Regression classifier from the pipeline."""
    if not hasattr(model, "named_steps"):
        raise TypeError("Expected a trained sklearn Pipeline with named_steps.")

    if "tfidf" not in model.named_steps or "classifier" not in model.named_steps:
        raise ValueError(
            "Expected pipeline steps named 'tfidf' and 'classifier' for baseline explainability."
        )

    return model.named_steps["tfidf"], model.named_steps["classifier"]


def explain_prediction(
    model: Any,
    text: str,
    top_n: int = 5,
) -> tuple[str, list[tuple[str, float]]]:
    """
    Explain a baseline prediction using feature contributions from the linear model.

    The contribution score for each feature is:
    TF-IDF feature value * Logistic Regression coefficient for the predicted class.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string.")
    if top_n <= 0:
        raise ValueError("top_n must be greater than 0.")

    vectorizer, classifier = _get_pipeline_components(model)
    transformed_text = vectorizer.transform([text])
    predicted_label = model.predict([text])[0]

    if not hasattr(classifier, "coef_") or not hasattr(classifier, "classes_"):
        raise TypeError("The classifier does not expose linear coefficients for explainability.")

    class_index = int(np.where(classifier.classes_ == predicted_label)[0][0])
    coefficients = classifier.coef_[class_index]
    row = transformed_text[0]

    if row.nnz == 0:
        return predicted_label, []

    feature_names = vectorizer.get_feature_names_out()
    contributions = []
    for feature_index, feature_value in zip(row.indices, row.data, strict=True):
        contribution = float(feature_value * coefficients[feature_index])
        if contribution > 0:
            contributions.append((feature_names[feature_index], contribution))

    contributions.sort(key=lambda item: item[1], reverse=True)
    return predicted_label, contributions[:top_n]


def format_feature_contributions(contributions: list[tuple[str, float]]) -> str:
    """Format feature contributions for the console demo."""
    if not contributions:
        return "  No strong positive feature contributions were found."

    lines = []
    for feature, score in contributions:
        lines.append(f"  - {feature} ({score:.4f})")
    return "\n".join(lines)
