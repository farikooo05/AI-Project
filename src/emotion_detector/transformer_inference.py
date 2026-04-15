"""Inference architecture for future transformer models trained outside this repository."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from emotion_detector.transformer_utils import (
    format_output,
    preprocess_text,
    validate_transformer_artifacts,
)


class TransformerPredictor:
    """
    Load a transformer model exported from Google Colab for local inference.

    The baseline pipeline remains fully independent. This class is only for
    inference after a transformer model has already been trained elsewhere and
    its artifacts have been copied into the local project.
    """

    def __init__(
        self,
        model_dir: str = "artifacts/models/transformer",
    ) -> None:
        """Store the expected directory for exported transformer artifacts."""
        self.model_dir = Path(model_dir)
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.artifact_paths: dict[str, Path] | None = None

    def load_model(self) -> None:
        """
        Load local transformer artifacts exported from Google Colab.

        Expected files include:
        - config.json
        - model.safetensors or pytorch_model.bin
        - tokenizer.json
        - tokenizer_config.json
        - special_tokens_map.json
        """
        self.artifact_paths = validate_transformer_artifacts(self.model_dir)

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Transformer inference dependencies are not installed locally. "
                "Install Hugging Face transformers when you are ready to run "
                "a Colab-trained model inside this project."
            ) from error

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)

    def predict_proba(self, text: str) -> dict[str, float]:
        """
        Return class probabilities for a single text using the loaded transformer.

        If the model or dependencies are missing, the method fails with a clear
        error instead of silently falling back to placeholder behavior.
        """
        cleaned_text = preprocess_text(text)
        if not cleaned_text:
            raise ValueError("text must be a non-empty string.")

        if self.model is None or self.tokenizer is None:
            self.load_model()

        try:
            import torch
        except ImportError as error:
            raise RuntimeError(
                "PyTorch is required for local transformer inference but is not installed."
            ) from error

        encoded = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        with torch.no_grad():
            outputs = self.model(**encoded)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0].tolist()

        id_to_label = getattr(self.model.config, "id2label", None)
        if not id_to_label:
            raise ValueError(
                "The exported transformer model is missing id2label metadata in config.json."
            )

        return {
            str(id_to_label[index]).lower(): float(score)
            for index, score in enumerate(probabilities)
        }

    def predict(self, text: str) -> dict[str, str]:
        """
        Return the top predicted label from a Colab-trained exported transformer.

        This class remains inference-only. Transformer training is expected to
        happen separately in Google Colab.
        """
        probabilities = self.predict_proba(text)
        predicted_label = max(probabilities, key=probabilities.get)
        return format_output(predicted_label)
