"""Utility helpers for future transformer-based emotion models."""

from __future__ import annotations

import json
from pathlib import Path


def get_expected_transformer_files() -> dict[str, tuple[str, ...]]:
    """
    Return the expected exported files for local transformer inference.

    The model weights may be stored in either `model.safetensors` or
    `pytorch_model.bin`, depending on how the model was exported from Colab.
    """
    return {
        "required": (
            "config.json",
            "label_mappings.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ),
        "model_weights": (
            "model.safetensors",
            "pytorch_model.bin",
        ),
    }


def preprocess_text(text: str) -> str:
    """
    Apply minimal text cleanup before transformer tokenization.

    Transformer tokenizers usually handle most normalization internally, so this
    function intentionally stays lightweight and only trims whitespace.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    return text.strip()


def validate_transformer_artifacts(model_dir: str | Path) -> dict[str, Path]:
    """
    Validate that the expected transformer inference artifacts exist locally.

    Returns resolved file paths for the files that were found.
    """
    directory = Path(model_dir)
    if not directory.exists():
        raise FileNotFoundError(
            f"Transformer model directory was not found at '{directory}'. "
            "Copy the exported Colab model artifacts into this directory first."
        )
    if not directory.is_dir():
        raise NotADirectoryError(f"Transformer model path is not a directory: '{directory}'.")

    expected_files = get_expected_transformer_files()
    resolved_paths: dict[str, Path] = {}
    missing_required_files = []

    for filename in expected_files["required"]:
        file_path = directory / filename
        if not file_path.exists():
            missing_required_files.append(filename)
        else:
            resolved_paths[filename] = file_path

    weight_path = None
    for filename in expected_files["model_weights"]:
        candidate = directory / filename
        if candidate.exists():
            weight_path = candidate
            resolved_paths[filename] = candidate
            break

    if missing_required_files or weight_path is None:
        missing_messages = []
        if missing_required_files:
            missing_messages.append("Missing required files: " + ", ".join(missing_required_files))
        if weight_path is None:
            missing_messages.append(
                "Missing model weights file: one of model.safetensors or pytorch_model.bin"
            )
        raise FileNotFoundError(
            "Transformer artifacts are incomplete in "
            f"'{directory}'. " + " | ".join(missing_messages)
        )

    label_mapping_path = resolved_paths["label_mappings.json"]

    with resolved_paths["config.json"].open("r", encoding="utf-8") as config_file:
        config_data = json.load(config_file)
    with label_mapping_path.open("r", encoding="utf-8") as mapping_file:
        mapping_data = json.load(mapping_file)

    config_label2id = {
        str(label).lower(): int(index)
        for label, index in (config_data.get("label2id") or {}).items()
    }
    config_id2label = {
        int(index): str(label).lower()
        for index, label in (config_data.get("id2label") or {}).items()
    }
    saved_label2id = {
        str(label).lower(): int(index)
        for label, index in (mapping_data.get("label2id") or {}).items()
    }
    saved_id2label = {
        int(index): str(label).lower()
        for index, label in (mapping_data.get("id2label") or {}).items()
    }

    if not config_label2id or not config_id2label or not saved_label2id or not saved_id2label:
        raise ValueError(
            "Transformer label metadata is incomplete. Both config.json and "
            "label_mappings.json must contain label2id and id2label entries."
        )

    if config_label2id != saved_label2id or config_id2label != saved_id2label:
        raise ValueError(
            "Transformer label mapping is inconsistent between config.json and "
            "label_mappings.json. Re-export the model artifacts from Colab to "
            "keep training and local inference aligned."
        )

    return resolved_paths


def format_output(prediction: str) -> dict[str, str]:
    """
    Format a transformer prediction in a simple structured form.

    This placeholder output format can later be extended with probabilities,
    logits, or metadata without affecting the baseline pipeline.
    """
    if not isinstance(prediction, str) or not prediction.strip():
        raise ValueError("prediction must be a non-empty string.")

    return {"predicted_label": prediction.strip().lower()}
