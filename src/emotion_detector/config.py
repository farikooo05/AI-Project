"""Configuration helpers for the project."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class BaselineConfig:
    dataset_path: str
    text_column: str
    label_column: str
    labels: list[str]
    validation_size: float
    test_size: float
    remove_duplicates: bool
    random_state: int
    max_features: int
    ngram_range: tuple[int, int]
    min_df: int
    max_iter: int
    model_output_path: str
    labels_output_path: str
    metrics_output_path: str
    confusion_matrix_output_path: str


def load_config(config_path: Path) -> BaselineConfig:
    """Load JSON config from disk and convert it into a typed object."""
    with config_path.open("r", encoding="utf-8") as file:
        raw_config = json.load(file)

    return BaselineConfig(
        dataset_path=raw_config["dataset_path"],
        text_column=raw_config["text_column"],
        label_column=raw_config["label_column"],
        labels=list(raw_config["labels"]),
        validation_size=raw_config.get("validation_size", 0.1),
        test_size=raw_config["test_size"],
        remove_duplicates=raw_config.get("remove_duplicates", False),
        random_state=raw_config["random_state"],
        max_features=raw_config["max_features"],
        ngram_range=tuple(raw_config["ngram_range"]),
        min_df=raw_config["min_df"],
        max_iter=raw_config["max_iter"],
        model_output_path=raw_config["model_output_path"],
        labels_output_path=raw_config["labels_output_path"],
        metrics_output_path=raw_config["metrics_output_path"],
        confusion_matrix_output_path=raw_config["confusion_matrix_output_path"],
    )
