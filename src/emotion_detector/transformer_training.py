"""Transformer training helpers designed for Colab or other GPU-backed environments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from emotion_detector.data_loader import EXPECTED_LABELS, load_dataset, split_dataset
from emotion_detector.utils.io import save_json


@dataclass(frozen=True)
class TransformerTrainingConfig:
    """Minimal, explicit config for Colab-friendly transformer fine-tuning."""

    dataset_path: str
    text_column: str = "text"
    label_column: str = "label"
    labels: tuple[str, ...] = EXPECTED_LABELS
    validation_size: float = 0.25
    test_size: float = 0.25
    remove_duplicates: bool = True
    random_state: int = 42
    model_name: str = "distilbert-base-uncased"
    output_dir: str = "artifacts/models/transformer"
    reports_dir: str = "artifacts/reports/transformer_training"
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    train_batch_size: int = 16
    eval_batch_size: int = 16
    weight_decay: float = 0.01
    max_length: int = 128
    use_class_weights: bool = True


class WeightedTransformerTrainer:
    """Thin wrapper around Hugging Face Trainer with weighted cross-entropy support."""

    def __init__(self, trainer_cls: type, class_weights: Any | None, *args: Any, **kwargs: Any) -> None:
        self._trainer = trainer_cls(*args, **kwargs)
        self._class_weights = class_weights

        original_compute_loss = self._trainer.compute_loss

        def compute_loss(model: Any, inputs: dict[str, Any], return_outputs: bool = False, **extra_kwargs: Any) -> Any:
            if self._class_weights is None:
                return original_compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    **extra_kwargs,
                )

            import torch

            labels = inputs["labels"]
            model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
            outputs = model(**model_inputs)
            logits = outputs.logits
            loss_function = torch.nn.CrossEntropyLoss(
                weight=self._class_weights.to(logits.device)
            )
            loss = loss_function(logits.view(-1, model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

        self._trainer.compute_loss = compute_loss

    def __getattr__(self, name: str) -> Any:
        return getattr(self._trainer, name)


class TransformerTrainer:
    """
    Train a Hugging Face sequence-classification model on the prepared 6-class dataset.

    This module is intended for Colab or another GPU-backed environment. It keeps
    the local baseline pipeline untouched while standardizing the transformer
    training contract, saved label mappings, class-weight artifacts, and reports.
    """

    def __init__(self, config: TransformerTrainingConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.reports_dir = Path(config.reports_dir)
        self.train_frame: pd.DataFrame | None = None
        self.validation_frame: pd.DataFrame | None = None
        self.test_frame: pd.DataFrame | None = None
        self.label2id = {label: index for index, label in enumerate(config.labels)}
        self.id2label = {index: label for label, index in self.label2id.items()}

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load the prepared CSV and create reproducible train/validation/test splits."""
        dataset = load_dataset(
            dataset_path=Path(self.config.dataset_path),
            text_column=self.config.text_column,
            label_column=self.config.label_column,
            allowed_labels=self.config.labels,
            remove_duplicates=self.config.remove_duplicates,
        )
        train_frame, validation_frame, test_frame = split_dataset(
            data_frame=dataset,
            text_column=self.config.text_column,
            label_column=self.config.label_column,
            test_size=self.config.test_size,
            validation_size=self.config.validation_size,
            random_state=self.config.random_state,
        )

        self.train_frame = train_frame.reset_index(drop=True)
        self.validation_frame = validation_frame.reset_index(drop=True)
        self.test_frame = test_frame.reset_index(drop=True)
        return self.train_frame, self.validation_frame, self.test_frame

    def compute_class_weights(self) -> dict[str, float]:
        """Compute inverse-frequency class weights from the mapped training split."""
        if self.train_frame is None:
            self.load_data()

        assert self.train_frame is not None
        counts = self.train_frame[self.config.label_column].value_counts()
        total_rows = len(self.train_frame)
        class_count = len(self.config.labels)

        return {
            label: round(total_rows / (class_count * int(counts[label])), 6)
            for label in self.config.labels
        }

    def _build_tokenized_datasets(self) -> tuple[Any, Any, Any]:
        """Convert the split DataFrames into tokenized Hugging Face datasets."""
        if self.train_frame is None or self.validation_frame is None or self.test_frame is None:
            self.load_data()

        assert self.train_frame is not None
        assert self.validation_frame is not None
        assert self.test_frame is not None

        try:
            from datasets import Dataset, DatasetDict
            from transformers import AutoTokenizer, DataCollatorWithPadding
        except ImportError as error:
            raise RuntimeError(
                "Transformer training dependencies are missing. Install datasets and "
                "transformers in Colab or your training environment first."
            ) from error

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        def encode_frame(frame: pd.DataFrame) -> pd.DataFrame:
            encoded = frame[[self.config.text_column, self.config.label_column]].copy()
            encoded["labels"] = encoded[self.config.label_column].map(self.label2id)
            return encoded.drop(columns=[self.config.label_column])

        dataset_dict = DatasetDict(
            {
                "train": Dataset.from_pandas(encode_frame(self.train_frame), preserve_index=False),
                "validation": Dataset.from_pandas(
                    encode_frame(self.validation_frame), preserve_index=False
                ),
                "test": Dataset.from_pandas(encode_frame(self.test_frame), preserve_index=False),
            }
        )

        def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
            return tokenizer(
                batch[self.config.text_column],
                truncation=True,
                max_length=self.config.max_length,
            )

        tokenized_dataset = dataset_dict.map(tokenize_batch, batched=True)
        tokenized_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )

        return tokenized_dataset, DataCollatorWithPadding(tokenizer=tokenizer), tokenizer

    def _build_compute_metrics(self) -> Any:
        """Create a Trainer-compatible metrics function."""

        def compute_metrics(eval_prediction: Any) -> dict[str, float]:
            logits, labels = eval_prediction
            predicted_ids = np.argmax(logits, axis=-1)
            return {
                "accuracy": accuracy_score(labels, predicted_ids),
                "weighted_f1": f1_score(labels, predicted_ids, average="weighted"),
                "macro_f1": f1_score(labels, predicted_ids, average="macro"),
            }

        return compute_metrics

    def _build_training_args(self) -> Any:
        """Create Hugging Face TrainingArguments with version-friendly defaults."""
        try:
            import torch
            from transformers import TrainingArguments
        except ImportError as error:
            raise RuntimeError(
                "Transformer training dependencies are missing. Install torch and "
                "transformers in your training environment first."
            ) from error

        return TrainingArguments(
            output_dir=str(self.output_dir),
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            num_train_epochs=self.config.num_train_epochs,
            weight_decay=self.config.weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            save_total_limit=2,
            report_to="none",
            fp16=bool(torch.cuda.is_available()),
        )

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax for saved prediction probabilities."""
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exponentiated = np.exp(shifted_logits)
        return exponentiated / exponentiated.sum(axis=1, keepdims=True)

    def _build_detailed_report(
        self,
        true_labels: list[str],
        predicted_labels: list[str],
    ) -> dict[str, Any]:
        """Create a per-class evaluation report for the transformer test set."""
        class_report = classification_report(
            true_labels,
            predicted_labels,
            labels=list(self.config.labels),
            output_dict=True,
            zero_division=0,
        )
        return {
            "accuracy": accuracy_score(true_labels, predicted_labels),
            "weighted_f1": f1_score(true_labels, predicted_labels, average="weighted"),
            "macro_f1": f1_score(true_labels, predicted_labels, average="macro"),
            "classification_report": class_report,
        }

    def _save_reports(
        self,
        metrics_report: dict[str, Any],
        true_labels: list[str],
        predicted_labels: list[str],
        probabilities: np.ndarray,
    ) -> dict[str, Path]:
        """Save class weights, per-class metrics, and error-analysis artifacts."""
        assert self.test_frame is not None

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        class_weights = self.compute_class_weights()
        class_weights_path = self.reports_dir / "class_weights.json"
        evaluation_report_path = self.reports_dir / "evaluation_report.json"
        confusion_matrix_path = self.reports_dir / "confusion_matrix.csv"
        misclassified_examples_path = self.reports_dir / "misclassified_examples.csv"
        top_confusions_path = self.reports_dir / "top_confusions.json"
        training_config_path = self.reports_dir / "training_config.json"
        label_mapping_path = self.output_dir / "label_mappings.json"

        save_json(class_weights, class_weights_path)
        save_json(metrics_report, evaluation_report_path)
        save_json(asdict(self.config), training_config_path)
        save_json(
            {
                "label2id": self.label2id,
                "id2label": {str(index): label for index, label in self.id2label.items()},
            },
            label_mapping_path,
        )

        matrix = confusion_matrix(true_labels, predicted_labels, labels=list(self.config.labels))
        pd.DataFrame(matrix, index=self.config.labels, columns=self.config.labels).to_csv(
            confusion_matrix_path
        )

        prediction_frame = self.test_frame[[self.config.text_column, self.config.label_column]].copy()
        prediction_frame = prediction_frame.rename(
            columns={
                self.config.text_column: "text",
                self.config.label_column: "true_label",
            }
        )
        prediction_frame["predicted_label"] = predicted_labels
        prediction_frame["confidence"] = probabilities.max(axis=1).round(4)
        prediction_frame["is_correct"] = (
            prediction_frame["true_label"] == prediction_frame["predicted_label"]
        )
        prediction_frame[~prediction_frame["is_correct"]].sort_values(
            by="confidence",
            ascending=False,
        ).to_csv(misclassified_examples_path, index=False)

        top_confusions = (
            prediction_frame[~prediction_frame["is_correct"]]
            .groupby(["true_label", "predicted_label"])
            .agg(
                count=("text", "size"),
                average_confidence=("confidence", "mean"),
            )
            .reset_index()
            .sort_values(by=["count", "average_confidence"], ascending=[False, False])
        )
        top_confusion_rows = [
            {
                "true_label": row["true_label"],
                "predicted_label": row["predicted_label"],
                "count": int(row["count"]),
                "average_confidence": round(float(row["average_confidence"]), 4),
            }
            for _, row in top_confusions.iterrows()
        ]
        save_json(top_confusion_rows, top_confusions_path)

        return {
            "class_weights": class_weights_path,
            "evaluation_report": evaluation_report_path,
            "confusion_matrix": confusion_matrix_path,
            "misclassified_examples": misclassified_examples_path,
            "top_confusions": top_confusions_path,
            "training_config": training_config_path,
            "label_mappings": label_mapping_path,
        }

    def train(self) -> dict[str, Path]:
        """Train the transformer and save model plus evaluation artifacts."""
        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                Trainer,
            )
        except ImportError as error:
            raise RuntimeError(
                "Transformer training dependencies are missing. Install torch and "
                "transformers in Colab or your training environment first."
            ) from error

        tokenized_dataset, data_collator, tokenizer = self._build_tokenized_datasets()
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=len(self.config.labels),
            id2label=self.id2label,
            label2id=self.label2id,
        )

        class_weight_tensor = None
        if self.config.use_class_weights:
            class_weight_values = [
                self.compute_class_weights()[label] for label in self.config.labels
            ]
            class_weight_tensor = torch.tensor(class_weight_values, dtype=torch.float32)

        trainer = WeightedTransformerTrainer(
            Trainer,
            class_weight_tensor,
            model=model,
            args=self._build_training_args(),
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=self._build_compute_metrics(),
        )

        trainer.train()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(self.output_dir))
        tokenizer.save_pretrained(str(self.output_dir))

        test_prediction_output = trainer.predict(tokenized_dataset["test"])
        test_logits = test_prediction_output.predictions
        test_label_ids = test_prediction_output.label_ids
        predicted_ids = np.argmax(test_logits, axis=1)
        true_labels = [self.id2label[int(index)] for index in test_label_ids]
        predicted_labels = [self.id2label[int(index)] for index in predicted_ids]
        probabilities = self._softmax(test_logits)

        metrics_report = self._build_detailed_report(true_labels, predicted_labels)
        validation_metrics = trainer.evaluate(tokenized_dataset["validation"])
        metrics_report["validation_metrics"] = validation_metrics

        return self._save_reports(
            metrics_report=metrics_report,
            true_labels=true_labels,
            predicted_labels=predicted_labels,
            probabilities=probabilities,
        )

    def save_model(self) -> None:
        """Model saving is handled inside `train()` after fine-tuning completes."""
        raise NotImplementedError(
            "Call train() to fine-tune and save the transformer artifacts together."
        )
