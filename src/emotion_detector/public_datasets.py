"""Utilities for downloading and converting public emotion datasets."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


GOEMOTIONS_RAW_URLS = (
    "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv",
    "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv",
    "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv",
)

GOEMOTIONS_LABEL_COLUMNS = (
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
)


def download_file(url: str, output_path: Path) -> Path:
    """Download a single file to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, output_path)
    return output_path


def download_goemotions_raw_files(output_directory: Path) -> list[Path]:
    """Download the official raw GoEmotions CSV files."""
    downloaded_files: list[Path] = []
    for url in GOEMOTIONS_RAW_URLS:
        filename = url.rsplit("/", maxsplit=1)[-1]
        downloaded_files.append(download_file(url, output_directory / filename))
    return downloaded_files


def convert_goemotions_to_text_label_csv(
    input_files: list[Path],
    output_path: Path,
) -> pd.DataFrame:
    """
    Convert raw GoEmotions files into a simple CSV with text and label columns.

    To keep the baseline pipeline simple, only single-label rows are kept.
    The saved labels remain the original GoEmotions labels and are not mapped.
    """
    frames = [pd.read_csv(path) for path in input_files]
    combined_frame = pd.concat(frames, ignore_index=True)

    required_columns = {"text", *GOEMOTIONS_LABEL_COLUMNS}
    missing_columns = sorted(required_columns - set(combined_frame.columns))
    if missing_columns:
        raise ValueError(
            "Downloaded GoEmotions files are missing expected columns: "
            + ", ".join(missing_columns)
        )

    label_matrix = combined_frame[list(GOEMOTIONS_LABEL_COLUMNS)].fillna(0).astype(int)
    single_label_mask = label_matrix.sum(axis=1) == 1
    filtered_frame = combined_frame.loc[single_label_mask, ["text"]].copy()
    filtered_labels = label_matrix.loc[single_label_mask]

    filtered_frame["label"] = filtered_labels.idxmax(axis=1)
    filtered_frame["text"] = filtered_frame["text"].astype(str).str.strip()
    filtered_frame["label"] = filtered_frame["label"].astype(str).str.strip().str.lower()
    filtered_frame = filtered_frame[filtered_frame["text"].ne("")].drop_duplicates().reset_index(
        drop=True
    )

    if filtered_frame.empty:
        raise ValueError("No single-label rows were found in the downloaded GoEmotions files.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_frame.to_csv(output_path, index=False)
    return filtered_frame


def download_and_prepare_goemotions(output_path: Path) -> pd.DataFrame:
    """
    Download GoEmotions raw files and create a training-ready raw CSV in data/raw/.

    The output CSV contains original labels and is intended to be used next with
    prepare_dataset.py for label mapping into the final 6 classes.
    """
    raw_directory = output_path.parent / "goemotions_source"
    downloaded_files = download_goemotions_raw_files(raw_directory)
    return convert_goemotions_to_text_label_csv(downloaded_files, output_path)
