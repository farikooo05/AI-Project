"""Helpers for downloading public emotion datasets into data/raw/."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


GOEMOTIONS_PART_URLS = (
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


def download_file(url: str, destination_path: Path) -> Path:
    """Download a file to disk and return its saved path."""
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, destination_path)
    return destination_path


def load_goemotions_parts(part_paths: list[Path]) -> pd.DataFrame:
    """Load and combine the official GoEmotions raw CSV parts."""
    frames: list[pd.DataFrame] = []
    for part_path in part_paths:
        if not part_path.exists():
            raise FileNotFoundError(f"GoEmotions part not found at '{part_path}'.")

        frame = pd.read_csv(part_path)
        if frame.empty:
            raise ValueError(f"Downloaded GoEmotions file is empty: '{part_path}'.")
        frames.append(frame)

    combined_frame = pd.concat(frames, ignore_index=True)
    if "text" not in combined_frame.columns:
        raise ValueError("GoEmotions raw files do not contain the expected 'text' column.")

    missing_label_columns = [
        column for column in GOEMOTIONS_LABEL_COLUMNS if column not in combined_frame.columns
    ]
    if missing_label_columns:
        raise ValueError(
            "GoEmotions raw files are missing expected label columns: "
            + ", ".join(missing_label_columns)
        )

    return combined_frame


def convert_goemotions_to_label_rows(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the wide GoEmotions raw format into a simple text,label CSV.

    A comment with multiple emotion labels becomes multiple rows.
    """
    records: list[dict[str, str]] = []
    for _, row in data_frame.iterrows():
        text = str(row["text"]).strip()
        if not text:
            continue

        for label in GOEMOTIONS_LABEL_COLUMNS:
            if int(row[label]) == 1:
                records.append({"text": text, "label": label})

    result = pd.DataFrame(records, columns=["text", "label"])
    if result.empty:
        raise ValueError("No labeled rows were produced from the GoEmotions dataset.")

    return result


def download_goemotions_dataset(raw_data_dir: Path) -> tuple[list[Path], Path]:
    """
    Download GoEmotions raw files and create a simple text,label CSV in data/raw.

    Returns the downloaded part paths and the converted CSV path.
    """
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    part_paths: list[Path] = []
    for index, url in enumerate(GOEMOTIONS_PART_URLS, start=1):
        part_path = raw_data_dir / f"goemotions_part_{index}.csv"
        download_file(url, part_path)
        part_paths.append(part_path)

    combined_frame = load_goemotions_parts(part_paths)
    simple_frame = convert_goemotions_to_label_rows(combined_frame)

    output_path = raw_data_dir / "goemotions_raw.csv"
    simple_frame.to_csv(output_path, index=False)
    return part_paths, output_path
