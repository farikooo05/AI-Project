"""Input/output helper functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def save_json(data: Any, output_path: Path) -> None:
    """Save JSON data with UTF-8 encoding."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def load_json(input_path: Path) -> Any:
    """Load JSON data from disk."""
    with input_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_joblib(data: Any, output_path: Path) -> None:
    """Save a Python object with joblib."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data, output_path)


def load_joblib(input_path: Path) -> Any:
    """Load a Python object stored with joblib."""
    return joblib.load(input_path)
