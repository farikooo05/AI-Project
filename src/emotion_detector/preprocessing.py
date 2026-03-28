"""Text preprocessing functions for the baseline model."""

from __future__ import annotations

import html
import re

import pandas as pd


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
NON_ALPHANUMERIC_PATTERN = re.compile(r"[^a-z0-9\s!?']")
REPEATED_PUNCTUATION_PATTERN = re.compile(r"([!?]){2,}")
ELONGATED_WORD_PATTERN = re.compile(r"(.)\1{2,}")
MULTISPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: object) -> str:
    """
    Apply lightweight normalization for social media text.

    The baseline keeps punctuation like ! and ? because it can carry emotion.
    """
    if text is None:
        return ""

    normalized = html.unescape(str(text)).lower().strip()
    normalized = HTML_TAG_PATTERN.sub(" ", normalized)
    normalized = URL_PATTERN.sub(" ", normalized)
    normalized = EMAIL_PATTERN.sub(" ", normalized)
    normalized = MENTION_PATTERN.sub(" ", normalized)
    normalized = HASHTAG_PATTERN.sub(r" \1 ", normalized)
    normalized = REPEATED_PUNCTUATION_PATTERN.sub(r"\1", normalized)
    normalized = ELONGATED_WORD_PATTERN.sub(r"\1\1", normalized)
    normalized = NON_ALPHANUMERIC_PATTERN.sub(" ", normalized)
    normalized = MULTISPACE_PATTERN.sub(" ", normalized)
    return normalized.strip()


def preprocess_texts(texts: pd.Series) -> pd.Series:
    """Clean a pandas Series of text values."""
    return texts.fillna("").map(clean_text)


def preprocess_dataframe(
    data_frame: pd.DataFrame,
    text_column: str,
    output_column: str | None = None,
    drop_empty: bool = False,
) -> pd.DataFrame:
    """
    Return a copy of the DataFrame with cleaned text.

    If output_column is omitted, the original text column is replaced.
    """
    if text_column not in data_frame.columns:
        raise ValueError(f"Text column '{text_column}' was not found in the DataFrame.")

    result = data_frame.copy()
    target_column = output_column or text_column
    result[target_column] = preprocess_texts(result[text_column])

    if drop_empty:
        result = result[result[target_column].ne("")].copy()

    return result
