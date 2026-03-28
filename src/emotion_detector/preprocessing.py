"""Text preprocessing functions for the baseline model."""

from __future__ import annotations

import html
import re
from typing import Any

import pandas as pd


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
NON_ALPHANUMERIC_PATTERN = re.compile(r"[^a-z0-9\s!?']")
MULTISPACE_PATTERN = re.compile(r"\s+")


def is_valid_text_row(text: Any) -> bool:
    """
    Return True when the input can be used as a non-empty text row.

    This is helpful before preprocessing a dataset column.
    """
    if text is None or pd.isna(text):
        return False

    return str(text).strip() != ""


def clean_text(
    text: Any,
    *,
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_mentions: bool = True,
    keep_hashtag_text: bool = True,
) -> str:
    """
    Apply lightweight normalization for English social media text.

    The function keeps punctuation like ! and ? because it can carry emotion.
    It is compatible with sklearn's TfidfVectorizer preprocessor callback.
    """
    if text is None or pd.isna(text):
        return ""

    normalized = html.unescape(str(text)).strip()
    if lowercase:
        normalized = normalized.lower()

    normalized = HTML_TAG_PATTERN.sub(" ", normalized)
    normalized = EMAIL_PATTERN.sub(" ", normalized)

    if remove_urls:
        normalized = URL_PATTERN.sub(" ", normalized)

    if remove_mentions:
        normalized = MENTION_PATTERN.sub(" ", normalized)

    if keep_hashtag_text:
        normalized = HASHTAG_PATTERN.sub(r" \1 ", normalized)
    else:
        normalized = HASHTAG_PATTERN.sub(" ", normalized)

    normalized = NON_ALPHANUMERIC_PATTERN.sub(" ", normalized)
    normalized = MULTISPACE_PATTERN.sub(" ", normalized)
    return normalized.strip()


def preprocess_texts(
    texts: pd.Series,
    *,
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_mentions: bool = True,
    keep_hashtag_text: bool = True,
) -> pd.Series:
    """Clean a pandas Series of text values using the same rules as clean_text."""
    if not isinstance(texts, pd.Series):
        raise TypeError("texts must be a pandas Series.")

    return texts.map(
        lambda value: clean_text(
            value,
            lowercase=lowercase,
            remove_urls=remove_urls,
            remove_mentions=remove_mentions,
            keep_hashtag_text=keep_hashtag_text,
        )
    )


def preprocess_dataframe(
    data_frame: pd.DataFrame,
    text_column: str,
    output_column: str | None = None,
    drop_empty: bool = False,
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_mentions: bool = True,
    keep_hashtag_text: bool = True,
) -> pd.DataFrame:
    """
    Return a copy of the DataFrame with cleaned text.

    If output_column is omitted, the original text column is replaced.
    """
    if not isinstance(data_frame, pd.DataFrame):
        raise TypeError("data_frame must be a pandas DataFrame.")
    if text_column not in data_frame.columns:
        raise ValueError(f"Text column '{text_column}' was not found in the DataFrame.")

    result = data_frame.copy()
    target_column = output_column or text_column
    valid_mask = result[text_column].map(is_valid_text_row)
    result.loc[~valid_mask, target_column] = ""
    result.loc[valid_mask, target_column] = preprocess_texts(
        result.loc[valid_mask, text_column],
        lowercase=lowercase,
        remove_urls=remove_urls,
        remove_mentions=remove_mentions,
        keep_hashtag_text=keep_hashtag_text,
    )

    if drop_empty:
        result = result[result[target_column].ne("")].copy()

    return result
