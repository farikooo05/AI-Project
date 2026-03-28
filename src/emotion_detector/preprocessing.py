"""Text preprocessing functions for the baseline model."""

from __future__ import annotations

import re


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
NON_ALPHANUMERIC_PATTERN = re.compile(r"[^a-z0-9\s!?']")
MULTISPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Apply lightweight normalization for social media text.

    The baseline keeps punctuation like ! and ? because it can carry emotion.
    """
    normalized = text.lower().strip()
    normalized = URL_PATTERN.sub(" ", normalized)
    normalized = MENTION_PATTERN.sub(" ", normalized)
    normalized = HASHTAG_PATTERN.sub(r" \1 ", normalized)
    normalized = NON_ALPHANUMERIC_PATTERN.sub(" ", normalized)
    normalized = MULTISPACE_PATTERN.sub(" ", normalized)
    return normalized.strip()
