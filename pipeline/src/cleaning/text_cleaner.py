"""Composable text cleaning pipeline.

This module is intentionally small; for richer logic we can delegate to
`utils.rag_preprocessing.clean_text`, but having this layer makes it
easy to swap in experiment-specific cleaning steps.
"""

from __future__ import annotations

import re
from typing import Callable, List

Transform = Callable[[str], str]


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim the ends."""
    return re.sub(r"\s+", " ", text).strip()


def strip_html_tags(text: str) -> str:
    """Very lightweight HTML tag stripper.

    For more robust behavior we can call into BeautifulSoup via
    `utils.rag_preprocessing.clean_text`.
    """
    return re.sub(r"<[^>]+>", " ", text)


def build_cleaning_pipeline(
    remove_html: bool = True,
    normalize_ws: bool = True,
) -> List[Transform]:
    """Build a list of transforms to apply in order."""
    steps: List[Transform] = []
    if remove_html:
        steps.append(strip_html_tags)
    if normalize_ws:
        steps.append(normalize_whitespace)
    return steps


def clean_text(
    text: str,
    remove_html: bool = True,
    normalize_ws: bool = True,
) -> str:
    """Clean text using a simple transform pipeline."""
    pipeline = build_cleaning_pipeline(
        remove_html=remove_html,
        normalize_ws=normalize_ws,
    )
    for fn in pipeline:
        text = fn(text)
    return text


