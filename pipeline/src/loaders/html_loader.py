"""Loader for HTML files."""

from __future__ import annotations

from pathlib import Path
from typing import List

from rag.models import Document

from .base import DocumentLoader, register_loader

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    BeautifulSoup = None


@register_loader([".html", ".htm"])
class HTMLLoader(DocumentLoader):
    """Load HTML files and convert them into cleaned text."""

    def load(self, path: Path) -> List[Document]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()

        if BeautifulSoup is not None:
            soup = BeautifulSoup(html_content, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
        else:
            # Fallback: very simple tag stripping; real cleaning happens later
            import re

            text = re.sub(r"<[^>]+>", " ", html_content)

        doc = Document(
            id=str(path),
            path=path,
            text=text,
            metadata={"source": "html", "suffix": path.suffix.lower()},
        )
        return [doc]


