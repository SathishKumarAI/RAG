"""Loader for plain text and markdown-like files."""

from __future__ import annotations

from pathlib import Path
from typing import List

from rag.models import Document

from .base import DocumentLoader, register_loader

try:
    # Prefer the existing robust reader if available
    from utils.io_utils import read_text  # type: ignore
except Exception:  # pragma: no cover - fallback for minimal environments
    def read_text(file_path: Path) -> str:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


@register_loader([".txt", ".md", ".markdown"])
class TextLoader(DocumentLoader):
    """Load plain text / markdown files as a single Document."""

    def load(self, path: Path) -> List[Document]:
        text = read_text(path)
        doc = Document(
            id=str(path),
            path=path,
            text=text,
            metadata={"source": "text", "suffix": path.suffix.lower()},
        )
        return [doc]


