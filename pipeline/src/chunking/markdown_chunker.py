"""Markdown / heading-based chunker."""

from __future__ import annotations

import re
from typing import List

from rag.models import Chunk, Document

from .base import Chunker


class MarkdownHeadingChunker(Chunker):
    """Chunk markdown text by top-level headings.

    This is a simple heuristic: it treats lines starting with one or more
    '#' characters as section boundaries and groups subsequent lines
    until the next heading.
    """

    HEADING_RE = re.compile(r"^(#+)\s+(.*)")

    def chunk(self, document: Document) -> List[Chunk]:
        lines = document.text.splitlines()
        chunks: List[Chunk] = []

        current_lines: List[str] = []
        current_heading: str | None = None
        idx = 0

        def flush() -> None:
            nonlocal idx, current_lines, current_heading
            if not current_lines:
                return
            text = "\n".join(current_lines).strip()
            if not text:
                current_lines = []
                return
            chunk = Chunk(
                id=f"{document.id}::md_{idx}",
                document_id=document.id,
                text=text,
                metadata={
                    "strategy": "markdown_heading",
                    "heading": current_heading,
                },
            )
            chunks.append(chunk)
            idx += 1
            current_lines = []

        for line in lines:
            m = self.HEADING_RE.match(line)
            if m:
                # Start a new section
                flush()
                current_heading = m.group(2).strip()
                current_lines.append(line)
            else:
                current_lines.append(line)

        flush()
        return chunks


