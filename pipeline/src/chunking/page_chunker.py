"""Page-aware chunker.

For now this is a very small helper that expects documents coming from
`PDFLoader` to contain `[PAGE N]` markers and splits on those.
"""

from __future__ import annotations

from typing import List

from rag.models import Chunk, Document

from .base import Chunker


class PageMarkerChunker(Chunker):
    """Chunk documents on `[PAGE N]` markers inserted by `PDFLoader`."""

    def chunk(self, document: Document) -> List[Chunk]:
        marker = "[PAGE "
        text = document.text
        if marker not in text:
            # Fallback: single chunk
            return [
                Chunk(
                    id=f"{document.id}::page_0",
                    document_id=document.id,
                    text=text,
                    metadata={"strategy": "page_marker"},
                )
            ]

        parts = text.split(marker)
        chunks: List[Chunk] = []

        # The first part is everything before the first marker
        if parts[0].strip():
            chunks.append(
                Chunk(
                    id=f"{document.id}::page_0",
                    document_id=document.id,
                    text=parts[0].strip(),
                    metadata={"strategy": "page_marker", "page": 0},
                )
            )

        for idx, part in enumerate(parts[1:], start=1):
            # Reattach the marker for readability
            chunk_text = f"{marker}{part}".strip()
            if not chunk_text:
                continue
            chunks.append(
                Chunk(
                    id=f"{document.id}::page_{idx}",
                    document_id=document.id,
                    text=chunk_text,
                    metadata={"strategy": "page_marker", "page": idx},
                )
            )

        return chunks


