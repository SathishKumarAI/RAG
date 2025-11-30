"""Fixed-window character-based chunker."""

from __future__ import annotations

from typing import List

from rag.models import Chunk, Document

from .base import Chunker


class FixedWindowChunker(Chunker):
    """Simple fixed-size window chunker with configurable overlap.

    This mirrors the behavior of `utils.rag_preprocessing._chunk_fixed_size`
    but returns rich `Chunk` objects instead of raw strings.
    """

    def __init__(self, size: int = 500, overlap: int = 50) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        if overlap < 0 or overlap >= size:
            raise ValueError("overlap must be in [0, size)")
        self.size = size
        self.overlap = overlap

    def chunk(self, document: Document) -> List[Chunk]:
        text = document.text
        chunks: List[Chunk] = []

        if not text:
            return chunks

        start = 0
        idx = 0
        length = len(text)

        while start < length:
            end = min(start + self.size, length)
            chunk_text = text[start:end]
            chunk = Chunk(
                id=f"{document.id}::chunk_{idx}",
                document_id=document.id,
                text=chunk_text,
                metadata={
                    "strategy": "fixed_window",
                    "start": start,
                    "end": end,
                    "overlap": self.overlap,
                },
            )
            chunks.append(chunk)
            idx += 1
            if end == length:
                break
            start = end - self.overlap

        return chunks


