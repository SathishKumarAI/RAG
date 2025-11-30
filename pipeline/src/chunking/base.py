"""Base interfaces for chunking strategies."""

from __future__ import annotations

from typing import List, Protocol

from rag.models import Chunk, Document


class Chunker(Protocol):
    """Abstract chunker interface."""

    def chunk(self, document: Document) -> List[Chunk]:
        """Split a Document into a list of Chunks."""
        ...


