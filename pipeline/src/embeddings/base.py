"""Abstract interfaces for embedding backends."""

from __future__ import annotations

from typing import List, Protocol


class Embedder(Protocol):
    """Abstract embedding interface."""

    model_name: str

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of document texts."""
        ...

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        ...


