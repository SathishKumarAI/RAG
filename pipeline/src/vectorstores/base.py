"""Abstract vector store interface for the experimentation layer.

For now this simply re-exports the existing `rag.storage.vector_store.VectorStore`
so higher-level code can depend on a stable abstraction without caring
about the concrete backend.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

from rag.storage.vector_store import VectorStore as _CoreVectorStore


class VectorStore(Protocol):
    """Abstract vector store interface."""

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:  # pragma: no cover - interface
        ...

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:  # pragma: no cover - interface
        ...


def as_core_store(store: VectorStore) -> _CoreVectorStore:
    """Helper to treat a local `VectorStore` as the core implementation."""
    assert isinstance(store, _CoreVectorStore)  # runtime safeguard
    return store


