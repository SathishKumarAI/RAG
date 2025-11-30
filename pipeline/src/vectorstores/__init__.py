"""Vector store abstractions for the RAG lab.

These thin wrappers delegate to the existing `rag.storage.vector_store`
implementations so we don't duplicate logic.
"""

from .base import VectorStore  # noqa: F401
from .faiss_store import FAISSStore  # noqa: F401


