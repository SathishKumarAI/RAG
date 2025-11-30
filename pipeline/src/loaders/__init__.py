"""Document loaders for the multi-format RAG lab.

Each loader converts a concrete file type (PDF, DOCX, CSV, etc.)
into one or more `rag.models.Document` instances.
"""

from .base import DocumentLoader, get_loader_for_path  # noqa: F401


