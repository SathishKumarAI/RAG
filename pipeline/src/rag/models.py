"""Core data models for the RAG experimentation platform.

These are intentionally lightweight so they can be reused across
loaders, chunkers, vector stores, and pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class Document:
    """Canonical in-memory representation of a source document.

    Attributes:
        id: Stable identifier for the document (often a path or UUID).
        path: Original filesystem path, if applicable.
        text: Raw or cleaned text content for the whole document.
        metadata: Arbitrary metadata such as source type, page count,
            mime type, timestamps, etc.
    """

    id: str
    path: Path
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A semantically meaningful slice of a Document.

    Attributes:
        id: Stable identifier for the chunk (e.g. "{document_id}::chunk_{i}").
        document_id: ID of the parent document.
        text: Chunk text content.
        metadata: Chunk-level metadata such as page number, section heading,
            chunking strategy, character offsets, etc.
    """

    id: str
    document_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


