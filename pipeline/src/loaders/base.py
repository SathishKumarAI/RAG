"""Base interfaces and helpers for document loaders.

Loaders are responsible for reading raw files from disk and producing
`rag.models.Document` objects with reasonable default metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Protocol, Type

from rag.models import Document


class DocumentLoader(Protocol):
    """Abstract interface for loading one or more Documents from a path."""

    def load(self, path: Path) -> List[Document]:
        """Load one or more documents from the given path."""
        ...


_LOADER_REGISTRY: Dict[str, Type[DocumentLoader]] = {}


def register_loader(suffixes: List[str]):
    """Class decorator to register a loader implementation by file suffix.

    Example:
        @register_loader([".txt", ".md"])
        class TextLoader(DocumentLoader):
            ...
    """

    def decorator(cls: Type[DocumentLoader]) -> Type[DocumentLoader]:
        for suffix in suffixes:
            _LOADER_REGISTRY[suffix.lower()] = cls
        return cls

    return decorator


def get_loader_for_path(path: Path) -> DocumentLoader:
    """Return a loader instance based on file suffix.

    If no specific loader is registered for the suffix, this raises a
    KeyError so callers can decide how to handle unsupported types.
    """
    suffix = path.suffix.lower()
    if suffix not in _LOADER_REGISTRY:
        raise KeyError(f"No loader registered for suffix: {suffix}")
    loader_cls = _LOADER_REGISTRY[suffix]
    return loader_cls()  # type: ignore[call-arg]


