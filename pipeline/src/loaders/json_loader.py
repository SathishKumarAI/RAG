"""Loader for JSON files with record-style content."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from rag.models import Document

from .base import DocumentLoader, register_loader


def _iter_records(obj: Any) -> Iterable[Dict[str, Any]]:
    """Yield record-like dicts from arbitrary JSON."""
    if isinstance(obj, dict):
        yield obj
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item


def _record_to_text(record: Dict[str, Any]) -> str:
    """Convert a JSON record into a simple human-readable text block."""
    lines = []
    for key, value in record.items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


@register_loader([".json"])
class JSONLoader(DocumentLoader):
    """Load JSON files and flatten record-like structures to text."""

    def load(self, path: Path) -> List[Document]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)

        records = list(_iter_records(data))

        if not records:
            text = json.dumps(data, indent=2, ensure_ascii=False)
        else:
            blocks = [_record_to_text(r) for r in records]
            text = "\n\n---\n\n".join(blocks)

        doc = Document(
            id=str(path),
            path=path,
            text=text,
            metadata={
                "source": "json",
                "suffix": ".json",
                "num_records": len(records) if records else 1,
            },
        )
        return [doc]


