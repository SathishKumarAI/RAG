"""Loader for CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import List

from rag.models import Document

from .base import DocumentLoader, register_loader

try:
    from utils.io_utils import read_csv  # type: ignore
except Exception:  # pragma: no cover - fallback
    import csv

    def read_csv(file_path: Path):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            return list(reader)


@register_loader([".csv"])
class CSVLoader(DocumentLoader):
    """Load CSV files and represent them as table-like text."""

    def load(self, path: Path) -> List[Document]:
        rows = read_csv(path)

        if not rows:
            text = ""
            headers: List[str] = []
        else:
            headers = list(rows[0].keys())
            lines = [", ".join(headers)]
            for row in rows:
                lines.append(", ".join(str(row.get(h, "")) for h in headers))
            text = "\n".join(lines)

        doc = Document(
            id=str(path),
            path=path,
            text=text,
            metadata={
                "source": "csv",
                "suffix": ".csv",
                "num_rows": len(rows),
                "columns": headers,
            },
        )
        return [doc]


