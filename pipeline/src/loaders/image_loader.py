"""Loader for image files via OCR.

This is a thin abstraction so we can plug in pytesseract or any other
OCR backend later without changing the rest of the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from rag.models import Document

from .base import DocumentLoader, register_loader

try:
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None
    Image = None


@register_loader([".png", ".jpg", ".jpeg"])
class ImageLoader(DocumentLoader):
    """Load image files and extract text via OCR."""

    def load(self, path: Path) -> List[Document]:
        if pytesseract is None or Image is None:
            raise ImportError(
                "pytesseract and pillow are required for ImageLoader. "
                "Install them with `pip install pytesseract pillow`."
            )

        image = Image.open(path)
        text = pytesseract.image_to_string(image)

        doc = Document(
            id=str(path),
            path=path,
            text=text,
            metadata={"source": "image", "suffix": path.suffix.lower()},
        )
        return [doc]


