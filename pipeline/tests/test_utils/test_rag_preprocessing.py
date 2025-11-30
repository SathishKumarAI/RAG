"""Tests for RAG preprocessing utility."""

import pytest

from src.utils.rag_preprocessing import (
    clean_text,
    chunk_text,
    enrich_metadata,
)


def test_clean_text_remove_html():
    """Test HTML removal."""
    text = "<html><body>Hello <b>world</b></body></html>"
    cleaned = clean_text(text, remove_html=True)
    assert "<html>" not in cleaned
    assert "Hello" in cleaned
    assert "world" in cleaned


def test_clean_text_normalize_whitespace():
    """Test whitespace normalization."""
    text = "Hello    world\n\n\nTest"
    cleaned = clean_text(text, normalize_whitespace=True)
    assert "  " not in cleaned
    assert "\n\n" not in cleaned


def test_chunk_text_fixed_size():
    """Test fixed-size chunking."""
    text = "A" * 1000
    chunks = chunk_text(text, strategy="fixed_size", chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_chunk_text_recursive():
    """Test recursive chunking."""
    text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
    chunks = chunk_text(text, strategy="recursive", chunk_size=50, chunk_overlap=10)
    assert len(chunks) >= 1


def test_enrich_metadata():
    """Test metadata enrichment."""
    chunk = {"text": "test", "metadata": {}}
    enriched = enrich_metadata(
        chunk,
        source="test.txt",
        page=1,
        section="Introduction",
    )
    assert enriched["metadata"]["source"] == "test.txt"
    assert enriched["metadata"]["page"] == 1
    assert enriched["metadata"]["section"] == "Introduction"
    assert "timestamp" in enriched["metadata"]

