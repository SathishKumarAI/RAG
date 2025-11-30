"""Tests for validation utility."""

import pytest

from src.utils.validation import (
    validate_document,
    validate_chunk,
    validate_query,
    validate_metadata,
    validate_embedding,
    ValidationError,
)


def test_validate_document_valid():
    """Test validation of valid document."""
    doc = {
        "text": "This is a valid document.",
        "metadata": {"source": "test.txt"},
    }
    validate_document(doc)  # Should not raise


def test_validate_document_missing_text():
    """Test validation fails for missing text."""
    doc = {"metadata": {}}
    with pytest.raises(ValidationError):
        validate_document(doc)


def test_validate_document_empty_text():
    """Test validation fails for empty text."""
    doc = {"text": "   ", "metadata": {}}
    with pytest.raises(ValidationError):
        validate_document(doc)


def test_validate_chunk_valid():
    """Test validation of valid chunk."""
    chunk = {
        "text": "This is a valid chunk.",
        "metadata": {"source": "test.txt"},
    }
    validate_chunk(chunk)  # Should not raise


def test_validate_query_valid():
    """Test validation of valid query."""
    validate_query("What is RAG?")  # Should not raise


def test_validate_query_empty():
    """Test validation fails for empty query."""
    with pytest.raises(ValidationError):
        validate_query("")


def test_validate_embedding_valid():
    """Test validation of valid embedding."""
    embedding = [0.1, 0.2, 0.3, 0.4]
    validate_embedding(embedding)  # Should not raise


def test_validate_embedding_empty():
    """Test validation fails for empty embedding."""
    with pytest.raises(ValidationError):
        validate_embedding([])

