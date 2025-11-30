"""
Input validation utilities for the RAG pipeline.

WHAT: Validates inputs for documents, chunks, queries, and metadata.
WHY: Prevents invalid data from propagating through the pipeline, catching errors early.
HOW: Provides schema validation functions that check required fields, types, and constraints.

Usage:
    from utils.validation import validate_document, validate_chunk, validate_query
    
    doc = {"text": "content", "metadata": {"source": "file.pdf"}}
    validate_document(doc)  # Raises ValueError if invalid
    
    chunk = {"text": "chunk text", "metadata": {}}
    validate_chunk(chunk)
    
    query = "What is RAG?"
    validate_query(query)
"""

from typing import Any, Dict, List, Optional


class ValidationError(ValueError):
    """Custom exception for validation errors."""
    pass


def validate_document(document: Dict[str, Any]) -> None:
    """
    Validate a document dictionary.
    
    Expected structure:
    {
        "text": str,  # Required
        "metadata": dict,  # Required
        "id": str,  # Optional
    }
    
    Args:
        document: Document dictionary
        
    Raises:
        ValidationError: If document is invalid
    """
    if not isinstance(document, dict):
        raise ValidationError("Document must be a dictionary")
    
    # Check required fields
    if "text" not in document:
        raise ValidationError("Document must have 'text' field")
    
    if not isinstance(document["text"], str):
        raise ValidationError("Document 'text' must be a string")
    
    if not document["text"].strip():
        raise ValidationError("Document 'text' cannot be empty")
    
    if "metadata" not in document:
        raise ValidationError("Document must have 'metadata' field")
    
    if not isinstance(document["metadata"], dict):
        raise ValidationError("Document 'metadata' must be a dictionary")
    
    # Validate metadata
    validate_metadata(document["metadata"])


def validate_chunk(chunk: Dict[str, Any]) -> None:
    """
    Validate a chunk dictionary.
    
    Expected structure:
    {
        "text": str,  # Required
        "metadata": dict,  # Required
        "id": str,  # Optional
        "embedding": list,  # Optional
    }
    
    Args:
        chunk: Chunk dictionary
        
    Raises:
        ValidationError: If chunk is invalid
    """
    if not isinstance(chunk, dict):
        raise ValidationError("Chunk must be a dictionary")
    
    # Check required fields
    if "text" not in chunk:
        raise ValidationError("Chunk must have 'text' field")
    
    if not isinstance(chunk["text"], str):
        raise ValidationError("Chunk 'text' must be a string")
    
    if not chunk["text"].strip():
        raise ValidationError("Chunk 'text' cannot be empty")
    
    if "metadata" not in chunk:
        raise ValidationError("Chunk must have 'metadata' field")
    
    if not isinstance(chunk["metadata"], dict):
        raise ValidationError("Chunk 'metadata' must be a dictionary")
    
    # Validate metadata
    validate_metadata(chunk["metadata"])
    
    # Validate embedding if present
    if "embedding" in chunk:
        validate_embedding(chunk["embedding"])


def validate_query(query: Any) -> None:
    """
    Validate a query string.
    
    Args:
        query: Query string
        
    Raises:
        ValidationError: If query is invalid
    """
    if not isinstance(query, str):
        raise ValidationError("Query must be a string")
    
    if not query.strip():
        raise ValidationError("Query cannot be empty")
    
    # Check reasonable length (prevent extremely long queries)
    if len(query) > 10000:
        raise ValidationError("Query is too long (max 10000 characters)")


def validate_metadata(metadata: Dict[str, Any]) -> None:
    """
    Validate metadata dictionary.
    
    Args:
        metadata: Metadata dictionary
        
    Raises:
        ValidationError: If metadata is invalid
    """
    if not isinstance(metadata, dict):
        raise ValidationError("Metadata must be a dictionary")
    
    # Check for required metadata fields (if any)
    # For now, we just ensure it's a dict and values are serializable
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise ValidationError(f"Metadata key must be a string: {key}")
        
        # Check that value is a basic type (str, int, float, bool, list, dict, None)
        if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
            raise ValidationError(
                f"Metadata value must be a basic type: {key}={type(value)}"
            )


def validate_embedding(embedding: Any) -> None:
    """
    Validate an embedding vector.
    
    Args:
        embedding: Embedding vector (list of floats)
        
    Raises:
        ValidationError: If embedding is invalid
    """
    if not isinstance(embedding, list):
        raise ValidationError("Embedding must be a list")
    
    if len(embedding) == 0:
        raise ValidationError("Embedding cannot be empty")
    
    # Check all elements are numbers
    for i, val in enumerate(embedding):
        if not isinstance(val, (int, float)):
            raise ValidationError(f"Embedding element {i} must be a number: {val}")
    
    # Check reasonable dimension (prevent extremely large embeddings)
    if len(embedding) > 10000:
        raise ValidationError(f"Embedding dimension too large: {len(embedding)}")


def validate_batch_documents(documents: List[Dict[str, Any]]) -> None:
    """
    Validate a batch of documents.
    
    Args:
        documents: List of document dictionaries
        
    Raises:
        ValidationError: If any document is invalid
    """
    if not isinstance(documents, list):
        raise ValidationError("Documents must be a list")
    
    if len(documents) == 0:
        raise ValidationError("Documents list cannot be empty")
    
    for i, doc in enumerate(documents):
        try:
            validate_document(doc)
        except ValidationError as e:
            raise ValidationError(f"Document {i} is invalid: {e}")


def validate_batch_chunks(chunks: List[Dict[str, Any]]) -> None:
    """
    Validate a batch of chunks.
    
    Args:
        chunks: List of chunk dictionaries
        
    Raises:
        ValidationError: If any chunk is invalid
    """
    if not isinstance(chunks, list):
        raise ValidationError("Chunks must be a list")
    
    if len(chunks) == 0:
        raise ValidationError("Chunks list cannot be empty")
    
    for i, chunk in enumerate(chunks):
        try:
            validate_chunk(chunk)
        except ValidationError as e:
            raise ValidationError(f"Chunk {i} is invalid: {e}")

