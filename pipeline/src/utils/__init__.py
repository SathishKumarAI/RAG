"""
Utility modules for the RAG pipeline.

This package provides reusable utilities for:
- Configuration management
- Logging
- File I/O
- Input validation
- LLM interactions
- Embeddings
- RAG preprocessing
"""

from .config import Config, get_config
from .logging_utils import get_logger, setup_logging
from .io_utils import read_text, write_text, read_json, write_json, safe_write
from .validation import validate_document, validate_chunk, validate_query
from .llm_utils import call_llm, format_prompt, retry_with_backoff
from .embeddings import generate_embeddings, batch_embeddings, normalize_embeddings
from .rag_preprocessing import clean_text, chunk_text, enrich_metadata

__all__ = [
    # Config
    "Config",
    "get_config",
    # Logging
    "get_logger",
    "setup_logging",
    # I/O
    "read_text",
    "write_text",
    "read_json",
    "write_json",
    "safe_write",
    # Validation
    "validate_document",
    "validate_chunk",
    "validate_query",
    # LLM
    "call_llm",
    "format_prompt",
    "retry_with_backoff",
    # Embeddings
    "generate_embeddings",
    "batch_embeddings",
    "normalize_embeddings",
    # Preprocessing
    "clean_text",
    "chunk_text",
    "enrich_metadata",
]

