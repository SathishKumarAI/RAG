"""
Structured logging utilities for the RAG pipeline.

WHAT: Provides structured logging with JSON output and standard fields.
WHY: Structured logs are easier to parse, search, and analyze in production environments.
     Standard fields enable better tracing and debugging.
HOW: Uses structlog for structured logging with JSON formatter. Provides a logger factory
     that creates loggers with standard fields (request_id, user_id, pipeline_stage, etc.).

Usage:
    from utils.logging_utils import get_logger
    
    logger = get_logger(__name__)
    logger.info("Processing document", document_id="doc123", stage="ingestion")
"""

import logging
import sys
from typing import Any, Dict, Optional
import structlog
from structlog.stdlib import LoggerFactory


def setup_logging(
    level: str = "INFO",
    json_output: bool = True,
    include_timestamp: bool = True,
) -> None:
    """
    Set up structured logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: If True, output JSON logs (for production)
        include_timestamp: If True, include timestamps in logs
    """
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
    ]
    
    if include_timestamp:
        processors.append(structlog.processors.TimeStamper(fmt="iso"))
    
    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.extend([
            structlog.dev.ConsoleRenderer(),
        ])
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Set standard logging level
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Structured logger instance
        
    Example:
        logger = get_logger(__name__)
        logger.info("Processing started", request_id="req123", stage="ingestion")
    """
    return structlog.get_logger(name)


def add_context(**kwargs: Any) -> None:
    """
    Add context variables that will be included in all subsequent log messages.
    
    Args:
        **kwargs: Context variables to add
        
    Example:
        add_context(request_id="req123", user_id="user456")
        logger.info("Processing")  # Will include request_id and user_id
    """
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all context variables."""
    structlog.contextvars.clear_contextvars()


# Initialize logging on import (can be overridden)
setup_logging(level="INFO", json_output=False)

