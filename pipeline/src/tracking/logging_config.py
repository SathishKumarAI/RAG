"""Basic logging configuration for experiments."""

from __future__ import annotations

import logging
from typing import Optional


def configure_logging(level: int = logging.INFO, name: Optional[str] = None) -> None:
    """Configure root or named logger with a simple, structured-friendly format."""
    logger = logging.getLogger(name)
    if logger.handlers:
        # Avoid duplicate handlers if called multiple times
        return

    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


