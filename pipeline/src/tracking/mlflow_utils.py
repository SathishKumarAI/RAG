"""Lightweight MLflow helpers for RAG experiments.

These are intentionally minimal; real experiments can extend them to
log richer metrics, artifacts, and visualizations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

try:
    import mlflow  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    mlflow = None


def start_run_with_config(config: Dict[str, Any], run_name: Optional[str] = None):
    """Start an MLflow run and log the provided config as params.

    If MLflow is not installed, this becomes a no-op and returns None.
    """
    if mlflow is None:
        return None

    run = mlflow.start_run(run_name=run_name)
    flat_config = _flatten_dict(config)
    for key, value in flat_config.items():
        mlflow.log_param(key, value)
    return run


def log_basic_stats(num_documents: int, num_chunks: int) -> None:
    """Log a few basic statistics about an experiment."""
    if mlflow is None:
        return
    mlflow.log_metric("num_documents", num_documents)
    mlflow.log_metric("num_chunks", num_chunks)


def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten a nested dict so it can be logged as MLflow params."""
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


