"""Local (SentenceTransformers) Embedder implementation."""

from __future__ import annotations

from typing import List

from utils.embeddings import (
    _batch_sentence_transformer_embeddings,  # type: ignore[attr-defined]
    _generate_sentence_transformer_embeddings,  # type: ignore[attr-defined]
)

from .base import Embedder


class LocalSentenceTransformerEmbedder(Embedder):
    """Embedder that uses a local SentenceTransformers model.

    Note: this is a light wrapper around the internal utilities in
    `utils.embeddings` and will raise ImportError if sentence-transformers
    is not installed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return _batch_sentence_transformer_embeddings(
            texts,
            model=self.model_name,
            batch_size=32,
        )

    def embed_query(self, text: str) -> List[float]:
        return _generate_sentence_transformer_embeddings(
            text,
            model=self.model_name,
        )


