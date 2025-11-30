"""OpenAI-based Embedder implementation.

This is a thin wrapper around `utils.embeddings` so experiments can
depend on a stable `Embedder` protocol rather than the raw functions.
"""

from __future__ import annotations

from typing import List, Optional

from utils.embeddings import batch_embeddings, generate_embeddings

from .base import Embedder


class OpenAIEmbedder(Embedder):
    """Embedder that uses OpenAI embedding models."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self._api_key = api_key

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return batch_embeddings(
            texts,
            model=self.model_name,
            provider="openai",
            api_key=self._api_key,
        )

    def embed_query(self, text: str) -> List[float]:
        return generate_embeddings(
            text,
            model=self.model_name,
            provider="openai",
            api_key=self._api_key,
        )


