"""High-level RAG lab pipeline for experimentation.

This module ties together:
    - loaders (multi-format Document ingestion)
    - cleaning
    - chunking
    - embeddings
    - a simple vector store

It is intentionally minimal and avoids hard dependencies on remote
services so tests can run without API keys. In notebooks you can plug
in real OpenAI / local models instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from rag.models import Chunk, Document
from rag.storage.vector_store import InMemoryVectorStore
from utils.io_utils import list_files

from cleaning import clean_text as default_clean_text
from chunking.fixed_window import FixedWindowChunker
from chunking.base import Chunker
from embeddings.base import Embedder
from embeddings.openai_embeddings import OpenAIEmbedder
from loaders.base import get_loader_for_path
from tracking.logging_config import configure_logging


@dataclass
class RAGLabConfig:
    """Small config object for the RAG lab pipeline."""

    data_root: Path
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5


class SimpleRAGPipeline:
    """Minimal, dependency-light RAG pipeline for local experiments.

    This class does *not* call an LLM by default; it focuses on
    retrieval and exposes the retrieved chunks so you can inspect them
    or build prompts yourself (or via existing `rag.prompt_builder`).
    """

    def __init__(
        self,
        config: RAGLabConfig,
        embedder: Optional[Embedder] = None,
        chunker: Optional[Chunker] = None,
    ) -> None:
        configure_logging()
        self.config = config
        self.embedder: Embedder = embedder or OpenAIEmbedder()
        self.chunker: Chunker = chunker or FixedWindowChunker(
            size=config.chunk_size,
            overlap=config.chunk_overlap,
        )
        self.store = InMemoryVectorStore()
        self._indexed = False

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def index_directory(self, exts: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """Scan the data_root, load documents, chunk, embed, and index.

        Args:
            exts: Optional list of suffixes (e.g. [".pdf", ".txt"]). If
                omitted, all files are considered and dispatch is
                entirely handled by the loader registry.
        """
        root = self.config.data_root
        if exts:
            files: List[Path] = []
            for ext in exts:
                files.extend(list_files(root, pattern=f"*{ext}", recursive=True))
        else:
            files = list_files(root, recursive=True)

        documents: List[Document] = []
        for path in files:
            try:
                loader = get_loader_for_path(path)
            except KeyError:
                # Skip unsupported types
                continue
            docs = loader.load(path)
            documents.extend(docs)

        chunks = self._chunk_documents(documents)
        embeddings = self.embedder.embed_documents([c.text for c in chunks])

        payloads: List[Dict[str, Any]] = []
        for chunk, embedding in zip(chunks, embeddings):
            payloads.append(
                {
                    "text": chunk.text,
                    "embedding": embedding,
                    "metadata": chunk.metadata,
                }
            )

        self.store.add_documents(payloads)
        self._indexed = True

        return {
            "num_files": len(files),
            "num_documents": len(documents),
            "num_chunks": len(chunks),
        }

    def _chunk_documents(self, documents: Iterable[Document]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for doc in documents:
            cleaned = default_clean_text(doc.text)
            clean_doc = Document(
                id=doc.id,
                path=doc.path,
                text=cleaned,
                metadata=doc.metadata,
            )
            chunks.extend(self.chunker.chunk(clean_doc))
        return chunks

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def run_query(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Embed the query, retrieve top-k chunks, and build a simple prompt.

        Returns:
            dict with:
                - query
                - retrieved: list of {text, metadata, score}
                - prompt: concatenated context + question
        """
        if not self._indexed:
            raise RuntimeError("Pipeline has no index; call index_directory() first.")

        k = top_k or self.config.top_k
        query_embedding = self.embedder.embed_query(query)
        retrieved = self.store.search(query_embedding=query_embedding, top_k=k)

        context = "\n\n---\n\n".join(r["text"] for r in retrieved)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

        return {
            "query": query,
            "retrieved": retrieved,
            "prompt": prompt,
        }


