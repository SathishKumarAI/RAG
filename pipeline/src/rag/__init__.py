"""
RAG (Retrieval-Augmented Generation) pipeline modules.

This package provides a complete RAG implementation with:
- Index building from raw documents
- Vector and keyword retrieval
- Re-ranking
- Prompt building
- End-to-end inference
- Evaluation metrics
"""

from .pipelines import build_index, run_rag_query
from .index_builder import IndexBuilder
from .retriever import Retriever
from .re_ranker import ReRanker
from .prompt_builder import PromptBuilder
from .inference import RAGInference
from .evaluation import RAGEvaluator

__all__ = [
    "build_index",
    "run_rag_query",
    "IndexBuilder",
    "Retriever",
    "ReRanker",
    "PromptBuilder",
    "RAGInference",
    "RAGEvaluator",
]

