"""Tests for prompt builder module."""

import pytest

from src.rag.prompt_builder import PromptBuilder


def test_prompt_builder_standard():
    """Test standard prompt building."""
    builder = PromptBuilder(style="standard")
    context_docs = [
        {
            "text": "RAG is Retrieval-Augmented Generation",
            "metadata": {"source": "doc1.txt"},
        }
    ]
    prompt = builder.build("What is RAG?", context_docs)
    
    assert "What is RAG?" in prompt
    assert "RAG is Retrieval-Augmented Generation" in prompt


def test_prompt_builder_concise():
    """Test concise prompt building."""
    builder = PromptBuilder(style="concise")
    context_docs = [{"text": "Test context", "metadata": {}}]
    prompt = builder.build("Test query", context_docs)
    
    assert "Test query" in prompt
    assert "Test context" in prompt


def test_prompt_builder_include_sources():
    """Test prompt building with sources."""
    builder = PromptBuilder(include_sources=True)
    context_docs = [
        {
            "text": "Test text",
            "metadata": {"source": "test.txt"},
        }
    ]
    prompt = builder.build("Test query", context_docs)
    
    assert "Source: test.txt" in prompt or "[Source: test.txt]" in prompt

