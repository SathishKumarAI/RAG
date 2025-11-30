# RAG Pipeline Overview

## Introduction

This RAG (Retrieval-Augmented Generation) pipeline is a production-ready system for building question-answering and knowledge retrieval applications. It combines the power of large language models with semantic search to provide accurate, context-aware answers.

## Main Use Cases

1. **Document Q&A**: Answer questions based on a collection of documents
2. **Knowledge Base Search**: Search and retrieve relevant information from a knowledge base
3. **Contextual Chatbots**: Build chatbots that can reference specific documents
4. **Research Assistance**: Help researchers find and synthesize information from large document collections

## Key Features

- **Modular Architecture**: Clean separation of concerns with reusable components
- **Multiple Vector Stores**: Support for Pinecone, FAISS, Chroma, and in-memory stores
- **Flexible Chunking**: Multiple chunking strategies (fixed-size, recursive, sentence-based)
- **Re-ranking Support**: Optional re-ranking for improved relevance
- **Comprehensive Evaluation**: Built-in metrics for measuring system performance
- **Production Ready**: Structured logging, error handling, and configuration management

## RAG Pipeline Introduction

The RAG pipeline consists of two main phases:

### 1. Indexing Phase
- **Load Documents**: Read documents from various sources (files, S3, web, etc.)
- **Preprocess**: Clean and normalize text
- **Chunk**: Split documents into smaller, manageable pieces
- **Embed**: Generate vector embeddings for each chunk
- **Store**: Save embeddings and metadata in a vector store

### 2. Query Phase
- **Query**: Receive user question
- **Retrieve**: Find relevant document chunks using vector similarity
- **Re-rank** (optional): Improve ranking using cross-encoder models
- **Generate**: Build prompt with context and generate answer using LLM
- **Return**: Provide answer with source citations

## Quick Start

```python
from rag.pipelines import build_index, run_rag_query

# Build index
index = build_index(
    documents=["doc1.txt", "doc2.pdf"],
    output_path="./my_index"
)

# Query
result = run_rag_query(
    query="What is RAG?",
    index_path="./my_index"
)

print(result["answer"])
```

## Architecture

See [architecture.md](./architecture.md) for detailed component architecture.

## Getting Started

See [implementation_guide.md](./implementation_guide.md) for step-by-step implementation details.

