"""Tests for vector store usage in the lab pipeline."""

from rag.storage.vector_store import InMemoryVectorStore


def test_in_memory_vector_store_roundtrip():
    store = InMemoryVectorStore()
    docs = [
        {"text": "hello world", "embedding": [1.0, 0.0], "metadata": {"id": 1}},
        {"text": "goodbye world", "embedding": [0.0, 1.0], "metadata": {"id": 2}},
    ]
    store.add_documents(docs)

    results = store.search(query_embedding=[1.0, 0.0], top_k=1)

    assert len(results) == 1
    assert "hello" in results[0]["text"]


