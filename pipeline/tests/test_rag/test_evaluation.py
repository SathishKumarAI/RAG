"""Tests for evaluation module."""

import pytest

from src.rag.evaluation import RAGEvaluator


def test_evaluator_basic():
    """Test basic evaluation."""
    evaluator = RAGEvaluator()
    
    queries = ["What is RAG?"]
    ground_truth = ["RAG is Retrieval-Augmented Generation"]
    predictions = ["RAG stands for Retrieval-Augmented Generation"]
    
    metrics = evaluator.evaluate(queries, ground_truth, predictions)
    
    assert "answer_quality" in metrics
    assert "exact_match_rate" in metrics["answer_quality"]
    assert isinstance(metrics["answer_quality"]["exact_match_rate"], float)


def test_evaluator_retrieval_metrics():
    """Test retrieval metrics."""
    evaluator = RAGEvaluator()
    
    queries = ["test"]
    ground_truth = ["answer"]
    predictions = ["answer"]
    retrieved_docs_list = [
        [
            {"text": "doc1", "id": "1", "metadata": {}},
            {"text": "doc2", "id": "2", "metadata": {}},
        ]
    ]
    relevant_docs_list = [["1"]]
    
    metrics = evaluator.evaluate(
        queries, ground_truth, predictions, retrieved_docs_list, relevant_docs_list
    )
    
    assert "retrieval_quality" in metrics
    assert "hit_rate" in metrics["retrieval_quality"]


def test_evaluator_latency_stats():
    """Test latency statistics computation."""
    evaluator = RAGEvaluator()
    latencies = [0.1, 0.2, 0.3, 0.4, 0.5]
    stats = evaluator.compute_latency_stats(latencies)
    
    assert "mean" in stats
    assert "median" in stats
    assert stats["mean"] == 0.3
    assert stats["median"] == 0.3

