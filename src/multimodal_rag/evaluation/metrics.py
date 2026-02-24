"""Evaluation metrics for RAG system."""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class RAGMetrics:
    """
    RAG evaluation metrics (dataclass per coding standards).
    
    Tracks retrieval and generation quality.
    """
    # Retrieval metrics
    retrieval_precision: float
    retrieval_recall: float
    retrieval_f1: float
    mrr: float  # Mean Reciprocal Rank
    ndcg: float  # Normalized Discounted Cumulative Gain
    
    # Generation metrics
    answer_relevancy: float
    answer_correctness: float
    faithfulness: float  # Answer grounded in context
    
    # Context metrics
    context_relevancy: float
    context_precision: float
    context_recall: float
    
    # Overall
    hallucination_rate: float
    avg_latency_ms: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "retrieval_precision": self.retrieval_precision,
            "retrieval_recall": self.retrieval_recall,
            "retrieval_f1": self.retrieval_f1,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "answer_relevancy": self.answer_relevancy,
            "answer_correctness": self.answer_correctness,
            "faithfulness": self.faithfulness,
            "context_relevancy": self.context_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "hallucination_rate": self.hallucination_rate,
            "avg_latency_ms": self.avg_latency_ms,
        }
    
    def __str__(self) -> str:
        """Format metrics for display."""
        return (
            f"RAG Metrics:\n"
            f"  Retrieval: P={self.retrieval_precision:.3f}, R={self.retrieval_recall:.3f}, F1={self.retrieval_f1:.3f}\n"
            f"  Ranking: MRR={self.mrr:.3f}, NDCG={self.ndcg:.3f}\n"
            f"  Generation: Relevancy={self.answer_relevancy:.3f}, Correctness={self.answer_correctness:.3f}\n"
            f"  Faithfulness: {self.faithfulness:.3f}\n"
            f"  Context: Relevancy={self.context_relevancy:.3f}, P={self.context_precision:.3f}, R={self.context_recall:.3f}\n"
            f"  Hallucination Rate: {self.hallucination_rate:.1%}\n"
            f"  Avg Latency: {self.avg_latency_ms:.1f}ms"
        )


def calculate_precision_recall_f1(
    retrieved: List[str],
    relevant: List[str]
) -> tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        retrieved: List of retrieved document IDs
        relevant: List of relevant document IDs
        
    Returns:
        Tuple of (precision, recall, f1)
    """
    if not retrieved:
        return 0.0, 0.0, 0.0
    
    retrieved_set = set(retrieved)
    relevant_set = set(relevant)
    
    tp = len(retrieved_set & relevant_set)
    
    precision = tp / len(retrieved_set) if retrieved_set else 0.0
    recall = tp / len(relevant_set) if relevant_set else 0.0
    
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def calculate_mrr(retrieved: List[str], relevant: List[str]) -> float:
    """
    Calculate Mean Reciprocal Rank.
    
    Args:
        retrieved: Ordered list of retrieved document IDs
        relevant: List of relevant document IDs
        
    Returns:
        MRR score
    """
    relevant_set = set(relevant)
    
    for i, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant_set:
            return 1.0 / i
    
    return 0.0


def calculate_ndcg(retrieved: List[str], relevant: List[str], k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain.
    
    Args:
        retrieved: Ordered list of retrieved document IDs
        relevant: List of relevant document IDs (with implicit ranking)
        k: Cutoff for NDCG@k
        
    Returns:
        NDCG@k score
    """
    relevant_set = set(relevant)
    
    # Calculate DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], 1):
        if doc_id in relevant_set:
            dcg += 1.0 / np.log2(i + 1)
    
    # Calculate IDCG (ideal DCG)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    
    return dcg / idcg if idcg > 0 else 0.0
