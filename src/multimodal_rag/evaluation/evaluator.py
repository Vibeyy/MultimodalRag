"""RAG evaluator using RAGAS and custom metrics."""

from typing import List, Dict, Any, Optional
import time
import os
from datetime import datetime

try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        faithfulness,
        context_recall,
        context_precision,
    )
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

from .metrics import (
    RAGMetrics,
    calculate_precision_recall_f1,
    calculate_mrr,
    calculate_ndcg,
)
from ..utils.logger import setup_logger
from ..utils.tracing import trace_function
from ..utils.config import get_config

logger = setup_logger(__name__)


class RAGEvaluator:
    """
    RAG system evaluator (PascalCase per coding standards).
    
    Evaluates retrieval and generation quality using multiple metrics.
    """
    
    def __init__(self, use_ragas: bool = True):
        """
        Initialize evaluator.
        
        Args:
            use_ragas: Whether to use RAGAS metrics
        """
        self._use_ragas = use_ragas and RAGAS_AVAILABLE
        self._llm = None
        self._embeddings = None
        
        if self._use_ragas:
            # Set OpenAI API key from config for RAGAS
            config = get_config()
            if config.openai_api_key:
                os.environ["OPENAI_API_KEY"] = config.openai_api_key
                # Initialize with gpt-4o-mini for better quality at lower cost
                self._llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                self._embeddings = OpenAIEmbeddings()
                logger.info("Initialized RAG evaluator with RAGAS (using gpt-4o-mini)")
            else:
                logger.warning("OPENAI_API_KEY not configured - RAGAS evaluation will be skipped")
                self._use_ragas = False
        
        if not self._use_ragas:
            logger.info("Initialized RAG evaluator (RAGAS disabled)")
    
    @trace_function(name="evaluate_retrieval")
    def evaluate_retrieval(
        self,
        queries: List[str],
        retrieved_docs: List[List[str]],
        relevant_docs: List[List[str]],
    ) -> Dict[str, float]:
        """
        Evaluate retrieval quality.
        
        Args:
            queries: List of queries
            retrieved_docs: List of retrieved document IDs per query
            relevant_docs: List of relevant document IDs per query (ground truth)
            
        Returns:
            Dictionary of retrieval metrics
        """
        if len(queries) != len(retrieved_docs) != len(relevant_docs):
            raise ValueError("Queries, retrieved_docs, and relevant_docs must have same length")
        
        precisions = []
        recalls = []
        f1_scores = []
        mrr_scores = []
        ndcg_scores = []
        
        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            p, r, f1 = calculate_precision_recall_f1(retrieved, relevant)
            precisions.append(p)
            recalls.append(r)
            f1_scores.append(f1)
            
            mrr_scores.append(calculate_mrr(retrieved, relevant))
            ndcg_scores.append(calculate_ndcg(retrieved, relevant))
        
        metrics = {
            "retrieval_precision": sum(precisions) / len(precisions) if precisions else 0.0,
            "retrieval_recall": sum(recalls) / len(recalls) if recalls else 0.0,
            "retrieval_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
            "mrr": sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0,
            "ndcg": sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0,
        }
        
        logger.info(f"Retrieval evaluation: {len(queries)} queries, P={metrics['retrieval_precision']:.3f}")
        return metrics
    
    @trace_function(name="evaluate_generation")
    def evaluate_generation(
        self,
        queries: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate generation quality using RAGAS.
        
        Args:
            queries: List of queries
            answers: Generated answers
            contexts: Retrieved contexts per query
            ground_truths: Optional ground truth answers
            
        Returns:
            Dictionary of generation metrics
        """
        if not self._use_ragas:
            logger.warning("RAGAS not available, skipping generation evaluation")
            return {
                "answer_relevancy": 0.0,
                "faithfulness": 0.0,
                "context_precision": 0.0,
            }
        
        try:
            # Prepare dataset for RAGAS
            data = {
                "question": queries,
                "answer": answers,
                "contexts": contexts,
            }
            
            # Determine which metrics to run
            if ground_truths:
                # With ground truth: run all 4 metrics
                data["ground_truth"] = ground_truths
                metrics_to_use = [
                    answer_relevancy,
                    faithfulness,
                    context_precision,
                    context_recall,  # Only works with real ground truth
                ]
            else:
                # Without ground truth: run 3 metrics that work with self-validation
                # Use generated answer as reference for context_precision
                data["ground_truth"] = answers
                metrics_to_use = [
                    answer_relevancy,
                    faithfulness,
                    context_precision,  # Works with generated answer as reference
                    # context_recall EXCLUDED - requires real ground truth
                ]
            
            logger.info(f"Creating RAGAS dataset with {len(queries)} queries")
            dataset = Dataset.from_dict(data)
            
            logger.info("Starting RAGAS evaluation with gpt-4o-mini...")
            # Run evaluation with gpt-4o-mini
            result = evaluate(dataset, metrics=metrics_to_use, llm=self._llm, embeddings=self._embeddings)
            
            logger.info(f"Generation evaluation complete: {len(queries)} queries")
            logger.info(f"RAGAS results: {result}")
            return result
            
        except Exception as e:
            import traceback
            logger.error(f"RAGAS evaluation failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "answer_relevancy": 0.0,
                "faithfulness": 0.0,
                "context_precision": 0.0,
            }
    
    @trace_function(name="evaluate_end_to_end")
    def evaluate_end_to_end(
        self,
        test_cases: List[Dict[str, Any]],
        rag_function: callable,
    ) -> RAGMetrics:
        """
        Run end-to-end evaluation.
        
        Args:
            test_cases: List of test cases with queries and ground truth
                Each test case: {
                    "query": str,
                    "relevant_docs": List[str],
                    "ground_truth_answer": Optional[str]
                }
            rag_function: Function that takes query and returns result dict with:
                {
                    "answer": str,
                    "retrieved_chunks": List[Dict],
                    "is_hallucinated": bool,
                    "latency_ms": float
                }
                
        Returns:
            Complete RAG metrics
        """
        logger.info(f"Starting end-to-end evaluation with {len(test_cases)} test cases")
        
        queries = []
        answers = []
        contexts = []
        ground_truths = []
        retrieved_doc_ids = []
        relevant_doc_ids = []
        hallucinations = []
        latencies = []
        
        for i, test_case in enumerate(test_cases, 1):
            query = test_case["query"]
            relevant = test_case.get("relevant_docs", [])
            ground_truth = test_case.get("ground_truth_answer")
            
            logger.info(f"Evaluating test case {i}/{len(test_cases)}: {query[:50]}...")
            
            # Run RAG
            start_time = time.time()
            result = rag_function(query)
            latency = (time.time() - start_time) * 1000
            
            # Extract results
            queries.append(query)
            answers.append(result.get("answer", ""))
            
            chunks = result.get("retrieved_chunks", [])
            contexts.append([c.get("text", "") for c in chunks])
            retrieved_doc_ids.append([c.get("id", "") for c in chunks])
            relevant_doc_ids.append(relevant)
            
            if ground_truth:
                ground_truths.append(ground_truth)
            
            hallucinations.append(result.get("is_hallucinated", False))
            latencies.append(result.get("latency_ms", latency))
        
        # Evaluate retrieval
        retrieval_metrics = self.evaluate_retrieval(
            queries=queries,
            retrieved_docs=retrieved_doc_ids,
            relevant_docs=relevant_doc_ids,
        )
        
        # Evaluate generation
        generation_metrics = self.evaluate_generation(
            queries=queries,
            answers=answers,
            contexts=contexts,
            ground_truths=ground_truths if ground_truths else None,
        )
        
        # Combine metrics
        metrics = RAGMetrics(
            retrieval_precision=retrieval_metrics["retrieval_precision"],
            retrieval_recall=retrieval_metrics["retrieval_recall"],
            retrieval_f1=retrieval_metrics["retrieval_f1"],
            mrr=retrieval_metrics["mrr"],
            ndcg=retrieval_metrics["ndcg"],
            answer_relevancy=generation_metrics.get("answer_relevancy", 0.0),
            answer_correctness=0.0,  # Would need ground truth comparison
            faithfulness=generation_metrics.get("faithfulness", 0.0),
            context_relevancy=0.0,  # Custom metric
            context_precision=generation_metrics.get("context_precision", 0.0),
            context_recall=generation_metrics.get("context_recall", 0.0),
            hallucination_rate=sum(hallucinations) / len(hallucinations) if hallucinations else 0.0,
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
        )
        
        logger.info("End-to-end evaluation complete")
        logger.info(f"\n{metrics}")
        
        return metrics
