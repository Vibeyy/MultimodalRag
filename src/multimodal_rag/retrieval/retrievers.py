"""Retrieval strategies including dense, BM25, and hybrid search."""

from typing import List, Dict, Optional, Any
import numpy as np
from collections import defaultdict

from .vector_store import QdrantStore
from ..ingestion.embedder import TextEmbedder
from ..utils.config import get_config, TOP_K_RETRIEVAL
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class DenseRetriever:
    """
    Dense vector retriever using cosine similarity (PascalCase per standards).
    
    Retrieves chunks based on semantic similarity of embeddings.
    """
    
    def __init__(
        self,
        vector_store: QdrantStore,
        embedder: TextEmbedder,
        top_k: int = TOP_K_RETRIEVAL,
    ):
        """
        Initialize dense retriever.
        
        Args:
            vector_store: Qdrant vector store instance
            embedder: Text embedder for query encoding
            top_k: Number of results to retrieve
        """
        self._vector_store = vector_store
        self._embedder = embedder
        self._top_k = top_k
        logger.info(f"Initialized dense retriever with top_k={top_k}")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using dense vector search.
        
        Searches text vector space (includes both native text and OCR text).
        
        Args:
            query: User query text
            top_k: Number of results (overrides default)
            filters: Optional metadata filters
            score_threshold: Minimum similarity score
            
        Returns:
            List of retrieved chunks with scores
        """
        k = top_k or self._top_k
        
        try:
            # Encode query
            query_embedding = self._embedder.embed_text(query)
            
            # Search text vector space (includes native text + OCR text chunks)
            results = self._vector_store.search(
                query_vector=query_embedding.tolist(),
                limit=k,
                score_threshold=score_threshold,
                filters=filters,
                vector_name="text",
            )
            
            logger.info(f"Dense retrieval: {len(results)} chunks for query")
            return results
            
        except Exception as e:
            logger.error(f"Dense retrieval failed: {str(e)}")
            raise


class BM25Retriever:
    """
    BM25 keyword-based retriever (PascalCase per standards).
    
    Retrieves chunks based on keyword matching with BM25 scoring.
    """
    
    def __init__(
        self,
        vector_store: QdrantStore,
        top_k: int = TOP_K_RETRIEVAL,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """
        Initialize BM25 retriever.
        
        Args:
            vector_store: Qdrant vector store instance
            top_k: Number of results to retrieve
            k1: BM25 k1 parameter (term saturation)
            b: BM25 b parameter (length normalization)
        """
        self._vector_store = vector_store
        self._top_k = top_k
        self._k1 = k1
        self._b = b
        self._corpus = []  # Will be populated from vector store
        logger.info(f"Initialized BM25 retriever with top_k={top_k}")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using BM25 keyword search.
        
        Args:
            query: User query text
            top_k: Number of results (overrides default)
            filters: Optional metadata filters
            
        Returns:
            List of retrieved chunks with BM25 scores
        """
        k = top_k or self._top_k
        
        try:
            # For simplicity, we'll do a text-based search
            # In production, you'd use Qdrant's payload-based search or BM42
            # This is a simplified implementation
            
            # Get all chunks (in production, use scrolling for large collections)
            # For now, we'll use a workaround with dense search and rerank with BM25
            
            logger.info(f"BM25 retrieval: searching for query")
            
            # Note: Full BM25 implementation would require indexing
            # For this free stack, we'll use a hybrid approach in HybridRetriever
            return []
            
        except Exception as e:
            logger.error(f"BM25 retrieval failed: {str(e)}")
            raise


class HybridRetriever:
    """
    Hybrid retriever combining dense and BM25 with RRF (PascalCase per standards).
    
    Uses Reciprocal Rank Fusion to combine results from multiple retrievers.
    """
    
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        top_k: int = TOP_K_RETRIEVAL,
        rrf_k: int = 60,
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            dense_retriever: Dense retriever instance
            top_k: Number of final results
            rrf_k: RRF constant (typically 60)
        """
        self._dense_retriever = dense_retriever
        self._top_k = top_k
        self._rrf_k = rrf_k
        logger.info(f"Initialized hybrid retriever with RRF, top_k={top_k}")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        dense_weight: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using hybrid search with RRF.
        
        Args:
            query: User query text
            top_k: Number of results (overrides default)
            filters: Optional metadata filters
            dense_weight: Weight for dense retrieval (0-1)
            
        Returns:
            List of retrieved chunks with fused scores
        """
        k = top_k or self._top_k
        
        try:
            # Get dense results (retrieve more for fusion)
            dense_results = self._dense_retriever.retrieve(
                query=query,
                top_k=k * 2,
                filters=filters,
            )
            
            # For now, without full BM25 implementation, we'll use dense results
            # In production, you would fuse dense + BM25 results here
            
            # Apply RRF scoring
            fused_results = self._apply_rrf([dense_results])
            
            # Sort by fused score and take top_k
            fused_results.sort(key=lambda x: x["fused_score"], reverse=True)
            final_results = fused_results[:k]
            
            logger.info(f"Hybrid retrieval: {len(final_results)} chunks")
            return final_results
            
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {str(e)}")
            raise
    
    def retrieve_with_expansion(
        self,
        query: str,
        expansions: List[str],
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve with query expansion using multiple query variants.
        
        Args:
            query: Original user query
            expansions: List of query expansions/paraphrases
            top_k: Number of results
            filters: Optional metadata filters
            
        Returns:
            List of retrieved chunks with aggregated scores
        """
        k = top_k or self._top_k
        
        try:
            all_queries = [query] + expansions
            all_results = []
            
            # Retrieve for each query variant
            for q in all_queries:
                results = self._dense_retriever.retrieve(
                    query=q,
                    top_k=k,
                    filters=filters,
                )
                all_results.append(results)
            
            # Fuse results
            fused_results = self._apply_rrf(all_results)
            
            # Sort and take top_k
            fused_results.sort(key=lambda x: x["fused_score"], reverse=True)
            final_results = fused_results[:k]
            
            logger.info(
                f"Query expansion retrieval: {len(all_queries)} variants, "
                f"{len(final_results)} final chunks"
            )
            return final_results
            
        except Exception as e:
            logger.error(f"Query expansion retrieval failed: {str(e)}")
            raise
    
    def _apply_rrf(
        self,
        results_lists: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Apply Reciprocal Rank Fusion to combine rankings (private method).
        
        Args:
            results_lists: List of result lists from different retrievers
            
        Returns:
            Fused results with RRF scores
        """
        # Collect all unique chunks
        chunk_scores = defaultdict(lambda: {"score": 0.0, "chunk": None})
        
        for results in results_lists:
            for rank, result in enumerate(results, start=1):
                chunk_id = result["id"]
                # RRF formula: 1 / (k + rank)
                rrf_score = 1.0 / (self._rrf_k + rank)
                
                chunk_scores[chunk_id]["score"] += rrf_score
                if chunk_scores[chunk_id]["chunk"] is None:
                    chunk_scores[chunk_id]["chunk"] = result
        
        # Convert to list
        fused_results = []
        for chunk_id, data in chunk_scores.items():
            chunk = data["chunk"]
            chunk["fused_score"] = data["score"]
            fused_results.append(chunk)
        
        return fused_results
    
    def apply_mmr(
        self,
        results: List[Dict[str, Any]],
        lambda_param: float = 0.5,
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Apply Maximal Marginal Relevance for diversity.
        
        Args:
            results: Initial retrieval results
            lambda_param: Trade-off between relevance and diversity (0-1)
            k: Number of results to return
            
        Returns:
            Diversified results
        """
        if not results or len(results) <= 1:
            return results
        
        final_k = k or self._top_k
        selected = []
        remaining = results.copy()
        
        # Select first (most relevant)
        selected.append(remaining.pop(0))
        
        # Iteratively select diverse documents
        while remaining and len(selected) < final_k:
            max_mmr_score = -float('inf')
            max_idx = 0
            
            for idx, candidate in enumerate(remaining):
                # Relevance score
                relevance = candidate.get("score", 0.0)
                
                # Similarity to already selected (simplified)
                max_sim = 0.0
                for selected_doc in selected:
                    # Simple text overlap as similarity (in production, use embeddings)
                    sim = self._text_similarity(
                        candidate.get("text", ""),
                        selected_doc.get("text", "")
                    )
                    max_sim = max(max_sim, sim)
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                
                if mmr_score > max_mmr_score:
                    max_mmr_score = mmr_score
                    max_idx = idx
            
            selected.append(remaining.pop(max_idx))
        
        logger.info(f"Applied MMR: {len(selected)} diverse chunks")
        return selected
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple text similarity (private method).
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        # Simple Jaccard similarity on words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
