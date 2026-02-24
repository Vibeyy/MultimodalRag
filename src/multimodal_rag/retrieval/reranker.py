"""Reranking module for refining retrieval results."""

from typing import List, Dict, Any, Optional
import time

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from ..utils.config import get_config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class GeminiReranker:
    """
    Gemini-based reranker for context reranking (PascalCase per standards).
    
    Uses Gemini API to rerank retrieved chunks based on query relevance.
    """
    
    def __init__(self):
        """Initialize Gemini reranker."""
        if not GENAI_AVAILABLE:
            raise ImportError(
                "google-generativeai not installed. "
                "Run: pip install google-generativeai"
            )
        
        config = get_config()
        genai.configure(api_key=config.gemini_api_key)
        self._model = genai.GenerativeModel(config.gemini_model)
        
        logger.info("Initialized Gemini reranker")
    
    def rerank_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank chunks using Gemini for relevance assessment.
        
        Args:
            query: User query
            chunks: List of retrieved chunks
            top_k: Number of top chunks to return after reranking
            
        Returns:
            Reranked list of chunks
        """
        if not chunks:
            return []
        
        k = top_k or len(chunks)
        
        try:
            # For efficiency, we'll use a simpler approach:
            # Score each chunk individually
            scored_chunks = []
            
            for chunk in chunks:
                # Create prompt for relevance scoring
                prompt = self._create_relevance_prompt(query, chunk["text"])
                
                # Get relevance score
                try:
                    response = self._model.generate_content(prompt)
                    score = self._parse_score(response.text)
                except Exception as e:
                    logger.warning(f"Reranking failed for chunk, using original score: {str(e)}")
                    score = chunk.get("score", 0.0)
                
                chunk["rerank_score"] = score
                scored_chunks.append(chunk)
                
                # Small delay to avoid rate limits
                time.sleep(0.05)
            
            # Sort by rerank score
            scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
            
            # Take top k
            final_chunks = scored_chunks[:k]
            
            logger.info(f"Reranked {len(chunks)} chunks, returning top {len(final_chunks)}")
            return final_chunks
            
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            # Return original chunks on failure
            return chunks[:k]
    
    def _create_relevance_prompt(self, query: str, chunk_text: str) -> str:
        """
        Create prompt for relevance scoring (private method per standards).
        
        Args:
            query: User query
            chunk_text: Chunk text to score
            
        Returns:
            Prompt text
        """
        prompt = f"""Given the following query and text passage, rate the relevance of the passage to the query on a scale of 0.0 to 1.0, where:
- 1.0 = Highly relevant, directly answers the query
- 0.5 = Somewhat relevant, contains related information
- 0.0 = Not relevant, unrelated to the query

Query: {query}

Passage: {chunk_text[:500]}

Respond with ONLY a number between 0.0 and 1.0, nothing else."""

        return prompt
    
    def _parse_score(self, response_text: str) -> float:
        """
        Parse relevance score from response (private method per standards).
        
        Args:
            response_text: Response from Gemini
            
        Returns:
            Relevance score (0.0-1.0)
        """
        try:
            # Extract first number from response
            import re
            numbers = re.findall(r'0\.\d+|1\.0|0|1', response_text)
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(1.0, score))  # Clamp to [0, 1]
            else:
                return 0.5  # Default middle score
        except Exception:
            return 0.5


class SimpleReranker:
    """
    Simple rule-based reranker (PascalCase per standards).
    
    Reranks based on keyword matching and position. Free alternative to Gemini reranking.
    """
    
    def __init__(self):
        """Initialize simple reranker."""
        logger.info("Initialized simple keyword reranker")
    
    def rerank_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank chunks using keyword matching.
        
        Args:
            query: User query
            chunks: List of retrieved chunks
            top_k: Number of top chunks to return
            
        Returns:
            Reranked list of chunks
        """
        if not chunks:
            return []
        
        k = top_k or len(chunks)
        
        # Extract query keywords (simple tokenization)
        query_keywords = set(query.lower().split())
        
        # Score each chunk
        for chunk in chunks:
            text = chunk.get("text", "").lower()
            text_words = set(text.split())
            
            # Keyword overlap score
            overlap = len(query_keywords & text_words)
            total = len(query_keywords)
            keyword_score = overlap / total if total > 0 else 0.0
            
            # Combine with original score
            original_score = chunk.get("score", 0.0)
            chunk["rerank_score"] = 0.7 * original_score + 0.3 * keyword_score
        
        # Sort by rerank score
        chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        logger.info(f"Simple reranking complete: top {k} chunks")
        return chunks[:k]
