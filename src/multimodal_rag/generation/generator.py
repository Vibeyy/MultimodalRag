"""Generation module using Gemini API with citations."""

from typing import List, Dict, Any, Optional, Iterator
import re
import time

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from .prompts import PromptBuilder
from ..utils.config import get_config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class GeminiGenerator:
    """
    Gemini-based generator with citation support (PascalCase per standards).
    
    Generates responses using Google Gemini API with proper citations.
    """
    
    def __init__(self):
        """Initialize Gemini generator."""
        if not GENAI_AVAILABLE:
            raise ImportError(
                "google-generativeai not installed. "
                "Run: pip install google-generativeai"
            )
        
        config = get_config()
        genai.configure(api_key=config.gemini_api_key)
        
        # Initialize model with safety settings
        self._model = genai.GenerativeModel(
            model_name=config.gemini_model,
            generation_config={
                "temperature": 0.1,  # Low temperature for factual responses
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            },
        )
        
        self._prompt_builder = PromptBuilder()
        self._rate_limit_delay = 1.0  # Seconds between requests
        self._last_request_time = 0.0
        
        logger.info("Initialized Gemini generator")
    
    def generate_with_citations(
        self,
        query: str,
        context: List[Dict[str, Any]],
        max_context_length: int = 4000,
    ) -> Dict[str, Any]:
        """
        Generate response with citations based on context.
        
        Args:
            query: User query
            context: Retrieved context chunks
            max_context_length: Maximum context length
            
        Returns:
            Dictionary with answer, citations, and metadata
        """
        if not context:
            logger.warning("No context provided for generation")
            return {
                "answer": "I don't have enough information in the provided context to answer this question.",
                "citations": [],
                "has_citations": False,
                "token_count": 0,
            }
        
        try:
            # Build prompt
            prompt = self._prompt_builder.build_prompt_with_context(
                query=query,
                context_chunks=context,
                max_context_length=max_context_length,
            )
            
            # Rate limiting
            self._wait_for_rate_limit()
            
            # Generate response
            start_time = time.time()
            response = self._model.generate_content(prompt)
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract text
            answer = response.text
            
            # Extract citations
            citations = self._extract_citations(answer)
            
            # Estimate token count (rough approximation)
            token_count = len(prompt.split()) + len(answer.split())
            
            result = {
                "answer": answer,
                "citations": citations,
                "has_citations": len(citations) > 0,
                "token_count": token_count,
                "latency_ms": latency_ms,
            }
            
            logger.info(
                f"Generated response: {len(answer)} chars, "
                f"{len(citations)} citations, {latency_ms:.2f}ms"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise
    
    def generate_stream(
        self,
        query: str,
        context: List[Dict[str, Any]],
        max_context_length: int = 4000,
    ) -> Iterator[str]:
        """
        Generate streaming response with citations.
        
        Args:
            query: User query
            context: Retrieved context chunks
            max_context_length: Maximum context length
            
        Yields:
            Text chunks as they are generated
        """
        if not context:
            yield "I don't have enough information in the provided context to answer this question."
            return
        
        try:
            # Build prompt
            prompt = self._prompt_builder.build_prompt_with_context(
                query=query,
                context_chunks=context,
                max_context_length=max_context_length,
            )
            
            # Rate limiting
            self._wait_for_rate_limit()
            
            # Generate streaming response
            response = self._model.generate_content(prompt, stream=True)
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
            
            logger.info("Streaming generation complete")
            
        except Exception as e:
            logger.error(f"Streaming generation failed: {str(e)}")
            yield f"Error: {str(e)}"
    
    def expand_query(self, query: str, num_variants: int = 2) -> List[str]:
        """
        Generate query expansions for better retrieval.
        
        Args:
            query: Original query
            num_variants: Number of variants to generate
            
        Returns:
            List of query variants
        """
        try:
            # Build expansion prompt
            prompt = self._prompt_builder.build_query_expansion_prompt(
                query=query,
                num_variants=num_variants,
            )
            
            # Rate limiting
            self._wait_for_rate_limit()
            
            # Generate
            response = self._model.generate_content(prompt)
            
            # Parse variants (one per line)
            variants = []
            for line in response.text.split('\n'):
                # Remove numbering and clean
                clean_line = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
                if clean_line and clean_line != query:
                    variants.append(clean_line)
            
            logger.info(f"Generated {len(variants)} query variants")
            return variants[:num_variants]
            
        except Exception as e:
            logger.error(f"Query expansion failed: {str(e)}")
            return []
    
    def _extract_citations(self, text: str) -> List[Dict[str, str]]:
        """
        Extract citations from generated text (private method per standards).
        
        Args:
            text: Generated text with citations
            
        Returns:
            List of citation dictionaries
        """
        # Pattern: [Source: filename, Page: X]
        pattern = r'\[Source:\s*([^,]+),\s*Page:\s*(\d+)(?:,\s*Type:\s*(\w+))?\]'
        matches = re.findall(pattern, text)
        
        citations = []
        for match in matches:
            citation = {
                "source_file": match[0].strip(),
                "page_num": int(match[1]),
            }
            if match[2]:  # Type field (optional)
                citation["type"] = match[2]
            citations.append(citation)
        
        return citations
    
    def _wait_for_rate_limit(self) -> None:
        """
        Wait to respect rate limits (private method per standards).
        
        Ensures we don't exceed Gemini free tier limits (60 req/min).
        """
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._rate_limit_delay:
            wait_time = self._rate_limit_delay - time_since_last
            time.sleep(wait_time)
        
        self._last_request_time = time.time()
