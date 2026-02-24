"""Generation module using OpenAI Chat API with citations."""

from typing import List, Dict, Any, Optional, Iterator
import re
import time
from openai import OpenAI

from .prompts import PromptBuilder
from ..utils.config import get_config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class OpenAIGenerator:
    """
    OpenAI-based generator with citation support (PascalCase per standards).
    
    Generates responses using OpenAI Chat API with proper citations.
    """
    
    def __init__(self):
        """Initialize OpenAI generator."""
        config = get_config()
        self._client = OpenAI(api_key=config.openai_api_key)
        self._model = config.openai_model
        self._prompt_builder = PromptBuilder()
        self._rate_limit_delay = 0.1  # Seconds between requests
        self._last_request_time = 0.0
        
        logger.info(f"Initialized OpenAI generator with model: {self._model}")
    
    def generate_with_citations(
        self,
        query: str,
        context: List[Dict[str, Any]],
        max_context_length: int = 4000,
        allow_general_knowledge: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate response with citations based on context.
        
        Args:
            query: User query
            context: Retrieved context chunks
            max_context_length: Maximum context length
            allow_general_knowledge: If True, use general knowledge when context is insufficient
            
        Returns:
            Dictionary with answer, citations, and metadata
        """
        if not context:
            logger.warning("No context provided for generation")
            if allow_general_knowledge:
                logger.info("Falling back to general knowledge mode")
                return self._generate_with_general_knowledge(query)
            return {
                "answer": "I don't have enough information in the provided context to answer this question.",
                "citations": [],
                "has_citations": False,
                "token_count": 0,
                "source": "none",
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
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant. When context is provided and contains the answer, use it and cite sources. When context is insufficient or the question is casual, use your general knowledge to be helpful."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for factual responses
                max_tokens=2048,
            )
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract text
            answer = response.choices[0].message.content
            
            # Extract citations
            citations = self._extract_citations(answer)
            
            # Get token count
            token_count = response.usage.total_tokens
            
            result = {
                "answer": answer,
                "citations": citations,
                "has_citations": len(citations) > 0,
                "token_count": token_count,
                "latency_ms": latency_ms,
                "source": "retrieval",
            }
            
            logger.info(
                f"Generated response: {len(answer)} chars, "
                f"{len(citations)} citations, {latency_ms:.2f}ms, "
                f"{token_count} tokens"
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
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides accurate answers based on the given context. Always cite your sources using the format [Source: filename, Page: X]."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=2048,
                stream=True,
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            
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
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates query variations for better search results."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=500,
            )
            
            # Parse variants (one per line)
            variants = []
            for line in response.choices[0].message.content.split('\n'):
                # Remove numbering and clean
                clean_line = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
                if clean_line and clean_line != query:
                    variants.append(clean_line)
            
            logger.info(f"Generated {len(variants)} query variants")
            return variants[:num_variants]
            
        except Exception as e:
            logger.error(f"Query expansion failed: {str(e)}")
            return []
    
    def _generate_with_general_knowledge(self, query: str) -> Dict[str, Any]:
        """
        Generate response using general knowledge (no context).
        
        Args:
            query: User query
            
        Returns:
            Dictionary with answer and metadata
        """
        from .prompts import GENERAL_KNOWLEDGE_SYSTEM_PROMPT
        
        try:
            # Rate limiting
            self._wait_for_rate_limit()
            
            # Generate response without context
            start_time = time.time()
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": GENERAL_KNOWLEDGE_SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                temperature=0.7,  # Higher temperature for more natural responses
                max_tokens=2048,
            )
            latency_ms = (time.time() - start_time) * 1000
            
            answer = response.choices[0].message.content
            token_count = response.usage.total_tokens
            
            result = {
                "answer": answer,
                "citations": [],
                "has_citations": False,
                "token_count": token_count,
                "latency_ms": latency_ms,
                "source": "general_knowledge",
            }
            
            logger.info(
                f"Generated general knowledge response: {len(answer)} chars, "
                f"{latency_ms:.2f}ms, {token_count} tokens"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"General knowledge generation failed: {str(e)}")
            raise
    
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
        
        Ensures we don't exceed OpenAI rate limits.
        """
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._rate_limit_delay:
            wait_time = self._rate_limit_delay - time_since_last
            time.sleep(wait_time)
        
        self._last_request_time = time.time()


# Alias for backward compatibility
GeminiGenerator = OpenAIGenerator
