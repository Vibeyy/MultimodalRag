"""Hallucination detection for generated responses."""

from typing import Dict, Any, List
import re

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from .prompts import PromptBuilder
from ..utils.config import get_config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class HallucinationDetector:
    """
    Hallucination detector for RAG responses (PascalCase per standards).
    
    Validates that generated responses are grounded in the provided context.
    """
    
    def __init__(self):
        """Initialize hallucination detector."""
        if not GENAI_AVAILABLE:
            raise ImportError(
                "google-generativeai not installed. "
                "Run: pip install google-generativeai"
            )
        
        config = get_config()
        genai.configure(api_key=config.gemini_api_key)
        self._model = genai.GenerativeModel(config.gemini_model)
        self._prompt_builder = PromptBuilder()
        
        logger.info("Initialized hallucination detector")
    
    def validate_response(
        self,
        query: str,
        answer: str,
        context: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Validate that response is grounded in context.
        
        Args:
            query: Original query
            answer: Generated answer
            context: Context chunks that were provided
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Check if answer has citations
            has_citations = self._check_citations(answer)
            
            # Build context string
            context_text = "\n\n".join([
                chunk.get("text", "") for chunk in context[:10]
            ])
            
            # Build verification prompt
            prompt = self._prompt_builder.build_hallucination_check_prompt(
                query=query,
                answer=answer,
                context=context_text,
            )
            
            # Get verification
            response = self._model.generate_content(prompt)
            verification_text = response.text
            
            # Parse result
            is_supported = "YES" in verification_text.upper()[:50]
            
            # Check for specific claims
            unsupported_claims = self._extract_unsupported_claims(verification_text)
            
            result = {
                "is_grounded": is_supported and has_citations,
                "has_citations": has_citations,
                "llm_verification": is_supported,
                "unsupported_claims": unsupported_claims,
                "verification_text": verification_text,
            }
            
            logger.info(
                f"Hallucination check: grounded={result['is_grounded']}, "
                f"citations={has_citations}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Hallucination detection failed: {str(e)}")
            return {
                "is_grounded": False,
                "has_citations": False,
                "llm_verification": False,
                "unsupported_claims": [],
                "error": str(e),
            }
    
    def _check_citations(self, text: str) -> bool:
        """
        Check if text contains citations (private method per standards).
        
        Args:
            text: Text to check
            
        Returns:
            True if citations are present
        """
        # Look for citation pattern
        pattern = r'\[Source:\s*[^,]+,\s*Page:\s*\d+\]'
        matches = re.findall(pattern, text)
        return len(matches) > 0
    
    def _extract_unsupported_claims(self, verification_text: str) -> List[str]:
        """
        Extract unsupported claims from verification (private method).
        
        Args:
            verification_text: Verification response text
            
        Returns:
            List of unsupported claims
        """
        # Simple extraction - look for bullet points or numbered claims
        claims = []
        
        # Pattern for bullet points or numbers
        lines = verification_text.split('\n')
        for line in lines:
            if line.strip().startswith(('-', '*', '•')) or re.match(r'^\d+\.', line.strip()):
                claim = re.sub(r'^[\-\*\•\d\.]\s*', '', line.strip())
                if claim:
                    claims.append(claim)
        
        return claims
    
    def check_citation_accuracy(
        self,
        answer: str,
        context: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Check if citations actually exist in the provided context.
        
        Args:
            answer: Generated answer with citations
            context: Context chunks
            
        Returns:
            Dictionary with citation accuracy results
        """
        # Extract citations from answer
        pattern = r'\[Source:\s*([^,]+),\s*Page:\s*(\d+)\]'
        citations = re.findall(pattern, answer)
        
        # Build set of available sources
        available_sources = set()
        for chunk in context:
            source = chunk.get("source_file", "")
            page = chunk.get("page_num", 0)
            available_sources.add((source, page))
        
        # Check each citation
        valid_citations = []
        invalid_citations = []
        
        for source, page_str in citations:
            page = int(page_str)
            if (source.strip(), page) in available_sources:
                valid_citations.append((source, page))
            else:
                invalid_citations.append((source, page))
        
        accuracy = len(valid_citations) / len(citations) if citations else 0.0
        
        logger.info(
            f"Citation accuracy: {len(valid_citations)}/{len(citations)} valid"
        )
        
        return {
            "total_citations": len(citations),
            "valid_citations": len(valid_citations),
            "invalid_citations": len(invalid_citations),
            "accuracy": accuracy,
            "invalid_refs": invalid_citations,
        }
