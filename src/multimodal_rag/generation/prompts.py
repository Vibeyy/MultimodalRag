"""Prompt templates for generation with citations."""

from typing import List, Dict, Any

# Constants for prompts (UPPERCASE per coding standards)
SYSTEM_PROMPT = """You are a factual AI assistant that answers questions based strictly on the provided context.

IMPORTANT RULES:
1. Use ONLY information from the provided context - never use external knowledge
2. Cite every fact using this format: [Source: filename, Page: X]
3. If the context doesn't contain enough information, say "I don't have enough information in the provided context to answer this question."
4. Be precise and concise in your responses
5. If information comes from multiple sources, cite all relevant sources
6. For images, cite as: [Source: filename, Page: X, Type: Image]

Context format:
- Each chunk has: text content, source file, page number, and type (text/image)
- Image chunks include OCR-extracted text from the image"""

CITATION_INSTRUCTION = """Remember to cite EVERY fact using the format [Source: filename, Page: X]. 
Multiple citations should be inline: "This fact [Source: doc.pdf, Page: 1] and this fact [Source: report.pdf, Page: 3]."
"""

NO_CONTEXT_RESPONSE = "I don't have enough information in the provided context to answer this question."


class PromptBuilder:
    """
    Prompt builder for RAG generation (PascalCase per standards).
    
    Constructs prompts with context and citation instructions.
    """
    
    def __init__(self):
        """Initialize prompt builder."""
        pass
    
    def build_prompt_with_context(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        max_context_length: int = 4000,
    ) -> str:
        """
        Build prompt with query and retrieved context.
        
        Args:
            query: User query
            context_chunks: List of retrieved context chunks
            max_context_length: Maximum context characters to include
            
        Returns:
            Complete prompt text
        """
        if not context_chunks:
            return f"{SYSTEM_PROMPT}\n\nNo context available.\n\nQuestion: {query}"
        
        # Build context section
        context_parts = []
        current_length = 0
        
        for idx, chunk in enumerate(context_chunks, start=1):
            chunk_text = chunk.get("text", "")
            source_file = chunk.get("source_file", "unknown")
            page_num = chunk.get("page_num", 0)
            chunk_type = chunk.get("chunk_type", "text")
            
            # Format chunk with metadata
            chunk_header = f"\n[Context {idx}]"
            chunk_meta = f"\nSource: {source_file}, Page: {page_num}, Type: {chunk_type}"
            chunk_content = f"\n{chunk_text}\n"
            
            chunk_full = chunk_header + chunk_meta + chunk_content
            
            # Check length limit
            if current_length + len(chunk_full) > max_context_length:
                break
            
            context_parts.append(chunk_full)
            current_length += len(chunk_full)
        
        # Combine everything
        context_section = "CONTEXT:\n" + "\n---".join(context_parts)
        
        prompt = f"""{SYSTEM_PROMPT}

{context_section}

{CITATION_INSTRUCTION}

Question: {query}

Answer:"""
        
        return prompt
    
    def build_query_expansion_prompt(self, query: str, num_variants: int = 2) -> str:
        """
        Build prompt for query expansion.
        
        Args:
            query: Original user query
            num_variants: Number of paraphrased variants to generate
            
        Returns:
            Prompt for query expansion
        """
        prompt = f"""Given the following question, generate {num_variants} alternative phrasings or related questions that would help retrieve relevant information.

Original question: {query}

Generate {num_variants} alternative questions (one per line, numbered):"""
        
        return prompt
    
    def build_hallucination_check_prompt(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> str:
        """
        Build prompt for hallucination detection.
        
        Args:
            query: Original query
            answer: Generated answer
            context: Context that was provided
            
        Returns:
            Prompt for hallucination checking
        """
        prompt = f"""Verify if the following answer is fully supported by the provided context.

Context:
{context[:2000]}

Question: {query}

Answer: {answer}

Is every claim in the answer supported by the context? Respond with:
- "YES" if all claims are supported
- "NO" if any claims are not supported or seem to use external knowledge

Also briefly explain which claims (if any) are not supported.

Verification:"""
        
        return prompt
    
    def build_summarization_prompt(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Build prompt for context summarization.
        
        Args:
            chunks: Context chunks to summarize
            
        Returns:
            Summarization prompt
        """
        # Combine chunk texts
        combined_text = "\n\n".join([
            f"[From {chunk.get('source_file', 'unknown')}, Page {chunk.get('page_num', 0)}]:\n{chunk.get('text', '')}"
            for chunk in chunks[:5]  # Limit to avoid token overflow
        ])
        
        prompt = f"""Summarize the following context passages while preserving key information and sources:

{combined_text[:3000]}

Summary:"""
        
        return prompt
