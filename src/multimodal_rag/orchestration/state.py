"""State management for RAG workflows."""

from typing import List, Dict, Any, Optional, TypedDict, Annotated
from operator import add


class RAGState(TypedDict):
    """
    State for RAG workflow (TypedDict per type safety standards).
    
    Tracks the flow from query to final answer with citations.
    """
    # Input
    query: str
    
    # Query processing
    expanded_queries: Optional[List[str]]
    
    # Retrieval
    retrieved_chunks: Annotated[List[Dict[str, Any]], add]
    retrieval_scores: List[float]
    
    # Generation
    generated_answer: Optional[str]
    citations: List[Dict[str, str]]
    
    # Evaluation
    hallucination_score: Optional[float]
    is_hallucinated: Optional[bool]
    
    # Metadata
    errors: Annotated[List[str], add]
    step: str
    metadata: Dict[str, Any]
