"""RAG agent with LangGraph orchestration."""

from typing import Dict, Any, Optional
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage

from .state import RAGState
from ..retrieval.retrievers import HybridRetriever
from ..generation.generator import GeminiGenerator
from ..generation.hallucination_detector import HallucinationDetector
from ..utils.logger import setup_logger
from ..utils.tracing import trace_function

logger = setup_logger(__name__)


class RAGAgent:
    """
    RAG Agent with orchestrated workflow (PascalCase per coding standards).
    
    Coordinates query expansion, retrieval, generation, and validation.
    """
    
    def __init__(
        self,
        retriever: HybridRetriever,
        generator: GeminiGenerator,
        hallucination_detector: Optional[HallucinationDetector] = None,
    ):
        """
        Initialize RAG agent.
        
        Args:
            retriever: Hybrid retriever instance
            generator: Gemini generator instance
            hallucination_detector: Optional hallucination detector
        """
        self._retriever = retriever
        self._generator = generator
        self._hallucination_detector = hallucination_detector
        logger.info("Initialized RAG agent with orchestration")
    
    @trace_function(name="query_expansion")
    def expand_query(self, state: RAGState) -> RAGState:
        """
        Expand user query into multiple variants (private method).
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated state with expanded queries
        """
        try:
            query = state["query"]
            logger.info(f"Expanding query: {query}")
            
            # Generate query expansions
            expansions = self._generator.expand_query(query, num_variants=2)
            
            state["expanded_queries"] = expansions or [query]
            state["step"] = "query_expansion_complete"
            logger.info(f"Generated {len(state['expanded_queries'])} query variants")
            
        except Exception as e:
            logger.error(f"Query expansion failed: {str(e)}")
            state["errors"].append(f"Query expansion error: {str(e)}")
            state["expanded_queries"] = [state["query"]]
        
        return state
    
    @trace_function(name="retrieval")
    def retrieve_context(self, state: RAGState) -> RAGState:
        """
        Retrieve relevant context chunks (private method).
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated state with retrieved chunks
        """
        try:
            queries = state.get("expanded_queries") or [state["query"]]
            logger.info(f"Retrieving context for {len(queries)} queries")
            
            all_chunks = []
            all_scores = []
            
            # Retrieve for each query variant
            for query in queries:
                chunks = self._retriever.retrieve(query)
                all_chunks.extend(chunks)
                all_scores.extend([c.get("score", 0.0) for c in chunks])
            
            # Deduplicate by chunk ID
            seen_ids = set()
            unique_chunks = []
            unique_scores = []
            
            for chunk, score in zip(all_chunks, all_scores):
                chunk_id = chunk.get("id")
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    unique_chunks.append(chunk)
                    unique_scores.append(score)
            
            # Sort by score descending
            sorted_pairs = sorted(
                zip(unique_chunks, unique_scores),
                key=lambda x: x[1],
                reverse=True
            )
            
            state["retrieved_chunks"] = [c for c, _ in sorted_pairs[:10]]
            state["retrieval_scores"] = [s for _, s in sorted_pairs[:10]]
            state["step"] = "retrieval_complete"
            
            logger.info(f"Retrieved {len(state['retrieved_chunks'])} unique chunks")
            
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            state["errors"].append(f"Retrieval error: {str(e)}")
            state["retrieved_chunks"] = []
            state["retrieval_scores"] = []
        
        return state
    
    @trace_function(name="generation")
    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Generate answer with citations (private method).
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated state with generated answer
        """
        try:
            query = state["query"]
            chunks = state["retrieved_chunks"]
            
            logger.info(f"Generating answer for query with {len(chunks)} chunks")
            
            if not chunks:
                state["generated_answer"] = "I don't have enough information to answer this question."
                state["citations"] = []
                state["step"] = "generation_complete"
                return state
            
            # Generate with citations
            result = self._generator.generate_with_citations(
                query=query,
                context=chunks,
            )
            
            state["generated_answer"] = result["answer"]
            state["citations"] = result.get("citations", [])
            state["step"] = "generation_complete"
            state["metadata"]["generation_metadata"] = result.get("metadata", {})
            
            logger.info(f"Generated answer with {len(state['citations'])} citations")
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            state["errors"].append(f"Generation error: {str(e)}")
            state["generated_answer"] = "Error generating answer. Please try again."
            state["citations"] = []
        
        return state
    
    @trace_function(name="hallucination_check")
    def check_hallucination(self, state: RAGState) -> RAGState:
        """
        Validate answer for hallucinations (private method).
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated state with hallucination scores
        """
        try:
            if not self._hallucination_detector:
                state["step"] = "hallucination_check_skipped"
                return state
            
            answer = state["generated_answer"]
            chunks = state["retrieved_chunks"]
            
            logger.info("Checking for hallucinations")
            
            # Detect hallucination
            result = self._hallucination_detector.detect_hallucination(
                answer=answer,
                contexts=chunks,
            )
            
            state["hallucination_score"] = result.get("hallucination_score", 0.0)
            state["is_hallucinated"] = result.get("is_hallucinated", False)
            state["step"] = "hallucination_check_complete"
            state["metadata"]["hallucination_metadata"] = result.get("metadata", {})
            
            if state["is_hallucinated"]:
                logger.warning(f"Potential hallucination detected (score: {state['hallucination_score']:.2f})")
            else:
                logger.info("No hallucination detected")
            
        except Exception as e:
            logger.error(f"Hallucination check failed: {str(e)}")
            state["errors"].append(f"Hallucination check error: {str(e)}")
        
        return state
    
    def run(
        self,
        query: str,
        expand_query: bool = True,
        check_hallucination: bool = False,
    ) -> Dict[str, Any]:
        """
        Run complete RAG workflow.
        
        Args:
            query: User query
            expand_query: Whether to expand query
            check_hallucination: Whether to check for hallucinations
            
        Returns:
            Final response with answer and metadata
        """
        # Get tracer for Langfuse integration
        from ..utils.tracing import get_tracer
        tracer = get_tracer()
        
        # Create Langfuse trace if enabled
        trace = None
        if tracer._enabled and tracer._client:
            try:
                trace = tracer._client.trace(
                    name="rag_query",
                    input={"query": query, "expand_query": expand_query, "check_hallucination": check_hallucination},
                    metadata={"timestamp": datetime.now().isoformat()}
                )
            except Exception as e:
                logger.warning(f"Failed to create Langfuse trace: {e}")
        
        # Initialize state
        state: RAGState = {
            "query": query,
            "expanded_queries": None,
            "retrieved_chunks": [],
            "retrieval_scores": [],
            "generated_answer": None,
            "citations": [],
            "hallucination_score": None,
            "is_hallucinated": None,
            "errors": [],
            "step": "initialized",
            "metadata": {},
        }
        
        # Execute workflow
        logger.info(f"Starting RAG workflow for query: {query}")
        
        if expand_query:
            state = self.expand_query(state)
        else:
            state["expanded_queries"] = [query]
        
        state = self.retrieve_context(state)
        state = self.generate_answer(state)
        
        if check_hallucination:
            state = self.check_hallucination(state)
        
        state["step"] = "complete"
        logger.info("RAG workflow complete")
        
        # Update Langfuse trace with output
        if trace:
            try:
                trace.update(
                    output={
                        "answer": state["generated_answer"],
                        "num_citations": len(state["citations"]),
                        "num_chunks": len(state["retrieved_chunks"]),
                        "is_hallucinated": state["is_hallucinated"],
                    },
                    metadata={
                        **state["metadata"],
                        "errors": state["errors"] if state["errors"] else None,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to update Langfuse trace: {e}")
        
        return {
            "answer": state["generated_answer"],
            "citations": state["citations"],
            "retrieved_chunks": state["retrieved_chunks"],
            "hallucination_score": state["hallucination_score"],
            "is_hallucinated": state["is_hallucinated"],
            "errors": state["errors"],
            "metadata": state["metadata"],
        }
