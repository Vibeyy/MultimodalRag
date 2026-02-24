"""Tracing and observability with Langfuse integration."""

import time
import os
from typing import Any, Dict, Optional
from datetime import datetime
from contextlib import contextmanager

try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False

from .config import get_config
from .logger import setup_logger

logger = setup_logger(__name__)


class LangfuseTracer:
    """
    Langfuse tracing wrapper for observability (PascalCase per standards).
    
    Attributes:
        _client: Langfuse client instance
        _enabled: Whether tracing is enabled
    """
    
    def __init__(self):
        """Initialize Langfuse tracer with configuration."""
        config = get_config()
        self._enabled = False
        self._client = None
        
        # Check if tracing is enabled via environment variable
        langfuse_enabled = os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"
        
        if LANGFUSE_AVAILABLE and langfuse_enabled and config.langfuse_public_key and config.langfuse_secret_key:
            try:
                self._client = Langfuse(
                    public_key=config.langfuse_public_key,
                    secret_key=config.langfuse_secret_key,
                    host=config.langfuse_host,
                )
                self._enabled = True
                logger.info("Langfuse tracing enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize Langfuse: {str(e)}")
        else:
            if not langfuse_enabled:
                logger.info("Langfuse tracing disabled (LANGFUSE_ENABLED not set to 'true')")
            elif not config.langfuse_public_key or not config.langfuse_secret_key:
                logger.info("Langfuse tracing disabled (credentials not configured)")
            else:
                logger.info("Langfuse tracing disabled (langfuse package not installed)")
    
    @contextmanager
    def trace_ingestion(self, file_path: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Trace document ingestion process.
        
        Args:
            file_path: Path to the document being ingested
            metadata: Additional metadata
            
        Yields:
            Trace object or None
        """
        if not self._enabled:
            yield None
            return
        
        start_time = time.time()
        trace = self._client.trace(
            name="document_ingestion",
            metadata={
                "file_path": file_path,
                "timestamp": datetime.utcnow().isoformat(),
                **(metadata or {})
            }
        )
        
        try:
            yield trace
        finally:
            latency_ms = (time.time() - start_time) * 1000
            trace.update(
                output={"latency_ms": latency_ms}
            )
    
    @contextmanager
    def trace_retrieval(
        self,
        query_id: str,
        query: str,
        method: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Trace retrieval process.
        
        Args:
            query_id: Unique query identifier
            query: User query text
            method: Retrieval method (dense, bm25, hybrid)
            metadata: Additional metadata
            
        Yields:
            Span object or None
        """
        if not self._enabled:
            yield None
            return
        
        start_time = time.time()
        span = self._client.span(
            name=f"retrieval_{method}",
            metadata={
                "query_id": query_id,
                "query": query,
                "method": method,
                **(metadata or {})
            }
        )
        
        try:
            yield span
        finally:
            latency_ms = (time.time() - start_time) * 1000
            span.end(
                output={"latency_ms": latency_ms}
            )
    
    @contextmanager
    def trace_generation(
        self,
        query_id: str,
        query: str,
        context_size: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Trace generation process.
        
        Args:
            query_id: Unique query identifier
            query: User query text
            context_size: Number of context chunks
            metadata: Additional metadata
            
        Yields:
            Generation object or None
        """
        if not self._enabled:
            yield None
            return
        
        start_time = time.time()
        generation = self._client.generation(
            name="gemini_generation",
            model="gemini-1.5-flash",
            input={"query": query, "context_size": context_size},
            metadata={
                "query_id": query_id,
                **(metadata or {})
            }
        )
        
        try:
            yield generation
        finally:
            latency_ms = (time.time() - start_time) * 1000
            generation.end(
                output={"latency_ms": latency_ms}
            )
    
    def flush(self) -> None:
        """Flush pending traces to Langfuse."""
        if self._enabled and self._client:
            self._client.flush()


# Global tracer instance (singleton)
_tracer: Optional[LangfuseTracer] = None


def get_tracer() -> LangfuseTracer:
    """Get or create the global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = LangfuseTracer()
    return _tracer


def trace_function(name: Optional[str] = None):
    """
    Decorator for tracing function calls.
    
    Args:
        name: Optional name for the trace (defaults to function name)
        
    Returns:
        Decorator function
    """
    def decorator(func):
        """Actual decorator."""
        def wrapper(*args, **kwargs):
            """Function wrapper with optional tracing."""
            # For now, just execute the function
            # Can add Langfuse tracing here if needed
            return func(*args, **kwargs)
        return wrapper
    return decorator
