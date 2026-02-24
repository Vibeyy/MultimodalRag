"""Structured logging utility for the multimodal RAG application."""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path


class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging (PascalCase per standards)."""
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            str: JSON-formatted log entry
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, "query_id"):
            log_data["query_id"] = record.query_id
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
        if hasattr(record, "latency_ms"):
            log_data["latency_ms"] = record.latency_ms
        if hasattr(record, "metadata"):
            log_data["metadata"] = record.metadata
            
        return json.dumps(log_data)


def setup_logger(
    name: str = "multimodal_rag",
    level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Setup structured logger with JSON formatting.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type ("json" or "text")
        log_file: Optional file path for logging
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    if log_format == "json":
        console_handler.setFormatter(JsonFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
    
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)
    
    return logger


def log_query(
    logger: logging.Logger,
    query_id: str,
    query: str,
    user_id: Optional[str] = None,
) -> None:
    """
    Log user query with structured metadata.
    
    Args:
        logger: Logger instance
        query_id: Unique query identifier
        query: User query text
        user_id: Optional user identifier
    """
    extra = {"query_id": query_id}
    if user_id:
        extra["user_id"] = user_id
    
    logger.info(
        f"Query received: {query[:100]}...",
        extra=extra
    )


def log_retrieval_quality(
    logger: logging.Logger,
    query_id: str,
    num_chunks: int,
    avg_score: float,
    retrieval_method: str,
) -> None:
    """
    Log retrieval quality metrics.
    
    Args:
        logger: Logger instance
        query_id: Unique query identifier
        num_chunks: Number of chunks retrieved
        avg_score: Average relevance score
        retrieval_method: Method used (dense, bm25, hybrid)
    """
    logger.info(
        f"Retrieval completed: {num_chunks} chunks, avg_score={avg_score:.3f}",
        extra={
            "query_id": query_id,
            "metadata": {
                "num_chunks": num_chunks,
                "avg_score": avg_score,
                "retrieval_method": retrieval_method,
            }
        }
    )


def log_generation_metrics(
    logger: logging.Logger,
    query_id: str,
    token_count: int,
    latency_ms: float,
    has_citations: bool,
) -> None:
    """
    Log generation metrics.
    
    Args:
        logger: Logger instance
        query_id: Unique query identifier
        token_count: Number of tokens generated
        latency_ms: Generation latency in milliseconds
        has_citations: Whether response includes citations
    """
    logger.info(
        f"Generation completed: {token_count} tokens in {latency_ms:.2f}ms",
        extra={
            "query_id": query_id,
            "latency_ms": latency_ms,
            "metadata": {
                "token_count": token_count,
                "has_citations": has_citations,
            }
        }
    )


def log_error(
    logger: logging.Logger,
    error: Exception,
    context: Dict[str, Any],
) -> None:
    """
    Log error with context information.
    
    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Additional context dictionary
    """
    logger.error(
        f"Error occurred: {str(error)}",
        extra={"metadata": context},
        exc_info=True
    )
