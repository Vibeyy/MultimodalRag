"""Configuration management for the multimodal RAG application."""

import os
from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()

# Constants (using UPPERCASE_WITH_UNDERSCORES per coding standards)
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
OPENAI_RATE_LIMIT = int(os.getenv("OPENAI_RATE_LIMIT", "60"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "10"))
EMBEDDING_DIM_TEXT = 1536  # OpenAI text-embedding-3-small dimension
EMBEDDING_DIM_IMAGE = 1536  # OpenAI compatible dimension


class AppConfig(BaseSettings):
    """Application configuration class (PascalCase per coding standards)."""
    
    # OpenAI API Configuration
    openai_api_key: str
    openai_model: str = "gpt-4o"
    openai_vision_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_rate_limit: int = OPENAI_RATE_LIMIT
    
    # Qdrant Configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "multimodal_rag"
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    
    # Firebase Configuration (for authentication and chat history)
    firebase_project_id: Optional[str] = None
    firebase_private_key: Optional[str] = None
    firebase_client_email: Optional[str] = None
    firebase_web_api_key: Optional[str] = None
    
    # Application Settings
    max_retries: int = MAX_RETRIES
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    top_k_retrieval: int = TOP_K_RETRIEVAL
    
    # Generation Settings
    allow_general_knowledge: bool = True
    retrieval_confidence_threshold: float = 0.3
    
    # PDF Processing Settings
    pdf_max_pages: Optional[int] = 100
    pdf_use_vision: Optional[str] = "true"
    pdf_vision_threshold: Optional[int] = 100
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def get_config() -> AppConfig:
    """
    Get application configuration with validation.
    
    Returns:
        AppConfig: Validated configuration object
        
    Raises:
        RuntimeError: If required environment variables are missing
    """
    try:
        config = AppConfig()
    except Exception as e:
        raise RuntimeError(
            f"Missing required environment variables. "
            f"Please check your .env file. Error: {str(e)}"
        ) from e
    
    # Validate critical secrets
    if not config.openai_api_key or config.openai_api_key == "your_openai_api_key_here":
        raise RuntimeError(
            "Missing OPENAI_API_KEY environment variable. "
            "Get your API key from https://platform.openai.com/api-keys"
        )
    
    return config


def validate_environment() -> bool:
    """
    Validate that all required environment variables are set.
    
    Returns:
        bool: True if environment is valid
        
    Raises:
        RuntimeError: If validation fails
    """
    try:
        config = get_config()
        return True
    except Exception as e:
        raise RuntimeError(f"Environment validation failed: {str(e)}") from e
