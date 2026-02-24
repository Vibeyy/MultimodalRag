"""Configuration management for the multimodal RAG application."""

import os
from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()

# Constants (using UPPERCASE_WITH_UNDERSCORES per coding standards)
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
GEMINI_RATE_LIMIT = int(os.getenv("GEMINI_RATE_LIMIT", "60"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "10"))
EMBEDDING_DIM_TEXT = 1024  # bge-large-en-v1.5 dimension
EMBEDDING_DIM_IMAGE = 768  # OpenCLIP ViT-L/14 dimension


class AppConfig(BaseSettings):
    """Application configuration class (PascalCase per coding standards)."""
    
    # Gemini API Configuration
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash"
    gemini_rate_limit: int = GEMINI_RATE_LIMIT
    
    # OpenAI API Configuration (for RAGAS evaluation)
    openai_api_key: Optional[str] = None
    
    # Qdrant Configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "multimodal_rag"
    
    # Langfuse Configuration
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    
    # Application Settings
    max_retries: int = MAX_RETRIES
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    top_k_retrieval: int = TOP_K_RETRIEVAL
    
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
    if not config.gemini_api_key or config.gemini_api_key == "your_gemini_api_key_here":
        raise RuntimeError(
            "Missing GEMINI_API_KEY environment variable. "
            "Get your free API key from https://makersuite.google.com/app/apikey"
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
