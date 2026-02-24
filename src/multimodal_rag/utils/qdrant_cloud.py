"""Qdrant Cloud configuration helper."""

import os
from qdrant_client import QdrantClient


def get_qdrant_client():
    """
    Get Qdrant client configured for cloud or local deployment.
    
    Returns:
        QdrantClient: Configured Qdrant client instance
    """
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if qdrant_url and qdrant_api_key:
        # Cloud configuration (Qdrant Cloud)
        return QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60,
        )
    else:
        # Local configuration (Docker or local instance)
        return QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333)),
        )
