"""Qdrant vector store wrapper for multimodal embeddings."""

from typing import List, Dict, Optional, Any
from pathlib import Path
import os
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    NamedVector,
    FieldCondition,
    MatchValue,
    SearchRequest,
)
import uuid

from ..utils.config import get_config, EMBEDDING_DIM_TEXT, EMBEDDING_DIM_IMAGE
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class QdrantStore:
    """
    Qdrant vector store manager (PascalCase per coding standards).
    
    Manages document storage and retrieval using Qdrant vector database.
    Supports both local and cloud deployments.
    """
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ):
        """
        Initialize Qdrant store.
        
        Args:
            collection_name: Name of the collection
            host: Qdrant server host (for local deployment)
            port: Qdrant server port (for local deployment)
        """
        config = get_config()
        
        self._collection_name = collection_name or config.qdrant_collection_name
        
        # Check for cloud configuration first
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if qdrant_url and qdrant_api_key:
            # Cloud configuration (Qdrant Cloud)
            self._client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
                timeout=60,
            )
            logger.info(f"Initialized Qdrant Cloud store: {qdrant_url}")
        else:
            # Local configuration (Docker or local instance)
            self._host = host or config.qdrant_host
            self._port = port or config.qdrant_port
            self._client = QdrantClient(host=self._host, port=self._port)
            logger.info(f"Initialized Qdrant local store: {self._host}:{self._port}")
        
        # Auto-ensure collection exists on initialization
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self) -> None:
        """
        Ensure collection exists, create if it doesn't.
        Called automatically on initialization and before operations.
        """
        try:
            collections = self._client.get_collections().collections
            exists = any(c.name == self._collection_name for c in collections)
            
            if not exists:
                logger.info(f"Collection doesn't exist, creating: {self._collection_name}")
                self.create_collection()
            else:
                logger.debug(f"Collection exists: {self._collection_name}")
        except Exception as e:
            logger.error(f"Failed to check/create collection: {str(e)}")
            raise
    
    def create_collection(
        self,
        vector_size: int = EMBEDDING_DIM_TEXT,
        distance: Distance = Distance.COSINE,
        recreate: bool = False,
    ) -> None:
        """
        Create vector collection in Qdrant with named vectors for multimodal support.
        
        Args:
            vector_size: Dimension of text embedding vectors (image is separate)
            distance: Distance metric (COSINE, EUCLID, DOT)
            recreate: Whether to recreate if exists
        """
        try:
            # Check if collection exists
            collections = self._client.get_collections().collections
            exists = any(c.name == self._collection_name for c in collections)
            
            if exists and recreate:
                logger.info(f"Deleting existing collection: {self._collection_name}")
                self._client.delete_collection(self._collection_name)
                exists = False
            
            if not exists:
                logger.info(f"Creating multimodal collection: {self._collection_name}")
                # Use named vectors to support both text and image embeddings
                self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config={
                        "text": VectorParams(size=EMBEDDING_DIM_TEXT, distance=distance),
                        "image": VectorParams(size=EMBEDDING_DIM_IMAGE, distance=distance),
                    },
                )
                logger.info(f"Collection created: {self._collection_name}")
            else:
                logger.info(f"Collection already exists: {self._collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}")
            raise
    
    def insert_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Insert document chunks into vector store.
        
        Args:
            chunks: List of chunk dictionaries with embeddings and metadata
            
        Returns:
            Number of chunks inserted
        """
        if not chunks:
            logger.warning("No chunks provided for insertion")
            return 0
        
        # Ensure collection exists before inserting
        self._ensure_collection_exists()
        
        try:
            points = []
            
            for chunk in chunks:
                # Generate unique ID
                point_id = str(uuid.uuid4())
                
                # Prepare payload (metadata + text)
                payload = {
                    "text": chunk.get("text", ""),
                    "chunk_type": chunk.get("chunk_type", "text"),
                    "source_file": chunk.get("metadata", {}).get("source_file", ""),
                    "page_num": chunk.get("metadata", {}).get("page_num", 0),
                    "document_id": chunk.get("metadata", {}).get("document_id", ""),
                    "created_at": chunk.get("metadata", {}).get("created_at", ""),
                    "tags": chunk.get("metadata", {}).get("tags", []),
                }
                
                # Add image-specific fields if present
                if "image_path" in chunk:
                    payload["image_path"] = chunk["image_path"]
                if "ocr_confidence" in chunk:
                    payload["ocr_confidence"] = chunk["ocr_confidence"]
                
                # Determine vector type (text or image)
                chunk_type = chunk.get("chunk_type", "text")
                embedding = chunk["embedding"]
                
                # Create named vector based on content type
                if chunk_type == "image":
                    vector = {"image": embedding}
                else:
                    vector = {"text": embedding}
                
                # Create point
                point = PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
                points.append(point)
            
            # Upsert points
            self._client.upsert(
                collection_name=self._collection_name,
                points=points,
            )
            
            logger.info(f"Inserted {len(points)} chunks into {self._collection_name}")
            return len(points)
            
        except Exception as e:
            logger.error(f"Failed to insert chunks: {str(e)}")
            raise
    
    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        vector_name: str = "text",
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using named vectors.
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filters: Optional metadata filters
            vector_name: Name of vector to search ("text" or "image")
            
        # Ensure collection exists before searching
        self._ensure_collection_exists()
        
        Returns:
            List of search results with scores and metadata
        """
        try:
            # Build filter if provided
            query_filter = None
            if filters:
                query_filter = self._build_filter(filters)
            
            # Perform search with named vector
            results = self._client.search(
                collection_name=self._collection_name,
                query_vector=NamedVector(name=vector_name, vector=query_vector),
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.id,
                    "score": result.score,
                    "text": result.payload.get("text", ""),
                    "chunk_type": result.payload.get("chunk_type", "text"),
                    "source_file": result.payload.get("source_file", ""),
                    "page_num": result.payload.get("page_num", 0),
                    "metadata": result.payload,
                })
            
            logger.info(f"Search returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise
    
    def _build_filter(self, filters: Dict[str, Any]) -> Filter:
        """
        Build Qdrant filter from dictionary (private method per standards).
        
        Args:
            filters: Filter conditions
            
        Returns:
            Qdrant Filter object
        """
        conditions = []
        
        # Filter by source file
        if "source_file" in filters:
            conditions.append(
                FieldCondition(
                    key="source_file",
                    match=MatchValue(value=filters["source_file"]),
                )
            )
        
        # Filter by chunk type
        if "chunk_type" in filters:
            conditions.append(
                FieldCondition(
                    key="chunk_type",
                    match=MatchValue(value=filters["chunk_type"]),
                )
            )
        
        # Filter by tags
        if "tags" in filters and filters["tags"]:
            for tag in filters["tags"]:
                conditions.append(
                    FieldCondition(
                        key="tags",
                        match=MatchValue(value=tag),
                    )
                )
        
        return Filter(must=conditions) if conditions else None
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get collection information and statistics.
        
        Returns:
            Dictionary with collection info
        """
        try:
            info = self._client.get_collection(self._collection_name)
            return {
                "name": info.config.name,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance,
                "points_count": info.points_count,
                "status": info.status,
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {str(e)}")
            return {}
    
    def delete_collection(self) -> None:
        """Delete the collection."""
        try:
            self._client.delete_collection(self._collection_name)
            logger.info(f"Deleted collection: {self._collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {str(e)}")
            raise
