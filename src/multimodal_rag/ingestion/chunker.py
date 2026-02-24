"""Text chunking strategies for RAG."""

from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..utils.config import CHUNK_SIZE, CHUNK_OVERLAP
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class SemanticChunker:
    """
    Semantic text chunker (PascalCase per coding standards).
    
    Uses recursive character splitting for optimal chunk creation.
    """
    
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ):
        """
        Initialize semantic chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        
        # Initialize text splitter
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        
        logger.info(
            f"Initialized semantic chunker: size={chunk_size}, overlap={chunk_overlap}"
        )
    
    def chunk_text(
        self,
        text: str,
        metadata: Dict[str, any] = None
    ) -> List[Dict[str, any]]:
        """
        Chunk text into semantic segments.
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        try:
            # Split text into chunks
            chunks = self._splitter.split_text(text)
            
            # Create chunk dictionaries
            chunk_dicts = []
            for chunk_index, chunk in enumerate(chunks):
                chunk_dict = {
                    "text": chunk,
                    "chunk_index": chunk_index,
                    "chunk_size": len(chunk),
                    "metadata": metadata or {},
                }
                chunk_dicts.append(chunk_dict)
            
            logger.info(f"Created {len(chunk_dicts)} chunks from text")
            return chunk_dicts
            
        except Exception as e:
            logger.error(f"Failed to chunk text: {str(e)}")
            raise
    
    def chunk_documents(
        self,
        documents: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of document dictionaries with 'text' and 'metadata' keys
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for doc_index, doc in enumerate(documents):
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            
            # Add document index to metadata
            metadata["document_index"] = doc_index
            
            # Chunk document
            chunks = self.chunk_text(text, metadata)
            all_chunks.extend(chunks)
        
        logger.info(
            f"Chunked {len(documents)} documents into {len(all_chunks)} chunks"
        )
        return all_chunks
    
    def get_chunk_stats(self, chunks: List[Dict[str, any]]) -> Dict[str, any]:
        """
        Calculate statistics about chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
            }
        
        chunk_sizes = [chunk["chunk_size"] for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
        }
