"""Main ingestion pipeline orchestrating all processing components."""

from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
import uuid

from .pdf_processor import PdfProcessor
from .image_processor import ImageProcessor
from .chunker import SemanticChunker
from .embedder import TextEmbedder, ImageEmbedder
from ..utils.logger import setup_logger
from ..utils.config import get_config

logger = setup_logger(__name__)


class IngestionPipeline:
    """
    Main ingestion pipeline (PascalCase per coding standards).
    
    Orchestrates PDF processing, OCR, chunking, and embedding generation.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize ingestion pipeline.
        
        Args:
            output_dir: Directory for processed outputs
        """
        config = get_config()
        self._output_dir = output_dir or Path("data/processed")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize processors
        self._pdf_processor = PdfProcessor()
        self._image_processor = ImageProcessor()
        self._chunker = SemanticChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        self._text_embedder = TextEmbedder()
        self._image_embedder = ImageEmbedder()
        
        logger.info("Initialized ingestion pipeline")
    
    def process_document(
        self,
        file_path: Path,
        enable_image_extraction: bool = False,
        tags: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Process a single document (PDF or image).
        
        Args:
            file_path: Path to document file
            enable_image_extraction: Whether to extract and process images (expensive for text PDFs)
            tags: Optional tags for categorization
            
        Returns:
            Dictionary containing processed chunks with embeddings and metadata
            
        Raises:
            ValueError: If file type is not supported
            FileNotFoundError: If file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_suffix = file_path.suffix.lower()
        
        if file_suffix == '.pdf':
            return self._process_pdf(file_path, tags, enable_image_extraction)
        elif file_suffix in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            return self._process_image(file_path, tags)
        elif file_suffix in ['.txt', '.md']:
            return self._process_text(file_path, tags)
        else:
            raise ValueError(f"Unsupported file type: {file_suffix}")
    
    def _process_text(
        self,
        text_path: Path,
        tags: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Process plain text file (private method per standards).
        
        Args:
            text_path: Path to text file
            tags: Optional tags
            
        Returns:
            Processed document data
        """
        logger.info(f"Processing text file: {text_path.name}")
        document_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat() + "Z"
        
        # Read text content
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Chunk the text
        text_chunks = self._chunker.chunk_text(text)
        
        all_chunks = []
        for i, chunk_dict in enumerate(text_chunks):
            # Extract text from chunk dictionary
            chunk_text = chunk_dict["text"]
            
            # Generate embedding
            embedding = self._text_embedder.embed_text(chunk_text)
            
            chunk_data = {
                "id": f"{document_id}_chunk_{i}",
                "document_id": document_id,
                "chunk_index": i,
                "text": chunk_text,
                "chunk_type": "text",
                "embedding": embedding,
                "metadata": {
                    "source_file": text_path.name,
                    "file_type": "text",
                    "page_num": 1,
                    "total_pages": 1,
                    "tags": tags or [],
                    "created_at": created_at,
                    "document_id": document_id
                }
            }
            all_chunks.append(chunk_data)
        
        return {
            "document_id": document_id,
            "source_file": text_path.name,
            "chunks": all_chunks,
            "total_chunks": len(text_chunks),
            "text_chunks": len(text_chunks),
            "image_chunks": 0
        }
    
    def _process_pdf(
        self,
        pdf_path: Path,
        tags: Optional[List[str]] = None,
        enable_image_extraction: bool = False
    ) -> Dict[str, any]:
        """
        Process PDF document (private method per standards).
        
        Args:
            pdf_path: Path to PDF file
            tags: Optional tags
            
        Returns:
            Processed document data
        """
        logger.info(f"Processing PDF: {pdf_path.name}")
        document_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat() + "Z"
        
        # Extract text from pages
        pages_data = self._pdf_processor.extract_text(pdf_path)
        
        # Chunk text
        all_chunks = []
        for page_data in pages_data:
            page_chunks = self._chunker.chunk_text(
                text=page_data["text"],
                metadata={
                    "source_file": pdf_path.name,
                    "page_num": page_data["page_num"],
                    "chunk_type": "text",
                    "document_id": document_id,
                    "created_at": created_at,
                    "tags": tags or [],
                }
            )
            all_chunks.extend(page_chunks)
        
        # Generate text embeddings
        chunk_texts = [chunk["text"] for chunk in all_chunks]
        embeddings = self._text_embedder.embed_batch(chunk_texts)
        
        # Attach embeddings to chunks
        for chunk, embedding in zip(all_chunks, embeddings):
            chunk["embedding"] = embedding.tolist()
            chunk["embedding_model"] = "bge-large-en-v1.5"
        
        # Extract images (optional - expensive for text-only PDFs)
        image_chunks = []
        ocr_text_chunks = []
        
        if enable_image_extraction:
            images_dir = self._output_dir / "images" / pdf_path.stem
            images_data = self._pdf_processor.extract_images(pdf_path, images_dir)
            
            # Process extracted images
        
            for img_idx, img_data in enumerate(images_data):
                # Extract text from image via OCR
                ocr_text = self._image_processor.extract_text_from_image(
                    img_data["image_path"]
                )
                
                # Generate image embedding (for visual similarity search)
                image_embedding = self._image_embedder.embed_image(img_data["image_path"])
            
                # Create image chunk (stores visual features)
                image_chunk = {
                "text": ocr_text,
                "image_path": str(img_data["image_path"]),
                "page_num": img_data["page_num"],
                "chunk_type": "image",
                "embedding": image_embedding.tolist(),
                "embedding_model": "openclip-vit-l-14",
                "metadata": {
                    "source_file": pdf_path.name,
                    "page_num": img_data["page_num"],
                    "chunk_type": "image",
                    "document_id": document_id,
                    "created_at": created_at,
                        "tags": tags or [],
                    }
                }
                image_chunks.append(image_chunk)
                
                # ALSO create text chunk from OCR (for semantic text search)
                # Only if OCR extracted meaningful text
                if ocr_text.strip():
                    text_embedding = self._text_embedder.embed_text(ocr_text)
                    
                    ocr_text_chunk = {
                    "text": ocr_text,
                    "image_path": str(img_data["image_path"]),
                    "page_num": img_data["page_num"],
                    "chunk_type": "text",  # Text type for text search!
                    "embedding": text_embedding.tolist(),
                    "embedding_model": "bge-large-en-v1.5",
                    "source_type": "ocr",  # Mark this as OCR-derived
                    "metadata": {
                        "source_file": pdf_path.name,
                        "page_num": img_data["page_num"],
                        "chunk_type": "text",
                        "source_type": "ocr",
                        "document_id": document_id,
                        "created_at": created_at,
                            "tags": tags or [],
                        }
                    }
                    ocr_text_chunks.append(ocr_text_chunk)
        
        # Combine all chunks
        all_chunks.extend(image_chunks)
        all_chunks.extend(ocr_text_chunks)
        
        logger.info(
            f"Processed {pdf_path.name}: "
            f"{len(chunk_texts)} native text chunks, "
            f"{len(ocr_text_chunks)} OCR text chunks, "
            f"{len(image_chunks)} image chunks"
        )
        
        return {
            "document_id": document_id,
            "source_file": pdf_path.name,
            "file_path": str(pdf_path),
            "total_chunks": len(all_chunks),
            "text_chunks": len(chunk_texts) + len(ocr_text_chunks),
            "image_chunks": len(image_chunks),
            "chunks": all_chunks,
            "created_at": created_at,
        }
    
    def _process_image(
        self,
        image_path: Path,
        tags: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Process standalone image file (private method per standards).
        
        Args:
            image_path: Path to image file
            tags: Optional tags
            
        Returns:
            Processed image data
        """
        logger.info(f"Processing image: {image_path.name}")
        document_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat() + "Z"
        
        # Extract text via OCR
        ocr_text = self._image_processor.extract_text_from_image(image_path)
        
        # Generate image embedding
        image_embedding = self._image_embedder.embed_image(image_path)
        
        # Create chunk
        chunk = {
            "text": ocr_text,
            "image_path": str(image_path),
            "chunk_type": "image",
            "embedding": image_embedding.tolist(),
            "embedding_model": "openclip-vit-l-14",
            "metadata": {
                "source_file": image_path.name,
                "chunk_type": "image",
                "document_id": document_id,
                "created_at": created_at,
                "tags": tags or [],
            }
        }
        
        logger.info(f"Processed image: {image_path.name}")
        
        return {
            "document_id": document_id,
            "source_file": image_path.name,
            "file_path": str(image_path),
            "total_chunks": 1,
            "text_chunks": 0,
            "image_chunks": 1,
            "chunks": [chunk],
            "created_at": created_at,
        }
    
    def process_batch(
        self,
        file_paths: List[Path],
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, any]]:
        """
        Process multiple documents in batch.
        
        Args:
            file_paths: List of file paths to process
            tags: Optional tags for all documents
            
        Returns:
            List of processed document data
        """
        results = []
        
        for file_path in file_paths:
            try:
                result = self.process_document(file_path, tags)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")
                continue
        
        total_chunks = sum(r["total_chunks"] for r in results)
        logger.info(
            f"Batch processing complete: "
            f"{len(results)}/{len(file_paths)} documents, "
            f"{total_chunks} total chunks"
        )
        
        return results
