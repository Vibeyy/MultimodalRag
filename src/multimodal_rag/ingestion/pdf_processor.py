"""PDF document processor for text and image extraction."""

from typing import List, Dict, Tuple
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class PdfProcessor:
    """
    PDF document processor (PascalCase per coding standards).
    
    Extracts text and images from PDF files using PyMuPDF.
    """
    
    def __init__(self):
        """Initialize PDF processor."""
        logger.info("Initialized PDF processor")
    
    def extract_text(self, pdf_path: Path) -> List[Dict[str, any]]:
        """
        Extract text from PDF document with page-level metadata.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of dictionaries containing page text and metadata
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If PDF processing fails
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        pages_data = []
        
        try:
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc, start=1):
                    # Extract text
                    text = page.get_text()
                    
                    # Clean text
                    cleaned_text = self._clean_text(text)
                    
                    if cleaned_text.strip():
                        pages_data.append({
                            "page_num": page_num,
                            "text": cleaned_text,
                            "metadata": {
                                "source_file": pdf_path.name,
                                "page_count": len(doc),
                                "file_path": str(pdf_path),
                            }
                        })
                        
            logger.info(f"Extracted text from {len(pages_data)} pages in {pdf_path.name}")
            return pages_data
            
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {str(e)}")
            raise
    
    def extract_images(
        self,
        pdf_path: Path,
        output_dir: Path,
        dpi: int = 200
    ) -> List[Dict[str, any]]:
        """
        Extract embedded images from PDF pages using PyMuPDF.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save extracted images
            dpi: Resolution for image extraction (not used, kept for compatibility)
            
        Returns:
            List of dictionaries containing image paths and metadata
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If image extraction fails
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        images_data = []
        
        try:
            with fitz.open(pdf_path) as doc:
                image_count = 0
                skipped_count = 0
                
                for page_num, page in enumerate(doc, start=1):
                    # Get list of images on this page
                    image_list = page.get_images(full=True)
                    
                    for img_index, img in enumerate(image_list):
                        try:
                            # Get image XREF
                            xref = img[0]
                            
                            # Extract image
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            image_ext = base_image["ext"]
                            
                            # Convert to PIL Image and save as PNG
                            pil_image = Image.open(io.BytesIO(image_bytes))
                            image_count += 1
                            image_path = output_dir / f"{pdf_path.stem}_page_{page_num}_img_{img_index + 1}.png"
                            pil_image.save(image_path, "PNG")
                            
                            images_data.append({
                                "page_num": page_num,
                                "image_path": image_path,
                                "metadata": {
                                    "source_file": pdf_path.name,
                                    "page_count": len(doc),
                                    "file_path": str(pdf_path),
                                    "original_format": image_ext,
                                }
                            })
                        except Exception as img_error:
                            # Skip images that cannot be decoded (e.g., JBIG2, JPX formats)
                            skipped_count += 1
                            logger.warning(
                                f"Skipped image {img_index + 1} on page {page_num} "
                                f"of {pdf_path.name}: {str(img_error)}"
                            )
                            continue
            
            if skipped_count > 0:
                logger.info(
                    f"Extracted {len(images_data)} images from {pdf_path.name} "
                    f"({skipped_count} skipped due to unsupported formats)"
                )
            else:
                logger.info(f"Extracted {len(images_data)} images from {pdf_path.name}")
            
            return images_data
            
        except Exception as e:
            logger.error(f"Failed to extract images from {pdf_path}: {str(e)}")
            # Return empty list instead of raising to allow text-only processing
            return []
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text (private method per coding standards).
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove control characters
        text = "".join(char for char in text if char.isprintable() or char.isspace())
        
        return text.strip()
    
    def get_metadata(self, pdf_path: Path) -> Dict[str, any]:
        """
        Extract PDF metadata.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary containing PDF metadata
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            with fitz.open(pdf_path) as doc:
                metadata = {
                    "title": doc.metadata.get("title", ""),
                    "author": doc.metadata.get("author", ""),
                    "subject": doc.metadata.get("subject", ""),
                    "keywords": doc.metadata.get("keywords", ""),
                    "creator": doc.metadata.get("creator", ""),
                    "producer": doc.metadata.get("producer", ""),
                    "page_count": len(doc),
                    "file_size_bytes": pdf_path.stat().st_size,
                }
                
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to extract metadata from {pdf_path}: {str(e)}")
            return {}
