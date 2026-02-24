"""PDF document processor with smart Vision API usage."""

from typing import List, Dict, Tuple, Optional
from pathlib import Path
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image
import io
import base64
from openai import OpenAI

from ..utils.config import get_config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class PdfProcessor:
    """
    Smart PDF processor with hybrid text extraction (PascalCase per coding standards).
    
    Uses traditional text extraction first, only falls back to Vision API for:
    - Pages with little/no extractable text
    - Image-heavy pages
    - Scanned documents
    
    This saves costs and time for large documents.
    """
    
    def __init__(
        self,
        use_vision: bool = True,
        vision_fallback_threshold: int = 100,
        max_pages: Optional[int] = None,
    ):
        """
        Initialize PDF processor with smart extraction.
        
        Args:
            use_vision: Whether to use Vision API at all
            vision_fallback_threshold: Min chars needed to skip Vision (default: 100)
            max_pages: Maximum pages to process (None = all pages)
        """
        config = get_config()
        self._client = OpenAI(api_key=config.openai_api_key) if use_vision else None
        self._model = config.openai_vision_model if use_vision else None
        self._use_vision = use_vision
        self._vision_threshold = vision_fallback_threshold
        self._max_pages = max_pages
        logger.info(
            f"Initialized PDF processor - Vision: {use_vision}, "
            f"Threshold: {vision_fallback_threshold} chars, "
            f"Max pages: {max_pages or 'unlimited'}"
        )
    
    def _encode_image(self, image: Image.Image) -> str:
        """
        Encode PIL Image to base64 string (private method).
        
        Args:
            image: PIL Image object
            
        Returns:
            Base64 encoded image string
        """
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def extract_text(self, pdf_path: Path, page_range: Optional[Tuple[int, int]] = None) -> List[Dict[str, any]]:
        """
        Smart text extraction: tries traditional first, uses Vision only when needed.
        
        Args:
            pdf_path: Path to PDF file
            page_range: Optional (start_page, end_page) tuple (1-indexed)
            
        Returns:
            List of dictionaries containing page text and metadata
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If PDF processing fails
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        pages_data = []
        vision_pages = 0
        traditional_pages = 0
        
        try:
            with fitz.open(pdf_path) as doc:
                total_pages = len(doc)
                
                # Apply page limits
                start = (page_range[0] - 1) if page_range else 0
                end = min(page_range[1] if page_range else total_pages, self._max_pages or total_pages)
                
                if self._max_pages and (end - start) > self._max_pages:
                    end = start + self._max_pages
                    logger.warning(f"Limiting to {self._max_pages} pages (set max_pages to increase)")
                
                logger.info(f"Processing pages {start + 1}-{end} of {total_pages} from {pdf_path.name}")
                
                for page_num in range(start, end):
                    page = doc[page_num]
                    
                    # Try traditional text extraction first
                    traditional_text = page.get_text()
                    cleaned_text = self._clean_text(traditional_text)
                    
                    # Decide whether to use Vision API
                    text_length = len(cleaned_text.strip())
                    use_vision_for_page = (
                        self._use_vision and 
                        text_length < self._vision_threshold
                    )
                    
                    if use_vision_for_page:
                        # Use Vision API for this page
                        try:
                            vision_text = self._extract_page_with_vision(pdf_path, page_num + 1)
                            extracted_text = vision_text
                            method = "openai_vision"
                            vision_pages += 1
                        except Exception as e:
                            logger.warning(f"Vision failed for page {page_num + 1}, using traditional: {str(e)}")
                            extracted_text = cleaned_text
                            method = "traditional_fallback"
                            traditional_pages += 1
                    else:
                        # Use traditional extraction
                        extracted_text = cleaned_text
                        method = "traditional"
                        traditional_pages += 1
                    
                    if extracted_text and extracted_text.strip():
                        pages_data.append({
                            "page_num": page_num + 1,
                            "text": extracted_text,
                            "metadata": {
                                "source_file": pdf_path.name,
                                "page_count": total_pages,
                                "file_path": str(pdf_path),
                                "extraction_method": method,
                                "char_count": len(extracted_text),
                            }
                        })
                    
                    if (page_num + 1) % 10 == 0:
                        logger.info(f"Processed {page_num + 1}/{end} pages")
            
            logger.info(
                f"Extracted text from {len(pages_data)} pages: "
                f"{traditional_pages} traditional, {vision_pages} vision"
            )
            
            if vision_pages > 0:
                estimated_cost = vision_pages * 0.01
                logger.info(f"Estimated Vision API cost: ${estimated_cost:.2f}")
            
            return pages_data
            
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {str(e)}")
            raise
    
    def _extract_page_with_vision(self, pdf_path: Path, page_num: int) -> str:
        """
        Extract single page using Vision API (private method).
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (1-indexed)
            
        Returns:
            Extracted text
        """
        # Convert single page to image
        images = convert_from_path(
            str(pdf_path),
            dpi=200,
            first_page=page_num,
            last_page=page_num
        )
        
        if not images:
            raise ValueError(f"Failed to convert page {page_num}")
        
        # Encode image
        base64_image = self._encode_image(images[0])
        
        # Call Vision API
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all text from this PDF page. Include text in images, diagrams, and tables. Maintain structure."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=4096,
            temperature=0.0,
        )
        
        return response.choices[0].message.content
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text (private method).
        
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
    
    def estimate_cost(self, pdf_path: Path) -> Dict[str, any]:
        """
        Estimate processing cost before running.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with cost estimates
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            with fitz.open(pdf_path) as doc:
                total_pages = len(doc)
                pages_to_process = min(total_pages, self._max_pages or total_pages)
                
                # Quick sample: check first 5 pages
                sample_size = min(5, pages_to_process)
                low_text_pages = 0
                
                for page_num in range(sample_size):
                    page = doc[page_num]
                    text = page.get_text()
                    if len(self._clean_text(text).strip()) < self._vision_threshold:
                        low_text_pages += 1
                
                # Estimate vision pages needed
                vision_ratio = low_text_pages / sample_size
                estimated_vision_pages = int(pages_to_process * vision_ratio)
                estimated_traditional_pages = pages_to_process - estimated_vision_pages
                
                # Cost calculation
                vision_cost = estimated_vision_pages * 0.01  # ~$0.01 per image
                embedding_cost = pages_to_process * 0.0001  # ~$0.0001 per page
                total_cost = vision_cost + embedding_cost
                
                return {
                    "total_pages": total_pages,
                    "pages_to_process": pages_to_process,
                    "estimated_vision_pages": estimated_vision_pages,
                    "estimated_traditional_pages": estimated_traditional_pages,
                    "estimated_cost_usd": round(total_cost, 2),
                    "vision_cost_usd": round(vision_cost, 2),
                    "embedding_cost_usd": round(embedding_cost, 4),
                }
                
        except Exception as e:
            logger.error(f"Failed to estimate cost: {str(e)}")
            return {}
    
    def extract_images(
        self,
        pdf_path: Path,
        output_dir: Path,
        dpi: int = 200
    ) -> List[Dict[str, any]]:
        """
        Extract images by converting PDF pages to images.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save extracted images
            dpi: Resolution for image conversion
            
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
            # Get total page count
            with fitz.open(pdf_path) as doc:
                total_pages = len(doc)
            
            # Convert PDF pages to images
            logger.info(f"Converting {total_pages} pages from {pdf_path.name} to images...")
            images = convert_from_path(str(pdf_path), dpi=dpi)
            
            # Save each page as image
            for page_num, image in enumerate(images, start=1):
                image_path = output_dir / f"{pdf_path.stem}_page_{page_num}.png"
                image.save(image_path, "PNG")
                
                images_data.append({
                    "page_num": page_num,
                    "image_path": image_path,
                    "metadata": {
                        "source_file": pdf_path.name,
                        "page_count": total_pages,
                        "file_path": str(pdf_path),
                        "dpi": dpi,
                    }
                })
            
            logger.info(f"Extracted {len(images_data)} page images from {pdf_path.name}")
            return images_data
            
        except Exception as e:
            logger.error(f"Failed to extract images from {pdf_path}: {str(e)}")
            return []
    
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
