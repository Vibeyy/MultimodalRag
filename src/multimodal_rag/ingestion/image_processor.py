"""Image processor for OCR and preprocessing."""

from typing import Dict, Optional
from pathlib import Path
import easyocr
from PIL import Image, ImageEnhance
import numpy as np

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class ImageProcessor:
    """
    Image processor with OCR capabilities (PascalCase per coding standards).
    
    Uses EasyOCR for text extraction from images.
    """
    
    def __init__(self, languages: list = None):
        """
        Initialize image processor with OCR.
        
        Args:
            languages: List of language codes for OCR (default: ['en'])
        """
        self._languages = languages or ['en']
        self._reader = None
        logger.info(f"Initialized image processor with languages: {self._languages}")
    
    def _init_reader(self) -> None:
        """Initialize EasyOCR reader (lazy loading, private method)."""
        if self._reader is None:
            logger.info("Loading EasyOCR model...")
            self._reader = easyocr.Reader(self._languages, gpu=False)
            logger.info("EasyOCR model loaded successfully")
    
    def extract_text_from_image(self, image_path: Path) -> str:
        """
        Extract text from image using OCR.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text from image
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            Exception: If OCR processing fails
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Initialize reader if needed
            self._init_reader()
            
            # Preprocess image
            processed_image = self._preprocess_image(image_path)
            
            # Perform OCR
            result = self._reader.readtext(np.array(processed_image))
            
            # Extract text from results
            extracted_text = " ".join([text for (_, text, _) in result])
            
            logger.info(f"Extracted {len(extracted_text)} characters from {image_path.name}")
            return extracted_text
            
        except Exception as e:
            logger.error(f"Failed to extract text from {image_path}: {str(e)}")
            raise
    
    def extract_text_with_confidence(
        self,
        image_path: Path,
        min_confidence: float = 0.5
    ) -> Dict[str, any]:
        """
        Extract text from image with confidence scores.
        
        Args:
            image_path: Path to image file
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dictionary with text, confidence scores, and bounding boxes
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Initialize reader if needed
            self._init_reader()
            
            # Preprocess image
            processed_image = self._preprocess_image(image_path)
            
            # Perform OCR
            results = self._reader.readtext(np.array(processed_image))
            
            # Filter by confidence
            filtered_results = []
            full_text_parts = []
            
            for bbox, text, confidence in results:
                if confidence >= min_confidence:
                    filtered_results.append({
                        "text": text,
                        "confidence": confidence,
                        "bbox": bbox,
                    })
                    full_text_parts.append(text)
            
            return {
                "full_text": " ".join(full_text_parts),
                "results": filtered_results,
                "avg_confidence": np.mean([r["confidence"] for r in filtered_results]) if filtered_results else 0.0,
            }
            
        except Exception as e:
            logger.error(f"Failed to extract text with confidence from {image_path}: {str(e)}")
            raise
    
    def _preprocess_image(self, image_path: Path) -> Image.Image:
        """
        Preprocess image for better OCR accuracy (private method).
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed PIL Image
        """
        try:
            # Load image
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.5)
            
            return image
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed, using original: {str(e)}")
            return Image.open(image_path)
    
    def get_image_metadata(self, image_path: Path) -> Dict[str, any]:
        """
        Extract image metadata.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary containing image metadata
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            with Image.open(image_path) as img:
                metadata = {
                    "format": img.format,
                    "mode": img.mode,
                    "size": img.size,
                    "width": img.width,
                    "height": img.height,
                    "file_size_bytes": image_path.stat().st_size,
                }
                
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to extract metadata from {image_path}: {str(e)}")
            return {}
