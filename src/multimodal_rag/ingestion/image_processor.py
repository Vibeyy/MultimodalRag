"""Image processor using OpenAI Vision API."""

from typing import Dict, Optional
from pathlib import Path
import base64
from openai import OpenAI
from PIL import Image

from ..utils.config import get_config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class ImageProcessor:
    """
    Image processor using OpenAI Vision API (PascalCase per coding standards).
    
    Uses GPT-4 Vision for text extraction from images.
    """
    
    def __init__(self):
        """Initialize image processor with OpenAI client."""
        config = get_config()
        self._client = OpenAI(api_key=config.openai_api_key)
        self._model = config.openai_vision_model
        logger.info(f"Initialized OpenAI image processor with model: {self._model}")
    
    def _encode_image(self, image_path: Path) -> str:
        """
        Encode image to base64 string (private method).
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded image string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def extract_text_from_image(self, image_path: Path) -> str:
        """
        Extract text from image using OpenAI Vision API.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text from image
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            Exception: If API processing fails
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Encode image
            base64_image = self._encode_image(image_path)
            
            # Call OpenAI Vision API
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract all text from this image. Return only the text content, maintaining the original structure and formatting as much as possible."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4096,
                temperature=0.0,
            )
            
            extracted_text = response.choices[0].message.content
            
            logger.info(f"Extracted {len(extracted_text)} characters from {image_path.name}")
            return extracted_text
            
        except Exception as e:
            logger.error(f"Failed to extract text from {image_path}: {str(e)}")
            raise
    
    def extract_text_with_description(
        self,
        image_path: Path,
        include_description: bool = True
    ) -> Dict[str, any]:
        """
        Extract text and optional description from image.
        
        Args:
            image_path: Path to image file
            include_description: Whether to include image description
            
        Returns:
            Dictionary with text and optional description
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Encode image
            base64_image = self._encode_image(image_path)
            
            # Prepare prompt based on options
            if include_description:
                prompt = """Analyze this image and provide:
1. All text present in the image
2. A brief description of what the image shows (charts, diagrams, photos, etc.)

Format your response as:
TEXT: [extracted text]
DESCRIPTION: [image description]"""
            else:
                prompt = "Extract all text from this image. Return only the text content."
            
            # Call OpenAI Vision API
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4096,
                temperature=0.0,
            )
            
            content = response.choices[0].message.content
            
            # Parse response
            if include_description:
                parts = content.split("DESCRIPTION:")
                text_part = parts[0].replace("TEXT:", "").strip()
                description = parts[1].strip() if len(parts) > 1 else ""
                
                return {
                    "text": text_part,
                    "description": description,
                    "full_content": content,
                }
            else:
                return {
                    "text": content,
                    "description": "",
                    "full_content": content,
                }
            
        except Exception as e:
            logger.error(f"Failed to extract text with description from {image_path}: {str(e)}")
            raise
    
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
