"""Embedding models using OpenAI API."""

from typing import List
from pathlib import Path
import numpy as np
from openai import OpenAI

from ..utils.config import get_config, EMBEDDING_DIM_TEXT, EMBEDDING_DIM_IMAGE
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class TextEmbedder:
    """
    Text embedding model using OpenAI (PascalCase per standards).
    
    Uses OpenAI text-embedding-3-small for high-quality text embeddings.
    """
    
    def __init__(self):
        """Initialize text embedder with OpenAI client."""
        config = get_config()
        self._client = OpenAI(api_key=config.openai_api_key)
        self._model = config.openai_embedding_model
        self._embedding_dim = EMBEDDING_DIM_TEXT
        logger.info(f"Initialized OpenAI text embedder: {self._model}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for single text.
        
        Args:
            text: Input text
            
        Returns:
            Numpy array of embedding vector
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return np.zeros(self._embedding_dim)
        
        try:
            response = self._client.embeddings.create(
                input=text,
                model=self._model
            )
            
            embedding = np.array(response.data[0].embedding)
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to embed text: {str(e)}")
            raise
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            batch_size: Maximum batch size (OpenAI supports up to 2048)
            
        Returns:
            Numpy array of embedding vectors (shape: [num_texts, embedding_dim])
        """
        if not texts:
            return np.array([])
        
        # Filter out empty texts
        filtered_texts = [t if t and t.strip() else " " for t in texts]
        
        all_embeddings = []
        
        try:
            # Process in batches
            for i in range(0, len(filtered_texts), batch_size):
                batch = filtered_texts[i:i + batch_size]
                
                response = self._client.embeddings.create(
                    input=batch,
                    model=self._model
                )
                
                # Extract embeddings in order
                batch_embeddings = [np.array(item.embedding) for item in response.data]
                all_embeddings.extend(batch_embeddings)
            
            result = np.array(all_embeddings)
            logger.info(f"Generated embeddings for {len(result)} texts")
            return result
            
        except Exception as e:
            logger.error(f"Failed to embed batch: {str(e)}")
            raise
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self._embedding_dim


class ImageEmbedder:
    """
    Image embedding using OpenAI Vision + Embeddings (PascalCase per standards).
    
    Extracts text from images using Vision API, then embeds the text.
    """
    
    def __init__(self):
        """Initialize image embedder with OpenAI client."""
        config = get_config()
        self._client = OpenAI(api_key=config.openai_api_key)
        self._vision_model = config.openai_vision_model
        self._embedding_model = config.openai_embedding_model
        self._embedding_dim = EMBEDDING_DIM_IMAGE
        logger.info(f"Initialized OpenAI image embedder with vision: {self._vision_model}")
    
    def _extract_image_description(self, image_path: Path) -> str:
        """
        Extract description from image using Vision API (private method).
        
        Args:
            image_path: Path to image file
            
        Returns:
            Text description of the image
        """
        import base64
        
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        response = self._client.chat.completions.create(
            model=self._vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in detail, including any text, diagrams, charts, or visual elements. Be comprehensive."
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
            max_tokens=1000,
            temperature=0.0,
        )
        
        return response.choices[0].message.content
    
    def embed_image(self, image_path: Path) -> np.ndarray:
        """
        Generate embedding for single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Numpy array of embedding vector
            
        Raises:
            FileNotFoundError: If image file doesn't exist
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Extract description from image
            description = self._extract_image_description(image_path)
            
            # Embed the description
            response = self._client.embeddings.create(
                input=description,
                model=self._embedding_model
            )
            
            embedding = np.array(response.data[0].embedding)
            logger.info(f"Generated embedding for image: {image_path.name}")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to embed image {image_path}: {str(e)}")
            raise
    
    def embed_batch(self, image_paths: List[Path], batch_size: int = 10) -> np.ndarray:
        """
        Generate embeddings for multiple images.
        
        Args:
            image_paths: List of image file paths
            batch_size: Number of images to process at once
            
        Returns:
            Numpy array of embedding vectors (shape: [num_images, embedding_dim])
        """
        if not image_paths:
            return np.array([])
        
        all_embeddings = []
        
        try:
            for i, img_path in enumerate(image_paths):
                if img_path.exists():
                    embedding = self.embed_image(img_path)
                    all_embeddings.append(embedding)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i + 1}/{len(image_paths)} images")
            
            if all_embeddings:
                result = np.array(all_embeddings)
                logger.info(f"Generated embeddings for {len(result)} images")
                return result
            else:
                return np.array([])
                
        except Exception as e:
            logger.error(f"Failed to embed batch: {str(e)}")
            raise
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self._embedding_dim
