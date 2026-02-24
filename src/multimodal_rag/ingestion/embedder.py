"""Embedding models for text and images."""

from typing import List, Union
import torch
from sentence_transformers import SentenceTransformer
import open_clip
from PIL import Image
from pathlib import Path
import numpy as np

from ..utils.config import EMBEDDING_DIM_TEXT, EMBEDDING_DIM_IMAGE
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class TextEmbedder:
    """
    Text embedding model using sentence-transformers (PascalCase per standards).
    
    Uses BAAI/bge-large-en-v1.5 for high-quality text embeddings.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        """
        Initialize text embedder.
        
        Args:
            model_name: HuggingFace model name
        """
        self._model_name = model_name
        self._model = None
        self._embedding_dim = EMBEDDING_DIM_TEXT
        logger.info(f"Initialized text embedder: {model_name}")
    
    def _load_model(self) -> None:
        """Load embedding model (lazy loading, private method)."""
        if self._model is None:
            logger.info(f"Loading text embedding model: {self._model_name}...")
            self._model = SentenceTransformer(self._model_name)
            logger.info("Text embedding model loaded successfully")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for single text.
        
        Args:
            text: Input text
            
        Returns:
            Numpy array of embedding vector
        """
        self._load_model()
        
        try:
            embedding = self._model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to embed text: {str(e)}")
            raise
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embedding vectors (shape: [num_texts, embedding_dim])
        """
        self._load_model()
        
        if not texts:
            return np.array([])
        
        try:
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 100
            )
            
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to embed batch: {str(e)}")
            raise
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self._embedding_dim


class ImageEmbedder:
    """
    Image embedding model using OpenCLIP (PascalCase per standards).
    
    Uses OpenCLIP ViT-L/14 for high-quality image embeddings.
    """
    
    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai"
    ):
        """
        Initialize image embedder.
        
        Args:
            model_name: OpenCLIP model architecture
            pretrained: Pretrained weights source
        """
        self._model_name = model_name
        self._pretrained = pretrained
        self._model = None
        self._preprocess = None
        self._embedding_dim = EMBEDDING_DIM_IMAGE
        logger.info(f"Initialized image embedder: {model_name}/{pretrained}")
    
    def _load_model(self) -> None:
        """Load embedding model (lazy loading, private method)."""
        if self._model is None:
            logger.info(f"Loading image embedding model: {self._model_name}...")
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                self._model_name,
                pretrained=self._pretrained
            )
            self._model.eval()
            logger.info("Image embedding model loaded successfully")
    
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
        
        self._load_model()
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self._preprocess(image).unsqueeze(0)
            
            # Generate embedding
            with torch.no_grad():
                embedding = self._model.encode_image(image_tensor)
                embedding = embedding.squeeze().numpy()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to embed image {image_path}: {str(e)}")
            raise
    
    def embed_batch(self, image_paths: List[Path], batch_size: int = 16) -> np.ndarray:
        """
        Generate embeddings for multiple images.
        
        Args:
            image_paths: List of image file paths
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embedding vectors (shape: [num_images, embedding_dim])
        """
        self._load_model()
        
        if not image_paths:
            return np.array([])
        
        all_embeddings = []
        
        try:
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                
                # Load and preprocess batch
                batch_tensors = []
                for img_path in batch_paths:
                    if img_path.exists():
                        image = Image.open(img_path).convert('RGB')
                        tensor = self._preprocess(image)
                        batch_tensors.append(tensor)
                
                if not batch_tensors:
                    continue
                
                # Stack and embed
                batch_tensor = torch.stack(batch_tensors)
                with torch.no_grad():
                    embeddings = self._model.encode_image(batch_tensor)
                    all_embeddings.append(embeddings.numpy())
            
            # Concatenate all batches
            if all_embeddings:
                result = np.vstack(all_embeddings)
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
