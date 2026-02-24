"""Test ingestion pipeline components."""

import pytest
from pathlib import Path

from multimodal_rag.ingestion.chunker import SemanticChunker
from multimodal_rag.ingestion.embedder import TextEmbedder


def test_semantic_chunker_initialization():
    """Test semantic chunker initialization."""
    chunker = SemanticChunker(chunk_size=512, chunk_overlap=50)
    assert chunker._chunk_size == 512
    assert chunker._chunk_overlap == 50


def test_semantic_chunker_chunk_text(sample_text):
    """Test text chunking."""
    chunker = SemanticChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_text(sample_text)
    
    assert len(chunks) > 0
    assert all("text" in chunk for chunk in chunks)
    assert all("chunk_index" in chunk for chunk in chunks)


def test_semantic_chunker_empty_text():
    """Test chunking with empty text."""
    chunker = SemanticChunker()
    chunks = chunker.chunk_text("")
    
    assert len(chunks) == 0


def test_text_embedder_initialization():
    """Test text embedder initialization."""
    embedder = TextEmbedder()
    assert embedder._model_name == "BAAI/bge-large-en-v1.5"
    assert embedder.embedding_dim == 1024


def test_text_embedder_embed_single(sample_text):
    """Test single text embedding."""
    embedder = TextEmbedder()
    embedding = embedder.embed_text(sample_text)
    
    assert embedding.shape[0] == 1024
    assert embedding.dtype == "float32" or embedding.dtype == "float64"


def test_text_embedder_embed_batch():
    """Test batch text embedding."""
    embedder = TextEmbedder()
    texts = [
        "This is the first text.",
        "This is the second text.",
        "This is the third text.",
    ]
    
    embeddings = embedder.embed_batch(texts)
    
    assert embeddings.shape[0] == 3
    assert embeddings.shape[1] == 1024


def test_chunk_stats():
    """Test chunk statistics calculation."""
    chunker = SemanticChunker()
    chunks = [
        {"text": "a" * 100, "chunk_size": 100},
        {"text": "b" * 200, "chunk_size": 200},
        {"text": "c" * 150, "chunk_size": 150},
    ]
    
    stats = chunker.get_chunk_stats(chunks)
    
    assert stats["total_chunks"] == 3
    assert stats["avg_chunk_size"] == 150.0
    assert stats["min_chunk_size"] == 100
    assert stats["max_chunk_size"] == 200
