"""Test retrieval components."""

import pytest
import numpy as np

from multimodal_rag.retrieval.vector_store import QdrantStore
from multimodal_rag.retrieval.retrievers import DenseRetriever, HybridRetriever
from multimodal_rag.ingestion.embedder import TextEmbedder


def test_qdrant_store_initialization():
    """Test Qdrant store initialization."""
    store = QdrantStore(collection_name="test_init")
    assert store._collection_name == "test_init"
    assert store._host == "localhost"
    assert store._port == 6333


def test_qdrant_create_collection(vector_store):
    """Test collection creation."""
    vector_store.create_collection(vector_size=1024, recreate=True)
    
    info = vector_store.get_collection_info()
    assert info["name"] == "test_collection"
    assert info["vector_size"] == 1024


def test_qdrant_insert_chunks(vector_store, sample_chunks):
    """Test chunk insertion."""
    vector_store.create_collection(vector_size=1024, recreate=True)
    
    count = vector_store.insert_chunks(sample_chunks)
    assert count == len(sample_chunks)
    
    info = vector_store.get_collection_info()
    assert info["points_count"] == count


def test_qdrant_search(vector_store, sample_chunks, text_embedder):
    """Test vector search."""
    # Setup
    vector_store.create_collection(vector_size=1024, recreate=True)
    vector_store.insert_chunks(sample_chunks)
    
    # Search
    query = "artificial intelligence"
    query_embedding = text_embedder.embed_text(query)
    
    results = vector_store.search(
        query_vector=query_embedding.tolist(),
        limit=2,
    )
    
    assert len(results) <= 2
    assert all("score" in r for r in results)
    assert all("text" in r for r in results)


def test_dense_retriever(vector_store, sample_chunks, text_embedder):
    """Test dense retriever."""
    # Setup
    vector_store.create_collection(vector_size=1024, recreate=True)
    vector_store.insert_chunks(sample_chunks)
    
    # Create retriever
    retriever = DenseRetriever(
        vector_store=vector_store,
        embedder=text_embedder,
        top_k=2,
    )
    
    # Retrieve
    results = retriever.retrieve("machine learning")
    
    assert len(results) <= 2
    assert all("text" in r for r in results)


def test_hybrid_retriever_rrf():
    """Test RRF fusion logic."""
    retriever = HybridRetriever(None, rrf_k=60)
    
    results_list = [
        [
            {"id": "1", "score": 0.9, "text": "first"},
            {"id": "2", "score": 0.8, "text": "second"},
        ],
        [
            {"id": "2", "score": 0.85, "text": "second"},
            {"id": "3", "score": 0.75, "text": "third"},
        ],
    ]
    
    fused = retriever._apply_rrf(results_list)
    
    assert len(fused) == 3  # Three unique documents
    assert all("fused_score" in r for r in fused)


def test_mmr_diversity():
    """Test MMR diversity algorithm."""
    retriever = HybridRetriever(None)
    
    results = [
        {"id": "1", "score": 0.9, "text": "machine learning algorithms"},
        {"id": "2", "score": 0.85, "text": "machine learning models"},
        {"id": "3", "score": 0.7, "text": "natural language processing"},
    ]
    
    diverse_results = retriever.apply_mmr(results, lambda_param=0.5, k=2)
    
    assert len(diverse_results) == 2
    # First should be most relevant
    assert diverse_results[0]["id"] == "1"
