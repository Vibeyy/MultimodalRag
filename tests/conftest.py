"""Pytest configuration and shared fixtures."""

import pytest
from pathlib import Path
import tempfile
import shutil

from multimodal_rag.ingestion.pipeline import IngestionPipeline
from multimodal_rag.retrieval.vector_store import QdrantStore
from multimodal_rag.ingestion.embedder import TextEmbedder
from multimodal_rag.generation.generator import OpenAIGenerator


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def text_embedder():
    """Create text embedder instance."""
    return TextEmbedder()


@pytest.fixture(scope="session")
def vector_store():
    """Create vector store instance for testing."""
    store = QdrantStore(collection_name="test_collection")
    yield store
    # Cleanup
    try:
        store.delete_collection()
    except Exception:
        pass


@pytest.fixture(scope="session")
def ingestion_pipeline(test_data_dir):
    """Create ingestion pipeline for testing."""
    return IngestionPipeline(output_dir=test_data_dir / "processed")


@pytest.fixture(scope="function")
def sample_pdf(test_data_dir):
    """Create a sample PDF file for testing."""
    # Note: In real tests, you'd create or copy an actual PDF
    pdf_path = test_data_dir / "sample.pdf"
    # For now, just return a path (actual PDF creation requires reportlab or similar)
    return pdf_path


@pytest.fixture(scope="function")
def sample_text():
    """Sample text for testing."""
    return """
    Artificial Intelligence (AI) has revolutionized many industries.
    Machine learning algorithms can now process vast amounts of data.
    Natural language processing enables computers to understand human language.
    """


@pytest.fixture(scope="function")
def sample_chunks():
    """Sample document chunks for testing."""
    return [
        {
            "text": "Artificial intelligence is transforming healthcare.",
            "embedding": [0.1] * 1024,
            "metadata": {
                "source_file": "test.pdf",
                "page_num": 1,
                "chunk_type": "text",
                "document_id": "test-doc-1",
                "created_at": "2024-01-01T00:00:00Z",
                "tags": ["test"],
            },
            "chunk_type": "text",
        },
        {
            "text": "Machine learning models require large datasets.",
            "embedding": [0.2] * 1024,
            "metadata": {
                "source_file": "test.pdf",
                "page_num": 2,
                "chunk_type": "text",
                "document_id": "test-doc-1",
                "created_at": "2024-01-01T00:00:00Z",
                "tags": ["test"],
            },
            "chunk_type": "text",
        },
    ]
