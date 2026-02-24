"""Test generation components."""

import pytest

from multimodal_rag.generation.prompts import PromptBuilder
from multimodal_rag.generation.hallucination_detector import HallucinationDetector


def test_prompt_builder_initialization():
    """Test prompt builder initialization."""
    builder = PromptBuilder()
    assert builder is not None


def test_build_prompt_with_context():
    """Test prompt building with context."""
    builder = PromptBuilder()
    
    query = "What is machine learning?"
    context = [
        {
            "text": "Machine learning is a subset of AI.",
            "source_file": "ml_guide.pdf",
            "page_num": 1,
            "chunk_type": "text",
        },
        {
            "text": "It uses algorithms to learn from data.",
            "source_file": "ml_guide.pdf",
            "page_num": 2,
            "chunk_type": "text",
        },
    ]
    
    prompt = builder.build_prompt_with_context(query, context)
    
    assert "Question: What is machine learning?" in prompt
    assert "Machine learning is a subset of AI" in prompt
    assert "Source: ml_guide.pdf" in prompt
    assert "cite" in prompt.lower() or "citation" in prompt.lower()


def test_build_prompt_no_context():
    """Test prompt building without context."""
    builder = PromptBuilder()
    
    query = "What is AI?"
    prompt = builder.build_prompt_with_context(query, [])
    
    assert "Question: What is AI?" in prompt
    assert "No context available" in prompt


def test_query_expansion_prompt():
    """Test query expansion prompt."""
    builder = PromptBuilder()
    
    query = "How does neural network work?"
    prompt = builder.build_query_expansion_prompt(query, num_variants=3)
    
    assert query in prompt
    assert "3" in prompt or "three" in prompt.lower()
    assert "alternative" in prompt.lower()


def test_hallucination_check_prompt():
    """Test hallucination check prompt."""
    builder = PromptBuilder()
    
    query = "What is AI?"
    answer = "AI is artificial intelligence [Source: ai.pdf, Page: 1]"
    context = "Artificial intelligence is the simulation of human intelligence."
    
    prompt = builder.build_hallucination_check_prompt(query, answer, context)
    
    assert query in prompt
    assert answer in prompt
    assert "supported" in prompt.lower() or "verify" in prompt.lower()


def test_citation_extraction():
    """Test citation pattern extraction."""
    import re
    
    text = "AI is growing [Source: report.pdf, Page: 5]. ML is a subset [Source: guide.pdf, Page: 2]."
    pattern = r'\[Source:\s*([^,]+),\s*Page:\s*(\d+)\]'
    
    matches = re.findall(pattern, text)
    
    assert len(matches) == 2
    assert matches[0] == ("report.pdf", "5")
    assert matches[1] == ("guide.pdf", "2")


def test_hallucination_detector_check_citations():
    """Test citation checking."""
    detector = HallucinationDetector()
    
    text_with_citation = "This is a fact [Source: doc.pdf, Page: 1]."
    text_without_citation = "This is a fact without citation."
    
    assert detector._check_citations(text_with_citation) == True
    assert detector._check_citations(text_without_citation) == False
