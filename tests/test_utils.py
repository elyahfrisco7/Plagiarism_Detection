# tests/test_utils.py
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from app.utils import compute_cosine_similarity, extract_text_from_pdf

def test_compute_cosine_similarity_identical():
    vec = [1, 0, 0]
    assert compute_cosine_similarity(vec, vec) == pytest.approx(1.0)

def test_compute_cosine_similarity_orthogonal():
    vec1 = [1, 0, 0]
    vec2 = [0, 1, 0]
    assert compute_cosine_similarity(vec1, vec2) == pytest.approx(0.0)

def test_compute_cosine_similarity_opposite():
    vec1 = [1, 0, 0]
    vec2 = [-1, 0, 0]
    assert compute_cosine_similarity(vec1, vec2) == pytest.approx(-1.0)

def test_compute_cosine_similarity_zero_vector():
    vec1 = [0, 0, 0]
    vec2 = [1, 2, 3]
    assert compute_cosine_similarity(vec1, vec2) == 0.0

@patch("PyPDF2.PdfReader")
@patch("builtins.open", new_callable=MagicMock)
def test_extract_text_from_pdf(mock_open, mock_pdf_reader):
    # Setup mock
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Hello World"
    
    mock_reader_instance = mock_pdf_reader.return_value
    mock_reader_instance.pages = [mock_page]
    
    result = extract_text_from_pdf("dummy.pdf")
    
    assert "Hello World" in result
    mock_open.assert_called_once_with("dummy.pdf", "rb")
