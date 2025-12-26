# tests/test_controllers.py
import pytest
from app.controllers import compute_global_similarity, compute_similarity_breakdown

def test_compute_similarity_breakdown(two_mock_theses):
    thesis1, thesis2 = two_mock_theses
    sims, overall = compute_similarity_breakdown(thesis1, thesis2)
    
    assert isinstance(sims, dict)
    assert "theme" in sims
    assert "images" in sims
    assert 0 <= overall <= 100

def test_compute_global_similarity(two_mock_theses):
    thesis1, thesis2 = two_mock_theses
    overall = compute_global_similarity(thesis1, thesis2)
    
    assert 0 <= overall <= 100
    # Since they are identical in embeddings (from fixture), similarity should be 100%
    assert overall == pytest.approx(100.0)
