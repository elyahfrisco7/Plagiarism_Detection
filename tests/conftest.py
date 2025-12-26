# tests/conftest.py
import pytest
import json
import numpy as np
from app import create_app
from app.models import Base, engine, SessionLocal, Thesis

@pytest.fixture
def app():
    """Create and configure a test Flask application"""
    app = create_app()
    app.config['TESTING'] = True
    yield app

@pytest.fixture
def client(app):
    """Create a test client for the Flask application"""
    return app.test_client()

@pytest.fixture
def sample_thesis_data():
    """Sample thesis data for testing"""
    return {
        'title': 'Test Thesis',
        'theme': 'Machine Learning',
        'author': 'Test Author',
        'university': 'Test University',
        'thesis_type': 'research',
        'stage_location': 'Test Lab',
        'methodology': 'Test methodology description',
        'results': 'Test results description',
        'pdf_path': 'test.pdf'
    }

@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing"""
    embedding_dim = 384
    return {
        'theme_embedding': json.dumps(np.random.rand(embedding_dim).tolist()),
        'stage_embedding': json.dumps(np.random.rand(embedding_dim).tolist()),
        'methodology_embedding': json.dumps(np.random.rand(embedding_dim).tolist()),
        'results_embedding': json.dumps(np.random.rand(embedding_dim).tolist()),
        'content_embedding': json.dumps(np.random.rand(embedding_dim).tolist()),
        'images_embedding': json.dumps(np.random.rand(512).tolist())  # CLIP dimension
    }

@pytest.fixture
def mock_thesis(sample_thesis_data, sample_embeddings):
    """Create a mock Thesis object for testing"""
    thesis_data = {**sample_thesis_data, **sample_embeddings}
    thesis = Thesis(**thesis_data)
    thesis.id = 1
    return thesis

@pytest.fixture
def two_mock_theses(sample_thesis_data, sample_embeddings):
    """Create two mock Thesis objects for comparison testing"""
    thesis1_data = {**sample_thesis_data, **sample_embeddings}
    thesis1 = Thesis(**thesis1_data)
    thesis1.id = 1
    
    thesis2_data = {**sample_thesis_data, **sample_embeddings}
    thesis2_data['title'] = 'Test Thesis 2'
    thesis2_data['author'] = 'Test Author 2'
    thesis2 = Thesis(**thesis2_data)
    thesis2.id = 2
    
    return thesis1, thesis2
