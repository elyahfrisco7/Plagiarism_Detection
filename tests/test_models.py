# tests/test_models.py
from app.models import Thesis

def test_thesis_model_creation(sample_thesis_data):
    thesis = Thesis(**sample_thesis_data)
    assert thesis.title == 'Test Thesis'
    assert thesis.author == 'Test Author'
    assert thesis.thesis_type == 'research'

def test_thesis_model_repr(sample_thesis_data):
    thesis = Thesis(**sample_thesis_data)
    # Just check it can be instantiated and fields accessed
    assert hasattr(thesis, 'theme_embedding')
