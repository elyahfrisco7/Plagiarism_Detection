# app/config.py
import os
import torch
import clip
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# ----------------------------------------------------------------------------
# Flask Configuration
# ----------------------------------------------------------------------------
UPLOAD_FOLDER = 'uploads'

# ----------------------------------------------------------------------------
# Database Configuration
# ----------------------------------------------------------------------------
DATABASE_URI = 'mysql+pymysql://root:@127.0.0.1:3306/thesis_db'

# ----------------------------------------------------------------------------
# ChromaDB Configuration
# ----------------------------------------------------------------------------
PERSIST_DIRECTORY = "./chroma_db"

# ----------------------------------------------------------------------------
# Model Configuration
# ----------------------------------------------------------------------------
# Embedding dimension for SentenceTransformer
EMBEDDING_DIM = 384

# Device for CLIP
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------------------------------
# Model Loading (singleton pattern)
# ----------------------------------------------------------------------------
_sentence_model = None
_clip_model = None
_clip_preprocess = None
_chroma_client = None
_collection = None

def get_sentence_model():
    """Get or create SentenceTransformer model"""
    global _sentence_model
    if _sentence_model is None:
        _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _sentence_model

def get_clip_model():
    """Get or create CLIP model and preprocessor"""
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
    return _clip_model, _clip_preprocess

def get_chroma_client():
    """Get or create ChromaDB client"""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = PersistentClient(path=PERSIST_DIRECTORY)
    return _chroma_client

def get_collection():
    """Get or create ChromaDB collection"""
    global _collection
    if _collection is None:
        client = get_chroma_client()
        try:
            _collection = client.get_collection("thesis_full_content")
        except:
            _collection = client.create_collection("thesis_full_content")
    return _collection
