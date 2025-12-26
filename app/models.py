# app/models.py
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import DATABASE_URI

# ----------------------------------------------------------------------------
# Database Setup
# ----------------------------------------------------------------------------
engine = create_engine(DATABASE_URI, echo=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ----------------------------------------------------------------------------
# Thesis Model
# ----------------------------------------------------------------------------
class Thesis(Base):
    __tablename__ = 'theses'
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255))
    theme = Column(String(255))
    author = Column(String(255))
    university = Column(String(255))
    thesis_type = Column(String(50))      # "research" ou "professional"
    stage_location = Column(String(255))  # Lieu de stage ou d'étude
    methodology = Column(Text)            # Méthodologie et objectifs
    results = Column(Text)                # Résultats (technologies, outils, etc.)
    pdf_path = Column(String(255))        # Nom du fichier PDF

    # Embeddings pour chaque critère (stockés sous forme de JSON)
    theme_embedding = Column(Text)
    stage_embedding = Column(Text)
    methodology_embedding = Column(Text)
    results_embedding = Column(Text)
    content_embedding = Column(Text)      # Embedding du contenu complet
    images_embedding = Column(Text)       # Embedding moyen des images

# Create tables
Base.metadata.create_all(bind=engine)
