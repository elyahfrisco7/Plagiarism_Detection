# Script pour mettre à jour les embeddings d'images des thèses existantes
import os
import json
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Text
import fitz  # PyMuPDF
import io
from PIL import Image
import torch
import clip

# Configuration de la base de données
DATABASE_URI = 'mysql+pymysql://root:@127.0.0.1:3306/thesis_db'
engine = create_engine(DATABASE_URI, echo=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Thesis(Base):
    __tablename__ = 'theses'
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255))
    pdf_path = Column(String(255))
    images_embedding = Column(Text)

# Charger CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
embedding_dim = 384  # Pour compatibilité avec le reste

def extract_images_from_pdf(pdf_filepath):
    """Extrait les images d'un PDF via PyMuPDF"""
    images = []
    try:
        doc = fitz.open(pdf_filepath)
        for page_index in range(len(doc)):
            page = doc[page_index]
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                images.append(image)
                
        print(f"  Extracted {len(images)} images")
    except Exception as e:
        print(f"  Error: {e}")
    return images

def get_image_embedding(image):
    """Calcule l'embedding d'une image avec CLIP"""
    image_input = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()[0].tolist()

# Récupérer toutes les thèses
session = SessionLocal()
theses = session.query(Thesis).all()

print(f"Found {len(theses)} theses in database\n")

for thesis in theses:
    print(f"Processing: {thesis.title} (ID: {thesis.id})")
    
    pdf_path = os.path.join('uploads', thesis.pdf_path)
    if not os.path.exists(pdf_path):
        print(f"  PDF not found: {pdf_path}")
        continue
    
    # Extraire les images
    images = extract_images_from_pdf(pdf_path)
    
    if images:
        # Calculer les embeddings
        image_embeddings = []
        for img in images:
            emb = get_image_embedding(img)
            image_embeddings.append(emb)
        
        # Moyenne des embeddings
        images_emb = np.mean(np.array(image_embeddings), axis=0).tolist()
        
        # Mettre à jour dans la base
        thesis.images_embedding = json.dumps(images_emb)
        print(f"  Updated with {len(images)} images")
    else:
        # Pas d'images - mettre un vecteur nul
        images_emb = [0.0] * 512  # CLIP dimension
        thesis.images_embedding = json.dumps(images_emb)
        print(f"  No images found - set to zero vector")
    
    print()

session.commit()
session.close()

print("Done! All thesis image embeddings have been updated.")
