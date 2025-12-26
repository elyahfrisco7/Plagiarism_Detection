# app/utils.py
import io
import numpy as np
import PyPDF2
import fitz  # PyMuPDF
from PIL import Image
from app.config import get_clip_model, DEVICE

def extract_text_from_pdf(pdf_filepath):
    """Extrait le texte complet d'un PDF à l'aide de PyPDF2."""
    text = ""
    try:
        with open(pdf_filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print("Erreur d'extraction PDF:", e)
    return text

def extract_images_from_pdf(pdf_filepath):
    """
    Extrait les images d'un PDF via PyMuPDF (fitz).
    Retourne une liste d'objets PIL.Image.
    """
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
                
        print(f"DEBUG: Extracted {len(images)} images from {pdf_filepath}")
    except Exception as e:
        print(f"Erreur d'extraction d'images (PyMuPDF): {e}")
    return images

def get_image_embedding(image):
    """
    Calcule l'embedding d'une image en utilisant CLIP.
    """
    clip_model, clip_preprocess = get_clip_model()
    image_input = clip_preprocess(image).unsqueeze(0).to(DEVICE)
    import torch
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()[0].tolist()

def compute_cosine_similarity(vec1, vec2):
    """Calcule la similarité cosinus entre deux vecteurs numpy."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)
