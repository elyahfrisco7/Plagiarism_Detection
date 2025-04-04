# app.py
import os
import json
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker, declarative_base
import PyPDF2
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Pour l'extraction d'images depuis le PDF
from pdf2image import convert_from_path
from PIL import Image

# Pour encoder les images avec CLIP
import torch
import clip

# ------------------------------
# Configuration de Flask
# ------------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# ------------------------------
# Configuration de la base de données MySQL
# ------------------------------
# Remplacez 'username', 'password', 'localhost' et 'thesis_db' par vos informations MySQL.
DATABASE_URI = 'mysql+pymysql://username:password@localhost/thesis_db'
engine = create_engine(DATABASE_URI, echo=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ------------------------------
# Modèle de données pour un mémoire (Thesis)
# ------------------------------
class Thesis(Base):
    __tablename__ = 'theses'
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255))
    theme = Column(String(255))
    author = Column(String(255))
    university = Column(String(255))
    thesis_type = Column(String(50))      # "research" ou "professional"
    stage_location = Column(String(255))    # Lieu de stage ou d'étude
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

# Création des tables dans la base
Base.metadata.create_all(bind=engine)

# ------------------------------
# Chargement des modèles d'embeddings
# ------------------------------
# Pour le texte
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_dim = 384

# Pour les images avec CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# ------------------------------
# Initialisation de ChromaDB pour le contenu complet
# ------------------------------
chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db"))
collection = chroma_client.get_or_create_collection(name="thesis_full_content")

# ------------------------------
# Fonctions utilitaires
# ------------------------------

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
    Extrait les images d'un PDF en convertissant chaque page en image.
    Retourne une liste d'objets PIL.Image.
    """
    try:
        images = convert_from_path(pdf_filepath)
        return images
    except Exception as e:
        print("Erreur d'extraction d'images:", e)
        return []

def get_image_embedding(image):
    """
    Calcule l'embedding d'une image en utilisant le modèle CLIP.
    """
    image_input = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()[0].tolist()

def compute_cosine_similarity(vec1, vec2):
    """Calcule la similarité cosinus entre deux vecteurs."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def compute_global_similarity(thesis_a, thesis_b):
    """
    Calcule une similarité globale entre deux mémoires en combinant les critères suivants :
      - Thème (0.2)
      - Lieu de stage/d'étude (0.2)
      - Méthodologie et objectifs (0.2)
      - Résultats (0.2)
      - Contenu complet (0.2)
      - Images (0.2)
      
    (Ici, la somme des poids est 1 pour chaque critère, mais nous avons 6 critères.
    Pour obtenir une moyenne, nous appliquons 1/6 à chacun.)
    """
    # Récupération des embeddings stockés (conversion depuis JSON)
    theme_emb_a = np.array(json.loads(thesis_a.theme_embedding))
    theme_emb_b = np.array(json.loads(thesis_b.theme_embedding))
    stage_emb_a = np.array(json.loads(thesis_a.stage_embedding))
    stage_emb_b = np.array(json.loads(thesis_b.stage_embedding))
    meth_emb_a = np.array(json.loads(thesis_a.methodology_embedding))
    meth_emb_b = np.array(json.loads(thesis_b.methodology_embedding))
    results_emb_a = np.array(json.loads(thesis_a.results_embedding))
    results_emb_b = np.array(json.loads(thesis_b.results_embedding))
    content_emb_a = np.array(json.loads(thesis_a.content_embedding))
    content_emb_b = np.array(json.loads(thesis_b.content_embedding))
    images_emb_a = np.array(json.loads(thesis_a.images_embedding)) if thesis_a.images_embedding else np.zeros(embedding_dim)
    images_emb_b = np.array(json.loads(thesis_b.images_embedding)) if thesis_b.images_embedding else np.zeros(embedding_dim)
    
    # Calcul des similarités cosinus pour chaque critère
    theme_sim = compute_cosine_similarity(theme_emb_a, theme_emb_b)
    location_sim = compute_cosine_similarity(stage_emb_a, stage_emb_b)
    meth_sim = compute_cosine_similarity(meth_emb_a, meth_emb_b)
    results_sim = compute_cosine_similarity(results_emb_a, results_emb_b)
    content_sim = compute_cosine_similarity(content_emb_a, content_emb_b)
    images_sim = compute_cosine_similarity(images_emb_a, images_emb_b)
    
    # Pondération égale de 1/6 pour chacun
    overall = ( (1/6) * theme_sim +
                (1/6) * location_sim +
                (1/6) * meth_sim +
                (1/6) * results_sim +
                (1/6) * content_sim +
                (1/6) * images_sim )
    return overall * 100

# ------------------------------
# Routes Flask
# ------------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """
    Récupère les informations du formulaire, extrait le texte et les images du PDF,
    calcule les embeddings pour chaque critère ainsi que pour le contenu complet et les images,
    puis stocke les données dans MySQL et l'embedding du contenu complet dans ChromaDB.
    """
    title = request.form.get('title')
    theme = request.form.get('theme')
    author = request.form.get('author')
    university = request.form.get('university')
    thesis_type = request.form.get('thesis_type')
    stage_location = request.form.get('stage_location')
    methodology = request.form.get('methodology')
    results_text = request.form.get('results')
    file = request.files.get('pdf_file')

    if not file:
        return "Aucun fichier fourni", 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Extraction du texte complet depuis le PDF
    full_text = extract_text_from_pdf(file_path)
    # Extraction des images depuis le PDF
    images = extract_images_from_pdf(file_path)
    
    # Calcul des embeddings pour chacun des critères
    theme_emb = model.encode(theme).tolist()
    stage_emb = model.encode(stage_location).tolist()
    meth_emb = model.encode(methodology).tolist()
    results_emb = model.encode(results_text).tolist()
    full_content_emb = model.encode(full_text).tolist()
    
    # Traitement des images : calculer l'embedding pour chaque image puis la moyenne
    image_embeddings = []
    for img in images:
        emb = get_image_embedding(img)
        image_embeddings.append(emb)
    if image_embeddings:
        # Moyenne des embeddings d'images
        images_emb = np.mean(np.array(image_embeddings), axis=0).tolist()
    else:
        images_emb = [0.0] * embedding_dim  # Valeur par défaut si aucune image n'est extraite

    # Stockage dans MySQL
    session = SessionLocal()
    thesis = Thesis(
        title=title,
        theme=theme,
        author=author,
        university=university,
        thesis_type=thesis_type,
        stage_location=stage_location,
        methodology=methodology,
        results=results_text,
        pdf_path=filename,
        theme_embedding=json.dumps(theme_emb),
        stage_embedding=json.dumps(stage_emb),
        methodology_embedding=json.dumps(meth_emb),
        results_embedding=json.dumps(results_emb),
        content_embedding=json.dumps(full_content_emb),
        images_embedding=json.dumps(images_emb)
    )
    session.add(thesis)
    session.commit()
    thesis_id = thesis.id
    session.close()

    # Stockage de l'embedding du contenu complet dans ChromaDB pour la recherche sémantique
    collection.add(
        ids=[str(thesis_id)],
        embeddings=[full_content_emb],
        metadatas=[{
            "title": title,
            "theme": theme,
            "author": author,
            "university": university,
            "thesis_type": thesis_type,
            "stage_location": stage_location
        }]
    )

    return redirect(url_for('index'))

@app.route('/compare', methods=['GET'])
def compare():
    """
    Compare deux mémoires par leurs IDs et renvoie un score global de similarité (en %).
    Exemple d'URL : /compare?id1=1&id2=2
    """
    id1 = request.args.get('id1')
    id2 = request.args.get('id2')
    if not id1 or not id2:
        return "Les deux IDs sont requis pour la comparaison", 400

    session = SessionLocal()
    thesis1 = session.query(Thesis).filter(Thesis.id == int(id1)).first()
    thesis2 = session.query(Thesis).filter(Thesis.id == int(id2)).first()
    session.close()

    if not thesis1 or not thesis2:
        return "Un ou plusieurs documents n'ont pas été trouvés", 404

    similarity = compute_global_similarity(thesis1, thesis2)
    return jsonify({
        "thesis1": thesis1.title,
        "thesis2": thesis2.title,
        "similarity_percentage": similarity
    })

@app.route('/search', methods=['GET'])
def search():
    """
    Recherche sémantique : calcule l'embedding de la requête et interroge ChromaDB pour
    trouver jusqu'à 100 documents similaires (basé sur le contenu complet du PDF).
    Affiche ensuite tous les documents trouvés avec leur score de similarité (basé sur le contenu).
    """
    query = request.args.get('query', '')
    if not query:
        return "Une requête est nécessaire", 400
    
    query_embedding = model.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=100,
        include=["metadatas", "distances", "ids"]
    )
    
    session = SessionLocal()
    theses_found = []
    for idx, thesis_id in enumerate(results['ids'][0]):
        t = session.query(Thesis).filter(Thesis.id == int(thesis_id)).first()
        if t:
            # Transformation simplifiée de la distance en score de similarité (pour le contenu complet)
            content_score = max(0, 100 - results['distances'][0][idx] * 100)
            theses_found.append({
                "id": t.id,
                "title": t.title,
                "theme": t.theme,
                "author": t.author,
                "university": t.university,
                "thesis_type": t.thesis_type,
                "stage_location": t.stage_location,
                "methodology": t.methodology,
                "results": t.results,
                "content_score": content_score,
                "pdf_path": t.pdf_path
            })
    session.close()
    return render_template('results.html', results=theses_found)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
