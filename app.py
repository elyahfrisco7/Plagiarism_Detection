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

# Remplacer l'import de chromadb.Client(...) par PersistentClient
from chromadb import PersistentClient

# Pour l'extraction d'images depuis le PDF
from pdf2image import convert_from_path
from PIL import Image

# Pour encoder les images avec CLIP
import torch
import clip

# ----------------------------------------------------------------------------
# Configuration de Flask
# ----------------------------------------------------------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# ----------------------------------------------------------------------------
# Configuration de la base de données MySQL
# ----------------------------------------------------------------------------
DATABASE_URI = 'mysql+pymysql://root:@127.0.0.1:3306/thesis_db'
engine = create_engine(DATABASE_URI, echo=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ----------------------------------------------------------------------------
# Modèle de données pour un mémoire (Thesis)
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

# Création de la table `theses` si elle n'existe pas encore
Base.metadata.create_all(bind=engine)

# ----------------------------------------------------------------------------
# Chargement des modèles d'embeddings
# ----------------------------------------------------------------------------
# 1) SentenceTransformer pour le texte
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_dim = 384

# 2) CLIP pour les images
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# ----------------------------------------------------------------------------
# Initialisation de ChromaDB via PersistentClient
# ----------------------------------------------------------------------------
# Ce client stocke les embeddings dans le dossier ./chroma_db
PERSIST_DIRECTORY = "./chroma_db"
chroma_client = PersistentClient(path=PERSIST_DIRECTORY)

# Tenter de récupérer la collection "thesis_full_content"
# Si elle n'existe pas, on la crée.
try:
    collection = chroma_client.get_collection("thesis_full_content")
except:
    collection = chroma_client.create_collection("thesis_full_content")

# ----------------------------------------------------------------------------
# Fonctions utilitaires
# ----------------------------------------------------------------------------

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
    Extrait les images d'un PDF en convertissant chaque page en image PIL via pdf2image.
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
    Calcule l'embedding d'une image en utilisant CLIP.
    """
    image_input = clip_preprocess(image).unsqueeze(0).to(device)
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

def compute_global_similarity(thesis_a, thesis_b):
    """
    Calcule une similarité globale entre deux mémoires en combinant :
      - Thème
      - Lieu de stage/d'étude
      - Méthodologie
      - Résultats
      - Contenu complet
      - Images
    Chaque critère a un poids de 1/6 => somme = 1
    """
    # Récupération des embeddings depuis la base (JSON -> numpy array)
    theme_emb_a = np.array(json.loads(thesis_a.theme_embedding))
    theme_emb_b = np.array(json.loads(thesis_b.theme_embedding))

    stage_emb_a = np.array(json.loads(thesis_a.stage_embedding))
    stage_emb_b = np.array(json.loads(thesis_b.stage_embedding))

    meth_emb_a  = np.array(json.loads(thesis_a.methodology_embedding))
    meth_emb_b  = np.array(json.loads(thesis_b.methodology_embedding))

    results_emb_a = np.array(json.loads(thesis_a.results_embedding))
    results_emb_b = np.array(json.loads(thesis_b.results_embedding))

    content_emb_a = np.array(json.loads(thesis_a.content_embedding))
    content_emb_b = np.array(json.loads(thesis_b.content_embedding))

    images_emb_a = np.array(json.loads(thesis_a.images_embedding)) if thesis_a.images_embedding else np.zeros(embedding_dim)
    images_emb_b = np.array(json.loads(thesis_b.images_embedding)) if thesis_b.images_embedding else np.zeros(embedding_dim)

    # Calcul de la similarité pour chaque critère
    theme_sim = compute_cosine_similarity(theme_emb_a, theme_emb_b)
    location_sim = compute_cosine_similarity(stage_emb_a, stage_emb_b)
    meth_sim = compute_cosine_similarity(meth_emb_a, meth_emb_b)
    results_sim = compute_cosine_similarity(results_emb_a, results_emb_b)
    content_sim = compute_cosine_similarity(content_emb_a, content_emb_b)
    images_sim = compute_cosine_similarity(images_emb_a, images_emb_b)

    # Poids de 1/6 par critère => total 1
    overall = (
        (1/6)*theme_sim +
        (1/6)*location_sim +
        (1/6)*meth_sim +
        (1/6)*results_sim +
        (1/6)*content_sim +
        (1/6)*images_sim
    )
    return overall * 100

def compute_similarity_breakdown(thesis_a, thesis_b):
    """Retourne le détail des similarités par critère + le score global (%)"""
    # Embeddings (JSON -> np.array)
    theme_emb_a = np.array(json.loads(thesis_a.theme_embedding))
    theme_emb_b = np.array(json.loads(thesis_b.theme_embedding))

    stage_emb_a = np.array(json.loads(thesis_a.stage_embedding))
    stage_emb_b = np.array(json.loads(thesis_b.stage_embedding))

    meth_emb_a  = np.array(json.loads(thesis_a.methodology_embedding))
    meth_emb_b  = np.array(json.loads(thesis_b.methodology_embedding))

    results_emb_a = np.array(json.loads(thesis_a.results_embedding))
    results_emb_b = np.array(json.loads(thesis_b.results_embedding))

    content_emb_a = np.array(json.loads(thesis_a.content_embedding))
    content_emb_b = np.array(json.loads(thesis_b.content_embedding))

    images_emb_a = np.array(json.loads(thesis_a.images_embedding)) if thesis_a.images_embedding else np.zeros(embedding_dim)
    images_emb_b = np.array(json.loads(thesis_b.images_embedding)) if thesis_b.images_embedding else np.zeros(embedding_dim)

    # Similarités par critère (0..1)
    sims = {
        "theme":       compute_cosine_similarity(theme_emb_a, theme_emb_b),
        "stage":       compute_cosine_similarity(stage_emb_a, stage_emb_b),
        "methodology": compute_cosine_similarity(meth_emb_a,  meth_emb_b),
        "results":     compute_cosine_similarity(results_emb_a, results_emb_b),
        "content":     compute_cosine_similarity(content_emb_a, content_emb_b),
        "images":      compute_cosine_similarity(images_emb_a, images_emb_b),
    }
    # Moyenne pondérée (1/6 chacun) -> %
    overall = sum(sims.values()) / 6.0 * 100.0
    return sims, overall


# ----------------------------------------------------------------------------
# Routes Flask
# ----------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """
    Récupère le PDF et les champs du formulaire,
    extrait le texte + images, calcule les embeddings,
    stocke dans MySQL, puis indexe le contenu dans ChromaDB.
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

    # Extraction
    full_text = extract_text_from_pdf(file_path)
    images = extract_images_from_pdf(file_path)

    # Embeddings texte
    theme_emb = model.encode(theme).tolist()
    stage_emb = model.encode(stage_location).tolist()
    meth_emb = model.encode(methodology).tolist()
    results_emb = model.encode(results_text).tolist()
    full_content_emb = model.encode(full_text).tolist()

    # Embeddings images (moyenne si plusieurs)
    image_embeddings = []
    for img in images:
        emb = get_image_embedding(img)
        image_embeddings.append(emb)

    if image_embeddings:
        images_emb = np.mean(np.array(image_embeddings), axis=0).tolist()
    else:
        images_emb = [0.0] * embedding_dim

    # Stockage MySQL
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

    # Ajout dans la collection ChromaDB
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
    Compare deux mémoires par ID -> page HTML élégante (ou JSON si demandé).
    Ex: /compare?id1=1&id2=2
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

    sims, overall = compute_similarity_breakdown(thesis1, thesis2)

    # JSON si explicitement demandé
    if "application/json" in request.headers.get("Accept", ""):
        return jsonify({
            "thesis1": thesis1.title,
            "thesis2": thesis2.title,
            "similarity_percentage": overall,
            "breakdown": {k: float(v) for k, v in sims.items()}
        })

    # Sinon, rendu HTML
    return render_template(
        "compare.html",
        thesis1=thesis1,
        thesis2=thesis2,
        overall=overall,
        sims=sims
    )

@app.route('/search', methods=['GET'])
def search():
    """
    Recherche sémantique -> calcule l'embedding de la requête,
    interroge ChromaDB pour trouver jusqu'à 100 documents proches.
    Retourne la page 'results.html' avec la liste.
    """
    query = request.args.get('query', '')
    if not query:
        return "Une requête est nécessaire", 400

    print(f"DEBUG: Search query received: '{query}'")
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=100,
        include=["metadatas", "distances"]
    )
    
    print(f"DEBUG: ChromaDB returned {len(results['ids'][0])} results")

    session = SessionLocal()
    theses_found = []
    for idx, thesis_id in enumerate(results['ids'][0]):
        t = session.query(Thesis).filter(Thesis.id == int(thesis_id)).first()
        if t:
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
    
    print(f"DEBUG: Found {len(theses_found)} theses in MySQL matching ChromaDB IDs")

    return render_template('results.html', results=theses_found)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/theses')
def theses():
    session = SessionLocal()
    items = session.query(Thesis).order_by(Thesis.id.desc()).all()
    session.close()
    return render_template('theses.html', theses=items)

@app.route('/api/scan', methods=['GET'])
def api_scan():
    target_id = request.args.get('id')
    if not target_id:
        return jsonify({"error": "Paramètre id manquant"}), 400

    session = SessionLocal()
    all_theses = session.query(Thesis).all()
    target = None
    for t in all_theses:
        if str(t.id) == str(target_id):
            target = t
            break

    if target is None:
        session.close()
        return jsonify({"error": "Mémoire introuvable"}), 404

    results = []
    for t in all_theses:
        if t.id == target.id:
            continue
        sims, overall = compute_similarity_breakdown(target, t)
        results.append({
            "id": t.id,
            "title": t.title,
            "author": t.author,
            "university": t.university,
            "thesis_type": t.thesis_type,
            "stage_location": t.stage_location,
            "pdf_path": t.pdf_path,
            "overall": float(overall),
            "sims": {k: float(v) for k, v in sims.items()},
            "compare_url": url_for('compare', id1=target.id, id2=t.id)
        })

    session.close()

    # Tri décroissant par score global
    results.sort(key=lambda x: x["overall"], reverse=True)

    return jsonify({
        "target": {
            "id": target.id,
            "title": target.title,
            "author": target.author,
            "university": target.university,
            "thesis_type": target.thesis_type,
            "stage_location": target.stage_location,
            "pdf_path": target.pdf_path
        },
        "results": results
    })



if __name__ == '__main__':
    app.run(debug=True)