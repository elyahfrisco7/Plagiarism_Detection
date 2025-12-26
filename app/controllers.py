# app/controllers.py
import json
import numpy as np
from app.config import EMBEDDING_DIM

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

    images_emb_a = np.array(json.loads(thesis_a.images_embedding)) if thesis_a.images_embedding else np.zeros(EMBEDDING_DIM)
    images_emb_b = np.array(json.loads(thesis_b.images_embedding)) if thesis_b.images_embedding else np.zeros(EMBEDDING_DIM)

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

    images_emb_a = np.array(json.loads(thesis_a.images_embedding)) if thesis_a.images_embedding else np.zeros(EMBEDDING_DIM)
    images_emb_b = np.array(json.loads(thesis_b.images_embedding)) if thesis_b.images_embedding else np.zeros(EMBEDDING_DIM)

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
