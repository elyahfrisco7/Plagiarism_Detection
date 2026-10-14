# app/controllers.py
import json
import numpy as np
from app.config import EMBEDDING_DIM
from app.weights import get_weights

def compute_cosine_similarity(vec1, vec2):
    """Calcule la similarité cosinus entre deux vecteurs numpy."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def compute_similarity_breakdown(thesis_a, thesis_b):
    """
    Retourne le détail des similarités par critère + le score global (%).

    Les 6 critères (Thème, Lieu de stage/d'étude, Méthodologie, Résultats,
    Contenu complet, Images) sont combinés selon une pondération configurable
    (voir app/weights.py et la page /settings) : par défaut, poids égal de
    1/6 chacun, mais ajustable sans redéploiement.
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

    # Similarités par critère, calculées en 0..1 (cosinus brut)...
    sims_raw = {
        "theme":       compute_cosine_similarity(theme_emb_a, theme_emb_b),
        "stage":       compute_cosine_similarity(stage_emb_a, stage_emb_b),
        "methodology": compute_cosine_similarity(meth_emb_a,  meth_emb_b),
        "results":     compute_cosine_similarity(results_emb_a, results_emb_b),
        "content":     compute_cosine_similarity(content_emb_a, content_emb_b),
        "images":      compute_cosine_similarity(images_emb_a, images_emb_b),
    }

    # ...puis converties en pourcentage (0..100) : tous les gabarits et API
    # (compare.html, le scan de theses.html, le JSON de /compare et
    # /api/scan) affichent "sims" en supposant déjà une échelle 0-100, comme
    # "overall". Sans cette conversion, une similarité de 90 % s'affichait
    # "0.9 %" au lieu de "90.0 %".
    sims = {c: v * 100.0 for c, v in sims_raw.items()}

    # Moyenne pondérée (poids configurables, somme = 1) -> %
    weights = get_weights()
    overall = sum(sims_raw[c] * weights.get(c, 0.0) for c in sims_raw) * 100.0
    return sims, overall

def compute_global_similarity(thesis_a, thesis_b):
    """Ne renvoie que le score global (%) ; voir compute_similarity_breakdown pour le détail par critère."""
    _, overall = compute_similarity_breakdown(thesis_a, thesis_b)
    return overall
