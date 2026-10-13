# app/weights.py
"""
Pondération configurable des 6 critères de similarité (schéma THLME-Gre).

Par défaut, chaque critère pèse 1/6 (comme avant). Les poids peuvent être
ajustés depuis la page /settings ; ils sont persistés dans un fichier JSON à
la racine du projet (weights_config.json, non versionné) pour survivre aux
redémarrages de l'application sans nécessiter de migration de base de données.
"""
import json
import os

CRITERIA = ["theme", "stage", "methodology", "results", "content", "images"]

CRITERIA_LABELS = {
    "theme": "Thème",
    "stage": "Lieu de stage / d'étude",
    "methodology": "Méthodologie",
    "results": "Résultats",
    "content": "Contenu global",
    "images": "Images",
}

CRITERIA_ICONS = {
    "theme": "tag",
    "stage": "pin",
    "methodology": "flask",
    "results": "chart",
    "content": "document",
    "images": "image",
}

DEFAULT_WEIGHTS = {c: 1 / 6 for c in CRITERIA}

# Racine du projet = dossier parent de app/
WEIGHTS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights_config.json")


def _normalize(weights: dict) -> dict:
    total = sum(weights.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {c: weights[c] / total for c in CRITERIA}


def get_weights() -> dict:
    """Renvoie les poids actuels (normalisés, somme = 1). Valeurs par défaut si rien n'est configuré."""
    if not os.path.exists(WEIGHTS_FILE):
        return dict(DEFAULT_WEIGHTS)
    try:
        with open(WEIGHTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        weights = {}
        for c in CRITERIA:
            try:
                weights[c] = max(0.0, float(data.get(c, DEFAULT_WEIGHTS[c])))
            except (TypeError, ValueError):
                weights[c] = DEFAULT_WEIGHTS[c]
        return _normalize(weights)
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_WEIGHTS)


def save_weights(raw_weights: dict) -> dict:
    """
    Enregistre de nouveaux poids. `raw_weights` peut contenir des valeurs
    quelconques (ex. pourcentages 0-100 saisis dans le formulaire) : elles
    sont automatiquement normalisées pour que la somme fasse 1.
    Renvoie les poids effectivement enregistrés (normalisés).
    """
    cleaned = {}
    for c in CRITERIA:
        try:
            cleaned[c] = max(0.0, float(raw_weights.get(c, 0)))
        except (TypeError, ValueError):
            cleaned[c] = 0.0

    normalized = _normalize(cleaned)
    with open(WEIGHTS_FILE, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)
    return normalized


def reset_weights() -> dict:
    """Supprime la configuration personnalisée : retour aux poids égaux par défaut."""
    if os.path.exists(WEIGHTS_FILE):
        os.remove(WEIGHTS_FILE)
    return dict(DEFAULT_WEIGHTS)


def is_default() -> bool:
    return not os.path.exists(WEIGHTS_FILE)
