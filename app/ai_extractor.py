# app/ai_extractor.py
"""
Auto-remplissage de la fiche technique via l'API Claude.

On réutilise le texte déjà extrait du PDF (app.utils.extract_text_from_pdf)
plutôt que d'envoyer le PDF brut : Claude n'a pas besoin de "voir" la mise en
page pour retrouver titre / auteur / université / méthodologie / résultats,
et le coût en tokens d'entrée est 3 à 6 fois inférieur à l'envoi du document
complet (pas de rendu image par page).

Modèle utilisé : Claude Haiku 4.5 (claude-haiku-4-5) - le modèle le moins
coûteux de la gamme actuelle, largement suffisant pour une tâche d'extraction
structurée.
"""
import os
from typing import Literal, Optional

from anthropic import Anthropic
from pydantic import BaseModel, Field

# Un mémoire de 120 pages reste très en dessous de la fenêtre de contexte de
# Haiku (200K tokens) une fois converti en texte brut, mais on garde un
# garde-fou pour les documents anormalement denses.
MAX_INPUT_CHARS = 350_000

MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = (
    "Tu es un assistant documentaliste qui renseigne la fiche technique d'un "
    "mémoire universitaire francophone à partir du texte brut extrait de son PDF "
    "(pages de garde, résumé, sommaire, introduction, méthodologie, résultats, etc.). "
    "Déduis chaque champ du contenu réel du document. Si une information est "
    "introuvable, renvoie une chaîne vide plutôt que d'inventer une valeur. "
    "Les champs 'methodology' et 'results' doivent être de courtes synthèses "
    "rédigées en français (3 à 6 phrases), pas une copie brute du texte."
)


class ThesisMetadata(BaseModel):
    title: str = Field(description="Titre du mémoire")
    theme: str = Field(description="Thème ou sujet principal (court, quelques mots)")
    author: str = Field(description="Nom et prénom de l'auteur du mémoire")
    university: str = Field(description="Université ou établissement d'accueil")
    thesis_type: Literal["research", "professional"] = Field(
        description="'research' si mémoire de recherche, 'professional' si mémoire professionnel / de stage"
    )
    stage_location: str = Field(description="Lieu de stage ou d'étude")
    methodology: str = Field(description="Synthèse courte de la méthodologie et des objectifs")
    results: str = Field(description="Synthèse courte des résultats, outils et technologies utilisés")


class AIExtractionError(Exception):
    """Levée quand l'extraction automatique échoue (clé manquante, erreur API, etc.)."""


_client: Optional[Anthropic] = None


def get_ai_client() -> Anthropic:
    """Singleton paresseux, comme les autres modèles de app/config.py."""
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise AIExtractionError(
                "ANTHROPIC_API_KEY n'est pas configurée (fichier .env manquant ou incomplet)."
            )
        _client = Anthropic(api_key=api_key)
    return _client


def extract_metadata_with_ai(full_text: str) -> ThesisMetadata:
    """
    Envoie le texte extrait du PDF à Claude Haiku 4.5 et renvoie les 8 champs
    de la fiche technique sous forme structurée (validée par Pydantic).
    """
    text = (full_text or "").strip()
    if not text:
        raise AIExtractionError("Aucun texte n'a pu être extrait de ce PDF.")

    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]

    client = get_ai_client()

    try:
        response = client.messages.parse(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text}],
            output_format=ThesisMetadata,
        )
    except Exception as e:
        raise AIExtractionError(f"Échec de l'appel à l'API Claude : {e}") from e

    if response.parsed_output is None:
        raise AIExtractionError("La réponse de l'IA n'a pas pu être validée.")

    return response.parsed_output
