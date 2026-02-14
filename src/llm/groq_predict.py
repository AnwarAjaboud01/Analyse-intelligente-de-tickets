import os
import json
import re
from typing import Dict, Any, Optional

from openai import OpenAI

# Groq OpenAI-compatible base URL
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# ⚡ Modèles recommandés Groq (tu peux changer)
# - "llama-3.1-8b-instant" : très rapide
# - "llama-3.3-70b-versatile" : meilleur raisonnement (souvent encore rapide chez Groq)
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

ALLOWED_URGENCE = ["Basse", "Moyenne", "Haute"]
ALLOWED_TYPE = ["Demande", "Incident"]
ALLOWED_CATEGORIES = [
    "Applications & Logiciels",
    "Bureautique",
    "Réseau & Connexion",
    "Matériel / Poste de travail",
    "Sécurité",
    "Comptes & Accès",
    "Impression",
    "Email / Messagerie",
    "Téléphonie",
    "Partage & Droits",
    "Autre",
]

SYSTEM_PROMPT = (
    "Tu es un agent Helpdesk IT (entreprise informatique).\n"
    "Tu DOIS répondre uniquement en JSON strict (pas de texte, pas de markdown).\n"
    "Le JSON doit contenir EXACTEMENT les clés suivantes:\n"
    '  - "urgence": une des valeurs {"Basse","Moyenne","Haute"}\n'
    '  - "categorie": une des catégories autorisées\n'
    '  - "type_ticket": une des valeurs {"Demande","Incident"}\n'
    '  - "temps_resolution": nombre (heures) entre 0.25 et 72\n'
    "Catégories autorisées:\n"
    + ", ".join(ALLOWED_CATEGORIES)
    + "\n\n"
    "Règles métier importantes:\n"
    "- Sécurité (fuite, accès non autorisé, intrusion, tentative suspecte) => urgence Haute, categorie Sécurité, type_ticket Incident.\n"
    "- Coupure globale / service bloquant / production KO => urgence Haute, type_ticket Incident.\n"
    "- Mot de passe / réinitialisation / procédure => urgence Basse, categorie Comptes & Accès, type_ticket Demande.\n"
    "- Demande d'amélioration / ajouter filtre / fonctionnalité => urgence Basse, categorie Applications & Logiciels, type_ticket Demande.\n"
    "- Réseau / Wi-Fi / internet => categorie Réseau & Connexion, souvent Incident.\n"
    "- Email / messagerie => categorie Email / Messagerie, souvent Incident.\n"
)

def _clamp_hours(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        v = 8.0
    v = max(0.25, min(72.0, v))
    return round(v, 2)

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    # direct
    try:
        return json.loads(text)
    except Exception:
        pass
    # extract first {...}
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def _normalize(out: Dict[str, Any]) -> Dict[str, Any]:
    urgence = out.get("urgence", "Moyenne")
    if urgence not in ALLOWED_URGENCE:
        urgence = "Moyenne"

    categorie = out.get("categorie", "Autre")
    if categorie not in ALLOWED_CATEGORIES:
        categorie = "Autre"

    type_ticket = out.get("type_ticket", "Demande")
    if type_ticket not in ALLOWED_TYPE:
        type_ticket = "Demande"

    temps = _clamp_hours(out.get("temps_resolution", 8.0))

    return {
        "urgence": urgence,
        "categorie": categorie,
        "type_ticket": type_ticket,
        "temps_resolution": temps,
    }

def _hard_overrides(text_full: str, out: Dict[str, Any]) -> Dict[str, Any]:
    """
    Post-traitement minimal pour éviter des sorties illogiques sur cas critiques.
    (Même en LLM 100%, ça sécurise la démo.)
    """
    t = (text_full or "").lower()

    # Sécurité
    if any(k in t for k in [
        "fuite", "accès non autorisé", "acces non autorise", "intrusion",
        "tentatives de connexion", "connexion suspecte", "suspicious login",
        "pirat", "hack", "data breach"
    ]):
        return {"urgence": "Haute", "categorie": "Sécurité", "type_ticket": "Incident", "temps_resolution": 10.0}

    # Coupure / bloquant
    if any(k in t for k in ["coupure", "plus aucune", "panne", "bloquant", "production", "service down", "inaccessible"]):
        # Si c'est clairement bloquant, on pousse vers Haute/Incident
        out["type_ticket"] = "Incident"
        if out["urgence"] == "Basse":
            out["urgence"] = "Haute"
        if "réseau" in t or "internet" in t or "wifi" in t or "serveur" in t:
            out["categorie"] = "Réseau & Connexion"
        return out

    # Password/proc
    if any(k in t for k in ["mot de passe", "réinitialiser", "reinitialiser", "procedure", "procédure"]):
        return {"urgence": "Basse", "categorie": "Comptes & Accès", "type_ticket": "Demande", "temps_resolution": 2.0}

    return out

def predict_ticket_groq(titre: str, texte: str, model: Optional[str] = None) -> Dict[str, Any]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY manquant. Définis la variable d'environnement GROQ_API_KEY.")

    client = OpenAI(base_url=GROQ_BASE_URL, api_key=api_key)

    titre = (titre or "").strip()
    texte = (texte or "").strip()
    text_full = f"{titre} {texte}".strip()

    user_prompt = (
        f"TITRE: {titre}\n"
        f"DESCRIPTION: {texte}\n"
        "Réponds uniquement en JSON strict."
    )

    chosen_model = model or DEFAULT_MODEL

    # Appel Groq (OpenAI-compatible chat completions) :contentReference[oaicite:1]{index=1}
    resp = client.chat.completions.create(
        model=chosen_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=120,   # ⚡ rapide + évite blabla
        top_p=0.9,
    )

    content = resp.choices[0].message.content or ""
    data = _extract_json(content)

    if not data:
        # fallback robuste
        out = {"urgence": "Moyenne", "categorie": "Autre", "type_ticket": "Demande", "temps_resolution": 8.0}
        return _hard_overrides(text_full, out)

    out = _normalize(data)
    out = _hard_overrides(text_full, out)
    out["temps_resolution"] = _clamp_hours(out["temps_resolution"])
    return out
