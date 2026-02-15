import os
import json
import re
from typing import Dict, Any

from openai import OpenAI

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

# Heuristiques de catégorisation si le LLM sort une catégorie “proche” ou inconnue
KEYWORD_TO_CATEGORY = [
    (["wifi", "réseau", "internet", "vpn", "dns", "coupure", "routeur", "latence", "serveur", "ping"], "Réseau & Connexion"),
    (["mail", "email", "outlook", "messagerie", "smtp", "imap", "boite", "boîte"], "Email / Messagerie"),
    (["mot de passe", "password", "login", "connexion", "compte", "accès", "sso", "auth"], "Comptes & Accès"),
    (["imprim", "printer", "scanner", "toner", "cartouche"], "Impression"),
    (["téléphone", "telephonie", "softphone", "voip", "sip", "call"], "Téléphonie"),
    (["partage", "droit", "permission", "drive", "dossier partagé", "dossier partage"], "Partage & Droits"),
    (["virus", "malware", "phishing", "intrusion", "fuite", "données", "data leak", "ransom", "attaque", "suspect"], "Sécurité"),
    (["pc", "ordinateur", "écran", "clavier", "souris", "ram", "disque", "ssd", "redémarre", "démarrage"], "Matériel / Poste de travail"),
    (["excel", "word", "powerpoint", "teams", "office"], "Bureautique"),
    (["application", "bug", "erreur", "crash", "écran blanc", "service", "api"], "Applications & Logiciels"),
]


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    # JSON pur
    try:
        return json.loads(text)
    except Exception:
        pass
    # premier bloc {...}
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("Aucun JSON détecté dans la réponse.")
    return json.loads(m.group(0))


def _normalize_urgence(u: str) -> str:
    u = (u or "").strip().lower()
    if u in ["haute", "urgent", "urgente", "très haute", "tres haute", "critique"]:
        return "Haute"
    if u in ["moyenne", "medium", "moderee", "modérée"]:
        return "Moyenne"
    if u in ["basse", "low", "faible"]:
        return "Basse"
    return "Moyenne"


def _normalize_type(t: str) -> str:
    t = (t or "").strip().lower()
    if "incident" in t or "panne" in t or "erreur" in t:
        return "Incident"
    if "demande" in t or "request" in t:
        return "Demande"
    # fallback raisonnable : si urgence haute => Incident sinon Demande (sera ajusté plus bas si besoin)
    return "Demande"


def _guess_category_from_text(titre: str, texte: str) -> str:
    blob = f"{titre} {texte}".lower()
    for keywords, cat in KEYWORD_TO_CATEGORY:
        if any(k in blob for k in keywords):
            return cat
    return "Autre"


def _normalize_category(cat: str, titre: str, texte: str) -> str:
    cat = (cat or "").strip()
    # alias fréquents
    alias = {
        "Accès au partage": "Partage & Droits",
        "Partage & Accès": "Partage & Droits",
        "Partage et droits": "Partage & Droits",
        "Comptes et accès": "Comptes & Accès",
        "Réseau": "Réseau & Connexion",
        "Messagerie": "Email / Messagerie",
        "Securite": "Sécurité",
        "Sécurite": "Sécurité",
    }
    cat = alias.get(cat, cat)

    if cat in ALLOWED_CATEGORIES:
        return cat

    # Si hors liste → heuristique par mots-clés
    guess = _guess_category_from_text(titre, texte)
    return guess if guess in ALLOWED_CATEGORIES else "Autre"


def _normalize_time(t: Any, urgence: str) -> float:
    try:
        v = float(t)
    except Exception:
        v = None

    # fallback: temps par urgence (simple + cohérent)
    if v is None or v <= 0:
        if urgence == "Haute":
            v = 10.0
        elif urgence == "Moyenne":
            v = 4.0
        else:
            v = 2.0

    # bornes raisonnables
    v = max(0.25, min(v, 72.0))
    return round(v, 2)


def predict_ticket_groq(titre: str, texte: str) -> Dict[str, Any]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY manquant. PowerShell: $env:GROQ_API_KEY='...'\n")

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    system = (
        "Tu es un expert Helpdesk IT (entreprise informatique). "
        "Tu dois classifier un ticket (titre + texte) et répondre STRICTEMENT en JSON."
    )

    user = f"""
RÈGLES OBLIGATOIRES (respect strict):
- urgence ∈ {ALLOWED_URGENCE}
- type_ticket ∈ {ALLOWED_TYPE}
- categorie ∈ {ALLOWED_CATEGORIES}
- temps_resolution = nombre d'heures (0.25 à 72), cohérent avec l'urgence.

Réponds UNIQUEMENT en JSON, sans texte autour:
{{
  "urgence": "Basse|Moyenne|Haute",
  "categorie": "<une catégorie autorisée>",
  "type_ticket": "Demande|Incident",
  "temps_resolution": 0.25
}}

Indications métier:
- Sécurité / fuite / intrusion / compte admin compromis => urgence Haute, type Incident
- Coupure réseau complète / service bloqué pour tous => urgence Haute, type Incident
- Problème partiel / intermittent => urgence Moyenne
- Amélioration / demande d'information / changement => urgence Basse, type Demande

TITRE: {titre}
TEXTE: {texte}
""".strip()

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=180,
    )

    content = resp.choices[0].message.content or ""
    raw = _extract_json(content)

    urgence = _normalize_urgence(str(raw.get("urgence", "")))
    type_ticket = _normalize_type(str(raw.get("type_ticket", "")))
    categorie = _normalize_category(str(raw.get("categorie", "")), titre, texte)
    temps = _normalize_time(raw.get("temps_resolution", None), urgence)

    # Ajustement final: si Haute + Demande incohérent => Incident
    if urgence == "Haute" and type_ticket == "Demande":
        type_ticket = "Incident"

    return {
        "urgence": urgence,
        "categorie": categorie,
        "type_ticket": type_ticket,
        "temps_resolution": temps,
    }
