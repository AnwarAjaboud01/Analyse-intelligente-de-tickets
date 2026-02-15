# app.py
import os
import csv
import time
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory

from src.llm.groq_predict import predict_ticket_groq  # ton modèle distant

# Chatbot imports
from openai import OpenAI

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(APP_ROOT, "web")
DATA_DIR = os.path.join(APP_ROOT, "data")
CSV_PATH = os.path.join(DATA_DIR, "data.csv")

CSV_HEADER = ["id", "created_at", "titre", "texte", "urgence", "categorie", "type_ticket", "temps_resolution", "statue"]

app = Flask(__name__, static_folder=WEB_DIR, static_url_path="")

def ensure_csv():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(CSV_HEADER)

def read_tickets():
    ensure_csv()
    rows = []
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            # ensure keys exist
            for k in CSV_HEADER:
                row.setdefault(k, "")
            # cast temps_resolution
            try:
                row["temps_resolution"] = float(row["temps_resolution"])
            except Exception:
                row["temps_resolution"] = 0.0
            rows.append(row)
    return rows

def write_tickets(rows):
    ensure_csv()
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in CSV_HEADER})

@app.get("/")
def home():
    return send_from_directory(WEB_DIR, "index.html")

@app.get("/add")
def add_page():
    return send_from_directory(WEB_DIR, "add.html")

@app.get("/styles.css")
def styles():
    return send_from_directory(WEB_DIR, "styles.css")

@app.get("/app.js")
def js():
    return send_from_directory(WEB_DIR, "app.js")

@app.get("/api/tickets")
def api_get_tickets():
    tickets = read_tickets()
    return jsonify({"tickets": tickets})

@app.post("/api/tickets")
def api_add_ticket():
    payload = request.get_json(force=True, silent=True) or {}
    titre = (payload.get("titre") or "").strip()
    texte = (payload.get("texte") or "").strip()
    if not titre or not texte:
        return jsonify({"error": "titre et texte sont obligatoires"}), 400

    # prediction via modèle distant
    pred = predict_ticket_groq(titre, texte)

    ticket_id = f"t_{int(time.time()*1000)}"
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "id": ticket_id,
        "created_at": created_at,
        "titre": titre,
        "texte": texte,
        "urgence": pred.get("urgence", "Basse"),
        "categorie": pred.get("categorie", "Autre"),
        "type_ticket": pred.get("type_ticket", "Demande"),
        "temps_resolution": float(pred.get("temps_resolution", 2.0)),
        "statue": "en_attente",
    }

    tickets = read_tickets()
    tickets.append(row)
    write_tickets(tickets)
    return jsonify({"ok": True, "ticket": row})

@app.post("/api/tickets/<ticket_id>/done")
def api_done_ticket(ticket_id):
    tickets = read_tickets()
    found = False
    for t in tickets:
        if t.get("id") == ticket_id:
            t["statue"] = "termine"
            found = True
            break
    if not found:
        return jsonify({"error": "ticket introuvable"}), 404
    write_tickets(tickets)
    return jsonify({"ok": True})

@app.delete("/api/tickets/<ticket_id>")
def api_delete_ticket(ticket_id):
    tickets = read_tickets()
    new_rows = [t for t in tickets if t.get("id") != ticket_id]
    if len(new_rows) == len(tickets):
        return jsonify({"error": "ticket introuvable"}), 404
    write_tickets(new_rows)
    return jsonify({"ok": True})

# Chatbot configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
SYSTEM_PROMPT = """
Tu es un assistant support utile pour l'application de gestion de tickets IT "Analyse intelligente de tickets".
Tu connais le contexte marocain (français, arabe dialecte) et aides les utilisateurs avec les tickets de support IT.

Contexte de l'application :
- C'est un tableau de bord web pour créer, visualiser et gérer des tickets de support.
- Les tickets ont : titre, description, urgence (Basse/Moyenne/Haute), catégorie, type (Demande/Incident), temps de résolution estimé.
- Pour supprimer un ticket : Dans l'interface web, trouvez le ticket et cliquez sur le bouton de suppression (si autorisé).
- Pour créer un ticket : Cliquez sur "Ajouter" en haut à droite.
- Les tickets sont classés par urgence : Haute (rouge), Moyenne (orange), Basse (verte).
- Utilisez un langage simple et helpful. Si la question n'est pas liée aux tickets, redirigez poliment.
"""

def get_chatbot_response(user_message):
    if not GROQ_API_KEY:
        return "Erreur: Clé API GROQ non configurée."

    client = OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Erreur lors de l'appel à l'API: {str(e)}"

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({'error': 'Message vide'}), 400

    bot_response = get_chatbot_response(user_message)
    return jsonify({'response': bot_response})

if __name__ == "__main__":
    ensure_csv()
    app.run(debug=True)
