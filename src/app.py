import streamlit as st
import os
import sys
import pandas as pd
import plotly.express as px
from datetime import datetime
import json

# Ajouter le chemin racine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Imports
from llm.simple_rag_bot import ask_bot
from llm.transformer_engine import predict_inference_pipeline as predict_ticket

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI

# -----------------------------------------------------------------------------
# CONFIGURATION DE LA PAGE
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Ticket AI - Dashboard",
    page_icon="🎫",
    layout="wide"
)

# CSS Custom
st.markdown("""
<style>
    .main-header {font-size: 2.2rem; color: #4F46E5; font-weight: 700; margin-bottom: 0.5rem;}
    .sub-header {font-size: 1.3rem; color: #374151; font-weight: 600; margin-top: 1rem;}
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #e5e7eb;
    }
    .metric-val {font-size: 2rem; font-weight: bold; color: #111827;}
    .metric-lbl {font-size: 0.9rem; color: #6B7280; text-transform: uppercase; letter-spacing: 0.05em;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# INITIALISATION
# -----------------------------------------------------------------------------
@st.cache_resource
def load_resources():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("🚨 Clé API GROQ manquante ! Définissez GROQ_API_KEY.")
        return None, None
    
    client_llm = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
    
    persist_directory = "./chroma_db"
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if os.path.exists(persist_directory):
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        db = None
        
    return client_llm, db

client_llm, db = load_resources()

# -----------------------------------------------------------------------------
# GESTION DES DONNÉES LIVE (Stockage local simple JSON)
# -----------------------------------------------------------------------------
DB_FILE = "tickets_db.json"

def load_live_data():
    if os.path.exists(DB_FILE):
        try:
            return pd.read_json(DB_FILE)
        except:
            return pd.DataFrame()
    return pd.DataFrame()

def save_ticket(ticket_data):
    df = load_live_data()
    
    new_row = pd.DataFrame([ticket_data])
    
    if not df.empty:
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row
        
    df.to_json(DB_FILE, orient='records', date_format='iso')
    return df

# Charger les données au démarrage
df_tickets = load_live_data()

# -----------------------------------------------------------------------------
# INTERFACE
# -----------------------------------------------------------------------------

st.markdown('<div class="main-header">🎫 Analyse Intelligente & Dashboard IT</div>', unsafe_allow_html=True)

# Tabs
tab_new, tab_dash, tab_bot = st.tabs(["📝 Nouveau Ticket", "📊 Dashboard Analytics", "🤖 Assistant IA"])

# --- TAB 1 : NOUVEAU TICKET (ANALYSE + SAVE) ---
with tab_new:
    st.markdown('<div class="sub-header">Qualification & Enregistrement</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        titre = st.text_input("Titre du ticket", placeholder="Ex: Panne wifi salle réunion")
        description = st.text_area("Description", height=150, placeholder="Détails du problème...")
        analyze_btn = st.button("🚀 Analyser et Enregistrer", type="primary")

    if analyze_btn and titre and description:
        with st.spinner("Analyse sémantique en cours..."):
            try:
                # Appel API Groq
                result = predict_ticket(titre, description)
                
                # Sauvegarde du ticket
                new_ticket = {
                    "Date": datetime.now().isoformat(),
                    "Titre": titre,
                    "Description": description,
                    "Catégorie": result.get('categorie', 'Autre'),
                    "Urgence": result.get('urgence', 'Moyenne'),
                    "Type": result.get('type_ticket', 'Demande'),
                    "Temps Résolution (h)": result.get('temps_resolution', 0),
                    "Statut": "Nouveau"
                }
                save_ticket(new_ticket)
                
                # Feedback UI
                st.success("Ticket analysé et enregistré avec succès !")
                st.balloons()
                
                # Cartes de résultats
                c1, c2, c3, c4 = st.columns(4)
                
                def metric_card(label, value, color="black"):
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-lbl">{label}</div>
                        <div class="metric-val" style="color: {color};">{value}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with c1:
                    color = "#ef4444" if result.get('urgence') == "Haute" else "#3b82f6"
                    metric_card("Urgence", result.get('urgence', '-'), color)
                with c2:
                    metric_card("Catégorie", result.get('categorie', '-'))
                with c3:
                    metric_card("Type", result.get('type_ticket', '-'))
                with c4:
                    metric_card("Temps Est.", f"{result.get('temps_resolution', 0)} h")
                
                # Recharger les données pour que le dashboard soit à jour au prochain clic
                st.cache_data.clear()

            except Exception as e:
                st.error(f"Erreur: {e}")

# --- TAB 2 : DASHBOARD ANALYTICS ---
with tab_dash:
    st.markdown('<div class="sub-header">Vue d\'ensemble (Données en temps réel)</div>', unsafe_allow_html=True)
    
    # Relecture fraîche des données
    df_live = load_live_data()
    
    if df_live.empty:
        st.warning("Aucun ticket enregistré pour le moment. Allez dans l'onglet 'Nouveau Ticket' !")
    else:
        # Conversion de date
        if 'Date' in df_live.columns:
            df_live['Date'] = pd.to_datetime(df_live['Date'])

        # KPIs en haut
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Total Tickets", len(df_live))
        with k2:
            open_tickets = len(df_live[df_live['Statut'].isin(["Nouveau", "En cours"])])
            st.metric("Tickets Ouverts", open_tickets)
        with k3:
            avg_time = df_live["Temps Résolution (h)"].mean()
            st.metric("Temps Moyen Résol.", f"{avg_time:.1f} h")
        with k4:
            critical = len(df_live[df_live['Urgence'] == "Haute"])
            st.metric("Urgences Hautes", critical, delta_color="inverse")

        st.markdown("---")

        # GRAPHIQUES (Plotly)
        g1, g2 = st.columns(2)
        
        with g1:
            st.markdown("**Répartition par Catégorie**")
            fig_cat = px.pie(df_live, names='Catégorie', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_cat, use_container_width=True)
            
        with g2:
            st.markdown("**Urgence par Statut**")
            fig_bar = px.bar(df_live, x="Statut", color="Urgence", barmode="group", 
                             color_discrete_map={"Haute": "#ef4444", "Moyenne": "#f59e0b", "Basse": "#10b981"})
            st.plotly_chart(fig_bar, use_container_width=True)

        # Tableau des derniers tickets
        st.markdown("**Derniers Tickets Enregistrés**")
        st.dataframe(df_live.sort_values(by="Date", ascending=False).head(10), use_container_width=True)

# --- TAB 3 : CHATBOT RAG ---
with tab_bot:
    st.markdown('<div class="sub-header">Assistant Virtuel</div>', unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Posez une question technique..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if db and client_llm:
                with st.spinner("Recherche..."):
                    docs = db.similarity_search(prompt, k=3)
                    context = "\n\n".join([d.page_content for d in docs])
                    
                    rag_prompt = f"""CONTEXTE:\n{context}\n\nQUESTION:\n{prompt}\n\nINSTRUCTIONS:\nRéponds en français, de manière concise. Base-toi UNIQUEMENT sur le contexte fourni. Si tu ne sais pas, dis-le."""
                    
                    try:
                        resp = client_llm.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": rag_prompt}],
                            temperature=0.0
                        )
                        response_text = resp.choices[0].message.content
                    except Exception as e:
                        response_text = f"Erreur API: {e}"
            else:
                response_text = "⚠️ Base de connaissances déconnectée."
            
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
