import os
import sys
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

# Configuration
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

RAG_TEMPLATE = """CONTEXTE:
{context}

QUESTION:
{question}

INSTRUCTIONS:
Réponds en te basant UNIQUEMENT sur le contexte fourni. Si tu ne trouves pas la réponse dis "Je ne trouve pas cette information dans les documents".
"""

# Initialisation de ChromaDB (en mémoire pour ce test, ou persistant si besoin)
# Pour ce test simple, on va créer une petite base temporaire
def setup_chroma():
    client = chromadb.Client()
    # Utilisation d'un modèle d'embedding par défaut (sentence-transformers/all-MiniLM-L6-v2)
    # Note: Cela téléchargera le modèle au premier lancement
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    collection = client.create_collection(name="tickets_knowledge", embedding_function=ef)
    
    # Ajout de documents de test "Maroc Telecom"
    documents = [
        "Pour les problèmes de connexion Maroc Telecom (IAM), vérifiez d'abord que le routeur Livebox n'a pas de voyant rouge clignotant.",
        "Si la connexion fibre optique est lente, redémarrez l'ONT (le petit boîtier blanc) avant le routeur.",
        "Le support technique Maroc Telecom est joignable au 115 pour les particuliers et 124 pour les pro.",
        "Pour configurer l'APN 4G IAM sur mobile: Nom=IAM, APN=www.iam.net.ma, le reste par défaut.",
        "En cas de coupure généralisée dans le quartier, attendre 1h avant d'appeler le support."
    ]
    
    ids = [f"doc_{i}" for i in range(len(documents))]
    
    collection.add(
        documents=documents,
        ids=ids
    )
    return collection

def ask_bot(query, collection, client_llm):
    print(f"\n🔍 Question: {query}")
    
    # Étape A : Chercher dans la base Chroma
    # On récupère les 3 documents les plus proches
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    
    documents_trouves = results['documents'][0]
    context_text = "\n\n".join(documents_trouves)
    print(f"📄 Contexte trouvé ({len(documents_trouves)} docs):\n{context_text[:200]}...") # Aperçu
    
    # Étape B : Construire le prompt
    prompt = RAG_TEMPLATE.format(context=context_text, question=query)
    
    # Étape C : Envoyer au LLM
    try:
        response = client_llm.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "Tu es un assistant support expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        answer = response.choices[0].message.content
        return answer
        
    except Exception as e:
        return f"Erreur LLM: {e}"

def main():
    if not GROQ_API_KEY:
        print("Erreur: GROQ_API_KEY manquant.")
        sys.exit(1)

    print("⏳ Initialisation de la base de connaissances (ChromaDB)...")
    collection = setup_chroma()
    
    client_llm = OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)
    
    # Test
    query = "Comment résoudre un problème de connexion Maroc Telecom?"
    reponse = ask_bot(query, collection, client_llm)
    
    print(f"\n🤖 Réponse du Bot:\n{reponse}\n")

if __name__ == "__main__":
    main()
