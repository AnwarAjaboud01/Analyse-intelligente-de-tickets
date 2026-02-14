import os
import sys
from openai import OpenAI

# Configuration
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Use the model set in env or default to a safe choice
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

RAG_TEMPLATE = """CONTEXTE:
{context}

QUESTION:
{question}

INSTRUCTIONS:
Réponds en te basant UNIQUEMENT sur le contexte fourni. Si tu ne trouves pas la réponse dis "Je ne trouve pas cette information dans les documents".
"""

def query_llm(context, question, client):
    prompt = RAG_TEMPLATE.format(context=context, question=question)
    
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "Tu es un assistant précis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0 # Low temperature for factual RAG
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erreur: {e}"

def main():
    if not GROQ_API_KEY:
        print("Erreur: La variable d'environnement GROQ_API_KEY n'est pas définie.")
        sys.exit(1)

    client = OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)

    print("=== TEST DU TEMPLATE RAG ===\n")

    # TEST 1: Information présente dans le contexte
    context_1 = "Le service informatique est ouvert de 9h à 18h du lundi au vendredi. Pour toute urgence le week-end, contactez le 0600000000."
    question_1 = "Quels sont les horaires d'ouverture ?"
    
    print(f"--- Test 1 (Info présente) ---")
    print(f"Contexte: {context_1}")
    print(f"Question: {question_1}")
    response_1 = query_llm(context_1, question_1, client)
    print(f"Réponse: {response_1}\n")

    # TEST 2: Information ABSENTE du contexte
    context_2 = "La procédure de réinitialisation de mot de passe se fait via le portail Self-Service."
    question_2 = "Quelle est la capitale du Maroc ?"
    
    print(f"--- Test 2 (Info absente) ---")
    print(f"Contexte: {context_2}")
    print(f"Question: {question_2}")
    response_2 = query_llm(context_2, question_2, client)
    print(f"Réponse: {response_2}\n")

if __name__ == "__main__":
    main()
