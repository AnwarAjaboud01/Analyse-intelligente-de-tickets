import os
import sys

# Ensure we can import from src if needed, or just use standard libs
# Using the Groq setup consistent with the project

from openai import OpenAI

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Use the model set in env or default to a safe choice
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

SYSTEM_PROMPT = "Tu es un assistant support utile qui connaît le contexte marocain (français arabe dialecte)."

def main():
    if not GROQ_API_KEY:
        print("Erreur: La variable d'environnement GROQ_API_KEY n'est pas définie.")
        sys.exit(1)

    client = OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)
    
    question = "Quelle est la capitale du Maroc ?"
    print(f"Question: {question}")
    
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ],
            temperature=0.7
        )
        
        answer = response.choices[0].message.content
        print(f"Réponse: {answer}")
        
        if "Rabat" in answer or "rabat" in answer:
            print("\nRESULTAT: LLM fonctionne")
        else:
             print("\nRESULTAT: Réponse inattendue, vérifiez le contenu.")

    except Exception as e:
        print(f"Erreur lors de l'appel à l'API: {e}")

if __name__ == "__main__":
    main()
