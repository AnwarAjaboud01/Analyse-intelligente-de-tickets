import sys
import os
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.llm.groq_predict import predict_ticket_groq

def main():
    print("\n==================================================")
    print("   PREDICTION DE TICKET - Mode Interactif (GROQ API)")
    print("==================================================\n")

    titre = input("Entrez le TITRE du ticket:\n> ").strip()
    texte = input("\nEntrez le TEXTE/DESCRIPTION du ticket:\n> ").strip()

    print("\nAnalyse en cours...\n")
    t0 = time.time()

    out = predict_ticket_groq(titre, texte)

    dt = time.time() - t0

    print("==================================================")
    print("📋 RÉSULTATS DE LA PRÉDICTION (GROQ)")
    print("==================================================")
    print(f"Urgence   : {out['urgence']}")
    print(f"Categorie : {out['categorie']}")
    print(f"Type      : {out['type_ticket']}")
    print(f"Temps(h)  : {out['temps_resolution']:.2f}")
    print("==================================================")
    print(f"Temps de réponse: {dt:.2f} sec")
    print("\nTermine!\n")

if __name__ == "__main__":
    main()
