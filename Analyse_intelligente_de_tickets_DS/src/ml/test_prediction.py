# =============================================================================
# Script de test rapide - Prédiction d'un ticket
# =============================================================================
"""
Usage: python src/ml/test_prediction.py
"""

import sys
import os

# Ajouter le chemin
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predict_pipeline import predict_ticket, display_results

# =============================================================================
# MODE INTERACTIF
# =============================================================================
if __name__ == "__main__":
    print()
    print("=" * 50)
    print("   PREDICTION DE TICKET - Mode Interactif")
    print("=" * 50)
    print()
    
    # Demander le titre
    print("Entrez le TITRE du ticket:")
    titre = input("> ")
    
    print()
    
    # Demander le texte
    print("Entrez le TEXTE/DESCRIPTION du ticket:")
    texte = input("> ")
    
    # Faire la prédiction
    print()
    print("Analyse en cours...")
    result = predict_ticket(titre, texte)
    display_results(result)
    
    print()
    print("Termine!")
