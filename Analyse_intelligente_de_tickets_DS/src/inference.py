import sys
import os
from typing import Dict, Any

# =============================================================================
# Inference Module - Text Analysis Pipeline
# =============================================================================

# This module handles the text analysis and classification logic.
# It uses a specialized transformer-based pipeline for ticket classification.

# Add parent directory to path to access core libraries
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up to project root (../../ from src/inference.py inside DS folder)
project_root = os.path.abspath(os.path.join(current_dir, "../../"))

if project_root not in sys.path:
    sys.path.append(project_root)

# Import the inference engine from the core module
try:
    from src.llm.transformer_engine import predict_inference_pipeline as _predict_internal
    from src.llm.transformer_engine import get_chatbot_response as _chat_internal
except ImportError:
    # Fallback in case of path issues during development
    print("Warning: Could not load transformer engine. Ensure project root is in PYTHONPATH.")
    
    def _predict_internal(*args, **kwargs):
        return {
            "urgence": "Moyenne",
            "categorie": "Autre",
            "type_ticket": "Demande",
            "temps_resolution": 1.0
        }
        
    def _chat_internal(msg):
        return "Service indisponible."

def predict_ticket(titre: str, texte: str) -> Dict[str, Any]:
    """
    Main entry point for ticket prediction.
    Categorizes the ticket based on the title and description.
    
    Args:
        titre (str): The subject of the ticket
        texte (str): The body/description of the ticket
        
    Returns:
        Dict[str, Any]: A dictionary containing normalized prediction results:
            - urgence: "Basse", "Moyenne", or "Haute"
            - categorie: Assigned category
            - type_ticket: "Incident" or "Demande"
            - temps_resolution: Estimated resolution time in hours
    """
    # Use the internal engine for prediction
    return _predict_internal(titre, texte)

def get_chatbot_response(message: str) -> str:
    """
    Get response from the AI assistant.
    """
    return _chat_internal(message)
