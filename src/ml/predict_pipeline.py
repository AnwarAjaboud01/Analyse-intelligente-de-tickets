# =============================================================================
# Pipeline d'Inférence - Prédiction de Tickets
# =============================================================================
"""
Script d'inférence pour prédire les caractéristiques d'un ticket à partir
de son titre et texte.

Prédictions :
    1. Urgence (Basse, Moyenne, Haute)
    2. Catégorie (50 classes)
    3. Type de ticket (Demande, Incident)
    4. Temps de résolution estimé (en heures)

Usage :
    python src/ml/predict_pipeline.py
    
    Puis entrer le titre et le texte du ticket quand demandé.

Modèles requis (dans models/) :
    - urgency_pipeline.pkl
    - category_pipeline.pkl
    - type_pipeline.pkl
    - time_pipeline.pkl
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

# Chemin vers le dossier des modèles (relatif à la racine du projet)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Fichiers des pipelines
PIPELINE_FILES = {
    'urgency': 'urgency_pipeline.pkl',
    'category': 'category_pipeline.pkl',
    'type': 'type_pipeline.pkl',
    'time': 'time_pipeline.pkl'
}


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def check_model_exists(model_name: str) -> str:
    """
    Vérifie qu'un fichier de modèle existe et retourne son chemin complet.
    
    Args:
        model_name: Nom du modèle ('urgency', 'category', 'type', 'time')
        
    Returns:
        Chemin complet vers le fichier du modèle
        
    Raises:
        FileNotFoundError: Si le fichier n'existe pas
    """
    if model_name not in PIPELINE_FILES:
        raise ValueError(f"Modèle inconnu: {model_name}. "
                        f"Modèles disponibles: {list(PIPELINE_FILES.keys())}")
    
    filename = PIPELINE_FILES[model_name]
    filepath = os.path.join(MODELS_DIR, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\n{'='*60}\n"
            f"❌ ERREUR: Modèle '{model_name}' introuvable!\n"
            f"{'='*60}\n"
            f"Fichier manquant: {filepath}\n\n"
            f"Pour résoudre ce problème, exécutez d'abord les scripts d'entraînement:\n"
            f"  1. python src/ml/train_urgency.py\n"
            f"  2. python src/ml/train_category.py\n"
            f"  3. python src/ml/train_type_ticket.py\n"
            f"  4. python src/ml/train_time_regression.py\n"
            f"{'='*60}"
        )
    
    return filepath


def load_pipeline(model_name: str) -> dict:
    """
    Charge un pipeline de modèle depuis un fichier .pkl.
    
    Args:
        model_name: Nom du modèle à charger
        
    Returns:
        Dictionnaire contenant le modèle et ses composants (TF-IDF, encoder, etc.)
    """
    filepath = check_model_exists(model_name)
    pipeline = joblib.load(filepath)
    return pipeline


def prepare_features(titre: str, texte: str) -> pd.DataFrame:
    """
    Prépare les features à partir du titre et du texte du ticket.
    
    Args:
        titre: Titre du ticket
        texte: Corps du texte du ticket
        
    Returns:
        DataFrame avec une ligne contenant text_full et nb_mots
    """
    # Concaténer titre et texte
    text_full = f"{titre} {texte}".strip()
    
    # Compter le nombre de mots
    nb_mots = len(text_full.split()) if text_full else 0
    
    # Créer le DataFrame
    df = pd.DataFrame({
        'text_full': [text_full],
        'nb_mots': [nb_mots]
    })
    
    return df


def validate_input(titre: str, texte: str) -> None:
    """
    Valide que l'entrée utilisateur n'est pas vide.
    
    Args:
        titre: Titre du ticket
        texte: Corps du texte du ticket
        
    Raises:
        ValueError: Si le texte combiné est vide ou trop court
    """
    text_full = f"{titre} {texte}".strip()
    
    if not text_full:
        raise ValueError(
            "\n❌ ERREUR: Le titre et le texte sont vides!\n"
            "Veuillez entrer au moins un titre ou un texte significatif."
        )
    
    if len(text_full.split()) < 2:
        raise ValueError(
            "\n⚠️ AVERTISSEMENT: Le texte est trop court (moins de 2 mots).\n"
            "Pour de meilleures prédictions, fournissez plus de détails."
        )


# =============================================================================
# FONCTIONS DE PRÉDICTION
# =============================================================================

def predict_urgency(pipeline: dict, df: pd.DataFrame) -> str:
    """
    Prédit l'urgence du ticket.
    
    Args:
        pipeline: Pipeline chargé pour l'urgence
        df: DataFrame avec text_full et nb_mots
        
    Returns:
        Urgence prédite (Basse, Moyenne, Haute)
    """
    from scipy.sparse import hstack, csr_matrix
    
    tfidf = pipeline['tfidf']
    model = pipeline['model']
    text_col = pipeline['text_column']
    num_cols = pipeline.get('numeric_columns', ['nb_mots'])
    
    # Vectoriser le texte
    X_text = tfidf.transform(df[text_col].fillna(''))
    
    # Ajouter les features numériques (nb_mots)
    X_num = df[num_cols].values
    X_combined = hstack([X_text, csr_matrix(X_num)])
    
    # Prédire
    prediction = model.predict(X_combined)[0]
    
    return prediction


def predict_category(pipeline: dict, df: pd.DataFrame) -> str:
    """
    Prédit la catégorie du ticket.
    
    Args:
        pipeline: Pipeline chargé pour la catégorie
        df: DataFrame avec text_full, nb_mots, urgence_pred
        
    Returns:
        Catégorie prédite
    """
    from scipy.sparse import hstack, csr_matrix
    
    tfidf = pipeline['tfidf']
    model = pipeline['model']
    encoder = pipeline['encoder']
    text_col = pipeline['text_column']
    num_cols = pipeline.get('numeric_columns', ['nb_mots'])
    cat_cols = pipeline.get('categorical_pred_columns', ['urgence_pred'])
    
    # Vectoriser le texte
    X_text = tfidf.transform(df[text_col].fillna(''))
    
    # Features numériques
    X_num = df[num_cols].values
    
    # Encoder les prédictions catégorielles (urgence_pred)
    X_cat = encoder.transform(df[cat_cols])
    
    # Combiner toutes les features
    X_combined = hstack([X_text, csr_matrix(X_num), X_cat])
    
    # Prédire
    prediction = model.predict(X_combined)[0]
    
    return prediction


def predict_type(pipeline: dict, df: pd.DataFrame) -> str:
    """
    Prédit le type de ticket.
    
    Args:
        pipeline: Pipeline chargé pour le type
        df: DataFrame avec text_full, nb_mots, urgence_pred, categorie_pred
        
    Returns:
        Type prédit (Demande, Incident)
    """
    from scipy.sparse import hstack, csr_matrix
    
    tfidf = pipeline['tfidf']
    model = pipeline['model']
    encoder = pipeline['encoder']
    text_col = pipeline['text_column']
    num_cols = pipeline.get('numeric_columns', ['nb_mots'])
    cat_cols = pipeline.get('categorical_pred_columns', ['urgence_pred', 'categorie_pred'])
    
    # Vectoriser le texte
    X_text = tfidf.transform(df[text_col].fillna(''))
    
    # Features numériques
    X_num = df[num_cols].values
    
    # Encoder les prédictions catégorielles (urgence_pred + categorie_pred)
    X_cat = encoder.transform(df[cat_cols])
    
    # Combiner toutes les features
    X_combined = hstack([X_text, csr_matrix(X_num), X_cat])
    
    # Prédire
    prediction = model.predict(X_combined)[0]
    
    return prediction


def predict_time(pipeline: dict, df: pd.DataFrame) -> float:
    """
    Prédit le temps de résolution du ticket.
    
    Args:
        pipeline: Pipeline chargé pour le temps
        df: DataFrame avec text_full, nb_mots, urgence_pred, categorie_pred, type_ticket_pred
        
    Returns:
        Temps de résolution prédit (en heures)
    """
    from scipy.sparse import hstack, csr_matrix
    
    tfidf = pipeline['tfidf']
    model = pipeline['model']
    encoder = pipeline['encoder']
    text_col = pipeline['text_column']
    num_cols = pipeline['numeric_columns']
    cat_cols = pipeline['categorical_pred_columns']
    
    # Vectoriser le texte
    X_text = tfidf.transform(df[text_col].fillna(''))
    
    # Features numériques
    X_num = df[num_cols].values
    
    # Encoder les prédictions catégorielles
    X_cat = encoder.transform(df[cat_cols])
    
    # Combiner toutes les features
    X_combined = hstack([X_text, csr_matrix(X_num), X_cat])
    
    # Prédire
    prediction = model.predict(X_combined)[0]
    
    # Assurer une valeur positive
    prediction = max(0, prediction)
    
    return prediction


# =============================================================================
# PIPELINE PRINCIPAL D'INFÉRENCE
# =============================================================================

def predict_ticket(titre: str, texte: str) -> dict:
    """
    Pipeline complet de prédiction pour un ticket.
    
    Chaîne séquentielle :
    1. Urgence → 2. Catégorie → 3. Type → 4. Temps
    
    Args:
        titre: Titre du ticket
        texte: Corps du texte du ticket
        
    Returns:
        Dictionnaire avec toutes les prédictions
    """
    # Valider l'entrée
    validate_input(titre, texte)
    
    # Préparer les features de base
    df = prepare_features(titre, texte)
    
    # -------------------------------------------------------------------------
    # ÉTAPE 1 : Prédiction de l'urgence
    # -------------------------------------------------------------------------
    urgency_pipeline = load_pipeline('urgency')
    urgence_pred = predict_urgency(urgency_pipeline, df)
    df['urgence_pred'] = urgence_pred
    
    # -------------------------------------------------------------------------
    # ÉTAPE 2 : Prédiction de la catégorie
    # -------------------------------------------------------------------------
    category_pipeline = load_pipeline('category')
    categorie_pred = predict_category(category_pipeline, df)
    df['categorie_pred'] = categorie_pred
    
    # -------------------------------------------------------------------------
    # ÉTAPE 3 : Prédiction du type de ticket
    # -------------------------------------------------------------------------
    type_pipeline = load_pipeline('type')
    type_ticket_pred = predict_type(type_pipeline, df)
    df['type_ticket_pred'] = type_ticket_pred
    
    # -------------------------------------------------------------------------
    # ÉTAPE 4 : Prédiction du temps de résolution
    # -------------------------------------------------------------------------
    time_pipeline = load_pipeline('time')
    temps_resolution_pred = predict_time(time_pipeline, df)
    
    # Construire le résultat (temps arrondi à 2 décimales)
    result = {
        'urgence_pred': urgence_pred,
        'categorie_pred': categorie_pred,
        'type_ticket_pred': type_ticket_pred,
        'temps_resolution_pred': float(f"{temps_resolution_pred:.2f}")
    }
    
    return result


def display_results(result: dict) -> None:
    """
    Affiche les résultats de prédiction de manière formatée.
    
    Args:
        result: Dictionnaire des prédictions
    """
    print("\n" + "=" * 50)
    print("📋 RÉSULTATS DE LA PRÉDICTION")
    print("=" * 50)
    print(f"Urgence   : {result['urgence_pred']}")
    print(f"Categorie : {result['categorie_pred']}")
    print(f"Type      : {result['type_ticket_pred']}")
    print(f"Temps(h)  : {result['temps_resolution_pred']:.2f}")
    print("=" * 50)


# =============================================================================
# INTERFACE CLI
# =============================================================================

def main():
    """
    Point d'entrée CLI pour l'inférence interactive.
    """
    print("\n" + "=" * 50)
    print("🎫 SYSTÈME DE PRÉDICTION DE TICKETS")
    print("=" * 50)
    print("Entrez les informations du ticket ci-dessous.")
    print()
    
    try:
        # Demander les entrées utilisateur
        print("📌 Titre du ticket:")
        titre = input("   > ").strip()
        
        print("\n📝 Description/Texte du ticket:")
        texte = input("   > ").strip()
        
        # Exécuter la prédiction
        result = predict_ticket(titre, texte)
        
        # Afficher les résultats
        display_results(result)
        
        return result
        
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)
        
    except ValueError as e:
        print(str(e))
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Annulé par l'utilisateur.")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        sys.exit(1)


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    main()
