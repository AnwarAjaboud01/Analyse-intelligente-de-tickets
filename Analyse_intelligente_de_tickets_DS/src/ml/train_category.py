# =============================================================================
# Entraînement du Modèle de Catégorie (Classification multi-classes)
# =============================================================================
"""
MODÈLE 2 : Prédiction de la catégorie des tickets
- Classes : Multiples catégories (variable selon le dataset)
- Features : TF-IDF(text_full) + nb_mots + urgence_pred (encodé OneHot)
- Algorithme : LogisticRegression ou LinearSVC

ANTI-FUITE : utilise urgence_pred OOF (pas les vraies valeurs d'urgence)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import clone
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

from ml_utils import (
    load_data, ensure_models_dir, save_model, save_metadata, load_model,
    evaluate_classification, generate_oof_predictions_with_categorical,
    TFIDF_PARAMS, RANDOM_STATE, N_FOLDS, MODELS_DIR
)

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_NAME = "category_model"
PIPELINE_FILE = "category_pipeline.pkl"
TEXT_COLUMN = "text_full"
NUMERIC_COLUMNS = ["nb_mots"]
CATEGORICAL_PRED_COLUMNS = ["urgence_pred"]  # Prédiction upstream (pas le vrai label!)
TARGET_COLUMN = "categorie"

# =============================================================================
# FONCTION PRINCIPALE D'ENTRAÎNEMENT
# =============================================================================
def train_category_model(df_train: pd.DataFrame = None, 
                        df_val: pd.DataFrame = None, 
                        df_test: pd.DataFrame = None):
    """
    Entraîne le modèle de prédiction de catégorie.
    
    IMPORTANT : Nécessite que urgence_pred soit déjà présent dans les DataFrames.
    Si non fourni, charge les fichiers avec prédictions d'urgence.
    
    Args:
        df_train, df_val, df_test: DataFrames avec urgence_pred (optionnel)
        
    Returns:
        Tuple: (pipeline_dict, df_train avec oof_pred, df_val avec pred, df_test avec pred)
    """
    print("=" * 70)
    print("ENTRAÎNEMENT DU MODÈLE DE CATÉGORIE")
    print("=" * 70)
    print(f"Target : {TARGET_COLUMN}")
    print(f"Features : TF-IDF({TEXT_COLUMN}) + {NUMERIC_COLUMNS} + {CATEGORICAL_PRED_COLUMNS}")
    print(f"Algorithme : LogisticRegression (multi-class)")
    print()
    
    # -------------------------------------------------------------------------
    # ÉTAPE 1 : Chargement des données avec prédictions d'urgence
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("ÉTAPE 1 : Chargement des données")
    print("-" * 70)
    
    if df_train is None or df_val is None or df_test is None:
        # Charger les fichiers avec prédictions d'urgence
        try:
            df_train = pd.read_csv("data/train_with_urgence_pred.csv")
            df_val = pd.read_csv("data/val_with_urgence_pred.csv")
            df_test = pd.read_csv("data/test_with_urgence_pred.csv")
            print("✅ Chargé depuis les fichiers *_with_urgence_pred.csv")
        except FileNotFoundError:
            print("❌ Fichiers avec prédictions d'urgence non trouvés.")
            print("   Veuillez d'abord exécuter train_urgency.py")
            return None
    
    # Vérifier que urgence_pred existe
    if 'urgence_pred' not in df_train.columns:
        print("❌ Colonne 'urgence_pred' manquante dans le train set.")
        print("   Veuillez d'abord exécuter train_urgency.py")
        return None
    
    print(f"   Train      : {len(df_train)} lignes")
    print(f"   Validation : {len(df_val)} lignes")
    print(f"   Test       : {len(df_test)} lignes")
    
    # Identifier les classes de catégorie
    category_labels = sorted(df_train[TARGET_COLUMN].unique())
    n_classes = len(category_labels)
    print(f"\n📊 Nombre de catégories : {n_classes}")
    print(f"   Classes : {category_labels[:5]}..." if n_classes > 5 else f"   Classes : {category_labels}")
    
    # Distribution
    print(f"\n📊 Distribution des catégories (top 5):")
    dist = df_train[TARGET_COLUMN].value_counts()
    for label in dist.index[:5]:
        pct = dist[label] / len(df_train) * 100
        print(f"   {label}: {dist[label]} ({pct:.2f}%)")
    
    # -------------------------------------------------------------------------
    # ÉTAPE 2 : Génération des prédictions OOF sur le train set
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("ÉTAPE 2 : Génération des prédictions Out-Of-Fold (OOF)")
    print("-" * 70)
    print("""
    ANTI-FUITE : On utilise urgence_pred (pas urgence réelle) comme feature.
    Ces prédictions ont été générées en OOF donc pas de fuite.
    
    Maintenant on génère les prédictions OOF de CATÉGORIE pour les utiliser
    dans les modèles downstream (type_ticket, temps_resolution).
    """)
    
    # Définir le modèle de base
    base_model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
        solver='lbfgs',
        n_jobs=-1
    )
    
    # Générer les prédictions OOF
    oof_predictions = generate_oof_predictions_with_categorical(
        model=base_model,
        X=df_train,
        y=df_train[TARGET_COLUMN],
        text_column=TEXT_COLUMN,
        numeric_columns=NUMERIC_COLUMNS,
        categorical_pred_columns=CATEGORICAL_PRED_COLUMNS,
        n_folds=N_FOLDS,
        random_state=RANDOM_STATE
    )
    
    # Ajouter les prédictions OOF au DataFrame train
    df_train['categorie_pred'] = oof_predictions
    
    # Évaluation OOF
    print("\n📊 Performance OOF sur le Train Set:")
    oof_metrics = evaluate_classification(
        df_train[TARGET_COLUMN], 
        oof_predictions, 
        "Train (OOF)",
        labels=category_labels
    )
    
    # -------------------------------------------------------------------------
    # ÉTAPE 3 : Entraînement du modèle final sur tout le train set
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("ÉTAPE 3 : Entraînement du modèle FINAL sur tout le train set")
    print("-" * 70)
    
    # TF-IDF
    tfidf_final = TfidfVectorizer(**TFIDF_PARAMS)
    X_text_train = tfidf_final.fit_transform(df_train[TEXT_COLUMN].fillna(''))
    
    # Features numériques
    X_num_train = df_train[NUMERIC_COLUMNS].values
    
    # Encoder les prédictions catégorielles (urgence_pred)
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    X_cat_train = encoder.fit_transform(df_train[CATEGORICAL_PRED_COLUMNS])
    
    # Combiner toutes les features
    X_train_combined = hstack([X_text_train, csr_matrix(X_num_train), X_cat_train])
    
    print(f"   Shape features combinées : {X_train_combined.shape}")
    print(f"   - TF-IDF features : {X_text_train.shape[1]}")
    print(f"   - Numeric features : {len(NUMERIC_COLUMNS)}")
    print(f"   - Categorical features (urgence_pred encoded) : {X_cat_train.shape[1]}")
    
    # Entraîner le modèle final
    final_model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
        solver='lbfgs',
        n_jobs=-1
    )
    final_model.fit(X_train_combined, df_train[TARGET_COLUMN])
    print("   ✅ Modèle final entraîné")
    
    # -------------------------------------------------------------------------
    # ÉTAPE 4 : Prédictions et évaluation sur Validation et Test
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("ÉTAPE 4 : Évaluation sur Validation et Test")
    print("-" * 70)
    
    # Prédictions sur validation
    X_text_val = tfidf_final.transform(df_val[TEXT_COLUMN].fillna(''))
    X_num_val = df_val[NUMERIC_COLUMNS].values
    X_cat_val = encoder.transform(df_val[CATEGORICAL_PRED_COLUMNS])
    X_val_combined = hstack([X_text_val, csr_matrix(X_num_val), X_cat_val])
    
    val_predictions = final_model.predict(X_val_combined)
    df_val['categorie_pred'] = val_predictions
    
    val_metrics = evaluate_classification(
        df_val[TARGET_COLUMN],
        val_predictions,
        "VALIDATION",
        labels=category_labels
    )
    
    # Prédictions sur test
    X_text_test = tfidf_final.transform(df_test[TEXT_COLUMN].fillna(''))
    X_num_test = df_test[NUMERIC_COLUMNS].values
    X_cat_test = encoder.transform(df_test[CATEGORICAL_PRED_COLUMNS])
    X_test_combined = hstack([X_text_test, csr_matrix(X_num_test), X_cat_test])
    
    test_predictions = final_model.predict(X_test_combined)
    df_test['categorie_pred'] = test_predictions
    
    test_metrics = evaluate_classification(
        df_test[TARGET_COLUMN],
        test_predictions,
        "TEST",
        labels=category_labels
    )
    
    # -------------------------------------------------------------------------
    # ÉTAPE 5 : Sauvegarde du pipeline et des métadonnées
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("ÉTAPE 5 : Sauvegarde du modèle")
    print("-" * 70)
    
    ensure_models_dir()
    
    # Créer le dictionnaire pipeline
    pipeline_dict = {
        'model': final_model,
        'tfidf': tfidf_final,
        'encoder': encoder,
        'text_column': TEXT_COLUMN,
        'numeric_columns': NUMERIC_COLUMNS,
        'categorical_pred_columns': CATEGORICAL_PRED_COLUMNS,
        'target_column': TARGET_COLUMN,
        'labels': category_labels
    }
    
    save_model(pipeline_dict, PIPELINE_FILE, "Pipeline Catégorie")
    
    # Sauvegarder les métadonnées
    metadata = {
        'category_model': {
            'training_date': datetime.now().isoformat(),
            'algorithm': 'LogisticRegression (multinomial)',
            'features': {
                'text': TEXT_COLUMN,
                'numeric': NUMERIC_COLUMNS,
                'categorical_pred': CATEGORICAL_PRED_COLUMNS,
                'tfidf_params': TFIDF_PARAMS
            },
            'n_classes': n_classes,
            'classes': category_labels,
            'metrics': {
                'oof_train': oof_metrics,
                'validation': val_metrics,
                'test': test_metrics
            },
            'n_train_samples': len(df_train),
            'n_features': X_train_combined.shape[1]
        }
    }
    save_metadata(metadata)
    
    # -------------------------------------------------------------------------
    # RÉSUMÉ FINAL
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RÉSUMÉ - MODÈLE DE CATÉGORIE")
    print("=" * 70)
    print(f"""
    ┌─────────────────────────────────────────────────────────────────────┐
    │  MODÈLE DE CATÉGORIE (Classification {n_classes} classes)               │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Features : TF-IDF + nb_mots + urgence_pred (OneHot)                │
    │  Algorithme : LogisticRegression (multinomial)                      │
    │                                                                     │
    │  📊 PERFORMANCES :                                                  │
    │  ─────────────────────────────────────────────────────────────────  │
    │  Train (OOF)  : Accuracy={oof_metrics['accuracy']:.4f}  F1-Macro={oof_metrics['f1_macro']:.4f}  │
    │  Validation   : Accuracy={val_metrics['accuracy']:.4f}  F1-Macro={val_metrics['f1_macro']:.4f}  │
    │  Test         : Accuracy={test_metrics['accuracy']:.4f}  F1-Macro={test_metrics['f1_macro']:.4f}  │
    │                                                                     │
    │  ✅ Pipeline sauvegardé : models/{PIPELINE_FILE}            │
    └─────────────────────────────────────────────────────────────────────┘
    """)
    
    return pipeline_dict, df_train, df_val, df_test


# =============================================================================
# FONCTION DE PRÉDICTION POUR UTILISATION EXTERNE
# =============================================================================
def predict_category(pipeline_dict: dict, X: pd.DataFrame) -> np.ndarray:
    """
    Fait des prédictions de catégorie sur de nouvelles données.
    
    Args:
        pipeline_dict: Dictionnaire contenant le modèle et les transformers
        X: DataFrame avec text_full, nb_mots, et urgence_pred
        
    Returns:
        Array de prédictions
    """
    tfidf = pipeline_dict['tfidf']
    model = pipeline_dict['model']
    encoder = pipeline_dict['encoder']
    text_col = pipeline_dict['text_column']
    num_cols = pipeline_dict['numeric_columns']
    cat_cols = pipeline_dict['categorical_pred_columns']
    
    X_text = tfidf.transform(X[text_col].fillna(''))
    X_num = X[num_cols].values
    X_cat = encoder.transform(X[cat_cols])
    X_combined = hstack([X_text, csr_matrix(X_num), X_cat])
    
    return model.predict(X_combined)


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================
if __name__ == "__main__":
    result = train_category_model()
    
    if result is not None:
        pipeline, df_train, df_val, df_test = result
        
        # Sauvegarder les DataFrames avec les prédictions pour les modèles suivants
        df_train.to_csv("data/train_with_categorie_pred.csv", index=False)
        df_val.to_csv("data/val_with_categorie_pred.csv", index=False)
        df_test.to_csv("data/test_with_categorie_pred.csv", index=False)
        
        print("\n✅ DataFrames avec prédictions de catégorie sauvegardés pour les modèles downstream")
    
    print("=" * 70)
