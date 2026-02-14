# =============================================================================
# Entraînement du Modèle d'Urgence (Classification 3 classes)
# =============================================================================
"""
MODÈLE 1 : Prédiction de l'urgence des tickets
- Classes : Basse, Moyenne, Haute
- Features : TF-IDF(text_full) + nb_mots
- Algorithme : LogisticRegression avec class_weight='balanced'

Ce modèle est le PREMIER de la chaîne. Ses prédictions OOF seront utilisées
comme features pour les modèles downstream (category, type, time).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import clone
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

from ml_utils import (
    load_data, ensure_models_dir, save_model, save_metadata,
    evaluate_classification, generate_oof_predictions_classification,
    TFIDF_PARAMS, RANDOM_STATE, N_FOLDS, MODELS_DIR
)

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_NAME = "urgency_model"
PIPELINE_FILE = "urgency_pipeline.pkl"
TEXT_COLUMN = "text_full"
NUMERIC_COLUMNS = ["nb_mots"]
TARGET_COLUMN = "urgence"
URGENCY_LABELS = ['Basse', 'Moyenne', 'Haute']

# =============================================================================
# FONCTION PRINCIPALE D'ENTRAÎNEMENT
# =============================================================================
def train_urgency_model():
    """
    Entraîne le modèle de prédiction d'urgence.
    
    Steps:
    1. Charger les données
    2. Générer les prédictions OOF sur le train set
    3. Entraîner le modèle final sur tout le train set
    4. Évaluer sur validation et test
    5. Sauvegarder le pipeline et les métadonnées
    
    Returns:
        Tuple: (pipeline_dict, df_train avec oof_pred, df_val avec pred, df_test avec pred)
    """
    print("=" * 70)
    print("ENTRAÎNEMENT DU MODÈLE D'URGENCE")
    print("=" * 70)
    print(f"Target : {TARGET_COLUMN}")
    print(f"Classes : {URGENCY_LABELS}")
    print(f"Features : TF-IDF({TEXT_COLUMN}) + {NUMERIC_COLUMNS}")
    print(f"Algorithme : LogisticRegression (class_weight='balanced')")
    print()
    
    # -------------------------------------------------------------------------
    # ÉTAPE 1 : Chargement des données
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("ÉTAPE 1 : Chargement des données")
    print("-" * 70)
    
    df_train, df_val, df_test = load_data()
    
    # Vérifier les classes
    print(f"\n📊 Distribution d'urgence dans le train set:")
    dist = df_train[TARGET_COLUMN].value_counts()
    for label in URGENCY_LABELS:
        pct = dist[label] / len(df_train) * 100
        print(f"   {label}: {dist[label]} ({pct:.2f}%)")
    
    # -------------------------------------------------------------------------
    # ÉTAPE 2 : Génération des prédictions OOF sur le train set
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("ÉTAPE 2 : Génération des prédictions Out-Of-Fold (OOF)")
    print("-" * 70)
    print("""
    POURQUOI OOF ?
    Les prédictions d'urgence seront utilisées comme features pour les modèles
    downstream (category, type, time). Pour éviter la FUITE DE DONNÉES, on ne
    peut pas utiliser les prédictions du modèle entraîné sur tout le train set.
    
    Solution : K-Fold CV où chaque prédiction est faite par un modèle qui n'a
    JAMAIS vu cette donnée.
    """)
    
    # Définir le modèle de base
    base_model = LogisticRegression(
        max_iter=2000,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        solver='lbfgs',
        n_jobs=-1
    )
    
    # Générer les prédictions OOF
    oof_predictions = generate_oof_predictions_classification(
        model=base_model,
        X=df_train,
        y=df_train[TARGET_COLUMN],
        feature_columns=NUMERIC_COLUMNS,
        text_column=TEXT_COLUMN,
        n_folds=N_FOLDS,
        random_state=RANDOM_STATE
    )
    
    # Ajouter les prédictions OOF au DataFrame train
    df_train['urgence_pred'] = oof_predictions
    
    # Évaluation OOF
    print("\n📊 Performance OOF sur le Train Set:")
    oof_metrics = evaluate_classification(
        df_train[TARGET_COLUMN], 
        oof_predictions, 
        "Train (OOF)",
        labels=URGENCY_LABELS
    )
    
    # -------------------------------------------------------------------------
    # ÉTAPE 3 : Entraînement du modèle final sur tout le train set
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("ÉTAPE 3 : Entraînement du modèle FINAL sur tout le train set")
    print("-" * 70)
    
    # Créer et fitter le vectorizer TF-IDF final
    tfidf_final = TfidfVectorizer(**TFIDF_PARAMS)
    X_text_train = tfidf_final.fit_transform(df_train[TEXT_COLUMN].fillna(''))
    
    # Ajouter les features numériques
    X_num_train = df_train[NUMERIC_COLUMNS].values
    X_train_combined = hstack([X_text_train, csr_matrix(X_num_train)])
    
    print(f"   Shape features combinées : {X_train_combined.shape}")
    print(f"   - TF-IDF features : {X_text_train.shape[1]}")
    print(f"   - Numeric features : {len(NUMERIC_COLUMNS)}")
    
    # Entraîner le modèle final
    final_model = LogisticRegression(
        max_iter=2000,
        class_weight='balanced',
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
    X_val_combined = hstack([X_text_val, csr_matrix(X_num_val)])
    
    val_predictions = final_model.predict(X_val_combined)
    df_val['urgence_pred'] = val_predictions
    
    val_metrics = evaluate_classification(
        df_val[TARGET_COLUMN],
        val_predictions,
        "VALIDATION",
        labels=URGENCY_LABELS
    )
    
    # Prédictions sur test
    X_text_test = tfidf_final.transform(df_test[TEXT_COLUMN].fillna(''))
    X_num_test = df_test[NUMERIC_COLUMNS].values
    X_test_combined = hstack([X_text_test, csr_matrix(X_num_test)])
    
    test_predictions = final_model.predict(X_test_combined)
    df_test['urgence_pred'] = test_predictions
    
    test_metrics = evaluate_classification(
        df_test[TARGET_COLUMN],
        test_predictions,
        "TEST",
        labels=URGENCY_LABELS
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
        'text_column': TEXT_COLUMN,
        'numeric_columns': NUMERIC_COLUMNS,
        'target_column': TARGET_COLUMN,
        'labels': URGENCY_LABELS
    }
    
    save_model(pipeline_dict, PIPELINE_FILE, "Pipeline Urgence")
    
    # Sauvegarder les métadonnées
    metadata = {
        'urgency_model': {
            'training_date': datetime.now().isoformat(),
            'algorithm': 'LogisticRegression',
            'features': {
                'text': TEXT_COLUMN,
                'numeric': NUMERIC_COLUMNS,
                'tfidf_params': TFIDF_PARAMS
            },
            'classes': URGENCY_LABELS,
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
    print("RÉSUMÉ - MODÈLE D'URGENCE")
    print("=" * 70)
    print(f"""
    ┌─────────────────────────────────────────────────────────────────────┐
    │  MODÈLE D'URGENCE (Classification 3 classes)                        │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Features : TF-IDF(text_full) + nb_mots                             │
    │  Algorithme : LogisticRegression (balanced)                         │
    │                                                                     │
    │  📊 PERFORMANCES :                                                  │
    │  ─────────────────────────────────────────────────────────────────  │
    │  Train (OOF)  : Accuracy={oof_metrics['accuracy']:.4f}  F1-Macro={oof_metrics['f1_macro']:.4f}  │
    │  Validation   : Accuracy={val_metrics['accuracy']:.4f}  F1-Macro={val_metrics['f1_macro']:.4f}  │
    │  Test         : Accuracy={test_metrics['accuracy']:.4f}  F1-Macro={test_metrics['f1_macro']:.4f}  │
    │                                                                     │
    │  ✅ Pipeline sauvegardé : models/{PIPELINE_FILE}              │
    └─────────────────────────────────────────────────────────────────────┘
    """)
    
    return pipeline_dict, df_train, df_val, df_test


# =============================================================================
# FONCTION DE PRÉDICTION POUR UTILISATION EXTERNE
# =============================================================================
def predict_urgency(pipeline_dict: dict, X: pd.DataFrame) -> np.ndarray:
    """
    Fait des prédictions d'urgence sur de nouvelles données.
    
    Args:
        pipeline_dict: Dictionnaire contenant le modèle et les transformers
        X: DataFrame avec les colonnes text_full et nb_mots
        
    Returns:
        Array de prédictions
    """
    tfidf = pipeline_dict['tfidf']
    model = pipeline_dict['model']
    text_col = pipeline_dict['text_column']
    num_cols = pipeline_dict['numeric_columns']
    
    X_text = tfidf.transform(X[text_col].fillna(''))
    X_num = X[num_cols].values
    X_combined = hstack([X_text, csr_matrix(X_num)])
    
    return model.predict(X_combined)


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================
if __name__ == "__main__":
    pipeline, df_train, df_val, df_test = train_urgency_model()
    
    # Sauvegarder les DataFrames avec les prédictions pour les modèles suivants
    df_train.to_csv("data/train_with_urgence_pred.csv", index=False)
    df_val.to_csv("data/val_with_urgence_pred.csv", index=False)
    df_test.to_csv("data/test_with_urgence_pred.csv", index=False)
    
    print("\n✅ DataFrames avec prédictions d'urgence sauvegardés pour les modèles downstream")
    print("=" * 70)
