# =============================================================================
# Entraînement du Modèle de Type de Ticket (Classification binaire)
# =============================================================================
"""
MODÈLE 3 : Prédiction du type de ticket
- Classes : Demande, Incident
- Features : TF-IDF(text_full) + nb_mots + urgence_pred + categorie_pred
- Algorithme : LogisticRegression avec class_weight='balanced'

ANTI-FUITE : utilise urgence_pred et categorie_pred OOF (pas les vraies valeurs!)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

from ml_utils import (
    ensure_models_dir, save_model, save_metadata,
    evaluate_classification,
    TFIDF_PARAMS, RANDOM_STATE, N_FOLDS, MODELS_DIR
)

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_NAME = "type_model"
PIPELINE_FILE = "type_pipeline.pkl"
TEXT_COLUMN = "text_full"
NUMERIC_COLUMNS = ["nb_mots"]
CATEGORICAL_PRED_COLUMNS = ["urgence_pred", "categorie_pred"]  # Prédictions upstream
TARGET_COLUMN = "type_ticket"
TYPE_LABELS = ['Demande', 'Incident']

# =============================================================================
# FONCTION DE GÉNÉRATION OOF POUR TYPE_TICKET
# =============================================================================
def generate_oof_type_predictions(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    text_column: str,
    numeric_columns: list,
    categorical_pred_columns: list,
    n_folds: int = N_FOLDS,
    random_state: int = RANDOM_STATE
) -> np.ndarray:
    """
    Génère des prédictions OOF pour le modèle de type de ticket.
    """
    oof_predictions = np.empty(len(X), dtype=object)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    print(f"   Génération OOF avec {n_folds} folds...")
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_fold_train = X.iloc[train_idx]
        X_fold_val = X.iloc[val_idx]
        y_fold_train = y.iloc[train_idx]
        
        # TF-IDF
        tfidf = TfidfVectorizer(**TFIDF_PARAMS)
        X_text_train = tfidf.fit_transform(X_fold_train[text_column].fillna(''))
        X_text_val = tfidf.transform(X_fold_val[text_column].fillna(''))
        
        # Features numériques
        X_num_train = X_fold_train[numeric_columns].values
        X_num_val = X_fold_val[numeric_columns].values
        
        # Encoder les prédictions catégorielles
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        X_cat_train = encoder.fit_transform(X_fold_train[categorical_pred_columns])
        X_cat_val = encoder.transform(X_fold_val[categorical_pred_columns])
        
        # Combiner
        X_train_combined = hstack([X_text_train, csr_matrix(X_num_train), X_cat_train])
        X_val_combined = hstack([X_text_val, csr_matrix(X_num_val), X_cat_val])
        
        # Entraîner et prédire
        model_fold = clone(model)
        model_fold.fit(X_train_combined, y_fold_train)
        oof_predictions[val_idx] = model_fold.predict(X_val_combined)
        
        print(f"      Fold {fold_idx + 1}/{n_folds} terminé")
    
    return oof_predictions


# =============================================================================
# FONCTION PRINCIPALE D'ENTRAÎNEMENT
# =============================================================================
def train_type_model(df_train: pd.DataFrame = None, 
                    df_val: pd.DataFrame = None, 
                    df_test: pd.DataFrame = None):
    """
    Entraîne le modèle de prédiction du type de ticket.
    
    IMPORTANT : Nécessite que urgence_pred ET categorie_pred soient déjà présents.
    Si non fourni, charge les fichiers avec prédictions de catégorie.
    
    Args:
        df_train, df_val, df_test: DataFrames avec urgence_pred et categorie_pred
        
    Returns:
        Tuple: (pipeline_dict, df_train avec oof_pred, df_val avec pred, df_test avec pred)
    """
    print("=" * 70)
    print("ENTRAÎNEMENT DU MODÈLE DE TYPE DE TICKET")
    print("=" * 70)
    print(f"Target : {TARGET_COLUMN}")
    print(f"Classes : {TYPE_LABELS}")
    print(f"Features : TF-IDF({TEXT_COLUMN}) + {NUMERIC_COLUMNS} + {CATEGORICAL_PRED_COLUMNS}")
    print(f"Algorithme : LogisticRegression (class_weight='balanced')")
    print()
    
    # -------------------------------------------------------------------------
    # ÉTAPE 1 : Chargement des données
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("ÉTAPE 1 : Chargement des données")
    print("-" * 70)
    
    if df_train is None or df_val is None or df_test is None:
        try:
            df_train = pd.read_csv("data/train_with_categorie_pred.csv")
            df_val = pd.read_csv("data/val_with_categorie_pred.csv")
            df_test = pd.read_csv("data/test_with_categorie_pred.csv")
            print("✅ Chargé depuis les fichiers *_with_categorie_pred.csv")
        except FileNotFoundError:
            print("❌ Fichiers avec prédictions de catégorie non trouvés.")
            print("   Veuillez d'abord exécuter train_category.py")
            return None
    
    # Vérifier les colonnes requises
    required_cols = ['urgence_pred', 'categorie_pred']
    missing = [col for col in required_cols if col not in df_train.columns]
    if missing:
        print(f"❌ Colonnes manquantes dans le train set: {missing}")
        return None
    
    print(f"   Train      : {len(df_train)} lignes")
    print(f"   Validation : {len(df_val)} lignes")
    print(f"   Test       : {len(df_test)} lignes")
    
    # Distribution
    print(f"\n📊 Distribution des types dans le train set:")
    dist = df_train[TARGET_COLUMN].value_counts()
    for label in TYPE_LABELS:
        if label in dist.index:
            pct = dist[label] / len(df_train) * 100
            print(f"   {label}: {dist[label]} ({pct:.2f}%)")
    
    # -------------------------------------------------------------------------
    # ÉTAPE 2 : Génération des prédictions OOF
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("ÉTAPE 2 : Génération des prédictions Out-Of-Fold (OOF)")
    print("-" * 70)
    print("""
    ANTI-FUITE : On utilise urgence_pred et categorie_pred (pas les vraies valeurs).
    Ces prédictions ont été générées en OOF donc pas de fuite.
    
    On génère maintenant les prédictions OOF de TYPE pour les utiliser
    dans le modèle de régression (temps_resolution).
    """)
    
    base_model = LogisticRegression(
        max_iter=2000,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        solver='lbfgs',
        n_jobs=-1
    )
    
    oof_predictions = generate_oof_type_predictions(
        model=base_model,
        X=df_train,
        y=df_train[TARGET_COLUMN],
        text_column=TEXT_COLUMN,
        numeric_columns=NUMERIC_COLUMNS,
        categorical_pred_columns=CATEGORICAL_PRED_COLUMNS,
        n_folds=N_FOLDS,
        random_state=RANDOM_STATE
    )
    
    df_train['type_ticket_pred'] = oof_predictions
    
    print("\n📊 Performance OOF sur le Train Set:")
    oof_metrics = evaluate_classification(
        df_train[TARGET_COLUMN], 
        oof_predictions, 
        "Train (OOF)",
        labels=TYPE_LABELS
    )
    
    # -------------------------------------------------------------------------
    # ÉTAPE 3 : Entraînement du modèle final
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("ÉTAPE 3 : Entraînement du modèle FINAL sur tout le train set")
    print("-" * 70)
    
    # TF-IDF
    tfidf_final = TfidfVectorizer(**TFIDF_PARAMS)
    X_text_train = tfidf_final.fit_transform(df_train[TEXT_COLUMN].fillna(''))
    
    # Features numériques
    X_num_train = df_train[NUMERIC_COLUMNS].values
    
    # Encoder les prédictions catégorielles
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    X_cat_train = encoder.fit_transform(df_train[CATEGORICAL_PRED_COLUMNS])
    
    X_train_combined = hstack([X_text_train, csr_matrix(X_num_train), X_cat_train])
    
    print(f"   Shape features combinées : {X_train_combined.shape}")
    print(f"   - TF-IDF features : {X_text_train.shape[1]}")
    print(f"   - Numeric features : {len(NUMERIC_COLUMNS)}")
    print(f"   - Categorical features : {X_cat_train.shape[1]}")
    
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
    # ÉTAPE 4 : Évaluation sur Validation et Test
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("ÉTAPE 4 : Évaluation sur Validation et Test")
    print("-" * 70)
    
    # Validation
    X_text_val = tfidf_final.transform(df_val[TEXT_COLUMN].fillna(''))
    X_num_val = df_val[NUMERIC_COLUMNS].values
    X_cat_val = encoder.transform(df_val[CATEGORICAL_PRED_COLUMNS])
    X_val_combined = hstack([X_text_val, csr_matrix(X_num_val), X_cat_val])
    
    val_predictions = final_model.predict(X_val_combined)
    df_val['type_ticket_pred'] = val_predictions
    
    val_metrics = evaluate_classification(
        df_val[TARGET_COLUMN],
        val_predictions,
        "VALIDATION",
        labels=TYPE_LABELS
    )
    
    # Test
    X_text_test = tfidf_final.transform(df_test[TEXT_COLUMN].fillna(''))
    X_num_test = df_test[NUMERIC_COLUMNS].values
    X_cat_test = encoder.transform(df_test[CATEGORICAL_PRED_COLUMNS])
    X_test_combined = hstack([X_text_test, csr_matrix(X_num_test), X_cat_test])
    
    test_predictions = final_model.predict(X_test_combined)
    df_test['type_ticket_pred'] = test_predictions
    
    test_metrics = evaluate_classification(
        df_test[TARGET_COLUMN],
        test_predictions,
        "TEST",
        labels=TYPE_LABELS
    )
    
    # -------------------------------------------------------------------------
    # ÉTAPE 5 : Sauvegarde
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("ÉTAPE 5 : Sauvegarde du modèle")
    print("-" * 70)
    
    ensure_models_dir()
    
    pipeline_dict = {
        'model': final_model,
        'tfidf': tfidf_final,
        'encoder': encoder,
        'text_column': TEXT_COLUMN,
        'numeric_columns': NUMERIC_COLUMNS,
        'categorical_pred_columns': CATEGORICAL_PRED_COLUMNS,
        'target_column': TARGET_COLUMN,
        'labels': TYPE_LABELS
    }
    
    save_model(pipeline_dict, PIPELINE_FILE, "Pipeline Type Ticket")
    
    metadata = {
        'type_model': {
            'training_date': datetime.now().isoformat(),
            'algorithm': 'LogisticRegression (balanced)',
            'features': {
                'text': TEXT_COLUMN,
                'numeric': NUMERIC_COLUMNS,
                'categorical_pred': CATEGORICAL_PRED_COLUMNS,
                'tfidf_params': TFIDF_PARAMS
            },
            'classes': TYPE_LABELS,
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
    # RÉSUMÉ
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RÉSUMÉ - MODÈLE DE TYPE DE TICKET")
    print("=" * 70)
    print(f"""
    ┌─────────────────────────────────────────────────────────────────────┐
    │  MODÈLE DE TYPE (Classification binaire)                            │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Features : TF-IDF + nb_mots + urgence_pred + categorie_pred        │
    │  Algorithme : LogisticRegression (balanced)                         │
    │                                                                     │
    │  📊 PERFORMANCES :                                                  │
    │  ─────────────────────────────────────────────────────────────────  │
    │  Train (OOF)  : Accuracy={oof_metrics['accuracy']:.4f}  F1-Macro={oof_metrics['f1_macro']:.4f}  │
    │  Validation   : Accuracy={val_metrics['accuracy']:.4f}  F1-Macro={val_metrics['f1_macro']:.4f}  │
    │  Test         : Accuracy={test_metrics['accuracy']:.4f}  F1-Macro={test_metrics['f1_macro']:.4f}  │
    │                                                                     │
    │  ✅ Pipeline sauvegardé : models/{PIPELINE_FILE}               │
    └─────────────────────────────────────────────────────────────────────┘
    """)
    
    return pipeline_dict, df_train, df_val, df_test


# =============================================================================
# FONCTION DE PRÉDICTION
# =============================================================================
def predict_type(pipeline_dict: dict, X: pd.DataFrame) -> np.ndarray:
    """Fait des prédictions de type de ticket."""
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
    result = train_type_model()
    
    if result is not None:
        pipeline, df_train, df_val, df_test = result
        
        # Sauvegarder pour le modèle de régression
        df_train.to_csv("data/train_with_type_pred.csv", index=False)
        df_val.to_csv("data/val_with_type_pred.csv", index=False)
        df_test.to_csv("data/test_with_type_pred.csv", index=False)
        
        print("\n✅ DataFrames avec prédictions de type sauvegardés pour le modèle de régression")
    
    print("=" * 70)
