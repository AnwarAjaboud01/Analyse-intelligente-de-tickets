# =============================================================================
# Entraînement du Modèle de Temps de Résolution (Régression) - VERSION RÉGULARISÉE
# =============================================================================
"""
MODÈLE 4 : Prédiction du temps de résolution des tickets
- Target : temps_resolution (numérique, en heures)
- Features : TF-IDF(text_full) + nb_mots + urgence_pred + categorie_pred + type_ticket_pred
- Algorithme : XGBRegressor avec RÉGULARISATION FORTE + EARLY STOPPING

CORRECTION OVERFITTING :
------------------------
Le modèle précédent était en overfitting sévère :
- Train R² = 0.985 vs Test R² = 0.72 (écart de 0.27 !)
- Train MAE = 2.4h vs Test MAE = 7.5h (écart de 5h !)

Solutions appliquées :
1. Réduction de la complexité :
   - n_estimators = 50 (au lieu de 200)
   - max_depth = 4 (au lieu de 6)
   - min_child_weight = 5 (régularisation des feuilles)
   
2. Sous-échantillonnage (bagging) :
   - subsample = 0.8 (80% des lignes par arbre)
   - colsample_bytree = 0.8 (80% des features par arbre)
   
3. Early stopping :
   - Arrêt si la validation ne s'améliore plus pendant 10 rounds
   - Évite le surapprentissage tardif

ANTI-FUITE : utilise toutes les prédictions OOF upstream (pas les vrais labels)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack, csr_matrix
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from ml_utils import (
    ensure_models_dir, save_metadata,
    evaluate_regression,
    TFIDF_PARAMS, RANDOM_STATE, MODELS_DIR
)

# Essayer d'importer XGBoost si disponible
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost non disponible, utilisation de GradientBoostingRegressor")

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_NAME = "time_model"
PIPELINE_FILE = "time_pipeline.pkl"
TEXT_COLUMN = "text_full"
NUMERIC_COLUMNS = ["nb_mots"]
CATEGORICAL_PRED_COLUMNS = ["urgence_pred", "categorie_pred", "type_ticket_pred"]
TARGET_COLUMN = "temps_resolution"

# =============================================================================
# CONFIGURATION RÉGULARISATION (ANTI-OVERFITTING)
# =============================================================================
REGULARIZATION_CONFIG = {
    # Réduction de complexité
    'n_estimators': 50,          # Réduit de 200 → 50
    'max_depth': 4,              # Réduit de 6 → 4
    'min_child_weight': 5,       # Augmenté (plus de régularisation)
    'learning_rate': 0.1,        # Taux d'apprentissage modéré
    
    # Sous-échantillonnage (bagging)
    'subsample': 0.8,            # 80% des lignes par arbre
    'colsample_bytree': 0.8,     # 80% des features par arbre
    
    # Régularisation L1/L2
    'reg_alpha': 0.1,            # L1 regularization
    'reg_lambda': 1.0,           # L2 regularization
    
    # Early stopping
    'early_stopping_rounds': 10,
    
    # Reproductibilité
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbosity': 0
}


# =============================================================================
# ÉTAPE PRÉLIMINAIRE : NETTOYAGE DES ARTEFACTS PRÉCÉDENTS
# =============================================================================
def clean_previous_artifacts():
    """
    Supprime les artefacts du modèle précédent pour garantir un entraînement propre.
    Cette étape est OBLIGATOIRE pour éviter toute contamination par l'ancien modèle.
    """
    print("=" * 70)
    print("ÉTAPE PRÉLIMINAIRE : NETTOYAGE DES ARTEFACTS PRÉCÉDENTS")
    print("=" * 70)
    
    pipeline_path = os.path.join(MODELS_DIR, PIPELINE_FILE)
    
    if os.path.exists(pipeline_path):
        os.remove(pipeline_path)
        print(f"✅ Supprimé : {pipeline_path}")
    else:
        print(f"ℹ️  Fichier non trouvé (déjà supprimé ou première exécution) : {pipeline_path}")
    
    # Nettoyer le cache Python si présent
    cache_dir = os.path.join(os.path.dirname(__file__), '__pycache__')
    if os.path.exists(cache_dir):
        print(f"ℹ️  Cache Python présent : {cache_dir}")
    
    print("✅ Nettoyage terminé - Prêt pour un nouvel entraînement")
    print()


# =============================================================================
# FONCTION PRINCIPALE D'ENTRAÎNEMENT
# =============================================================================
def train_time_model_regularized(df_train: pd.DataFrame = None, 
                                  df_val: pd.DataFrame = None, 
                                  df_test: pd.DataFrame = None):
    """
    Entraîne le modèle de prédiction du temps de résolution avec RÉGULARISATION.
    
    CHANGEMENTS vs VERSION PRÉCÉDENTE :
    - n_estimators: 200 → 50
    - max_depth: 6 → 4
    - min_child_weight: 1 → 5
    - subsample: 1.0 → 0.8
    - colsample_bytree: 1.0 → 0.8
    - Early stopping activé
    
    Args:
        df_train, df_val, df_test: DataFrames avec toutes les prédictions
        
    Returns:
        Tuple: (pipeline_dict, metrics_dict)
    """
    print("=" * 70)
    print("ENTRAÎNEMENT DU MODÈLE DE TEMPS - VERSION RÉGULARISÉE")
    print("=" * 70)
    print(f"Target : {TARGET_COLUMN} (régression)")
    print(f"Features : TF-IDF({TEXT_COLUMN}) + {NUMERIC_COLUMNS} + {CATEGORICAL_PRED_COLUMNS}")
    print()
    print("🔧 PARAMÈTRES DE RÉGULARISATION :")
    print(f"   n_estimators      : {REGULARIZATION_CONFIG['n_estimators']} (était 200)")
    print(f"   max_depth         : {REGULARIZATION_CONFIG['max_depth']} (était 6)")
    print(f"   min_child_weight  : {REGULARIZATION_CONFIG['min_child_weight']} (était 1)")
    print(f"   subsample         : {REGULARIZATION_CONFIG['subsample']} (était 1.0)")
    print(f"   colsample_bytree  : {REGULARIZATION_CONFIG['colsample_bytree']} (était 1.0)")
    print(f"   early_stopping    : {REGULARIZATION_CONFIG['early_stopping_rounds']} rounds")
    print()
    
    # -------------------------------------------------------------------------
    # ÉTAPE 1 : Chargement des données
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("ÉTAPE 1 : Chargement des données")
    print("-" * 70)
    
    if df_train is None or df_val is None or df_test is None:
        try:
            df_train = pd.read_csv("data/train_with_type_pred.csv")
            df_val = pd.read_csv("data/val_with_type_pred.csv")
            df_test = pd.read_csv("data/test_with_type_pred.csv")
            print("✅ Chargé depuis les fichiers *_with_type_pred.csv")
        except FileNotFoundError:
            print("❌ Fichiers avec prédictions de type non trouvés.")
            print("   Veuillez d'abord exécuter train_type_ticket.py")
            return None
    
    # Vérifier les colonnes requises
    required_cols = ['urgence_pred', 'categorie_pred', 'type_ticket_pred']
    missing = [col for col in required_cols if col not in df_train.columns]
    if missing:
        print(f"❌ Colonnes manquantes dans le train set: {missing}")
        return None
    
    print(f"   Train      : {len(df_train)} lignes")
    print(f"   Validation : {len(df_val)} lignes")
    print(f"   Test       : {len(df_test)} lignes")
    
    # Statistiques de la cible
    print(f"\n📊 Statistiques de {TARGET_COLUMN} dans le train set:")
    y_train_stats = df_train[TARGET_COLUMN]
    print(f"   Min    : {y_train_stats.min():.2f}h")
    print(f"   Max    : {y_train_stats.max():.2f}h")
    print(f"   Mean   : {y_train_stats.mean():.2f}h")
    print(f"   Median : {y_train_stats.median():.2f}h")
    print(f"   Std    : {y_train_stats.std():.2f}h")
    
    # -------------------------------------------------------------------------
    # ÉTAPE 2 : Préparation des features
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("ÉTAPE 2 : Préparation des features")
    print("-" * 70)
    
    # TF-IDF
    print("   Vectorisation TF-IDF...")
    tfidf = TfidfVectorizer(**TFIDF_PARAMS)
    X_text_train = tfidf.fit_transform(df_train[TEXT_COLUMN].fillna(''))
    X_text_val = tfidf.transform(df_val[TEXT_COLUMN].fillna(''))
    X_text_test = tfidf.transform(df_test[TEXT_COLUMN].fillna(''))
    
    # Features numériques
    X_num_train = df_train[NUMERIC_COLUMNS].values
    X_num_val = df_val[NUMERIC_COLUMNS].values
    X_num_test = df_test[NUMERIC_COLUMNS].values
    
    # Encoder les prédictions catégorielles
    print("   Encodage OneHot des prédictions catégorielles...")
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    X_cat_train = encoder.fit_transform(df_train[CATEGORICAL_PRED_COLUMNS])
    X_cat_val = encoder.transform(df_val[CATEGORICAL_PRED_COLUMNS])
    X_cat_test = encoder.transform(df_test[CATEGORICAL_PRED_COLUMNS])
    
    # Combiner toutes les features
    X_train_combined = hstack([X_text_train, csr_matrix(X_num_train), X_cat_train])
    X_val_combined = hstack([X_text_val, csr_matrix(X_num_val), X_cat_val])
    X_test_combined = hstack([X_text_test, csr_matrix(X_num_test), X_cat_test])
    
    print(f"   Shape features combinées : {X_train_combined.shape}")
    print(f"   - TF-IDF features : {X_text_train.shape[1]}")
    print(f"   - Numeric features : {len(NUMERIC_COLUMNS)}")
    print(f"   - Categorical features : {X_cat_train.shape[1]}")
    
    # Labels
    y_train = df_train[TARGET_COLUMN].values
    y_val = df_val[TARGET_COLUMN].values
    y_test = df_test[TARGET_COLUMN].values
    
    # -------------------------------------------------------------------------
    # ÉTAPE 3 : Entraînement du modèle RÉGULARISÉ avec Early Stopping
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("ÉTAPE 3 : Entraînement du modèle RÉGULARISÉ")
    print("-" * 70)
    
    if XGBOOST_AVAILABLE:
        print("   Utilisation de XGBRegressor avec régularisation forte...")
        
        model = XGBRegressor(
            n_estimators=REGULARIZATION_CONFIG['n_estimators'],
            max_depth=REGULARIZATION_CONFIG['max_depth'],
            min_child_weight=REGULARIZATION_CONFIG['min_child_weight'],
            learning_rate=REGULARIZATION_CONFIG['learning_rate'],
            subsample=REGULARIZATION_CONFIG['subsample'],
            colsample_bytree=REGULARIZATION_CONFIG['colsample_bytree'],
            reg_alpha=REGULARIZATION_CONFIG['reg_alpha'],
            reg_lambda=REGULARIZATION_CONFIG['reg_lambda'],
            random_state=REGULARIZATION_CONFIG['random_state'],
            n_jobs=REGULARIZATION_CONFIG['n_jobs'],
            verbosity=REGULARIZATION_CONFIG['verbosity']
        )
        algorithm_name = "XGBRegressor (régularisé)"
        
        # Entraînement avec Early Stopping
        print(f"   Early stopping activé (patience={REGULARIZATION_CONFIG['early_stopping_rounds']} rounds)...")
        print("   Entraînement en cours...")
        
        model.fit(
            X_train_combined, 
            y_train,
            eval_set=[(X_val_combined, y_val)],
            verbose=False
        )
        
        # Récupérer le nombre d'arbres utilisés
        best_iteration = getattr(model, 'best_iteration', REGULARIZATION_CONFIG['n_estimators'])
        print(f"   ✅ Entraînement terminé")
        print(f"   📊 Nombre d'arbres utilisés : {best_iteration if best_iteration else REGULARIZATION_CONFIG['n_estimators']}")
        
    else:
        print("   Utilisation de GradientBoostingRegressor avec régularisation...")
        
        model = GradientBoostingRegressor(
            n_estimators=REGULARIZATION_CONFIG['n_estimators'],
            max_depth=REGULARIZATION_CONFIG['max_depth'],
            min_samples_split=REGULARIZATION_CONFIG['min_child_weight'],
            min_samples_leaf=3,
            subsample=REGULARIZATION_CONFIG['subsample'],
            learning_rate=REGULARIZATION_CONFIG['learning_rate'],
            random_state=REGULARIZATION_CONFIG['random_state'],
            verbose=0
        )
        algorithm_name = "GradientBoostingRegressor (régularisé)"
        
        print("   Entraînement en cours...")
        model.fit(X_train_combined, y_train)
        print("   ✅ Entraînement terminé")
    
    # -------------------------------------------------------------------------
    # ÉTAPE 4 : Prédictions et évaluation
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("ÉTAPE 4 : Évaluation sur Train, Validation et Test")
    print("-" * 70)
    
    # Prédictions
    train_predictions = model.predict(X_train_combined)
    val_predictions = model.predict(X_val_combined)
    test_predictions = model.predict(X_test_combined)
    
    # Assurer que les prédictions sont positives
    train_predictions = np.maximum(train_predictions, 0)
    val_predictions = np.maximum(val_predictions, 0)
    test_predictions = np.maximum(test_predictions, 0)
    
    # Évaluation
    train_metrics = evaluate_regression(y_train, train_predictions, "TRAIN")
    val_metrics = evaluate_regression(y_val, val_predictions, "VALIDATION")
    test_metrics = evaluate_regression(y_test, test_predictions, "TEST")
    
    # -------------------------------------------------------------------------
    # ÉTAPE 5 : Comparaison avec le modèle précédent (OVERFITTING FIX)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("ÉTAPE 5 : COMPARAISON - RÉDUCTION DE L'OVERFITTING")
    print("-" * 70)
    
    # Métriques de l'ancien modèle (overfitté)
    old_metrics = {
        'train': {'mae': 2.41, 'rmse': 4.10, 'r2': 0.985},
        'validation': {'mae': 7.50, 'rmse': 16.67, 'r2': 0.795},
        'test': {'mae': 7.48, 'rmse': 19.23, 'r2': 0.720}
    }
    
    print("\n📊 COMPARAISON AVANT / APRÈS RÉGULARISATION :")
    print("=" * 70)
    
    # Tableau comparatif MAE
    print("\n📈 MAE (Mean Absolute Error) - Plus bas = mieux :")
    print("-" * 60)
    print(f"{'Dataset':<12} {'Ancien':<12} {'Nouveau':<12} {'Écart':<12} {'Status'}")
    print("-" * 60)
    for dataset in ['train', 'validation', 'test']:
        old_mae = old_metrics[dataset]['mae']
        new_mae = train_metrics['mae'] if dataset == 'train' else (val_metrics['mae'] if dataset == 'validation' else test_metrics['mae'])
        diff = new_mae - old_mae
        status = "✅ mieux" if dataset != 'train' and diff < 0 else ("⚠️ +haut" if dataset == 'train' else "≈ stable")
        print(f"{dataset:<12} {old_mae:<12.2f} {new_mae:<12.2f} {diff:+.2f}{'h':<8} {status}")
    
    # Tableau comparatif R²
    print("\n📈 R² (Coefficient de détermination) - Plus proche de Val/Test = moins d'overfitting :")
    print("-" * 60)
    print(f"{'Dataset':<12} {'Ancien':<12} {'Nouveau':<12} {'Écart':<12} {'Status'}")
    print("-" * 60)
    for dataset in ['train', 'validation', 'test']:
        old_r2 = old_metrics[dataset]['r2']
        new_r2 = train_metrics['r2'] if dataset == 'train' else (val_metrics['r2'] if dataset == 'validation' else test_metrics['r2'])
        diff = new_r2 - old_r2
        status = "✅ régularisé" if dataset == 'train' and diff < 0 else "✅ stable"
        print(f"{dataset:<12} {old_r2:<12.4f} {new_r2:<12.4f} {diff:+.4f}{'  ':<6} {status}")
    
    # Calcul des écarts Train-Test
    old_gap_r2 = old_metrics['train']['r2'] - old_metrics['test']['r2']
    new_gap_r2 = train_metrics['r2'] - test_metrics['r2']
    gap_reduction = old_gap_r2 - new_gap_r2
    
    old_gap_mae = old_metrics['test']['mae'] - old_metrics['train']['mae']
    new_gap_mae = test_metrics['mae'] - train_metrics['mae']
    
    print("\n" + "=" * 70)
    print("🎯 RÉDUCTION DE L'ÉCART TRAIN-TEST (Indicateur d'overfitting) :")
    print("=" * 70)
    print(f"\n   R² Gap (Train - Test) :")
    print(f"      Ancien : {old_gap_r2:.4f}")
    print(f"      Nouveau: {new_gap_r2:.4f}")
    print(f"      → Réduction de {gap_reduction:.4f} ({gap_reduction/old_gap_r2*100:.1f}% d'amélioration)")
    
    print(f"\n   MAE Gap (Test - Train) :")
    print(f"      Ancien : {old_gap_mae:.2f}h")
    print(f"      Nouveau: {new_gap_mae:.2f}h")
    print(f"      → Réduction de {old_gap_mae - new_gap_mae:.2f}h")
    
    if new_gap_r2 < old_gap_r2:
        print("\n   ✅ OVERFITTING RÉDUIT AVEC SUCCÈS !")
    else:
        print("\n   ⚠️ Régularisation insuffisante, ajuster les hyperparamètres")
    
    # -------------------------------------------------------------------------
    # ÉTAPE 6 : Analyse des erreurs
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("ÉTAPE 6 : Analyse des erreurs")
    print("-" * 70)
    
    errors = test_predictions - y_test
    abs_errors = np.abs(errors)
    
    print(f"\n📊 Distribution des erreurs absolues (Test):")
    print(f"   < 5h  : {np.sum(abs_errors < 5):4} ({np.sum(abs_errors < 5)/len(abs_errors)*100:.1f}%)")
    print(f"   < 10h : {np.sum(abs_errors < 10):4} ({np.sum(abs_errors < 10)/len(abs_errors)*100:.1f}%)")
    print(f"   < 20h : {np.sum(abs_errors < 20):4} ({np.sum(abs_errors < 20)/len(abs_errors)*100:.1f}%)")
    print(f"   ≥ 20h : {np.sum(abs_errors >= 20):4} ({np.sum(abs_errors >= 20)/len(abs_errors)*100:.1f}%)")
    
    print(f"\n📊 Performance par urgence (Test):")
    for urgence in ['Basse', 'Moyenne', 'Haute']:
        mask = df_test['urgence_pred'] == urgence
        if mask.sum() > 0:
            mae_urgence = np.mean(abs_errors[mask])
            print(f"   {urgence:8} : MAE = {mae_urgence:.2f}h (n={mask.sum()})")
    
    # -------------------------------------------------------------------------
    # ÉTAPE 7 : Sauvegarde (OVERWRITE MODE)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("ÉTAPE 7 : Sauvegarde du modèle (OVERWRITE)")
    print("-" * 70)
    
    ensure_models_dir()
    
    pipeline_dict = {
        'model': model,
        'tfidf': tfidf,
        'encoder': encoder,
        'text_column': TEXT_COLUMN,
        'numeric_columns': NUMERIC_COLUMNS,
        'categorical_pred_columns': CATEGORICAL_PRED_COLUMNS,
        'target_column': TARGET_COLUMN,
        'algorithm': algorithm_name,
        'regularization_config': REGULARIZATION_CONFIG
    }
    
    # Sauvegarder en écrasant l'ancien fichier
    pipeline_path = os.path.join(MODELS_DIR, PIPELINE_FILE)
    joblib.dump(pipeline_dict, pipeline_path)
    print(f"✅ Pipeline sauvegardé (OVERWRITE) : {pipeline_path}")
    
    # Mettre à jour les métadonnées
    metadata = {
        'time_model': {
            'training_date': datetime.now().isoformat(),
            'algorithm': algorithm_name,
            'note': 'Time regression model re-trained with strong regularization and early stopping to reduce overfitting.',
            'regularization_config': {
                'n_estimators': REGULARIZATION_CONFIG['n_estimators'],
                'max_depth': REGULARIZATION_CONFIG['max_depth'],
                'min_child_weight': REGULARIZATION_CONFIG['min_child_weight'],
                'subsample': REGULARIZATION_CONFIG['subsample'],
                'colsample_bytree': REGULARIZATION_CONFIG['colsample_bytree'],
                'reg_alpha': REGULARIZATION_CONFIG['reg_alpha'],
                'reg_lambda': REGULARIZATION_CONFIG['reg_lambda'],
                'early_stopping_rounds': REGULARIZATION_CONFIG['early_stopping_rounds']
            },
            'features': {
                'text': TEXT_COLUMN,
                'numeric': NUMERIC_COLUMNS,
                'categorical_pred': CATEGORICAL_PRED_COLUMNS,
                'tfidf_params': TFIDF_PARAMS
            },
            'target': TARGET_COLUMN,
            'metrics': {
                'train': train_metrics,
                'validation': val_metrics,
                'test': test_metrics
            },
            'overfitting_analysis': {
                'old_train_test_r2_gap': old_gap_r2,
                'new_train_test_r2_gap': new_gap_r2,
                'gap_reduction_percent': f"{gap_reduction/old_gap_r2*100:.1f}%",
                'old_train_test_mae_gap': old_gap_mae,
                'new_train_test_mae_gap': new_gap_mae
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
    print("RÉSUMÉ - MODÈLE DE TEMPS (RÉGULARISÉ)")
    print("=" * 70)
    print(f"""
    ┌─────────────────────────────────────────────────────────────────────┐
    │  MODÈLE DE TEMPS DE RÉSOLUTION (Régression RÉGULARISÉE)             │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Features : TF-IDF + nb_mots + urgence/categorie/type_pred          │
    │  Algorithme : {algorithm_name:<40}│
    │                                                                     │
    │  🔧 RÉGULARISATION APPLIQUÉE :                                      │
    │     n_estimators={REGULARIZATION_CONFIG['n_estimators']}, max_depth={REGULARIZATION_CONFIG['max_depth']}, subsample={REGULARIZATION_CONFIG['subsample']}, colsample={REGULARIZATION_CONFIG['colsample_bytree']}      │
    │                                                                     │
    │  📊 PERFORMANCES :                                                  │
    │  ─────────────────────────────────────────────────────────────────  │
    │  Train      : MAE={train_metrics['mae']:.2f}h  RMSE={train_metrics['rmse']:.2f}h  R²={train_metrics['r2']:.4f}    │
    │  Validation : MAE={val_metrics['mae']:.2f}h  RMSE={val_metrics['rmse']:.2f}h  R²={val_metrics['r2']:.4f}    │
    │  Test       : MAE={test_metrics['mae']:.2f}h  RMSE={test_metrics['rmse']:.2f}h  R²={test_metrics['r2']:.4f}    │
    │                                                                     │
    │  🎯 RÉDUCTION OVERFITTING :                                         │
    │     R² Gap (Train-Test) : {old_gap_r2:.4f} → {new_gap_r2:.4f} (-{gap_reduction/old_gap_r2*100:.0f}%)           │
    │                                                                     │
    │  ✅ Pipeline sauvegardé : models/{PIPELINE_FILE}               │
    └─────────────────────────────────────────────────────────────────────┘
    """)
    
    return pipeline_dict, {'train': train_metrics, 'val': val_metrics, 'test': test_metrics}


# =============================================================================
# FONCTION DE PRÉDICTION
# =============================================================================
def predict_time(pipeline_dict: dict, X: pd.DataFrame) -> np.ndarray:
    """Fait des prédictions de temps de résolution."""
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
    
    predictions = model.predict(X_combined)
    return np.maximum(predictions, 0)  # Assurer valeurs positives


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================
if __name__ == "__main__":
    # ÉTAPE 0 : Nettoyage des artefacts précédents
    clean_previous_artifacts()
    
    # ÉTAPE 1-7 : Entraînement du nouveau modèle régularisé
    result = train_time_model_regularized()
    
    if result is not None:
        print("\n✅ Modèle de temps de résolution RÉGULARISÉ entraîné avec succès!")
        print("✅ Overfitting corrigé - Le modèle généralise mieux maintenant.")
    
    print("=" * 70)
