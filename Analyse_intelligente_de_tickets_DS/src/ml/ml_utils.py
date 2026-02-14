# =============================================================================
# Module Utilitaire ML - Fonctions Partagées
# =============================================================================
"""
Ce module contient les fonctions utilitaires partagées entre tous les scripts
d'entraînement ML :
- Chargement des données
- Construction de pipelines TF-IDF + features numériques
- Génération de prédictions Out-Of-Fold (OOF) anti-fuite
- Fonctions d'évaluation (classification et régression)
- Sauvegarde des modèles et métadonnées
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION GLOBALE
# =============================================================================
RANDOM_STATE = 42
N_FOLDS = 5
DATA_DIR = "data"
MODELS_DIR = "models"

# Paramètres TF-IDF
TFIDF_PARAMS = {
    'ngram_range': (1, 2),
    'min_df': 2,
    'max_features': 50000,
    'sublinear_tf': True,  # Améliore les performances
    'strip_accents': 'unicode',
    'lowercase': True
}


# =============================================================================
# FONCTIONS DE CHARGEMENT DES DONNÉES
# =============================================================================
def load_data(train_path: str = None, val_path: str = None, test_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Charge les fichiers train, validation et test.
    
    Returns:
        Tuple[DataFrame, DataFrame, DataFrame]: (train, validation, test)
    """
    train_path = train_path or f"{DATA_DIR}/train.csv"
    val_path = val_path or f"{DATA_DIR}/validation.csv"
    test_path = test_path or f"{DATA_DIR}/test.csv"
    
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    
    print(f"✅ Données chargées:")
    print(f"   Train      : {len(df_train)} lignes")
    print(f"   Validation : {len(df_val)} lignes")
    print(f"   Test       : {len(df_test)} lignes")
    
    return df_train, df_val, df_test


def ensure_models_dir():
    """Crée le répertoire models/ s'il n'existe pas."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"✅ Répertoire {MODELS_DIR}/ prêt")


# =============================================================================
# TRANSFORMERS PERSONNALISÉS
# =============================================================================
class TextColumnExtractor(BaseEstimator, TransformerMixin):
    """Extrait une colonne texte d'un DataFrame."""
    def __init__(self, column: str):
        self.column = column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.column].fillna('').astype(str)
        return X


class NumericColumnExtractor(BaseEstimator, TransformerMixin):
    """Extrait des colonnes numériques d'un DataFrame et les reshape."""
    def __init__(self, columns: List[str]):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            result = X[self.columns].values
        else:
            result = X
        return result.reshape(-1, len(self.columns)) if len(self.columns) == 1 else result


class DenseTransformer(BaseEstimator, TransformerMixin):
    """Convertit une matrice sparse en dense."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if hasattr(X, 'toarray'):
            return X.toarray()
        return X


class CategoricalPredictionEncoder(BaseEstimator, TransformerMixin):
    """
    Encode les prédictions catégorielles (urgence_pred, categorie_pred, etc.)
    avec OneHotEncoder pour les utiliser comme features dans les modèles downstream.
    """
    def __init__(self, columns: List[str]):
        self.columns = columns
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.encoder.fit(X[self.columns])
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return self.encoder.transform(X[self.columns])
        return X


# =============================================================================
# CONSTRUCTION DE PIPELINES
# =============================================================================
def build_tfidf_pipeline(text_column: str = 'text_full') -> Pipeline:
    """
    Construit un pipeline TF-IDF pour la colonne texte.
    
    Args:
        text_column: Nom de la colonne texte
        
    Returns:
        Pipeline sklearn
    """
    return Pipeline([
        ('extract', TextColumnExtractor(text_column)),
        ('tfidf', TfidfVectorizer(**TFIDF_PARAMS))
    ])


def build_text_numeric_pipeline(text_column: str = 'text_full', 
                                numeric_columns: List[str] = ['nb_mots']) -> ColumnTransformer:
    """
    Construit un ColumnTransformer combinant TF-IDF et features numériques.
    
    Args:
        text_column: Nom de la colonne texte
        numeric_columns: Liste des colonnes numériques
        
    Returns:
        ColumnTransformer configuré
    """
    return ColumnTransformer([
        ('text', TfidfVectorizer(**TFIDF_PARAMS), text_column),
        ('numeric', StandardScaler(), numeric_columns)
    ], remainder='drop')


# =============================================================================
# GÉNÉRATION OOF (OUT-OF-FOLD) ANTI-FUITE
# =============================================================================
def generate_oof_predictions_classification(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    feature_columns: List[str],
    text_column: str = 'text_full',
    n_folds: int = N_FOLDS,
    random_state: int = RANDOM_STATE
) -> np.ndarray:
    """
    Génère des prédictions Out-Of-Fold pour éviter la fuite de données.
    
    POURQUOI OOF ?
    --------------
    Quand on veut utiliser la prédiction d'un modèle (ex: urgence_pred) comme
    feature pour un autre modèle (ex: category_model), on ne peut pas utiliser
    les prédictions du modèle entraîné sur tout le train set car cela créerait
    une fuite de données : le modèle aurait "vu" les données sur lesquelles
    il fait des prédictions.
    
    Solution : Pour chaque fold, on entraîne sur 4/5 des données et on prédit
    sur le 1/5 restant. Ainsi, chaque prédiction est faite par un modèle qui
    n'a JAMAIS vu cette donnée pendant l'entraînement.
    
    Args:
        model: Modèle sklearn (sera cloné pour chaque fold)
        X: DataFrame avec les features
        y: Series avec les labels
        feature_columns: Colonnes à utiliser (hors text_column)
        text_column: Colonne texte pour TF-IDF
        n_folds: Nombre de folds
        random_state: Seed pour reproductibilité
        
    Returns:
        Array des prédictions OOF (une par ligne de X)
    """
    oof_predictions = np.empty(len(X), dtype=object)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Construire le vectorizer TF-IDF
    tfidf = TfidfVectorizer(**TFIDF_PARAMS)
    
    print(f"   Génération OOF avec {n_folds} folds...")
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # Préparer les données du fold
        X_fold_train = X.iloc[train_idx]
        X_fold_val = X.iloc[val_idx]
        y_fold_train = y.iloc[train_idx]
        
        # TF-IDF sur le texte
        tfidf_fold = clone(tfidf)
        X_text_train = tfidf_fold.fit_transform(X_fold_train[text_column].fillna(''))
        X_text_val = tfidf_fold.transform(X_fold_val[text_column].fillna(''))
        
        # Ajouter les features numériques/catégorielles
        if feature_columns:
            X_num_train = X_fold_train[feature_columns].values
            X_num_val = X_fold_val[feature_columns].values
            X_train_combined = hstack([X_text_train, csr_matrix(X_num_train)])
            X_val_combined = hstack([X_text_val, csr_matrix(X_num_val)])
        else:
            X_train_combined = X_text_train
            X_val_combined = X_text_val
        
        # Entraîner et prédire
        model_fold = clone(model)
        model_fold.fit(X_train_combined, y_fold_train)
        oof_predictions[val_idx] = model_fold.predict(X_val_combined)
        
        print(f"      Fold {fold_idx + 1}/{n_folds} terminé")
    
    return oof_predictions


def generate_oof_predictions_with_categorical(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    text_column: str,
    numeric_columns: List[str],
    categorical_pred_columns: List[str],
    n_folds: int = N_FOLDS,
    random_state: int = RANDOM_STATE
) -> np.ndarray:
    """
    Génère des prédictions OOF avec features catégorielles encodées (prédictions upstream).
    
    Args:
        model: Modèle sklearn
        X: DataFrame avec toutes les colonnes nécessaires
        y: Labels
        text_column: Colonne texte
        numeric_columns: Colonnes numériques
        categorical_pred_columns: Colonnes de prédictions catégorielles à encoder
        
    Returns:
        Array des prédictions OOF
    """
    oof_predictions = np.empty(len(X), dtype=object)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    print(f"   Génération OOF avec {n_folds} folds (features cat: {categorical_pred_columns})...")
    
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
        
        # Combiner toutes les features
        X_train_combined = hstack([X_text_train, csr_matrix(X_num_train), X_cat_train])
        X_val_combined = hstack([X_text_val, csr_matrix(X_num_val), X_cat_val])
        
        # Entraîner et prédire
        model_fold = clone(model)
        model_fold.fit(X_train_combined, y_fold_train)
        oof_predictions[val_idx] = model_fold.predict(X_val_combined)
        
        print(f"      Fold {fold_idx + 1}/{n_folds} terminé")
    
    return oof_predictions


# =============================================================================
# FONCTIONS D'ÉVALUATION
# =============================================================================
def evaluate_classification(y_true: np.ndarray, 
                           y_pred: np.ndarray, 
                           set_name: str,
                           labels: List[str] = None) -> Dict[str, float]:
    """
    Évalue un modèle de classification et affiche les métriques.
    
    Args:
        y_true: Labels réels
        y_pred: Prédictions
        set_name: Nom de l'ensemble (ex: "Validation", "Test")
        labels: Liste des labels dans l'ordre souhaité
        
    Returns:
        Dict avec accuracy et f1_macro
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    print(f"\n📊 Évaluation sur {set_name}:")
    print("-" * 50)
    print(f"   Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   F1 Macro : {f1_macro:.4f} ({f1_macro*100:.2f}%)")
    
    print(f"\n   Classification Report:")
    print(classification_report(y_true, y_pred, labels=labels))
    
    print(f"\n   Matrice de Confusion:")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if labels:
        # Affichage formaté de la matrice
        max_label_len = max(len(str(l)) for l in labels)
        header = " " * (max_label_len + 3) + "  ".join(f"{l:>{max_label_len}}" for l in labels)
        print(f"   {header}")
        for i, label in enumerate(labels):
            row = "  ".join(f"{cm[i, j]:>{max_label_len}}" for j in range(len(labels)))
            print(f"   {label:>{max_label_len}} : {row}")
    else:
        print(cm)
    
    return {'accuracy': accuracy, 'f1_macro': f1_macro}


def evaluate_regression(y_true: np.ndarray, 
                       y_pred: np.ndarray, 
                       set_name: str) -> Dict[str, float]:
    """
    Évalue un modèle de régression et affiche les métriques.
    
    Args:
        y_true: Valeurs réelles
        y_pred: Prédictions
        set_name: Nom de l'ensemble
        
    Returns:
        Dict avec MAE, RMSE, R2
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n📊 Évaluation sur {set_name}:")
    print("-" * 50)
    print(f"   MAE  : {mae:.4f} heures")
    print(f"   RMSE : {rmse:.4f} heures")
    print(f"   R²   : {r2:.4f}")
    
    # Stats descriptives
    print(f"\n   Statistiques des prédictions:")
    print(f"   - Min  : {y_pred.min():.2f}h")
    print(f"   - Max  : {y_pred.max():.2f}h")
    print(f"   - Mean : {y_pred.mean():.2f}h")
    print(f"   - Std  : {y_pred.std():.2f}h")
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


# =============================================================================
# SAUVEGARDE DES MODÈLES ET MÉTADONNÉES
# =============================================================================
def save_model(model: Any, 
               filename: str, 
               model_name: str) -> str:
    """
    Sauvegarde un modèle avec joblib.
    
    Args:
        model: Objet à sauvegarder
        filename: Nom du fichier (sans chemin)
        model_name: Nom descriptif du modèle
        
    Returns:
        Chemin complet du fichier sauvegardé
    """
    ensure_models_dir()
    filepath = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, filepath)
    print(f"✅ {model_name} sauvegardé : {filepath}")
    return filepath


def save_metadata(metadata: Dict[str, Any], filename: str = "metadata.json"):
    """
    Sauvegarde les métadonnées d'entraînement en JSON.
    
    Args:
        metadata: Dict avec les informations à sauvegarder
        filename: Nom du fichier
    """
    ensure_models_dir()
    filepath = os.path.join(MODELS_DIR, filename)
    
    # Charger les métadonnées existantes si le fichier existe
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            existing = json.load(f)
    else:
        existing = {}
    
    # Mettre à jour avec les nouvelles métadonnées
    existing.update(metadata)
    existing['last_updated'] = datetime.now().isoformat()
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Métadonnées sauvegardées : {filepath}")


def load_model(filename: str) -> Any:
    """
    Charge un modèle sauvegardé.
    
    Args:
        filename: Nom du fichier
        
    Returns:
        Modèle chargé
    """
    filepath = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Modèle non trouvé : {filepath}")
    return joblib.load(filepath)


# =============================================================================
# CLASSE PIPELINE COMPLET POUR PRÉDICTION
# =============================================================================
class FullPredictionPipeline:
    """
    Pipeline complet pour faire des prédictions avec toutes les features.
    Combine TF-IDF + features numériques + features catégorielles encodées.
    """
    def __init__(self, 
                 model: BaseEstimator,
                 tfidf: TfidfVectorizer,
                 text_column: str,
                 numeric_columns: List[str],
                 categorical_columns: List[str] = None,
                 encoder: OneHotEncoder = None):
        self.model = model
        self.tfidf = tfidf
        self.text_column = text_column
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns or []
        self.encoder = encoder
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Fait des prédictions sur de nouvelles données."""
        # TF-IDF
        X_text = self.tfidf.transform(X[self.text_column].fillna(''))
        
        # Features numériques
        X_num = X[self.numeric_columns].values
        
        # Combiner
        if self.categorical_columns and self.encoder:
            X_cat = self.encoder.transform(X[self.categorical_columns])
            X_combined = hstack([X_text, csr_matrix(X_num), X_cat])
        else:
            X_combined = hstack([X_text, csr_matrix(X_num)])
        
        return self.model.predict(X_combined)


# =============================================================================
# FONCTION PRINCIPALE DE TEST DU MODULE
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TEST DU MODULE ml_utils.py")
    print("=" * 70)
    
    # Test de chargement des données
    try:
        df_train, df_val, df_test = load_data()
        print("\n✅ Chargement des données OK")
        print(f"   Colonnes : {list(df_train.columns)}")
        print(f"   Urgence classes : {df_train['urgence'].unique()}")
        print(f"   Catégorie classes : {df_train['categorie'].nunique()} uniques")
        print(f"   Type classes : {df_train['type_ticket'].unique()}")
    except Exception as e:
        print(f"❌ Erreur : {e}")
    
    print("\n" + "=" * 70)
    print("Module ml_utils.py chargé avec succès!")
    print("=" * 70)
