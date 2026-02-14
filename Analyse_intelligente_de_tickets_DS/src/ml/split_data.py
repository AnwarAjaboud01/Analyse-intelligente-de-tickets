# =============================================================================
# Script de Split Stratifié du Dataset Équilibré
# =============================================================================
#
# OBJECTIF :
# Diviser "tickets_balanced.csv" en 3 ensembles :
#   - train.csv      = 70% (entraînement)
#   - validation.csv = 15% (validation/tuning)
#   - test.csv       = 15% (évaluation finale)
#
# POURQUOI UN SPLIT STRATIFIÉ ?
# -----------------------------
# Un split stratifié garantit que la distribution des classes (urgence) est
# IDENTIQUE dans chaque ensemble (train, validation, test).
# 
# Sans stratification, on risque d'avoir par exemple :
#   - Train avec 60% Basse, 30% Moyenne, 10% Haute
#   - Test avec 40% Basse, 20% Moyenne, 40% Haute
# 
# Cela biaiserait l'évaluation : le modèle serait entraîné sur une distribution
# différente de celle sur laquelle il est testé.
#
# Avec stratification (notre cas) :
#   - Train : 50% Basse, 28% Moyenne, 22% Haute
#   - Val   : 50% Basse, 28% Moyenne, 22% Haute
#   - Test  : 50% Basse, 28% Moyenne, 22% Haute
#
# POURQUOI random_state=42 ?
# --------------------------
# Fixer le random_state garantit la REPRODUCTIBILITÉ :
#   - Chaque exécution du script produit exactement le même split
#   - Permet de comparer des expériences sur les mêmes données
#   - Facilite le debugging et la collaboration en équipe
#   - 42 est une convention (référence au "Guide du voyageur galactique")
#
# =============================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_FILE = "data/tickets_balanced.csv"
OUTPUT_DIR = "data"
TRAIN_FILE = f"{OUTPUT_DIR}/train.csv"
VAL_FILE = f"{OUTPUT_DIR}/validation.csv"
TEST_FILE = f"{OUTPUT_DIR}/test.csv"

# Proportions du split
TRAIN_RATIO = 0.70  # 70% pour l'entraînement
VAL_RATIO = 0.15    # 15% pour la validation
TEST_RATIO = 0.15   # 15% pour le test

# Seed pour la reproductibilité
RANDOM_STATE = 42

# Colonnes attendues (ordre exact)
EXPECTED_COLUMNS = ['ID', 'texte', 'titre', 'categorie', 'urgence', 
                    'temps_resolution', 'type_ticket', 'nb_mots', 'text_full']

# Valeurs valides pour urgence
VALID_URGENCE = {'Basse', 'Moyenne', 'Haute'}

# =============================================================================
# ÉTAPE 1 : CHARGEMENT DES DONNÉES
# =============================================================================
print("=" * 70)
print("ÉTAPE 1 : CHARGEMENT DES DONNÉES")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"✅ Fichier chargé : {INPUT_FILE}")
print(f"✅ Nombre de lignes : {len(df)}")
print(f"✅ Colonnes trouvées : {list(df.columns)}")

# =============================================================================
# ÉTAPE 2 : VALIDATION DES COLONNES
# =============================================================================
print("\n" + "=" * 70)
print("ÉTAPE 2 : VALIDATION DES COLONNES")
print("=" * 70)

# Vérifier que toutes les colonnes attendues sont présentes
missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)

if missing_cols:
    raise ValueError(f"❌ Colonnes manquantes : {missing_cols}")
if extra_cols:
    print(f"⚠️  Colonnes supplémentaires ignorées : {extra_cols}")

print(f"✅ Toutes les colonnes attendues sont présentes")

# Réordonner les colonnes selon l'ordre attendu
df = df[EXPECTED_COLUMNS]
print(f"✅ Colonnes réordonnées : {list(df.columns)}")

# =============================================================================
# ÉTAPE 3 : VALIDATION DES DONNÉES
# =============================================================================
print("\n" + "=" * 70)
print("ÉTAPE 3 : VALIDATION DES DONNÉES")
print("=" * 70)

# 3.1 Vérifier l'unicité des IDs
n_unique_ids = df['ID'].nunique()
n_total_rows = len(df)

if n_unique_ids != n_total_rows:
    duplicates = df[df['ID'].duplicated(keep=False)]['ID'].unique()
    raise ValueError(f"❌ IDs non uniques détectés ! {n_total_rows - n_unique_ids} doublons. "
                     f"Exemples : {list(duplicates[:5])}")
print(f"✅ Tous les IDs sont uniques ({n_unique_ids}/{n_total_rows})")

# 3.2 Vérifier les valeurs de urgence
urgence_values = set(df['urgence'].unique())
invalid_values = urgence_values - VALID_URGENCE

if invalid_values:
    raise ValueError(f"❌ Valeurs d'urgence invalides : {invalid_values}. "
                     f"Attendues : {VALID_URGENCE}")
print(f"✅ Valeurs d'urgence valides : {sorted(urgence_values)}")

# =============================================================================
# ÉTAPE 4 : SPLIT STRATIFIÉ EN DEUX ÉTAPES
# =============================================================================
print("\n" + "=" * 70)
print("ÉTAPE 4 : SPLIT STRATIFIÉ")
print("=" * 70)

print(f"\n📊 Distribution AVANT split :")
print("-" * 40)
dist_total = df['urgence'].value_counts()
dist_total_pct = df['urgence'].value_counts(normalize=True) * 100
for urgence in ['Basse', 'Moyenne', 'Haute']:
    if urgence in dist_total.index:
        print(f"   {urgence:10} : {dist_total[urgence]:4} ({dist_total_pct[urgence]:5.2f}%)")
print(f"   TOTAL     : {len(df)}")

# --- ÉTAPE 4a : Premier split - 70% train, 30% temp ---
print(f"\n🔄 Split 1 : 70% train, 30% temp (stratifié sur 'urgence')")

df_train, df_temp = train_test_split(
    df,
    test_size=0.30,           # 30% pour temp (validation + test)
    stratify=df['urgence'],   # Stratification sur la colonne urgence
    random_state=RANDOM_STATE,
    shuffle=True
)

print(f"   → Train : {len(df_train)} lignes ({len(df_train)/len(df)*100:.1f}%)")
print(f"   → Temp  : {len(df_temp)} lignes ({len(df_temp)/len(df)*100:.1f}%)")

# --- ÉTAPE 4b : Second split - 50% validation, 50% test (du temp) ---
print(f"\n🔄 Split 2 : 50% validation, 50% test du temp (stratifié sur 'urgence')")

df_val, df_test = train_test_split(
    df_temp,
    test_size=0.50,              # 50% du temp = 15% du total
    stratify=df_temp['urgence'], # Stratification sur la colonne urgence
    random_state=RANDOM_STATE,
    shuffle=True
)

print(f"   → Validation : {len(df_val)} lignes ({len(df_val)/len(df)*100:.1f}%)")
print(f"   → Test       : {len(df_test)} lignes ({len(df_test)/len(df)*100:.1f}%)")

# Réinitialiser les index
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# =============================================================================
# ÉTAPE 5 : AFFICHAGE DES DISTRIBUTIONS PAR ENSEMBLE
# =============================================================================
print("\n" + "=" * 70)
print("ÉTAPE 5 : DISTRIBUTION DE 'urgence' PAR ENSEMBLE")
print("=" * 70)

def print_distribution(df_set, set_name):
    """Affiche la distribution d'urgence pour un ensemble."""
    print(f"\n📊 {set_name} ({len(df_set)} lignes) :")
    print("-" * 40)
    dist = df_set['urgence'].value_counts()
    dist_pct = df_set['urgence'].value_counts(normalize=True) * 100
    for urgence in ['Basse', 'Moyenne', 'Haute']:
        if urgence in dist.index:
            bar = "█" * int(dist_pct[urgence] / 2)
            print(f"   {urgence:10} : {dist[urgence]:4} ({dist_pct[urgence]:5.2f}%) {bar}")

print_distribution(df_train, "TRAIN (70%)")
print_distribution(df_val, "VALIDATION (15%)")
print_distribution(df_test, "TEST (15%)")

# =============================================================================
# ÉTAPE 6 : VÉRIFICATION DE LA STRATIFICATION
# =============================================================================
print("\n" + "=" * 70)
print("ÉTAPE 6 : VÉRIFICATION DE LA STRATIFICATION")
print("=" * 70)

print("\n📈 Comparaison des proportions (%) :")
print("-" * 60)
print(f"{'Ensemble':<15} {'Basse':>10} {'Moyenne':>10} {'Haute':>10}")
print("-" * 60)

for name, df_set in [('Original', df), ('Train', df_train), ('Validation', df_val), ('Test', df_test)]:
    pct = df_set['urgence'].value_counts(normalize=True) * 100
    print(f"{name:<15} {pct.get('Basse', 0):>10.2f} {pct.get('Moyenne', 0):>10.2f} {pct.get('Haute', 0):>10.2f}")

print("-" * 60)
print("✅ Les proportions sont identiques dans tous les ensembles (stratification réussie)")

# =============================================================================
# ÉTAPE 7 : SAUVEGARDE DES FICHIERS CSV
# =============================================================================
print("\n" + "=" * 70)
print("ÉTAPE 7 : SAUVEGARDE DES FICHIERS")
print("=" * 70)

# Sauvegarder avec les mêmes colonnes et ordre que l'input
df_train.to_csv(TRAIN_FILE, index=False)
df_val.to_csv(VAL_FILE, index=False)
df_test.to_csv(TEST_FILE, index=False)

print(f"✅ {TRAIN_FILE:<25} : {len(df_train):5} lignes (70%)")
print(f"✅ {VAL_FILE:<25} : {len(df_val):5} lignes (15%)")
print(f"✅ {TEST_FILE:<25} : {len(df_test):5} lignes (15%)")

# Vérification finale
total_saved = len(df_train) + len(df_val) + len(df_test)
print(f"\n📊 Total sauvegardé : {total_saved} lignes")
print(f"📊 Total original   : {len(df)} lignes")
print(f"✅ Vérification : {total_saved} == {len(df)} : {total_saved == len(df)}")

# =============================================================================
# RÉSUMÉ FINAL
# =============================================================================
print("\n" + "=" * 70)
print("RÉSUMÉ DU SPLIT STRATIFIÉ")
print("=" * 70)

print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  FICHIER SOURCE : tickets_balanced.csv ({len(df)} lignes)              │
├─────────────────────────────────────────────────────────────────────┤
│  SPLIT STRATIFIÉ (random_state=42)                                  │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  📁 train.csv      : {len(df_train):5} lignes (70%)                        │
│  📁 validation.csv : {len(df_val):5} lignes (15%)                         │
│  📁 test.csv       : {len(df_test):5} lignes (15%)                         │
│                                                                     │
│  ✅ Distribution préservée dans tous les ensembles                  │
│  ✅ Reproductible (même résultat à chaque exécution)                │
└─────────────────────────────────────────────────────────────────────┘
""")

print("=" * 70)
print("✅ SPLIT STRATIFIÉ TERMINÉ AVEC SUCCÈS !")
print("=" * 70)
