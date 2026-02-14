# =============================================================================
# Script de Rééquilibrage du Dataset - Colonne "urgence"
# =============================================================================
# 
# PROBLÈME :
# La colonne "urgence" est très déséquilibrée :
#   - Basse     ≈ 94.78% (726 tickets)
#   - Moyenne   ≈  4.44% (34 tickets)
#   - Très haute ≈ 0.78% (6 tickets)
#
# NORMALISATION :
# La colonne "urgence" doit contenir EXACTEMENT 3 classes :
#   - "Basse", "Moyenne", "Haute"
# Toute occurrence de "Très haute" sera remplacée par "Haute"
#
# POURQUOI LE RÉÉQUILIBRAGE EST NÉCESSAIRE :
# Un modèle ML entraîné sur des données déséquilibrées apprend à prédire
# majoritairement la classe dominante ("Basse"), ignorant les classes rares.
# Résultat : le modèle prédit toujours "Basse" et rate les urgences réelles.
#
# POURQUOI ON NE SUPPRIME PAS LA CLASSE MAJORITAIRE :
# - On perdrait ~690 exemples précieux d'apprentissage
# - Le modèle aurait moins de données pour apprendre les patterns "Basse"
# - L'oversampling (sur-échantillonnage) est préférable : on AUGMENTE les
#   classes minoritaires sans perdre d'information.
#
# MÉTHODE : RandomOverSampler (imbalanced-learn)
# - Duplique aléatoirement des exemples de classes minoritaires
# - Plus simple que SMOTE pour les données textuelles
# - Préserve les vraies données (pas de synthèse artificielle)
#
# CONTRAINTES :
# - Ne PAS utiliser "categorie", "urgence", "type_ticket", "temps_resolution"
#   comme features d'entrée pour le modèle
# - Ne PAS utiliser de deep learning
# - Utiliser uniquement pandas, scikit-learn, imbalanced-learn
#
# =============================================================================

import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_FILE = "data/tickets_cleaned.csv"
OUTPUT_FILE = "data/tickets_balanced.csv"
RANDOM_STATE = 42

# Objectifs de distribution (chaque classe minoritaire ≥ 25-30%)
TARGET_MOYENNE_RATIO = 0.28  # ~28% pour "Moyenne"
TARGET_HAUTE_RATIO = 0.22    # ~22% pour "Haute" (moins car moins de données originales)

np.random.seed(RANDOM_STATE)

# =============================================================================
# ÉTAPE 1 : CHARGEMENT ET VALIDATION DES COLONNES
# =============================================================================
print("=" * 70)
print("ÉTAPE 1 : CHARGEMENT DES DONNÉES")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

# Colonnes attendues (ordre exact)
expected_columns = ['ID', 'texte', 'titre', 'categorie', 'urgence', 
                    'temps_resolution', 'type_ticket', 'nb_mots']

print(f"✅ Fichier chargé : {INPUT_FILE}")
print(f"✅ Nombre de lignes : {len(df)}")
print(f"✅ Colonnes trouvées : {list(df.columns)}")

# Validation de l'ordre des colonnes
if list(df.columns) != expected_columns:
    print(f"⚠️  Ordre des colonnes différent de l'attendu")
    print(f"   Attendu : {expected_columns}")
    print(f"   Trouvé  : {list(df.columns)}")
    
# Vérifier que toutes les colonnes sont présentes
missing_cols = set(expected_columns) - set(df.columns)
if missing_cols:
    raise ValueError(f"❌ Colonnes manquantes : {missing_cols}")
print(f"✅ Toutes les colonnes attendues sont présentes")

# =============================================================================
# ÉTAPE 2 : NETTOYAGE BASIQUE (texte et titre)
# =============================================================================
print("\n" + "=" * 70)
print("ÉTAPE 2 : NETTOYAGE DES DONNÉES")
print("=" * 70)

# Assurer que texte et titre sont des strings, remplir NaN avec ''
df['texte'] = df['texte'].fillna('').astype(str)
df['titre'] = df['titre'].fillna('').astype(str)

print(f"✅ Colonnes 'texte' et 'titre' converties en strings")
print(f"✅ Valeurs NaN remplacées par des chaînes vides")

# =============================================================================
# ÉTAPE 3 : CRÉATION DE LA COLONNE text_full
# =============================================================================
print("\n" + "=" * 70)
print("ÉTAPE 3 : CRÉATION DE 'text_full'")
print("=" * 70)

# Créer la colonne "text_full" = titre + " " + texte
df['text_full'] = df['titre'] + " " + df['texte']
df['text_full'] = df['text_full'].str.strip()

print(f"✅ Colonne 'text_full' créée (titre + ' ' + texte)")
print(f"   Exemple : '{df['text_full'].iloc[0][:60]}...'")

# =============================================================================
# ÉTAPE 4 : DISTRIBUTION AVANT NORMALISATION ET ÉQUILIBRAGE
# =============================================================================
print("\n" + "=" * 70)
print("ÉTAPE 4 : DISTRIBUTION DE 'urgence' AVANT TRAITEMENT")
print("=" * 70)

distribution_avant = df['urgence'].value_counts()
distribution_avant_pct = df['urgence'].value_counts(normalize=True) * 100

print("\n📊 Distribution ORIGINALE :")
print("-" * 40)
for urgence in distribution_avant.index:
    count = distribution_avant[urgence]
    pct = distribution_avant_pct[urgence]
    bar = "█" * int(pct / 2)
    print(f"   {urgence:12} : {count:4} ({pct:5.2f}%) {bar}")

total_avant = len(df)
print(f"\n   TOTAL       : {total_avant}")

# =============================================================================
# ÉTAPE 5 : NORMALISATION - "Très haute" → "Haute"
# =============================================================================
print("\n" + "=" * 70)
print("ÉTAPE 5 : NORMALISATION DES CLASSES D'URGENCE")
print("=" * 70)

# Règle obligatoire : "Très haute" → "Haute"
if 'Très haute' in df['urgence'].values:
    count_tres_haute = (df['urgence'] == 'Très haute').sum()
    print(f"⚠️  'Très haute' détectée : {count_tres_haute} occurrences")
    print(f"   → Remplacement par 'Haute' (règle de normalisation)")
    
    df['urgence'] = df['urgence'].replace('Très haute', 'Haute')
    
    print(f"✅ 'Très haute' remplacée par 'Haute'")
else:
    print("ℹ️  Pas de 'Très haute' dans les données")

# Vérifier qu'on a maintenant exactement 3 classes
classes_finales = df['urgence'].unique()
print(f"\n✅ Classes après normalisation : {sorted(classes_finales)}")

# Afficher distribution après normalisation
print("\n📊 Distribution APRÈS normalisation :")
print("-" * 40)
distribution_norm = df['urgence'].value_counts()
distribution_norm_pct = df['urgence'].value_counts(normalize=True) * 100

for urgence in ['Basse', 'Moyenne', 'Haute']:
    if urgence in distribution_norm.index:
        count = distribution_norm[urgence]
        pct = distribution_norm_pct[urgence]
        bar = "█" * int(pct / 2)
        print(f"   {urgence:12} : {count:4} ({pct:5.2f}%) {bar}")

# =============================================================================
# ÉTAPE 6 : OVERSAMPLING AVEC RandomOverSampler
# =============================================================================
print("\n" + "=" * 70)
print("ÉTAPE 6 : RÉÉQUILIBRAGE PAR OVERSAMPLING")
print("=" * 70)

# Compter les classes actuelles
count_basse = (df['urgence'] == 'Basse').sum()
count_moyenne = (df['urgence'] == 'Moyenne').sum()
count_haute = (df['urgence'] == 'Haute').sum()

print(f"\n📈 État actuel :")
print(f"   - Basse   : {count_basse}")
print(f"   - Moyenne : {count_moyenne}")
print(f"   - Haute   : {count_haute}")

# Calculer les cibles pour atteindre ~28% Moyenne et ~22% Haute
# Total final = Basse + Moyenne_target + Haute_target
# On résout : Moyenne_target / Total = 0.28 et Haute_target / Total = 0.22
# Donc Basse / Total = 0.50, soit Total = Basse / 0.50

# Pour avoir Basse = 50%, Moyenne = 28%, Haute = 22%
total_target = int(count_basse / 0.50)
target_moyenne = int(total_target * TARGET_MOYENNE_RATIO)
target_haute = int(total_target * TARGET_HAUTE_RATIO)

# S'assurer que les targets sont au moins égaux aux counts actuels
target_moyenne = max(target_moyenne, count_moyenne)
target_haute = max(target_haute, count_haute)

print(f"\n📈 Stratégie d'oversampling :")
print(f"   - Garder TOUS les {count_basse} exemples 'Basse' (aucune suppression)")
print(f"   - Augmenter 'Moyenne' : {count_moyenne} → {target_moyenne} exemples")
print(f"   - Augmenter 'Haute'   : {count_haute} → {target_haute} exemples")

# Définir la stratégie de sampling
sampling_strategy = {
    'Basse': count_basse,       # Garder tous (pas de suppression)
    'Moyenne': target_moyenne,   # Augmenter
    'Haute': target_haute        # Augmenter
}

# Préparer les données pour RandomOverSampler
# On utilise l'index comme X (on veut juste dupliquer des lignes entières)
X = df.index.values.reshape(-1, 1)
y = df['urgence'].values

# Appliquer RandomOverSampler
print("\n🔄 Application de RandomOverSampler...")
ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE)
X_resampled, y_resampled = ros.fit_resample(X, y)

print(f"✅ Oversampling effectué")
print(f"   - Avant : {len(X)} lignes")
print(f"   - Après : {len(X_resampled)} lignes")

# =============================================================================
# ÉTAPE 7 : RECONSTRUCTION DU DATAFRAME AVEC IDs UNIQUES
# =============================================================================
print("\n" + "=" * 70)
print("ÉTAPE 7 : RECONSTRUCTION DU DATAFRAME")
print("=" * 70)

# Récupérer les indices originaux
original_indices = X_resampled.flatten()

# Reconstruire le DataFrame avec toutes les colonnes
df_balanced = df.iloc[original_indices].copy()

# Générer des IDs uniques pour les lignes dupliquées
print("🔢 Génération d'IDs uniques pour les lignes dupliquées...")

new_ids = []
id_counter = {}

for i, idx in enumerate(original_indices):
    original_id = df.iloc[idx]['ID']
    
    if original_id not in id_counter:
        id_counter[original_id] = 0
        new_ids.append(original_id)  # Première occurrence : garder l'ID original
    else:
        id_counter[original_id] += 1
        # Pour les duplications : créer un nouvel ID unique avec suffixe
        new_id = f"{original_id}_dup_{id_counter[original_id]}"
        new_ids.append(new_id)

df_balanced['ID'] = new_ids

# Réinitialiser l'index
df_balanced = df_balanced.reset_index(drop=True)

# Réordonner les colonnes (colonnes originales + text_full)
final_columns = ['ID', 'texte', 'titre', 'categorie', 'urgence', 
                 'temps_resolution', 'type_ticket', 'nb_mots', 'text_full']
df_balanced = df_balanced[final_columns]

# Vérifier l'unicité des IDs
n_unique_ids = df_balanced['ID'].nunique()
n_total_rows = len(df_balanced)

print(f"✅ DataFrame reconstruit avec {n_total_rows} lignes")
print(f"✅ IDs uniques : {n_unique_ids} / {n_total_rows} (100% unique: {n_unique_ids == n_total_rows})")
print(f"✅ Colonnes finales : {list(df_balanced.columns)}")

# =============================================================================
# ÉTAPE 8 : DISTRIBUTION APRÈS ÉQUILIBRAGE
# =============================================================================
print("\n" + "=" * 70)
print("ÉTAPE 8 : DISTRIBUTION DE 'urgence' APRÈS ÉQUILIBRAGE")
print("=" * 70)

distribution_apres = df_balanced['urgence'].value_counts()
distribution_apres_pct = df_balanced['urgence'].value_counts(normalize=True) * 100

print("\n📊 Distribution FINALE :")
print("-" * 40)
for urgence in ['Basse', 'Moyenne', 'Haute']:
    if urgence in distribution_apres.index:
        count = distribution_apres[urgence]
        pct = distribution_apres_pct[urgence]
        bar = "█" * int(pct / 2)
        print(f"   {urgence:12} : {count:4} ({pct:5.2f}%) {bar}")

total_apres = len(df_balanced)
print(f"\n   TOTAL       : {total_apres}")

# =============================================================================
# RÉSUMÉ COMPARATIF
# =============================================================================
print("\n" + "=" * 70)
print("RÉSUMÉ : COMPARAISON AVANT / APRÈS")
print("=" * 70)

print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  AVANT ÉQUILIBRAGE                                                  │
│  ─────────────────────────────────────────────────────────────────  │
│  • Total      : {total_avant:4} tickets                                       │
│  • Basse      : {distribution_norm.get('Basse', 0):4} ({distribution_norm_pct.get('Basse', 0):5.2f}%)                                 │
│  • Moyenne    : {distribution_norm.get('Moyenne', 0):4} ({distribution_norm_pct.get('Moyenne', 0):5.2f}%)                                  │
│  • Haute      : {distribution_norm.get('Haute', 0):4} ({distribution_norm_pct.get('Haute', 0):5.2f}%)                                  │
├─────────────────────────────────────────────────────────────────────┤
│  APRÈS ÉQUILIBRAGE (RandomOverSampler)                              │
│  ─────────────────────────────────────────────────────────────────  │
│  • Total      : {total_apres:4} tickets                                      │
│  • Basse      : {distribution_apres.get('Basse', 0):4} ({distribution_apres_pct.get('Basse', 0):5.2f}%)                                 │
│  • Moyenne    : {distribution_apres.get('Moyenne', 0):4} ({distribution_apres_pct.get('Moyenne', 0):5.2f}%)                                 │
│  • Haute      : {distribution_apres.get('Haute', 0):4} ({distribution_apres_pct.get('Haute', 0):5.2f}%)                                 │
└─────────────────────────────────────────────────────────────────────┘
""")

# =============================================================================
# ÉTAPE 9 : SAUVEGARDE DU FICHIER ÉQUILIBRÉ
# =============================================================================
print("=" * 70)
print("ÉTAPE 9 : SAUVEGARDE")
print("=" * 70)

df_balanced.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Dataset équilibré sauvegardé : {OUTPUT_FILE}")
print(f"✅ Nombre de lignes : {len(df_balanced)}")
print(f"✅ Colonnes : {list(df_balanced.columns)}")

# Rappel des contraintes
print("\n" + "-" * 70)
print("📌 RAPPEL : Pour l'entraînement du modèle, utiliser UNIQUEMENT :")
print("   - INPUT  : 'text_full' (ou 'texte' + 'titre')")
print("   - OUTPUT : 'urgence' (Basse, Moyenne, Haute)")
print("   - NE PAS utiliser : categorie, type_ticket, temps_resolution")
print("-" * 70)

print("\n" + "=" * 70)
print("✅ RÉÉQUILIBRAGE TERMINÉ AVEC SUCCÈS !")
print("=" * 70)
