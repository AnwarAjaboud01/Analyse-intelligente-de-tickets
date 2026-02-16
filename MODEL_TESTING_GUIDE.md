# 🧪 Model Testing Guide

Complete guide to test all machine learning models in the Intelligent Ticket Analysis system.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start - Test All Models](#quick-start)
3. [Individual Model Testing](#individual-model-testing)
4. [Performance Metrics Explained](#performance-metrics-explained)
5. [Overfitting Analysis](#overfitting-analysis)
6. [Single Prediction Testing](#single-prediction-testing)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Files
Ensure these files exist in your project:

```
Analyse_intelligente_de_tickets_DS/
├── models/
│   ├── urgency_pipeline.pkl
│   ├── category_pipeline.pkl
│   ├── type_pipeline.pkl
│   ├── time_pipeline.pkl
│   └── metadata.json
├── evaluate_models.py
└── analyze_overfitting.py
```

### Check Python Installation

```powershell
# Check if Python is installed
py --version

# Expected output: Python 3.10.x or higher
```

---

## Quick Start - Test All Models

### Command 1: View All Model Performance Metrics

```powershell
cd "c:/Users/farou/OneDrive/Desktop/ai tickets/Analyse-intelligente-de-tickets"
py Analyse_intelligente_de_tickets_DS/evaluate_models.py
```

**What this does:**
- Loads all 4 trained models from `models/metadata.json`
- Displays accuracy, F1-score, and other metrics for each model
- Shows performance on Training, Validation, and Test sets

**Expected Output:**
```
================================================================================
🎯 MODEL PERFORMANCE EVALUATION
================================================================================

🏷️  MODEL 1: URGENCY CLASSIFICATION
    Algorithm: LogisticRegression
    Classes: Basse, Moyenne, Haute
    Training samples: 1015

📊 URGENCY
--------------------------------------------------------------------------------
  OOF_TRAIN       | Accuracy:  97.44% | F1-Score:  97.57%
  VALIDATION      | Accuracy:  96.33% | F1-Score:  96.58%
  TEST            | Accuracy:  98.17% | F1-Score:  98.10%

[... continues for all 4 models ...]
```

---

## Individual Model Testing

### Test 1: Urgency Classification Model

```powershell
# Create a temp Python file
@"
import json
m = json.load(open('Analyse_intelligente_de_tickets_DS/models/metadata.json'))
print('Urgency Model:')
print('Algorithm:', m['urgency_model']['algorithm'])
print('Test Accuracy:', f"{m['urgency_model']['metrics']['test']['accuracy']*100:.2f}%")
"@ | Out-File -Encoding utf8 temp_script.py

# Run it
py -3.10 temp_script.py

# Clean up
Remove-Item temp_script.py
```

**What this does:**
- Extracts urgency model metadata
- Shows algorithm used (Logistic Regression)
- Displays test accuracy

**Expected Output:**
```
Urgency Model:
 Algorithm: LogisticRegression
 Test Accuracy: 98.17%
```

---

### Test 2: Category Classification Model

```powershell
# Create a temp Python file to check category model metrics
@"
import json
m = json.load(open('Analyse_intelligente_de_tickets_DS/models/metadata.json'))
print('Category Model:')
print('Classes:', m['category_model']['n_classes'])
print('Test Accuracy:', f"{m['category_model']['metrics']['test']['accuracy']*100:.2f}%")
print('Test F1 Score:', f"{m['category_model']['metrics']['test']['f1_macro']*100:.2f}%")
"@ | Out-File -Encoding utf8 temp_script.py

# Run it
py -3.10 temp_script.py

# Clean up
Remove-Item temp_script.py
```

**What this does:**
- Shows number of categories (50)
- Displays accuracy and F1-score
- F1-score is important for imbalanced classes

**Expected Output:**
```
Category Model:
 Classes: 50
 Test Accuracy: 76.15%
 Test F1: 48.51%
```

---

### Test 3: Type Classification Model

```powershell
py -c "import json; m=json.load(open('Analyse_intelligente_de_tickets_DS/models/metadata.# Create a temp Python file to check type model metrics
@"
import json
m = json.load(open('Analyse_intelligente_de_tickets_DS/models/metadata.json'))
print('Type Model:')
print('Classes:', m['type_model']['classes'])
print('Test Accuracy:', f"{m['type_model']['metrics']['test']['accuracy']*100:.2f}%")
"@ | Out-File -Encoding utf8 temp_script.py

# Run it
py -3.10 temp_script.py

# Clean up
Remove-Item temp_script.py
```

**What this does:**
- Shows the two classes: Demande (Request) and Incident
- Displays binary classification accuracy

**Expected Output:**
```
Type Model:
 Classes: ['Demande', 'Incident']
 Test Accuracy: 95.41%
```

---

### Test 4: Time Regression Model

```powershell
# Create a temp Python file to check time model metrics
@"
import json
m = json.load(open('Analyse_intelligente_de_tickets_DS/models/metadata.json'))
t = m['time_model']['metrics']['test']
print('Time Model:')
print('Algorithm:', m['time_model']['algorithm'])
print(f'MAE: {t["mae"]:.2f} hours')
print(f'RMSE: {t["rmse"]:.2f} hours')
print(f'R²: {t["r2"]:.4f}')
"@ | Out-File -Encoding utf8 temp_script.py

# Run it
py -3.10 temp_script.py

# Clean up
Remove-Item temp_script.py
```

**What this does:**
- Shows regression metrics (not accuracy, since it's predicting a number)
- MAE = Mean Absolute Error (average prediction error in hours)
- RMSE = Root Mean Squared Error (penalizes large errors more)
- R² = Coefficient of determination (0-1, higher is better)

**Expected Output:**
```
Time Model:
 Algorithm: XGBRegressor (régularisé)
 MAE: 8.74 hours
 RMSE: 17.33 hours
 R²: 0.7722
```

---

## Performance Metrics Explained

### For Classification Models (Urgency, Category, Type)

| Metric | What it Means | Good Value |
|--------|---------------|------------|
| **Accuracy** | % of correct predictions | > 90% (excellent), > 75% (good) |
| **F1-Score** | Balance between precision and recall | > 0.90 (excellent), > 0.70 (good) |
| **Macro F1** | Average F1 across all classes (fair to rare classes) | > 0.80 (excellent) |

**Why F1 matters:**
- Accuracy can be misleading with imbalanced data
- F1-score considers both false positives and false negatives
- Macro F1 treats all classes equally (important for 50-class problem)

---

### For Regression Model (Time)

| Metric | What it Means | Interpretation |
|--------|---------------|----------------|
| **MAE** | Average error in hours | Lower is better. 8.74h means ±9h on average |
| **RMSE** | Error with penalty for large mistakes | Lower is better. Penalizes big errors more |
| **R²** | % of variance explained | 0.77 = model explains 77% of time variation |

**Example:**
- Actual resolution time: 24 hours
- Predicted time: 16 hours
- Error: 8 hours (within the MAE of 8.74h)

---

## Overfitting Analysis

### Command: Detect Overfitting

```powershell
py Analyse_intelligente_de_tickets_DS/analyze_overfitting.py
```

**What this does:**
- Compares Training vs Test performance
- Calculates the "generalization gap"
- Diagnoses overfitting, underfitting, or good fit

**Key Indicators:**

| Scenario | Train Acc | Test Acc | Gap | Diagnosis |
|----------|-----------|----------|-----|-----------|
| **Overfitting** | 99% | 82% | +17% | ⚠️ Model memorized training data |
| **Good Fit** | 95% | 94% | +1% | ✅ Excellent generalization |
| **Underfitting** | 94% | 96% | -2% | ✅ Model generalizes better than expected |

**Your Models:**
- Urgency: Train 97.44%, Test 98.17% → Gap: **-0.73%** ✅
- Type: Train 94.58%, Test 95.41% → Gap: **-0.83%** ✅
- Category: Train 77.73%, Test 76.15% → Gap: **+1.58%** ✅
- Time: Train R² 0.89, Test R² 0.77 → Gap: **+0.12** ⚠️ (mild, acceptable)

---

## Single Prediction Testing

### Test a Real Ticket Prediction

Create a test file `test_single_prediction.py`:

```python
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.ml.predict_pipeline import predict_ticket

# Test ticket
titre = "Problème connexion VPN urgent"
texte = "Je ne peux pas me connecter au VPN depuis ce matin. Travail bloqué."

print("Testing ticket prediction...")
print(f"Titre: {titre}")
print(f"Texte: {texte}\n")

result = predict_ticket(titre, texte)

print("Predictions:")
print(f"  Urgence: {result['urgence_pred']}")
print(f"  Catégorie: {result['categorie_pred']}")
print(f"  Type: {result['type_ticket_pred']}")
print(f"  Temps estimé: {result['temps_resolution_pred']} heures")
```

**Run it:**
```powershell
py Analyse_intelligente_de_tickets_DS/test_single_prediction.py
```

**Expected Output:**
```
Testing ticket prediction...
Titre: Problème connexion VPN urgent
Texte: Je ne peux pas me connecter au VPN depuis ce matin. Travail bloqué.

Predictions:
  Urgence: Haute
  Catégorie: VPN
  Type: Incident
  Temps estimé: 4.5 heures
```

---

## Verify Model Files Exist

```powershell
# Check all model files are present
Get-ChildItem "Analyse_intelligente_de_tickets_DS/models" | Select-Object Name, Length

# Expected output:
# Name                    Length
# ----                    ------
# category_pipeline.pkl   458707
# metadata.json           6201
# time_pipeline.pkl       102140
# type_pipeline.pkl       53115
# urgency_pipeline.pkl    67915
```

---

## Load and Inspect a Model

```powershell
py -c "import joblib; model = joblib.load('Analyse_intelligente_de_tickets_DS/models/urgency_pipeline.pkl'); print('Model loaded successfully'); print('Keys:', list(model.keys()))"
```

**Expected Output:**
```
Model loaded successfully
Keys: ['tfidf', 'model', 'text_column', 'numeric_columns', 'encoder']
```

---

## Troubleshooting

### Error: "No module named 'src'"

**Solution:**
```powershell
# Make sure you're in the project root
cd "c:/Users/farou/OneDrive/Desktop/ai tickets/Analyse-intelligente-de-tickets"

# Run with proper path
py Analyse_intelligente_de_tickets_DS/evaluate_models.py
```

---

### Error: "FileNotFoundError: metadata.json"

**Solution:**
```powershell
# Check if file exists
Test-Path "Analyse_intelligente_de_tickets_DS/models/metadata.json"

# If False, the models haven't been trained yet
# You need to train them first (not covered in this guide)
```

---

### Error: "ModuleNotFoundError: No module named 'joblib'"

**Solution:**
```powershell
# Install required packages
pip install joblib scikit-learn pandas numpy
```

---

## Summary of Commands

| Task | Command |
|------|---------|
| **Test all models** | `py Analyse_intelligente_de_tickets_DS/evaluate_models.py` |
| **Check overfitting** | `py Analyse_intelligente_de_tickets_DS/analyze_overfitting.py` |
| **List model files** | `Get-ChildItem Analyse_intelligente_de_tickets_DS/models` |
| **Test single prediction** | Create and run `test_single_prediction.py` |
| **View metadata** | `Get-Content Analyse_intelligente_de_tickets_DS/models/metadata.json` |

---

## Next Steps

1. ✅ Run `evaluate_models.py` to see all metrics
2. ✅ Run `analyze_overfitting.py` to verify no overfitting
3. ✅ Test a single prediction to see the pipeline in action
4. 📊 Use these results in your presentation to demonstrate model quality

---

**Last Updated:** 2026-02-16  
**Author:** AI Ticket Analysis Team
