# Analyse Intelligente de Tickets et Supports

## Description
Ce projet vise à développer une plateforme hybride combinant des modèles de Machine Learning classique et des techniques modernes de GenAI (RAG) pour analyser des documents tels que des tickets de support ou des CVs.

## Structure du Projet
- `data/` : Contient les fichiers de données (CSV, JSON, etc.).
- `notebooks/` : Contient les notebooks Jupyter pour l'analyse exploratoire.
- `src/ml/` : Contient les scripts pour les modèles de Machine Learning classique.
- `src/rag/` : Contient les scripts pour le chatbot basé sur RAG.
- `app/` : Contient les fichiers Streamlit pour l'interface utilisateur.
- `models/` : Contient les modèles entraînés sauvegardés.
- `docs/` : Contient la documentation technique.

## Installation
1. Clonez le dépôt :
   ```bash
   git clone <lien-du-repo>
   ```
2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Conventions de Commit
- `feat` : Ajout d'une nouvelle fonctionnalité.
- `fix` : Correction de bug.
- `docs` : Modification de la documentation.
- `refactor` : Refactorisation du code sans ajout de fonctionnalité.
- `test` : Ajout ou modification de tests.

## Branches
- `main` : Branche stable pour les versions finales.
- `dev` : Branche pour les développements en cours.

---

## Auteur
Personne C (Intégrateur Fullstack)
py -3.10 src/ml/predict_pipeline.py