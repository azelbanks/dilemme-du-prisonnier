# ♟️ Prisoner's Dilemma: AI vs Algorithms

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](TON_LIEN_STREAMLIT_ICI)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)

> **Expérience Comportementale :** Simulation de l'émergence de la coopération dans un environnement mixte (Algorithmes Déterministes vs LLM Génératifs).

---

## 🎯 Objectif
Reproduire l'expérience d'Axelrod (1981) en remplaçant les humains par des modèles de langage (**Mistral 7B** & **Llama 3**).
L'objectif est d'analyser si les IA sont capables de découvrir l'équilibre de Nash ou si elles cèdent à l'optimisation locale (Trahison).

## 🛠️ Architecture Technique (ETL)

Ce projet implémente un pipeline de données complet :

| Phase | Technologies | Description |
| :--- | :--- | :--- |
| **1. Extract** | `Python`, `Ollama`, `ThreadPool` | Simulation multi-agents parallélisée. Technique de **Prompt Masking** pour éviter le biais d'apprentissage. |
| **2. Transform** | `Pandas`, `TextBlob` | Feature Engineering (Lag Features) et **Analyse de Sentiment** (NLP) pour mesurer la dissonance cognitive. |
| **3. Load** | `Parquet`, `PyArrow` | Stockage colonnaire haute performance. |
| **4. Viz** | `Streamlit`, `Plotly` | Dashboard interactif déployé en SaaS. |

## 📊 Résultats Clés

* **Vainqueur :** L'algorithme `Grim_Bot` (Rancunier) domine grâce à une stratégie de dissuasion forte.
* **Performance IA :** L'agent `Machiavel_Llama` a échoué à maximiser ses gains, pénalisé par des tentatives de trahison mal calculées.
* **Phénomène :** Observation d'une **hypocrisie statistiquement significative** chez l'IA (Sentiment positif lors des trahisons).

## 🚀 Comment lancer le projet

### Pré-requis
* Python 3.9+
* Ollama installé localement (`ollama pull mistral` & `ollama pull llama3`)

### Installation
```bash
git clone [https://github.com/TON_USER/prisoner-dilemma-analytics.git](https://github.com/TON_USER/prisoner-dilemma-analytics.git)
cd prisoner-dilemma-analytics
pip install -r requirements.txt
