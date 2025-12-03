# ♟️ Prisoner's Dilemma: AI vs Algorithms

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dilemme-du-prisonnier.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-orange)](https://ollama.com/)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)

> 🔴 **LIVE DEMO :** [**Accéder au Dashboard Interactif (Streamlit)**](https://dilemme-du-prisonnier.streamlit.app/)

---

## 🎯 Objectif du Projet
Ce projet revisite l'expérience historique de **Robert Axelrod (1981)** sur l'émergence de la coopération, en remplaçant les humains par des **Agents IA Génératifs (LLM)**.

L'objectif est de construire un **Pipeline Data Engineering complet (ETL)** pour simuler, stocker et analyser si des modèles comme *Llama 3* ou *Mistral* sont capables de découvrir l'équilibre de Nash ou s'ils cèdent à l'optimisation locale (Trahison).

## 📊 Résultats Clés (Teaser)

* 🏆 **Vainqueur :** L'algorithme **Grim_Bot** (Rancunier) domine le tournoi grâce à une stratégie de dissuasion absolue.
* 📉 **Échec de l'IA :** L'agent **Machiavel_Llama** (Llama 3) a échoué à maximiser ses gains, pénalisé par des tentatives de trahison mal calculées.
* 🎭 **Phénomène NLP :** Détection d'une **hypocrisie statistiquement significative** chez l'IA (Sentiment positif détecté dans les justifications de trahison).

---

## 🛠️ Architecture Technique (ETL)

Le projet implémente une architecture **ELT (Extract, Load, Transform)** moderne et résiliente :

| Phase | Technologie | Description Technique |
| :--- | :--- | :--- |
| **1. EXTRACT** | `Python`, `Ollama`, `ThreadPool` | Simulation multi-agents parallélisée. Utilisation du **"Prompt Masking"** (Scénario énergie) pour éviter le biais d'apprentissage du LLM. |
| **2. TRANSFORM** | `Pandas`, `TextBlob` | Feature Engineering vectorisé (Lag Features, Memory) et **Analyse de Sentiment (NLP)** pour mesurer la dissonance cognitive. |
| **3. LOAD** | `Parquet`, `PyArrow` | Stockage colonnaire haute performance et typage strict des données. |
| **4. VIZ** | `Streamlit`, `Plotly` | Application interactive déployée en SaaS avec analyse comportementale avancée. |

---

## 🚀 Installation & Reproduction

Si vous souhaitez faire tourner la simulation sur votre machine (Mac/Linux recommandé) :

### 1. Pré-requis
* Python 3.9+
* [Ollama](https://ollama.com/) installé localement.
* Modèles téléchargés :
    ```bash
    ollama pull mistral
    ollama pull llama3.1:8b
    ```

### 2. Installation
```bash
git clone [https://github.com/VOTRE_USER/prisoner-dilemma-analytics.git](https://github.com/VOTRE_USER/prisoner-dilemma-analytics.git)
cd prisoner-dilemma-analytics
pip install -r requirements.txt
