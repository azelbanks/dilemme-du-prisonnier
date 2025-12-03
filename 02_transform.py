import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sys

# Gestion professionnelle des dépendances optionnelles
try:
    from textblob import TextBlob
    NLP_AVAILABLE = True
except ImportError:
    print("⚠️ Attention : 'textblob' n'est pas installé. L'analyse de sentiment sera ignorée.")
    print("   -> Installez-le via : pip install textblob")
    NLP_AVAILABLE = False

# =================CONFIGURATION=================
INPUT_FILE = "raw_prisoner_data.parquet"
OUTPUT_FILE = "clean_prisoner_dataset.parquet"

def load_data(filepath):
    print(f"📥 Chargement de {filepath}...")
    try:
        df = pd.read_parquet(filepath)
        # Tri indispensable pour que le calcul de mémoire (shift) fonctionne chronologiquement
        df = df.sort_values(by=['match_id', 'round_num'])
        return df
    except Exception as e:
        print(f"❌ Erreur critique de chargement : {e}")
        return pd.DataFrame()

def get_sentiment_score(text):
    """
    Fonction Helper NLP.
    Retourne un score de polarité : -1 (Très Négatif) à +1 (Très Positif).
    Retourne 0.0 si neutre ou vide.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    try:
        return TextBlob(text).sentiment.polarity
    except Exception:
        return 0.0

def feature_engineering(df):
    """
    Création des colonnes dérivées (Niveau Senior).
    Intègre : Lag Features, Psychologie comportementale et NLP.
    """
    print("⚙️ Enrichissement des données (Feature Engineering)...")
    
    # --- 1. CONVERSION NUMÉRIQUE ---
    # Binarisation des choix (C=1, T=0) pour les calculs statistiques
    df['p1_is_coop'] = (df['p1_move'] == 'C').astype(int)
    df['p2_is_coop'] = (df['p2_move'] == 'C').astype(int)
    
    # --- 2. MÉMOIRE (LAG FEATURES) ---
    # On regarde ce qu'il s'est passé au tour T-1 (Shift)
    # Le groupby('match_id') empêche de mélanger les parties entre elles
    df['p1_prev_move'] = df.groupby('match_id')['p1_move'].shift(1).fillna("START")
    df['p2_prev_move'] = df.groupby('match_id')['p2_move'].shift(1).fillna("START")
    
    # --- 3. ÉTAT PSYCHOLOGIQUE (CONTEXTUALISATION) ---
    # Coopération Mutuelle (CC) : Confiance / Paix
    df['is_mutual_coop'] = ((df['p1_move'] == 'C') & (df['p2_move'] == 'C')).astype(int)
    
    # Trahison Subie par P1 (P1=C, P2=T) -> P1 est le "Dindon de la farce" (Sucker)
    df['p1_betrayed'] = ((df['p1_move'] == 'C') & (df['p2_move'] == 'T')).astype(int)
    
    # Trahison Infligée par P1 (P1=T, P2=C) -> P1 est l'"Exploiteur"
    df['p1_exploits'] = ((df['p1_move'] == 'T') & (df['p2_move'] == 'C')).astype(int)
    
    # Conflit Mutuel (TT) -> Guerre / Punition
    df['is_mutual_defect'] = ((df['p1_move'] == 'T') & (df['p2_move'] == 'T')).astype(int)

    # --- 4. KPIs PERFORMANCE (CUMULATIFS) ---
    # Score Cumulé (Running Total) pour voir la "Course aux points"
    df['p1_cum_score'] = df.groupby('match_id')['p1_score'].cumsum()
    df['p2_cum_score'] = df.groupby('match_id')['p2_score'].cumsum()

    # Taux de Coopération Glissant (Évolution de la gentillesse)
    df['p1_rolling_coop'] = df.groupby('match_id')['p1_is_coop'].expanding().mean().reset_index(level=0, drop=True)

    # --- 5. ANALYSE RÉACTIONNELLE ---
    # PARDON : Est-ce que je coopère ALORS QUE j'ai été trahi juste avant ?
    df['p1_prev_betrayed'] = ((df['p1_prev_move'] == 'C') & (df['p2_prev_move'] == 'T'))
    df['p1_forgives'] = (df['p1_prev_betrayed'] & (df['p1_move'] == 'C')).astype(int)

    # --- 6. NLP & SENTIMENT ANALYSIS (JOKER SENIOR) ---
    if NLP_AVAILABLE:
        print("🧠 Exécution de l'Analyse de Sentiment (NLP) sur les raisonnements...")
        # On applique la fonction sur les colonnes de texte
        # Cela permet de voir si Machiavel utilise des mots "positifs" pour masquer ses trahisons
        df['p1_sentiment'] = df['p1_reasoning'].apply(get_sentiment_score)
        df['p2_sentiment'] = df['p2_reasoning'].apply(get_sentiment_score)
    else:
        df['p1_sentiment'] = 0.0
        df['p2_sentiment'] = 0.0

    return df

def save_clean_data(df, filepath):
    # Utilisation de PyArrow pour une écriture Parquet optimisée et typée
    table = pa.Table.from_pandas(df)
    pq.write_table(table, filepath)
    print(f"💾 Sauvegarde réussie : {filepath}")
    print(f"📊 Dimensions finales : {df.shape[0]} lignes x {df.shape[1]} colonnes")

def quality_check_transform(df):
    """Audit rapide post-transformation pour valider l'intégrité"""
    print("\n--- 🔍 Audit Rapide des Transformations ---")
    
    # 1. Vérifier la Mémoire
    print("1. Test Cohérence Mémoire (Shift) :")
    print(df[['match_id', 'round_num', 'p1_move', 'p1_prev_move']].iloc[1:3].to_string(index=False))
    
    # 2. Vérifier le NLP
    if 'p1_sentiment' in df.columns:
        mean_sent = df['p1_sentiment'].mean()
        print(f"2. Score de sentiment moyen global : {mean_sent:.4f} (-1=Négatif, +1=Positif)")
    
    # 3. Vérifier les KPIs
    if df['p1_cum_score'].isnull().sum() == 0:
        print("✅ Tous les KPIs sont calculés sans erreurs (NaN).")
    else:
        print("❌ ALERTE : Présence de valeurs nulles dans les scores cumulés.")

if __name__ == "__main__":
    # Exécution du Pipeline ETL - Phase 2
    df_raw = load_data(INPUT_FILE)
    
    if not df_raw.empty:
        df_clean = feature_engineering(df_raw)
        quality_check_transform(df_clean)
        save_clean_data(df_clean, OUTPUT_FILE)
