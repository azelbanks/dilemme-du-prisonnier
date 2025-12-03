import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Configuration "Senior Data Scientist" pour le style des graphiques
sns.set_theme(style="whitegrid", context="talk")
PALETTE = "viridis"

def load_clean_data(filename="clean_prisoner_dataset.parquet"):
    print(f"📥 Chargement des données enrichies : {filename}")
    return pd.read_parquet(filename)

def generate_leaderboard(df):
    """Génère le classement final par score moyen"""
    print("\n--- 🏆 CLASSEMENT GÉNÉRAL ---")
    
    # On combine les scores P1 et P2 pour avoir une vue globale par agent
    df_long = pd.concat([
        df[['p1_name', 'p1_score']].rename(columns={'p1_name': 'agent', 'p1_score': 'score'}),
        df[['p2_name', 'p2_score']].rename(columns={'p2_name': 'agent', 'p2_score': 'score'})
    ])
    
    ranking = df_long.groupby('agent')['score'].mean().sort_values(ascending=False)
    print(ranking)
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=ranking.values, y=ranking.index, hue=ranking.index, palette=PALETTE, legend=False)
    ax.bar_label(ax.containers[0], fmt='%.2f')
    plt.title("Score Moyen par Tour (Efficacité de la Stratégie)")
    plt.xlabel("Points moyens")
    plt.tight_layout()
    plt.savefig("viz_1_leaderboard.png")
    print("📸 Graphique sauvé : viz_1_leaderboard.png")

def analyze_nash_equilibrium(df):
    """Heatmap des interactions"""
    print("\n--- 🧠 ANALYSE DES DUELS (PAYOFF MATRIX) ---")
    
    pivot_matrix = df.pivot_table(
        index='p1_name', 
        columns='p2_name', 
        values='p1_score', 
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_matrix, annot=True, fmt=".1f", cmap="RdYlGn", center=2.25)
    plt.title("Matrice des Gains Moyens (Qui bat qui ?)")
    plt.ylabel("Joueur (Héros)")
    plt.xlabel("Adversaire")
    plt.tight_layout()
    plt.savefig("viz_2_heatmap.png")
    print("📸 Graphique sauvé : viz_2_heatmap.png")

def analyze_cooperation_timeline(df):
    """Montre si la coopération s'effondre avec le temps"""
    print("\n--- 📉 ÉVOLUTION TEMPORELLE ---")
    
    coop_per_round = df.groupby('round_num')[['p1_is_coop', 'p2_is_coop']].mean().mean(axis=1)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=coop_per_round.index, y=coop_per_round.values, linewidth=3, color="#2ecc71")
    plt.fill_between(coop_per_round.index, coop_per_round.values, alpha=0.3, color="#2ecc71")
    plt.ylim(0, 1)
    plt.title("Taux de Coopération Global au fil du temps")
    plt.xlabel("Numéro du Tour")
    plt.ylabel("% de Coopération")
    plt.tight_layout()
    plt.savefig("viz_3_timeline.png")
    print("📸 Graphique sauvé : viz_3_timeline.png")

def analyze_sentiment_hypocrisy(df):
    """
    NOUVEAU (Joker Senior) : Analyse de l'hypocrisie via NLP.
    On regarde le sentiment moyen quand l'agent TRAHIT.
    """
    print("\n--- 🎭 ANALYSE DE L'HYPOCRISIE (NLP) ---")
    
    if 'p1_sentiment' not in df.columns:
        print("⚠️ Colonne NLP manquante. Ignoré.")
        return

    # Préparation des données : On veut Agent + Action + Sentiment
    df_nlp = pd.concat([
        df[['p1_name', 'p1_move', 'p1_sentiment']].rename(columns={'p1_name': 'Agent', 'p1_move': 'Action', 'p1_sentiment': 'Sentiment'}),
        df[['p2_name', 'p2_move', 'p2_sentiment']].rename(columns={'p2_name': 'Agent', 'p2_move': 'Action', 'p2_sentiment': 'Sentiment'})
    ])
    
    # On filtre pour ne garder que les actions de TRAHISON (T)
    # et seulement les Agents IA (les algos ont un sentiment de 0.0)
    hypocrisy_df = df_nlp[
        (df_nlp['Action'] == 'T') & 
        (df_nlp['Agent'].str.contains("Mistral|Llama|AI"))
    ]
    
    if hypocrisy_df.empty:
        print("⚠️ Pas assez de données de trahison pour l'analyse NLP.")
        return

    plt.figure(figsize=(10, 6))
    # Correction du Warning : on ajoute hue et legend=False
sns.boxplot(data=hypocrisy_df, x='Agent', y='Sentiment', hue='Agent', palette="Reds", legend=False)
    plt.axhline(0, color='black', linestyle='--') # Ligne de neutralité
    plt.title("Tonalité des justifications lors d'une TRAHISON (Hypocrisie)")
    plt.ylabel("Sentiment (-1=Haineux, +1=Positif/Hypocrite)")
    plt.tight_layout()
    plt.savefig("viz_4_nlp_hypocrisy.png")
    print("📸 Graphique sauvé : viz_4_nlp_hypocrisy.png")

def executive_summary(df):
    """Résumé textuel pour le jury"""
    print("\n=============================================")
    print("📝 RAPPORT EXÉCUTIF (INSIGHTS)")
    print("=============================================")
    
    # Insight 1 : Taux de trahison global
    total_moves = len(df) * 2
    total_coop = df['p1_is_coop'].sum() + df['p2_is_coop'].sum()
    coop_rate = (total_coop / total_moves) * 100
    
    print(f"1. Taux de Coopération Global : {coop_rate:.1f}%")
    if coop_rate < 50:
        print("   -> L'environnement est HOSTILE (Majorité de trahisons).")
    else:
        print("   -> L'environnement est BIENVEILLANT (Majorité de coopérations).")

    # Insight 2 : Le plus grand traître
    df_long = pd.concat([
        df[['p1_name', 'p1_is_coop']].rename(columns={'p1_name': 'agent', 'p1_is_coop': 'coop'}),
        df[['p2_name', 'p2_is_coop']].rename(columns={'p2_name': 'agent', 'p2_is_coop': 'coop'})
    ])
    traitor = df_long.groupby('agent')['coop'].mean().idxmin()
    print(f"2. Le plus grand 'Machiavel' est : {traitor}")
    
    # Insight 3 : NLP
    if 'p1_sentiment' in df.columns:
        mean_sent = df['p1_sentiment'].mean()
        print(f"3. Tonalité moyenne des justifications : {mean_sent:.2f} (Positif = Poli)")

if __name__ == "__main__":
    df = load_clean_data()
    
    # Génération des VIZ
    generate_leaderboard(df)
    analyze_nash_equilibrium(df)
    analyze_cooperation_timeline(df)
    analyze_sentiment_hypocrisy(df) # <--- NOUVEAU
    
    # Conclusion textuelle
    executive_summary(df)