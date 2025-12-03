import pandas as pd
import pyarrow.parquet as pq

def check_data_quality(filename="raw_prisoner_data.parquet"):
    print(f"🕵️‍♂️ AUDIT AVANCÉ DU FICHIER : {filename}")
    print("="*60)
    
    # 1. Chargement Robustesse
    try:
        df = pd.read_parquet(filename)
        print(f"✅ Chargement réussi.")
        print(f"   Volume : {df.shape[0]} lignes (interactions)")
        print(f"   Colonnes : {df.shape[1]}")
    except Exception as e:
        print(f"❌ CRITIQUE : Impossible de lire le fichier. {e}")
        return

    print("\n--- 1. AUDIT DES POPULATIONS (AGENTS) ---")
    # On récupère tous les noms uniques apparus en P1 ou P2
    all_agents = set(df['p1_name'].unique()) | set(df['p2_name'].unique())
    print(f"👥 Agents détectés ({len(all_agents)}) :")
    for agent in sorted(list(all_agents)):
        # Compte combien de fois cet agent a joué (en P1 ou P2)
        count = len(df[df['p1_name'] == agent]) + len(df[df['p2_name'] == agent])
        print(f"   - {agent} : {count} participations")

    print("\n--- 2. AUDIT DES SOURCES (PROVENANCE) ---")
    if 'tournament_id' in df.columns:
        sources = df['tournament_id'].value_counts()
        print("🏟️ Répartition par Tournoi/Patch :")
        print(sources.to_string())
    else:
        print("⚠️ Colonne 'tournament_id' manquante.")

    print("\n--- 3. QUALITÉ DES DONNÉES (INTEGRITY) ---")
    # Check Nulls
    nulls = df.isnull().sum().sum()
    if nulls == 0:
        print("✅ Aucun NULL détecté.")
    else:
        print(f"⚠️ {nulls} valeurs nulles trouvées (Vérifier si critique).")

    # Check Moves
    invalid_moves = df[~df['p1_move'].isin(['C', 'T']) | ~df['p2_move'].isin(['C', 'T'])]
    if invalid_moves.empty:
        print("✅ Tous les coups sont valides ('C' ou 'T').")
    else:
        print(f"❌ {len(invalid_moves)} coups invalides détectés !")

    print("\n--- 4. INTELLIGENCE ARTIFICIELLE (NLP CHECK) ---")
    # On filtre pour ne garder que les agents qui ne sont pas des Algos (donc ceux qui ont du texte > 10 chars)
    # "Ras" est le placeholder des Algos, on l'ignore.
    ai_df = df[df['p1_reasoning'].str.len() > 15]
    
    unique_ai_agents = ai_df['p1_name'].unique()
    
    if len(unique_ai_agents) > 0:
        print(f"🧠 {len(unique_ai_agents)} Agents IA identifiés avec justifications : {list(unique_ai_agents)}")
        print("-" * 40)
        for agent in unique_ai_agents:
            # Prend un exemple aléatoire pour cet agent
            sample = ai_df[ai_df['p1_name'] == agent].iloc[0]
            print(f"🤖 [{agent}] (Round {sample['round_num']})")
            print(f"   Raisonnement : \"{sample['p1_reasoning'][:120]}...\"")
            print(f"   Action : {sample['p1_move']}")
            print("-" * 40)
    else:
        print("⚠️ Aucune justification d'IA complexe trouvée (Est-ce normal ?)")

if __name__ == "__main__":
    check_data_quality()