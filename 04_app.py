import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ==============================================================================
# 1. CONFIGURATION DE LA PAGE (Look & Feel Pro)
# ==============================================================================
st.set_page_config(
    page_title="Prisoner's Dilemma Analytics",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Palette de couleurs personnalisée (Psychologie des couleurs)
COLORS = {
    "Cooperate": "#2ecc71",  # Vert confiance
    "Defect": "#e74c3c",     # Rouge trahison
    "Background": "#0e1117",
    "Grim_Bot": "#3498db",
    "TitForTat_Bot": "#1abc9c",
    "Ghandi_AI": "#f1c40f",
    "Machiavel_AI": "#9b59b6"
}

# ==============================================================================
# 2. CHARGEMENT DES DONNÉES (Optimisé avec Cache)
# ==============================================================================
@st.cache_data
def load_data():
    """Charge les données nettoyées avec mise en cache pour la performance."""
    try:
        df = pd.read_parquet("clean_prisoner_dataset.parquet")
        # Création d'une colonne 'Winner' pour chaque round
        df['round_winner'] = np.where(
            df['p1_score'] > df['p2_score'], df['p1_name'],
            np.where(df['p2_score'] > df['p1_score'], df['p2_name'], "Draw")
        )
        return df
    except FileNotFoundError:
        st.error("⚠️ Fichier 'clean_prisoner_dataset.parquet' introuvable. Lancez d'abord l'ETL.")
        return pd.DataFrame()

df = load_data()

# ==============================================================================
# 3. SIDEBAR (Filtres)
# ==============================================================================
with st.sidebar:
    st.title("🎛️ Contrôles")
    st.markdown("---")
    
    # Filtre sur les agents
    all_agents = sorted(list(set(df['p1_name'].unique()) | set(df['p2_name'].unique())))
    selected_agents = st.multiselect("Filtrer par Agent", all_agents, default=all_agents)
    
    st.markdown("### ℹ️ À propos")
    st.info(
        """
        **Projet M1 Data Engineering**
        Simulation du Dilemme du Prisonnier Itératif mixant Algorithmes et LLM.
        """
    )
    st.caption("Auteur: Azélie Bernard")

# ==============================================================================
# 4. DASHBOARD HEADER (KPIs)
# ==============================================================================
st.title("♟️ Dilemme du Prisonnier : Analyse Comportementale")
st.markdown("### *Exploration des dynamiques de coopération entre IA et Algorithmes*")

if not df.empty:
    # Calcul des KPIs globaux
    total_rounds = len(df)
    
    # Taux de coopération global
    coop_rate = ((df['p1_is_coop'].sum() + df['p2_is_coop'].sum()) / (total_rounds * 2)) * 100
    
    # Meilleur Agent (Score moyen)
    df_long = pd.concat([
        df[['p1_name', 'p1_score']].rename(columns={'p1_name': 'agent', 'p1_score': 'score'}),
        df[['p2_name', 'p2_score']].rename(columns={'p2_name': 'agent', 'p2_score': 'score'})
    ])
    best_agent = df_long.groupby('agent')['score'].mean().idxmax()
    best_score = df_long.groupby('agent')['score'].mean().max()
    
    # Traître principal
    df_coop_long = pd.concat([
        df[['p1_name', 'p1_is_coop']].rename(columns={'p1_name': 'agent', 'p1_is_coop': 'coop'}),
        df[['p2_name', 'p2_is_coop']].rename(columns={'p2_name': 'agent', 'p2_is_coop': 'coop'})
    ])
    traitor = df_coop_long.groupby('agent')['coop'].mean().idxmin()

    # Affichage en 3 colonnes
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Interactions", f"{total_rounds}")
    col2.metric("Taux Coopération", f"{coop_rate:.1f}%", delta_color="normal")
    col3.metric("🏆 Meilleur Stratège", best_agent, f"{best_score:.2f} pts/tour")
    col4.metric("😈 Plus Machiavélique", traitor)

    st.markdown("---")

    # ==============================================================================
    # 5. ONGLETS D'ANALYSE (STRUCTURE ENRICHIE)
    # ==============================================================================
    # AJOUT DU PREMIER ONGLET "CONTEXTE"
    tab_context, tab1, tab2, tab3, tab4 = st.tabs([
        "📚 Contexte & Méthodologie", 
        "🏆 Performance & Stratégie", 
        "📈 Dynamique Temporelle", 
        "🧠 Cerveau de l'IA (NLP)", 
        "📄 Données Brutes"
    ])

    # --- TAB 0 : CONTEXTE & MÉTHODOLOGIE (NOUVEAU) ---
    with tab_context:
        st.subheader("📌 Cadre de l'Expérience")
        col_text, col_matrix = st.columns([2, 1])
        
        with col_text:
            st.markdown("""
            **Inspiré des travaux de Robert Axelrod (1981)**, ce projet vise à simuler l'émergence de la coopération dans un environnement mixte composé d'algorithmes déterministes et d'Intelligences Artificielles Génératives.
            
            **Le Dilemme :**
            Deux agents sont arrêtés. Ils ne peuvent pas communiquer.
            * S'ils coopèrent tous les deux : Gain modéré (3 pts).
            * Si l'un trahit et l'autre coopère : Le traître rafle tout (5 pts), la victime perd tout (0 pt).
            * S'ils se trahissent mutuellement : Perte commune (1 pt).
            """)
            
            st.info("""
            **Objectif Technique (ETL) :** Construire un pipeline robuste capable d'orchestrer des modèles **LLM locaux (Mistral/Llama)**, de structurer leurs réponses JSON et d'analyser les stratégies émergentes.
            """)

        with col_matrix:
            st.markdown("#### Matrice des Gains")
            matrix_df = pd.DataFrame(
                {"Coopère (B)": ["(3, 3) R", "(5, 0) T"], "Trahit (B)": ["(0, 5) S", "(1, 1) P"]},
                index=["Coopère (A)", "Trahit (A)"]
            )
            st.table(matrix_df)
            st.caption("*R=Reward, S=Sucker, T=Temptation, P=Punishment*")

        st.markdown("---")
        
        st.subheader("🛠️ Architecture & Innovation")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### 1. EXTRACT (Simulation)")
            st.markdown("""
            * **Hybridation :** Mix Algo (TitForTat, Grim) vs LLM (Mistral/Llama).
            * **Anti-Hallucination :** Parser JSON strict avec logique de *Retry*.
            * **Innovation :** Technique de **"Prompt Masking"**. Le terme "Prisonnier" est caché à l'IA et remplacé par un scénario de *Gestion d'Énergie* pour éviter le biais d'apprentissage.
            """)
        with c2:
            st.markdown("#### 2. TRANSFORM (Enrichissement)")
            st.markdown("""
            * **Feature Engineering :** Calcul vectorisé avec Pandas.
            * **Lag Features :** Création d'une "mémoire" (tours précédents).
            * **Psychologie :** Détection automatique des états (Trahison subie, Pardon, Rancune).
            """)
        with c3:
            st.markdown("#### 3. LOAD (Analyse)")
            st.markdown("""
            * **Stockage :** Format **Parquet** (Colonnaire) pour la performance.
            * **Visualisation :** Streamlit + Plotly pour l'interactivité.
            * **KPIs :** Équilibre de Nash et Taux de Coopération.
            """)

    # --- TAB 1 : PERFORMANCE ---
    with tab1:
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.subheader("Classement par Score Moyen")
            ranking = df_long.groupby('agent')['score'].mean().sort_values(ascending=True).reset_index()
            
            fig_rank = px.bar(
                ranking, x="score", y="agent", orientation='h',
                color="agent", text_auto='.2f',
                color_discrete_map=COLORS,
                title="Qui gagne à la fin ?"
            )
            fig_rank.update_layout(showlegend=False)
            st.plotly_chart(fig_rank, use_container_width=True)
            
        with c2:
            st.subheader("Matrice des Gains (Heatmap)")
            pivot_matrix = df.pivot_table(index='p1_name', columns='p2_name', values='p1_score', aggfunc='mean')
            
            fig_heat = px.imshow(
                pivot_matrix, 
                text_auto=".1f", 
                color_continuous_scale="RdYlGn",
                title="Équilibre de Nash (Vert = Domination)"
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        
        st.markdown("**Interprétation :** Les cases vert foncé indiquent une stratégie dominante. Les cases rouges indiquent qu'un agent se fait exploiter par l'autre.")

    # --- TAB 2 : DYNAMIQUE TEMPORELLE ---
    with tab2:
        st.subheader("Évolution de la Coopération au fil des tours")
        
        # Préparation des données temporelles
        coop_timeline = df.groupby('round_num')[['p1_is_coop', 'p2_is_coop']].mean().mean(axis=1).reset_index()
        coop_timeline.columns = ['round_num', 'coop_rate']
        
        fig_line = px.area(
            coop_timeline, x="round_num", y="coop_rate",
            line_shape="spline",
            color_discrete_sequence=["#2ecc71"],
            title="Stabilité de l'Alliance (1.0 = Paix Totale, 0.0 = Guerre Totale)"
        )
        fig_line.update_yaxes(range=[0, 1.1])
        st.plotly_chart(fig_line, use_container_width=True)

        st.subheader("La Course aux Points (Score Cumulé)")
        match_list = df['match_id'].unique()
        selected_match = st.selectbox("Sélectionner un duel à analyser :", match_list)
        
        match_data = df[df['match_id'] == selected_match].copy()
        
        p1_name = match_data['p1_name'].iloc[0]
        p2_name = match_data['p2_name'].iloc[0]
        
        cum_data = match_data[['round_num', 'p1_cum_score', 'p2_cum_score']].melt(
            id_vars='round_num', var_name='Player', value_name='Cumulative Score'
        )
        cum_data['Player'] = cum_data['Player'].map({'p1_cum_score': p1_name, 'p2_cum_score': p2_name})
        
        fig_race = px.line(
            cum_data, x="round_num", y="Cumulative Score", color="Player",
            markers=True, title=f"Duel : {p1_name} vs {p2_name}"
        )
        st.plotly_chart(fig_race, use_container_width=True)

    # --- TAB 3 : CERVEAU DE L'IA (NLP) ---
    with tab3:
        st.subheader("🧠 Analyse des Raisonnements (Prompt Engineering)")
        st.markdown("Plongée dans les justifications générées par les modèles LLM lors de leurs décisions.")
        
        # Filtre pour ne voir que les IA
        ai_df = df[df['p1_name'].str.contains("AI") | df['p2_name'].str.contains("AI")]
        
        if not ai_df.empty:
            col_filter, col_disp = st.columns([1, 3])
            
            with col_filter:
                agent_nlp = st.selectbox("Choisir l'Agent IA", [a for a in all_agents if "AI" in a])
                action_filter = st.radio("Filtrer par Action", ["Toutes", "Trahison (T)", "Coopération (C)"])
            
            # Filtrage des données
            mask = (ai_df['p1_name'] == agent_nlp)
            if action_filter == "Trahison (T)":
                mask = mask & (ai_df['p1_move'] == 'T')
            elif action_filter == "Coopération (C)":
                mask = mask & (ai_df['p1_move'] == 'C')
                
            nlp_data = ai_df[mask][['round_num', 'p2_name', 'p1_move', 'p1_reasoning', 'p1_score']].head(10)
            
            with col_disp:
                for index, row in nlp_data.iterrows():
                    color = "red" if row['p1_move'] == "T" else "green"
                    icon = "😈" if row['p1_move'] == "T" else "🤝"
                    
                    with st.expander(f"Tour {row['round_num']} vs {row['p2_name']} - Choix : {row['p1_move']} {icon}"):
                        st.markdown(f"**Gain :** {row['p1_score']}")
                        st.markdown(f"**Raisonnement :** _{row['p1_reasoning']}_")
                        if row['p1_move'] == "T":
                            st.caption("⚠️ Cet agent a choisi de trahir.")
        else:
            st.warning("Aucune donnée IA trouvée dans le dataset actuel.")

    # --- TAB 4 : DONNÉES BRUTES ---
    with tab4:
        st.subheader("Explorateur de Données")
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Télécharger en CSV",
            csv,
            "prisoner_results.csv",
            "text/csv",
            key='download-csv'
        )

else:
    st.info("En attente de données... Veuillez lancer le pipeline ETL.")
