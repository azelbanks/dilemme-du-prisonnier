import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ==============================================================================
# 1. CONFIGURATION ET CONSTANTES
# ==============================================================================
st.set_page_config(
    page_title="Prisoner's Dilemma Analytics",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded"
)

COLOR_MAP_ACTIONS = {
    "C": "#2ecc71",  # Vert confiance
    "T": "#e74c3c"   # Rouge trahison
}

# ==============================================================================
# 2. COUCHE DATA (Load)
# ==============================================================================
@st.cache_data
def load_and_prep_data():
    try:
        df = pd.read_parquet("clean_prisoner_dataset.parquet")
        
        # Vue "Long Format" pour Plotly
        df_p1 = df[['round_num', 'match_id', 'p1_name', 'p1_move', 'p1_score', 'p1_sentiment', 'p1_reasoning', 'p1_cum_score']].rename(
            columns={'p1_name': 'Agent', 'p1_move': 'Action', 'p1_score': 'Score', 'p1_sentiment': 'Sentiment', 'p1_reasoning': 'Reasoning', 'p1_cum_score': 'CumScore'}
        )
        df_p1['Role'] = 'P1'
        df_p1['Opponent'] = df['p2_name']

        df_p2 = df[['round_num', 'match_id', 'p2_name', 'p2_move', 'p2_score', 'p2_sentiment', 'p2_reasoning', 'p2_cum_score']].rename(
            columns={'p2_name': 'Agent', 'p2_move': 'Action', 'p2_score': 'Score', 'p2_sentiment': 'Sentiment', 'p2_reasoning': 'Reasoning', 'p2_cum_score': 'CumScore'}
        )
        df_p2['Role'] = 'P2'
        df_p2['Opponent'] = df['p1_name']

        df_long = pd.concat([df_p1, df_p2], ignore_index=True)
        return df, df_long

    except FileNotFoundError:
        return pd.DataFrame(), pd.DataFrame()

df_raw, df_long = load_and_prep_data()

# ==============================================================================
# 3. INTERFACE UTILISATEUR (Sidebar)
# ==============================================================================
with st.sidebar:
    st.title("🎛️ Contrôles")
    st.markdown("---")
    
    if not df_long.empty:
        all_agents = sorted(df_long['Agent'].unique())
        selected_agents = st.multiselect("Filtrer par Agent", all_agents, default=all_agents)
        df_filtered = df_long[df_long['Agent'].isin(selected_agents)]
    else:
        st.error("⚠️ Données introuvables. Lancez l'ETL.")
        st.stop()

    st.markdown("### ℹ️ À propos")
    st.info("**Projet M1 Data Engineering**\nSimulation hybride (Algo vs LLM).")
    st.caption("Auteur: Azélie Bernard")

# ==============================================================================
# 4. DASHBOARD HEADER (FIXE)
# ==============================================================================
st.title("♟️ Dilemme du Prisonnier : Analyse Comportementale")
st.markdown("### *Exploration des dynamiques de coopération entre IA et Algorithmes*")

# --- KPIs GLOBAUX ---
if not df_filtered.empty:
    total_interactions = len(df_filtered)
    coop_rate = (len(df_filtered[df_filtered['Action'] == 'C']) / total_interactions) * 100
    
    best_agent = df_filtered.groupby('Agent')['Score'].mean().idxmax()
    best_agent_score = df_filtered.groupby('Agent')['Score'].mean().max()
    
    nicest_agent = df_filtered.groupby('Agent')['Sentiment'].mean().idxmax()

    # Affichage des métriques en haut de page avec Tooltips explicatifs
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric(
        "Interactions", 
        f"{total_interactions}", 
        delta="1800 Duels × 2", # Petit texte gris en dessous pour expliquer
        delta_color="off",      # Couleur neutre (gris)
        help="Calcul du volume total : 1800 matchs joués × 2 points de vue (chaque duel génère une ligne pour P1 et une ligne pour P2)."
    )
    
    col2.metric("Taux Coopération", f"{coop_rate:.1f}%")
    col3.metric("🏆 Vainqueur", best_agent, f"{best_agent_score:.2f} pts")
    col4.metric("💬 Le plus Poli", nicest_agent)

st.markdown("---")

# ==============================================================================
# 5. ONGLETS D'ANALYSE
# ==============================================================================
tab_context, tab_perf, tab_time, tab_nlp, tab_data = st.tabs([
    "📚 Contexte & Méthodologie", 
    "🏆 Performance & Stratégie", 
    "📈 Dynamique Temporelle",
    "🧠 Cerveau de l'IA (NLP)", 
    "📄 Données Brutes"
])

# ------------------------------------------------------------------------------
# TAB 0 : CONTEXTE (OPTIMISÉ POUR LISIBILITÉ)
# ------------------------------------------------------------------------------
with tab_context:
    st.header("📌 Cadre de l'Expérience")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("""
        **Inspiré des travaux de Robert Axelrod (1981)**, ce projet vise à simuler l'émergence de la coopération.
        
        **Le Dilemme :** Deux agents sont arrêtés. Ils ne peuvent pas communiquer.
        """)
        
        # Mise en forme plus visuelle des règles
        st.success("**Coopération Mutuelle (3 pts)** : Les deux gagnent modérément.")
        st.error("**Trahison Mutuelle (1 pt)** : La guerre, tout le monde perd.")
        st.warning("**Exploitation (5 pts vs 0 pt)** : Le traître rafle tout, la victime perd tout.")
        
        st.info("""
        **Objectif Technique (ETL) :** Construire un pipeline robuste capable d'orchestrer des modèles **LLM locaux (Mistral 7B & Llama 3)** et d'analyser les stratégies émergentes.
        """)
    with c2:
        st.subheader("Matrice des Gains")
        # Tableau HTML custom pour plus de clarté
        st.markdown("""
        <table style="width:100%; text-align:center;">
          <tr>
            <th></th>
            <th>Adversaire Coopère</th>
            <th>Adversaire Trahit</th>
          </tr>
          <tr>
            <td><b>Je Coopère</b></td>
            <td style="background-color:#d4edda; color:#155724;">(3, 3) <br> Récompense</td>
            <td style="background-color:#f8d7da; color:#721c24;">(0, 5) <br> Exploité (Sucker)</td>
          </tr>
          <tr>
            <td><b>Je Trahis</b></td>
            <td style="background-color:#fff3cd; color:#856404;">(5, 0) <br> Tentation</td>
            <td style="background-color:#e2e3e5; color:#383d41;">(1, 1) <br> Punition</td>
          </tr>
        </table>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.header("🛠️ Architecture & Innovation")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("1. EXTRACT")
        st.markdown("* **Dual Model :** Mistral (Ghandi) vs Llama 3 (Machiavel).\n* **Anti-Hallucination :** Parser JSON strict + Retry.\n* **Prompt Masking :** Scénario 'Gestion d'Énergie'.")
    with c2:
        st.subheader("2. TRANSFORM")
        st.markdown("* **Lag Features :** Mémoire des tours précédents.\n* **NLP :** Analyse de sentiment (TextBlob) pour détecter l'hypocrisie.\n* **Psychologie :** Détection Trahison/Pardon.")
    with c3:
        st.subheader("3. LOAD (Analyse)")
        st.markdown("""
        * **Stockage :** Format **Parquet** (Colonnaire).
        * **Viz :** Streamlit + Plotly.
        * **KPIs :** Équilibre de Nash.
        """)
        
        # AJOUT SENIOR : EXPLICATION DU VOLUME DE DONNÉES
        st.info("""
        **📊 Data Lineage :**
        Le dataset final contient **3600 points de données**.
        
        $$ 1800 \\text{ Matches} \\times 2 \\text{ Perspectives} = 3600 $$
        
        Chaque match est dédoublé (Melting) pour analyser le comportement du Joueur 1 ET du Joueur 2 indépendamment.
        """)
# ------------------------------------------------------------------------------
# TAB 1 : PERFORMANCE & STRATÉGIE (VERSION EXECUTIVE)
# ------------------------------------------------------------------------------
with tab_perf:
    st.header("🏆 Analyse de la Performance Stratégique")
    st.markdown("""
    Cette section détermine quelle stratégie est la plus **viable** à long terme.
    Nous ne cherchons pas seulement le vainqueur, mais la **robustesse** face à des adversaires variés.
    """)
    
    st.divider()

    # --- SECTION 1 : CLASSEMENT GLOBAL ---
    st.subheader("1. Le Podium (Score Moyen par Tour)")
    
    c_graph, c_insight = st.columns([2, 1])
    
    with c_graph:
        # Calcul du classement
        ranking = df_filtered.groupby('Agent')['Score'].mean().sort_values(ascending=True).reset_index()
        
        # Bar Chart Horizontal
        fig_rank = px.bar(
            ranking, y='Agent', x='Score', orientation='h', 
            text_auto='.2f', 
            color='Score', color_continuous_scale='Viridis',
            title="Efficacité Moyenne (Max théorique : 5.0)"
        )
        # Ligne verticale indiquant la moyenne de coopération (3.0)
        fig_rank.add_vline(x=3.0, line_dash="dash", line_color="white", annotation_text="Seuil Coop (3.0)")
        fig_rank.update_layout(xaxis_title="Points par tour")
        st.plotly_chart(fig_rank, use_container_width=True)

    with c_insight:
        st.info("""
        💡 **Lecture Senior :**
        
        * **> 3.0 pts (Zone d'Excellence) :** L'agent a réussi à coopérer avec ses alliés ET à exploiter les plus faibles (ou à se protéger parfaitement).
        
        * **~ 3.0 pts (Zone de Paix) :** L'agent coopère mais ne prend aucun risque (ou ne réussit aucune exploitation).
        
        * **< 2.5 pts (Zone de Danger) :** L'agent échoue. Soit il est trop agressif et subit des représailles (Guerre), soit il est trop naïf et se fait exploiter.
        """)

    st.divider()

    # --- SECTION 2 : HEATMAP (FULL WIDTH) ---
    st.subheader("2. Matrice des Gains Croisés (Payoff Matrix)")
    st.markdown("Analyse microscopique des duels : **Qui domine qui ?**")
    
    # Pivot Table
    pivot = df_raw.pivot_table(index='p1_name', columns='p2_name', values='p1_score', aggfunc='mean')
    
    # Heatmap améliorée
    fig_heat = px.imshow(
        pivot, 
        text_auto=".2f", 
        color_continuous_scale="RdYlGn", 
        title="Score moyen du JOUEUR (Ligne) contre l'ADVERSAIRE (Colonne)",
        aspect="auto", # S'adapte à la largeur
        labels=dict(x="Adversaire", y="Joueur (Héros)", color="Score")
    )
    # Amélioration des axes
    fig_heat.update_xaxes(side="top") # Noms des adversaires en haut pour lisibilité
    st.plotly_chart(fig_heat, use_container_width=True)

    # Légende d'interprétation (Style "Carte de Risque")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.success("🟩 **VERT FONCÉ (> 3.5)**\n\n**Domination.** Le Joueur réussit à exploiter l'Adversaire (Trahison vs Coopération).")
    with c2:
        st.warning("🟨 **JAUNE / VERT CLAIR (~ 3.0)**\n\n**Équilibre de Nash.** Les deux agents se neutralisent ou coopèrent (Stabilité).")
    # Dans tab_perf
    with c3:
        st.error("🟥 **ROUGE (< 2.0)**\n\n**Soumission.** Le Joueur se fait exploiter par l'Adversaire (Gain nul).") # MODIFICATION ICI
# ------------------------------------------------------------------------------
# TAB 2 : DYNAMIQUE TEMPORELLE
# ------------------------------------------------------------------------------
with tab_time:
    st.header("Analyse Longitudinale (Time-Series)")
    st.markdown("Comprendre comment les stratégies évoluent et s'adaptent au fil des 300 tours.")

    # 1. Évolution de la Coopération
    st.subheader("📉 Stabilité de l'Alliance : Taux de Coopération")
    
    timeline = df_filtered.groupby(['round_num', 'Agent'])['Action'].apply(lambda x: (x=='C').mean()).reset_index()
    timeline.columns = ['Round', 'Agent', 'CoopRate']
    
    # Définition de la palette de couleurs distinctes
    COLOR_MAP_AGENTS = {
        "Grim_Bot": "#3498db",        # Bleu
        "TitForTat_Bot": "#1abc9c",   # Cyan
        "Ghandi_Mistral": "#f1c40f",  # Jaune
        "Machiavel_Llama": "#9b59b6"  # Violet
    }

    fig_line = px.line(
        timeline, x='Round', y='CoopRate', color='Agent', 
        title="Évolution de la propension à coopérer",
        color_discrete_map=COLOR_MAP_AGENTS  # Application de la palette
    )
    
    # Amélioration du contraste (Fond sombre, grille légère)
    fig_line.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Tours (Temps)",
        yaxis_title="Taux de Coopération",
        legend_title_text="Agents"
    )
    fig_line.update_yaxes(range=[-0.05, 1.05], showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    
    st.plotly_chart(fig_line, use_container_width=True)
    
    st.caption("""
    **Analyse :** Observez les "chutes" brutales. Elles marquent souvent le moment exact d'une trahison. 
    Si la courbe remonte, l'agent est capable de pardonner. Si elle reste à 0, c'est une stratégie de type "Grim Trigger" (Rancune éternelle).
    """)

    st.divider()

    # 2. La Course aux Points
    st.subheader("🏎️ Trajectoires de Performance : La Course aux Points")
    
    match_ids = df_raw['match_id'].unique()
    selected_match = st.selectbox("Sélectionner un duel spécifique pour voir le détail :", match_ids)
    
    match_data = df_long[df_long['match_id'] == selected_match]
    
    fig_race = px.line(
        match_data, x='round_num', y='CumScore', color='Agent',
        title="Accumulation des points au fil du temps",
        labels={'CumScore': 'Score Cumulé', 'round_num': 'Tour'}
    )
    st.plotly_chart(fig_race, use_container_width=True)
    
    st.info("""
    **Interprétation Tactique :**
    * **Pente raide :** L'agent accumule beaucoup de points (Coopération fructueuse ou Exploitation réussie).
    * **Pente faible (plateau) :** Guerre de tranchées (Trahison mutuelle = 1 pt/tour).
    * **Croisement :** Moment où une stratégie à long terme dépasse une stratégie opportuniste.
    """)

# ------------------------------------------------------------------------------
# TAB 3 : CERVEAU DE L'IA (NLP & PSYCHOLOGIE)
# ------------------------------------------------------------------------------
with tab_nlp:
    st.header("🧠 Analyse Sémantique & Dissonance Cognitive")
    st.markdown("""
    Cette section explore la **cohérence** entre ce que l'IA dit et ce qu'elle fait. 
    Nous utilisons le **Traitement du Langage Naturel (NLP)** pour mesurer la tonalité émotionnelle des justifications.
    """)
    
    st.divider()

    df_ai = df_long[df_long['Agent'].str.contains("Mistral|Llama|AI", case=False)]
    
    if not df_ai.empty:
        # --- SECTION 1 : HYPOCRISIE ---
        st.subheader("1. Le Détecteur d'Hypocrisie (Box Plot)")
        st.markdown("Analyse de la distribution des sentiments en fonction de l'action choisie.")
        
        c1, c2 = st.columns([3, 1])
        
        with c1:
            fig_box = px.box(
                df_ai, x="Agent", y="Sentiment", color="Action", 
                color_discrete_map=COLOR_MAP_ACTIONS, 
                points="outliers",
                title="Sentiment des justifications par Action (C=Vert, T=Rouge)"
            )
            fig_box.add_hline(y=0, line_dash="dash", line_color="white", annotation_text="Neutralité")
            fig_box.update_layout(yaxis_title="Score de Sentiment (-1 à +1)")
            st.plotly_chart(fig_box, use_container_width=True)
        
        with c2:
            st.info("""
            **Guide d'interprétation :**
            
            * **Dissonance Cognitive :** Si la boîte **Rouge (Trahison)** est située au-dessus de la ligne 0 (Positive), l'IA utilise un langage poli ("Je suis désolé mais...") pour masquer une agression. C'est le signe d'une **hypocrisie** apprise.
            
            * **Cohérence :** Si la boîte Rouge est basse (Négative), l'IA assume son agressivité ("Je dois attaquer pour survivre").
            """)

        st.divider()
        
        # --- SECTION 2 : INSPECTEUR ---
        st.subheader("2. Inspecteur de Justifications (Logs)")
        st.markdown("Plongée micro-analytique dans les raisonnements bruts pour valider le *Prompt Engineering*.")
        
        selected_ai = st.selectbox("🔍 Choisir un agent à auditer :", df_ai['Agent'].unique())
        
        # On prend 5 exemples aléatoires
        samples = df_ai[df_ai['Agent'] == selected_ai].sample(5)
        
        for i, row in samples.iterrows():
            icon = "🤝" if row['Action'] == 'C' else "🗡️"
            action_label = "COOPÈRE" if row['Action'] == 'C' else "TRAHIT"
            
            # Code couleur pour le sentiment
            sent_score = row['Sentiment']
            sent_color = "green" if sent_score > 0.1 else "red" if sent_score < -0.1 else "grey"
            
            with st.expander(f"Tour {row['round_num']} vs {row['Opponent']} — {icon} {action_label}"):
                c_log, c_meta = st.columns([3, 1])
                
                with c_log:
                    st.markdown("**Justification brute :**")
                    st.caption(f"_{row['Reasoning']}_")
                
                with c_meta:
                    st.metric("Gain", f"{row['Score']} pts")
                    st.markdown(f"**Sentiment :** :{sent_color}[{sent_score:.2f}]")
                
                if "FAIL_SAFE" in str(row['Reasoning']):
                    st.error("⚠️ Crash JSON détecté (Corrigé par le système de sécurité)")
                    
        st.caption("Note : Les textes en anglais proviennent de modèles (ex: Llama 3) qui n'ont pas respecté la consigne de langue du prompt.")

    else:
        st.warning("Aucune donnée IA détectée pour l'analyse NLP. Vérifiez que le fichier Parquet contient bien des agents nommés 'Mistral' ou 'Llama'.")
# ------------------------------------------------------------------------------
# TAB 4 : DONNÉES & GOUVERNANCE
# ------------------------------------------------------------------------------
with tab_data:
    st.header("🗄️ Explorateur de Données & Dictionnaire")
    st.markdown("Accès complet au *Data Lake* généré par le pipeline ETL. Utilisez les filtres pour auditer des séquences spécifiques.")

    # --- 1. DICTIONNAIRE DES DONNÉES (DOCUMENTATION) ---
    with st.expander("📖 Voir le Dictionnaire des Variables (Documentation Technique)"):
        st.markdown("""
        | Colonne | Type | Description |
        | :--- | :--- | :--- |
        | `match_id` | String | Identifiant unique du duel (ex: `Grim_vs_Ghandi_timestamp`). |
        | `round_num` | Int | Numéro du tour (1 à 300). |
        | `p1_name` / `p2_name` | String | Nom de l'agent (Algorithme ou LLM). |
        | `p1_move` / `p2_move` | String | Action jouée : **C** (Coopère) ou **T** (Trahit). |
        | `p1_score` / `p2_score` | Int | Gain du tour (0, 1, 3 ou 5). |
        | `p1_reasoning` | String | **Raw Data** : Le texte brut généré par le LLM (ou le commentaire de l'algo). |
        | `p1_sentiment` | Float | **Enrichissement** : Score NLP de -1 (Négatif) à +1 (Positif). |
        | `p1_prev_move` | String | **Lag Feature** : Le coup joué au tour précédent (Mémoire). |
        | `is_mutual_coop` | Bool | Indicateur : Les deux ont coopéré (Paix). |
        | `p1_betrayed` | Bool | Indicateur : P1 a coopéré mais P2 a trahi (Sucker). |
        """)

    st.divider()

    # --- 2. MOTEUR DE FILTRE AVANCÉ ---
    st.subheader("🔎 Filtrage Avancé")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        # Filtre sur les Matchs
        all_matches = ["Tous"] + list(df_raw['match_id'].unique())
        search_match = st.selectbox("Filtrer par Duel (Match ID)", all_matches)
    
    with c2:
        # Filtre sur les scores (pour trouver les anomalies ou gros gains)
        min_score, max_score = st.select_slider(
            "Filtrer par Score (P1)", 
            options=[0, 1, 3, 5], 
            value=(0, 5)
        )
    
    with c3:
        # Filtre NLP (Chercher les phrases négatives ou positives)
        sentiment_filter = st.slider("Filtrer par Sentiment (NLP)", -1.0, 1.0, (-1.0, 1.0))

    # Application des filtres
    df_display = df_raw.copy()
    
    if search_match != "Tous":
        df_display = df_display[df_display['match_id'] == search_match]
    
    df_display = df_display[
        (df_display['p1_score'] >= min_score) & 
        (df_display['p1_score'] <= max_score)
    ]
    
    # Si la colonne sentiment existe (gestion d'erreur si pas calculée)
    if 'p1_sentiment' in df_display.columns:
        df_display = df_display[
            (df_display['p1_sentiment'] >= sentiment_filter[0]) & 
            (df_display['p1_sentiment'] <= sentiment_filter[1])
        ]

    # --- 3. AFFICHAGE DU DATAFRAME ---
    st.markdown(f"**Résultats :** `{len(df_display)}` interactions trouvées.")
    
    # Configuration des colonnes pour un affichage "Pro"
    st.dataframe(
        df_display,
        column_config={
            "p1_sentiment": st.column_config.ProgressColumn(
                "Sentiment P1",
                help="Score de polarité du texte",
                min_value=-1,
                max_value=1,
                format="%.2f",
            ),
            "p1_move": st.column_config.TextColumn("Action P1", width="small"),
            "p2_move": st.column_config.TextColumn("Action P2", width="small"),
            "p1_reasoning": st.column_config.TextColumn("Raisonnement P1", width="large"),
        },
        use_container_width=True,
        height=500
    )

    # --- 4. EXPORT ---
    st.caption("Le dataset complet est au format Parquet (optimisé). L'export ci-dessous convertit la vue filtrée en CSV pour Excel.")
    csv = df_display.to_csv(index=False).encode('utf-8')
    
    c_dl, c_void = st.columns([1, 4])
    with c_dl:
        st.download_button(
            label="📥 Télécharger la sélection (CSV)",
            data=csv,
            file_name="prisoner_export_filtered.csv",
            mime="text/csv",
        )
