import json
import random
import time
import subprocess
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from abc import ABC, abstractmethod
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# =================CONFIGURATION=================
CONFIG = {
    "N_ROUNDS": 300,          # On garde 200 pour la vraie simu
    "TIMEOUT_SEC": 40,        # Augmenté car charger 2 modèles est plus lourd
    "MAX_RETRIES": 3,
    "CONTEXT_WINDOW": 10      
}

PAYOFF_MATRIX = {
    ("C", "C"): (3, 3),
    ("C", "T"): (0, 5),
    ("T", "C"): (5, 0),
    ("T", "T"): (1, 1)
}

# =================CLASSES=================

@dataclass
class RoundResult:
    tournament_id: str
    match_id: str
    round_num: int
    p1_name: str
    p2_name: str
    p1_move: str
    p2_move: str
    p1_score: int
    p2_score: int
    p1_reasoning: str
    p2_reasoning: str
    timestamp: float

class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def decide(self, history_self, history_opp) -> Tuple[str, str]:
        pass

class AlgoAgent(BaseAgent):
    def __init__(self, name: str, strategy_type: str):
        super().__init__(name)
        self.strategy = strategy_type

    def decide(self, history_self, history_opp) -> Tuple[str, str]:
        if self.strategy == "tit_for_tat":
            move = history_opp[-1] if history_opp else "C"
            return move, "J'imite ton dernier coup."
        elif self.strategy == "grim_trigger":
            if "T" in history_opp:
                return "T", "Tu m'as trahi, c'est fini."
            return "C", "Je coopère tant que tu es loyal."
        return random.choice(["C", "T"]), "Random fallback."

class LLMAgent(BaseAgent):
    # MODIFICATION SENIOR : On injecte le "model_name"
    def __init__(self, name: str, persona: str, model_name: str):
        super().__init__(name)
        self.persona = persona
        self.model_name = model_name

    def _clean_json(self, raw_output: str) -> Dict:
        try:
            start = raw_output.find('{')
            end = raw_output.rfind('}') + 1
            if start == -1 or end == 0: raise ValueError("No JSON found")
            return json.loads(raw_output[start:end])
        except Exception:
            raise ValueError(f"Invalid JSON content")

    def decide(self, history_self, history_opp) -> Tuple[str, str]:
        recent_history = []
        context_window = history_self[-CONFIG["CONTEXT_WINDOW"]:]
        opp_window = history_opp[-CONFIG["CONTEXT_WINDOW"]:]
        
        for my_move, opp_move in zip(context_window, opp_window):
            my_act = "J'ai PARTAGÉ" if my_move == "C" else "J'ai GARDÉ"
            opp_act = "Il a PARTAGÉ" if opp_move == "C" else "Il a GARDÉ"
            recent_history.append(f"({my_act}, {opp_act})")

        prompt = f"""
        CONTEXTE : Tu es un agent autonome chargé de la gestion énergétique.
        Ta personnalité : {self.persona}
        
        SITUATION :
        Tu rencontres un autre agent. Vous avez chacun un pack d'énergie.
        Vous devez choisir simultanément une action :
        - OPTION A : Mettre ton énergie dans le pot commun (Partager).
        - OPTION B : Garder ton énergie pour toi (Garder).
        
        RÈGLES DE GAIN (Points d'énergie reçus) :
        - Si vous PARTAGEZ tous les deux : Vous recevez chacun un bonus modéré (+3).
        - Si tu GARDES et l'autre PARTAGE : Tu rafles tout (+5), lui n'a rien (0).
        - Si tu PARTAGES et l'autre GARDE : Tu perds ton énergie (0), lui rafle tout (+5).
        - Si vous GARDEZ tous les deux : Gain minime de survie (+1 chacun).

        HISTORIQUE RÉCENT : {recent_history}

        TA MISSION : Maximiser ton score cumulé selon ta personnalité.
        
        RÉPONSE ATTENDUE (JSON UNIQUEMENT) :
        {{
            "reasoning": "Ton analyse en 1 phrase",
            "choice": "A" ou "B"
        }}
        """

        for attempt in range(CONFIG["MAX_RETRIES"]):
            try:
                # Appel dynamique du modèle spécifique à l'agent
                result = subprocess.run(
                    ["ollama", "run", self.model_name, prompt],
                    capture_output=True, text=True, encoding='utf-8', 
                    timeout=CONFIG["TIMEOUT_SEC"]
                )
                data = self._clean_json(result.stdout)
                
                choice_letter = data.get("choice", "").upper().strip()
                if "A" in choice_letter: return "C", data.get("reasoning", "Ras")
                elif "B" in choice_letter: return "T", data.get("reasoning", "Ras")
                else: raise ValueError(f"Invalid option {choice_letter}")

            except Exception:
                pass 

        return "C", "FAIL_SAFE"

# =================MOTEUR DE TOURNOI=================

class Tournament:
    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents
        self.results = []

    def play_match(self, agent1: BaseAgent, agent2: BaseAgent) -> List[RoundResult]:
        match_id = f"{agent1.name}_vs_{agent2.name}_{int(time.time())}"
        hist1, hist2 = [], []
        match_data = []

        print(f"🏁 Démarrage Match : {agent1.name} vs {agent2.name}")

        for r in range(1, CONFIG["N_ROUNDS"] + 1):
            m1, r1 = agent1.decide(hist1, hist2)
            m2, r2 = agent2.decide(hist2, hist1)

            s1, s2 = PAYOFF_MATRIX[(m1, m2)]
            hist1.append(m1)
            hist2.append(m2)
            
            match_data.append(RoundResult(
                tournament_id="DUAL_MODEL_CUP", match_id=match_id, round_num=r,
                p1_name=agent1.name, p2_name=agent2.name,
                p1_move=m1, p2_move=m2, p1_score=s1, p2_score=s2,
                p1_reasoning=r1, p2_reasoning=r2, timestamp=datetime.now().timestamp()
            ))
            # Petit log pour voir que ça avance
            if r % 50 == 0: print(f"   > {agent1.name} vs {agent2.name} : Tour {r}")

        return match_data

    def run_parallel(self):
        pairs = []
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                pairs.append((self.agents[i], self.agents[j]))

        print(f"🚀 Lancement de {len(pairs)} matchs (Mode Sécurisé 1 Worker)...")
        
        # SÉCURITÉ : 1 SEUL WORKER POUR ÉVITER LE CRASH MÉMOIRE AVEC 2 MODÈLES
        with ThreadPoolExecutor(max_workers=1) as executor: 
            futures = [executor.submit(self.play_match, p[0], p[1]) for p in pairs]
            for future in as_completed(futures):
                self.results.extend(future.result())

    def save_raw_data(self, filename="raw_prisoner_data.parquet"):
        df = pd.DataFrame([vars(r) for r in self.results])
        table = pa.Table.from_pandas(df)
        pq.write_table(table, filename)
        print(f"💾 Données brutes sauvegardées: {filename} ({len(df)} lignes)")

# =================EXECUTION=================
if __name__ == "__main__":
    # N'oublie pas de faire 'ollama pull llama3.1:8b' avant !
    roster = [
        AlgoAgent("TitForTat_Bot", "tit_for_tat"),
        
        # MISTRAL (Le modèle français/européen)
        LLMAgent("Ghandi_Mistral", "Tu es un pacifiste absolu, tu veux la paix.", "mistral:latest"),
        
        # LLAMA 3 (Le modèle Meta/Américain)
        LLMAgent("Machiavel_Llama", "Tu es un manipulateur opportuniste.", "llama3.1:8b")
    ]
    
    arena = Tournament(roster)
    arena.run_parallel()
    arena.save_raw_data()