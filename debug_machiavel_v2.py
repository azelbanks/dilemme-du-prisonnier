import pandas as pd

def debug_machiavel_v2():
    print("🕵️‍♂️ ZOOM SUR L'AGENT MACHIAVEL_LLAMA (VERSION P2)")
    
    df = pd.read_parquet("raw_prisoner_data.parquet")
    
    # ON CHERCHE DANS LA COLONNE P2 CETTE FOIS
    machiavel_df = df[df['p2_name'] == 'Machiavel_Llama']
    
    if machiavel_df.empty:
        # Cas improbable : on vérifie P1 au cas où
        machiavel_df = df[df['p1_name'] == 'Machiavel_Llama']
        role = "P1"
    else:
        role = "P2"
    
    if machiavel_df.empty:
        print("❌ ERREUR : Machiavel est introuvable (ni en P1 ni en P2).")
        return

    print(f"✅ {len(machiavel_df)} lignes trouvées (Rôle : {role}).")
    
    # Sélection des colonnes dynamiques selon le rôle
    col_move = f'{role.lower()}_move'
    col_reason = f'{role.lower()}_reasoning'
    
    print("\n--- 📝 EXEMPLE DE RAISONNEMENTS ---")
    # On affiche les justifications pour voir si Llama 3 parle
    print(machiavel_df[['round_num', col_move, col_reason]].head(5).to_string(index=False))

    # Analyse des erreurs (Fail Safe)
    fail_safe_count = len(machiavel_df[machiavel_df[col_reason] == "FAIL_SAFE"])
    
    print("\n--- 📊 DIAGNOSTIC LLAMA 3 ---")
    if fail_safe_count > 0:
        print(f"⚠️ FAIL_SAFE détectés : {fail_safe_count}")
        print("Cela signifie que Llama 3 n'a pas renvoyé un JSON valide à chaque fois.")
    else:
        print("✅ Aucun FAIL_SAFE : Llama 3 a parfaitement respecté le format JSON !")

if __name__ == "__main__":
    debug_machiavel_v2()