from pyswip import Prolog

# funzione principale per input dell'utente, interrogazione della KB e output
def run_personality_expert_system():
    prolog = Prolog()
    try:
        prolog.consult("KB_personality.pl")
    except Exception as e:
        print(f"ERRORE: Impossibile caricare 'KB_personality.pl'. Assicurati che il file sia nella stessa cartella.")
        print(f"Dettaglio errore: {e}")
        return
    
    # funzione per risposta si/no
    def ask_yes_no(question):
        while True:
            risposta = input(f"{question} (si/no): ").lower()
            if risposta in ['si', 'no']:
                return risposta
            print("ERRORE: Inserimento non valido, per favore inserisci 'si' o 'no'.")

    # funzione per risposta multipla
    def ask_multiple_choice(question, options):
        while True:
            print(f"\n{question} Opzioni possibili: {', '.join(options)}")
            risposta = input("La tua scelta: ").lower()
            if risposta in options:
                return risposta
            print("ERRORE: Inserimento non valido, per favore scegli una delle opzioni elencate.")

    # funzione per risposta numerica
    def ask_numeric(question):
        while True:
            risposta_str = input(f"{question} ").strip()
            try:
                numero = int(risposta_str)
                return numero
            except ValueError:
                print("ERRORE: Inserimento non valido, per favore inserisci un numero intero.")

    # inizio dialogo con sistema esperto
    print("-----Sistema Esperto per la Personalità-----")
    print("\nQuesto sistema sarà in grado di determinare la tua personalità (introverso oppure estroverso).\nRispondi alle domande per ottenere una previsione.\n")

    # domande con risposta si/no
    paura_palco = ask_yes_no("Provi paura da palcoscenico?")
    prolog.assertz(f"ha_tratto(utente, paura_palco({paura_palco}))")

    svuotato_dopo_socializzazione = ask_yes_no("\nTi senti svuotato dopo aver socializzato?")
    prolog.assertz(f"ha_tratto(utente, svuotato_dopo_socializzazione({svuotato_dopo_socializzazione}))")

    # domande con risposta multipla
    partecipazione_eventi = ask_multiple_choice(
        "Con quale frequenza partecipi a eventi sociali?",
        ["alta", "media", "bassa"])
    prolog.assertz(f"ha_tratto(utente, partecipazione_eventi({partecipazione_eventi}))")

    frequenza_uscite = ask_multiple_choice(
        "Con quale frequenza esci?",
        ["alta", "media", "bassa"])
    prolog.assertz(f"ha_tratto(utente, frequenza_uscite({frequenza_uscite}))")

    frequenza_post = ask_multiple_choice(
        "Con quale frequenza pubblichi post sui social media?",
        ["alta", "media", "bassa"])
    prolog.assertz(f"ha_tratto(utente, frequenza_post({frequenza_post}))")

    # domande con risposta numerica
    tempo_da_solo = ask_numeric("\nQuante ore al giorno trascorri da solo in media?")
    categoria_tempo = "alto" if tempo_da_solo >= 8 else "medio" if tempo_da_solo >= 4 else "basso"
    prolog.assertz(f"ha_tratto(utente, tempo_da_solo({categoria_tempo}))")
        
    dimensione_amici = ask_numeric("\nQuante persone ci sono nella tua cerchia di amici più stretta?")
    categoria_amici = "grande" if dimensione_amici >= 8 else "medio" if dimensione_amici >= 4 else "piccolo"
    prolog.assertz(f"ha_tratto(utente, dimensione_amici({categoria_amici}))")
    
    # interrogazione della KB per determinare la personalità
    print("\n--- Analisi della personalità in corso... ---")
    try:
        results = list(prolog.query("personalita_con_punteggio(utente, Tipo, Punteggio)"))

        if not results:
            print("Nessuna personalità ha ottenuto un punteggio. Risultato: INDETERMINATO")
            return
                        
        best_result = max(results, key=lambda x: x['Punteggio'])
        max_score = best_result['Punteggio']
        winners = [res['Tipo'] for res in results if res['Punteggio'] == max_score]

        # sistema a punteggio: se il punteggio è uguale, si ha la parità
        if len(winners) == 1:
            final_prediction = str(winners[0]).upper()
        else:
            winner_names = [str(w).upper() for w in winners]
            final_prediction = f"INDETERMINATO (Parità tra: {', '.join(winner_names)})"

        print(f"\n--- Risultato finale: ---")
        print(f"\nLa personalità dell'individuo è: {final_prediction}")
    
    except Exception as e:
        print(f"ERRORE durante l'interrogazione della KB: {e}")

if __name__ == "__main__":
    run_personality_expert_system()
        