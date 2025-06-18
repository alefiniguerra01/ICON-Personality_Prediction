/*features:
Time_spent_alone: tempo_da_solo (numerico da convertire in alto/media/basso)
Stage_fear: paura_palco (si/no)
Social_event_attendance: partecipazione_eventi (alta/media/bassa)
Going_outside: frequenza_uscite (alta/media/bassa)
Drained_after_socializing: svuotato_dopo_socializzazione (si/no)
Friends_circle_size: dimensione_amici (numerico da convertire in grande/medio/piccolo)
Post_frequency: frequenza_post (alta/media/bassa)
*/

:- dynamic ha_tratto/2. % indica che il predicato ha_tratto può essere modificato dinamicamente

tipo_personalita(introverso).
tipo_personalita(estroverso).

% regole per gli introversi
regola_punteggio(Persona, introverso, 3) :- 
    ha_tratto(Persona, tempo_da_solo(alto)).

regola_punteggio(Persona, introverso, 3) :-
    ha_tratto(Persona, paura_palco(si)).

regola_punteggio(Persona, introverso, 3) :-
    ha_tratto(Persona, frequenza_uscite(bassa)),
    ha_tratto(Persona, svuotato_dopo_socializzazione(si)).

regola_punteggio(Persona, introverso, 2) :-
    ha_tratto(Persona, dimensione_amici(piccolo)),
    ha_tratto(Persona, tempo_da_solo(alto)).

regola_punteggio(Persona, introverso, 2) :-
    ha_tratto(Persona, partecipazione_eventi(bassa)).

regola_punteggio(Persona, introverso, 1) :-
    ha_tratto(Persona, frequenza_post(bassa)).

% regole per gli estroversi
regola_punteggio(Persona, estroverso, 3) :-
    ha_tratto(Persona, dimensione_amici(grande)).

regola_punteggio(Persona, estroverso, 3) :-
    ha_tratto(Persona, frequenza_uscite(alta)),
    ha_tratto(Persona, partecipazione_eventi(alta)).

regola_punteggio(Persona, estroverso, 2) :-
    ha_tratto(Persona, svuotato_dopo_socializzazione(no)),
    ha_tratto(Persona, paura_palco(no)).

regola_punteggio(Persona, estroverso, 2) :-
    ha_tratto(Persona, tempo_da_solo(basso)).

regola_punteggio(Persona, estroverso, 1) :-
    ha_tratto(Persona, frequenza_post(alta)).

% regola che cacola il punteggio totale di ogni personalità
punteggio_totale(Persona, Tipo, PunteggioTotale) :-
    findall(Punteggio, regola_punteggio(Persona, Tipo, Punteggio), ListaPunteggi),
    sum_list(ListaPunteggi, PunteggioTotale).

% regola che determina la personalità
personalita_con_punteggio(Persona, Tipo, Punteggio) :-
    tipo_personalita(Tipo),
    punteggio_totale(Persona, Tipo, Punteggio),
    Punteggio > 0.