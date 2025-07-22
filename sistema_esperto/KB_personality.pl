:- dynamic ha_tratto/3.  % ha_tratto(Persona, Tratto, Valore)

tipo_personalita(introverso).
tipo_personalita(estroverso).

% ----- PESI -----

% tempo_da_solo
peso(introverso, tempo_da_solo, alto, 0.9).
peso(introverso, tempo_da_solo, medio, 0.5).
peso(introverso, tempo_da_solo, basso, 0.1).

peso(estroverso, tempo_da_solo, alto, 0.1).
peso(estroverso, tempo_da_solo, medio, 0.4).
peso(estroverso, tempo_da_solo, basso, 0.9).

% paura_palco
peso(introverso, paura_palco, si, 0.9).
peso(introverso, paura_palco, no, 0.2).

peso(estroverso, paura_palco, si, 0.1).
peso(estroverso, paura_palco, no, 0.8).

% partecipazione_eventi
peso(introverso, partecipazione_eventi, alta, 0.2).
peso(introverso, partecipazione_eventi, media, 0.5).
peso(introverso, partecipazione_eventi, bassa, 0.9).

peso(estroverso, partecipazione_eventi, alta, 0.9).
peso(estroverso, partecipazione_eventi, media, 0.6).
peso(estroverso, partecipazione_eventi, bassa, 0.2).

% frequenza_uscite
peso(introverso, frequenza_uscite, alta, 0.2).
peso(introverso, frequenza_uscite, media, 0.5).
peso(introverso, frequenza_uscite, bassa, 0.9).

peso(estroverso, frequenza_uscite, alta, 0.9).
peso(estroverso, frequenza_uscite, media, 0.6).
peso(estroverso, frequenza_uscite, bassa, 0.1).

% svuotato_dopo_socializzazione
peso(introverso, svuotato_dopo_socializzazione, si, 0.9).
peso(introverso, svuotato_dopo_socializzazione, no, 0.3).

peso(estroverso, svuotato_dopo_socializzazione, si, 0.2).
peso(estroverso, svuotato_dopo_socializzazione, no, 0.8).

% dimensione_amici
peso(introverso, dimensione_amici, grande, 0.2).
peso(introverso, dimensione_amici, medio, 0.5).
peso(introverso, dimensione_amici, piccolo, 0.9).

peso(estroverso, dimensione_amici, grande, 0.9).
peso(estroverso, dimensione_amici, medio, 0.6).
peso(estroverso, dimensione_amici, piccolo, 0.2).

% frequenza_post
peso(introverso, frequenza_post, alta, 0.3).
peso(introverso, frequenza_post, media, 0.6).
peso(introverso, frequenza_post, bassa, 0.9).

peso(estroverso, frequenza_post, alta, 0.9).
peso(estroverso, frequenza_post, media, 0.6).
peso(estroverso, frequenza_post, bassa, 0.2).

% somma pesi per ogni tipo
punteggio(Persona, Tipo, Totale) :-
    findall(Peso, (ha_tratto(Persona, Tratto, Valore), peso(Tipo, Tratto, Valore, Peso)), ListaPesi), media(ListaPesi, Totale).

% media dei pesi
media(Lista, Media) :- sum_list(Lista, Somma), length(Lista, Len), Len > 0, Media is Somma / Len.

% ----- CLASSIFICAZIONE -----
classificazione(Persona) :- 
    punteggio(Persona, introverso, P1),
    punteggio(Persona, estroverso, P2),
    Diff is abs(P1 - P2),
    (Diff < 0.1 -> Tipo = indefinito;
     P1 > P2 -> Tipo = introverso;
     Tipo = estroverso),
    upcase_atom(Tipo, TipoMaiuscolo),
    format("  \nRISULTATO: la personalita' dell'utente e' -> ~w~n", [TipoMaiuscolo]),
    format("  \nSpiegazione:~n"),
    format("    - Punteggio Introverso: ~2f~n", [P1]),
    format("    - Punteggio Estroverso: ~2f~n", [P2]),
    (Tipo == indefinito -> format("    - La differenza tra i punteggi (~2f) e' troppo bassa o uguale a 0 per una classificazione affidabile.~n", [Diff]); 
     format("    - La scelta e' stata fatta perche' il punteggio per ~w e' piu' alto di ~2f rispetto all'altro tipo.~n", [TipoMaiuscolo, Diff])).