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

% ----- COMPORTAMENTI -----
evita_socializzazione(Persona) :- 
    ha_tratto(Persona, partecipazione_eventi, bassa);
    ha_tratto(Persona, svuotato_dopo_socializzazione, si).

ama_stare_solo(Persona) :- 
    ha_tratto(Persona, tempo_da_solo, alto);
    ha_tratto(Persona, frequenza_uscite, bassa);
    ha_tratto(Persona, dimensione_amici, piccolo).

a_suo_agio(Persona) :- 
    ha_tratto(Persona, partecipazione_eventi, alta);
    ha_tratto(Persona, svuotato_dopo_socializzazione, no).

ama_compagnia(Persona) :- 
    ha_tratto(Persona, tempo_da_solo, basso);
    ha_tratto(Persona, frequenza_uscite, alta);
    ha_tratto(Persona, dimensione_amici, grande).

comunicazione_digitale_bassa(Persona) :- 
    ha_tratto(Persona, frequenza_post, bassa);
    ha_tratto(Persona, partecipazione_eventi, bassa).

comunicazione_digitale_alta(Persona) :- 
    ha_tratto(Persona, frequenza_post, alta);
    ha_tratto(Persona, partecipazione_eventi, alta).

riservatezza(Persona) :- 
    ama_stare_solo(Persona);
    comunicazione_digitale_bassa(Persona).

tendenza_sociale_alta(Persona) :- 
    comunicazione_digitale_alta(Persona);
    ama_compagnia(Persona).

% ----- CLASSIFICAZIONE TRATTI -----
comportamenti_introversi([evita_socializzazione, ama_stare_solo, comunicazione_digitale_bassa, riservatezza]).
comportamenti_estroversi([a_suo_agio, ama_compagnia, comunicazione_digitale_alta, tendenza_sociale_alta]).

% ----- CONTEGGIO -----
conteggio_veri(_, [], 0).
conteggio_veri(Persona, [Comportamento|Resto], Conteggio) :-
    (call(Comportamento, Persona) -> 
        conteggio_veri(Persona, Resto, ConteggioResto), 
        Conteggio is ConteggioResto + 1
    ; 
        conteggio_veri(Persona, Resto, Conteggio)
    ).

conteggio_comportamenti(Persona, introverso, N) :-
    comportamenti_introversi(ListaComportamenti),
    conteggio_veri(Persona, ListaComportamenti, N).

conteggio_comportamenti(Persona, estroverso, N) :-
    comportamenti_estroversi(ListaComportamenti),
    conteggio_veri(Persona, ListaComportamenti, N).

% ----- PUNTEGGIO -----
% somma pesi per ogni tipo
punteggio(Persona, Tipo, Totale) :-
    findall(Peso, (ha_tratto(Persona, Tratto, Valore), peso(Tipo, Tratto, Valore, Peso)), ListaPesi), media(ListaPesi, Totale).

% media dei pesi
media(Lista, Media) :- sum_list(Lista, Somma), length(Lista, Len), Len > 0, Media is Somma / Len.

% ----- CLASSIFICAZIONE -----
classificazione(Persona, Tipo, CountIntro, CountEstro, P1, P2) :-     
    % conteggio dei tratti
    conteggio_comportamenti(Persona, introverso, CountIntro),
    conteggio_comportamenti(Persona, estroverso, CountEstro),
    punteggio(Persona, introverso, P1),
    punteggio(Persona, estroverso, P2),

    Diff is abs(CountIntro - CountEstro),
    (Diff > 1 -> 
        (CountIntro > CountEstro -> Tipo = introverso; Tipo = estroverso)
    ;

    % logica a punteggio alternativa (situazione bilanciata)
    (P1 > P2 -> Tipo = 'tendente a introverso'; Tipo = 'tendente a estroverso')
    ).