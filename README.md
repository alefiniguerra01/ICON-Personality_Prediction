# 👥 Personality Prediction
***Personality Prediction*** è un progetto sviluppato per l'esame di Ingegneria della Conoscenza del corso di Informatica dell'Università degli Studi di Bari.  
Il suo scopo principale è quello di predire la personalità di un individuo, scegliendo tra *introverso* oppure *estroverso*.

---

## 🧠 Argomenti trattati
Nell'implementazione del progetto sono stati trattati i seguenti argomenti:
- **📘 Apprendimento Supervisionato**: il modello impara dal dataset ```personality_dataset.csv``` fornito in input e viene addestrato per effettuare le sue previsioni;
- **📗 Apprendimento Supervisionato con Iperparametri**: il modello viene affinato per migliorare l'accuratezza delle sue previsioni;
- **📕 Sistema Esperto**: tramite una base di conoscenza e un modello inferenziale, viene creato un *Knowledge Base System* che emula il ragionamento umano per arrivare ad una conclusione.

---

## 🔍 Struttura del repository
Il repository contiene:
- ```apprendimento_supervisionato```: file utilizzati per esplorare il dataset e visualizzare i risultati degli algoritmi di apprendimento supervisionato applicati al modello ***Personality Prediction***;
- ```dataset```: dataset utilizzato dal modello;
- ```documentazione```: documentazione del progetto;
- ```img```: grafici inseriti nella documentazione;
- ```sistema_esperto```: Knowledge Base System relativo all'argomento preso in esame;
- ```requirements.txt```: file con l'elenco di tutte le dipendenze necessarie.

---

## ▶️ Esecuzione
Innanzitutto, è necessario aprire il terminale e clonare il repository con il seguente comando:  

    git clone https://github.com/alefiniguerra01/ICON-Personality_Prediction.git

e navigare all'interno della cartella principale:

    cd ICON-Personality_Prediction

Prima di eseguire il progetto è necessario installare le dipendenze richieste (si consiglia di creare prima un ambiente virtuale e di attivarlo: https://aulab.it/guide-avanzate/come-creare-un-virtual-environment-in-python):

    pip install -r requirements.txt

### 📍 Apprendimento Supervisionato
Spostandoci nella cartella ```apprendimento supervisionato``` mediante il comando:

    cd apprendimento_supervisionato

è possibile eseguire nell'ordine i file ```data_exploration.py```, ```preprocessing.py``` e ```train_val.py``` per eseguire rispettivamente le fasi di *Data Exploration*, *Preprocessing*, *Training and Evaluation* che rappresentano le prime tipiche fasi di un progetto di Machine Learning.  
Il comando da digitare è il seguente:

    python nome_del_file.py

sostituendo **nome_del_file.py** con il file che si vuole eseguire (ad esempio: ```python data_exploration.py```).

❗️Se si vogliono visualizzare direttamente le informazioni iniziali del dataset, il risultato della fase di preprocessing e le valutazioni dell'addestramento del modello, si consiglia di eseguire il file ```train_val.py```; se, invece, si vogliono visualizzare dettagliatamente i risultati di ogni fase (compresi i grafici presenti nella documentazione e nella cartella ```img```) si consiglia di eseguire separatamente ogni file nell'ordine descritto sopra.

### 📍 Apprendimento Supervisionato con Iperparametri
Per la fase di tuning, si è deciso di migliorare i modelli *KNN*, *Random Forest* e *Decision Tree*.  
Per visualizzare i risultati dell'ottimizzazione di ogni modello, è possibile eseguire i file ```optimized_KNN.py```, ```optimized_random_forest.py``` e ```optimized_decision_tree.py``` digitando lo stesso comando descritto precedentemente.

### 📍 Sistema Esperto
Per eseguire il Knowledge Base System è necessario installare l'ambiente di sviluppo [SWI-Prolog](https://www.swi-prolog.org/download/devel) (❗️spuntare l'aggiunta alla variabile *path*).  
Successivamente, è necessario navigare all'interno della cartella ```sistema_esperto``` con il seguente comando (se ci si trova nella cartella principale ```ICON-Personality_Prediction```):

    cd sistema_esperto

oppure (se ci si trova nella cartella ```apprendimento_supervisionato```):

    cd ../sistema_esperto

e digitare il comando:

    python expert_system_personality.py

per lanciare l'interfaccia utente realizzata per il sistema esperto.

---

## 👤 Autrice
Realizzato da:
- **Finiguerra Alessia**: matricola: 735326, email: a.finiguerra1@studenti.uniba.it
