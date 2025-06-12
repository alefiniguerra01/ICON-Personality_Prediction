import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from train_val import X_train_scaled, y_train

# --- INIZIA LA FASE DI TUNING SPECIFICA PER KNN ---

print("----- Inizio del Tuning degli Iperparametri per KNN -----")

# 1. Definisci il modello base
knn = KNeighborsClassifier()

# 2. Definisci la griglia di iperparametri da testare
# Questi sono i "run con valori diversi" che hai richiesto
param_grid = {
    'n_neighbors': list(range(1, 41)),  # Prova un range ampio di "k" vicini
    'weights': ['uniform', 'distance'],   # Prova i due modi di pesare i vicini
    'metric': ['euclidean', 'manhattan']  # Prova le due principali metriche di distanza
}

# 3. Imposta ed esegui GridSearchCV
# cv=5: usa 5-fold cross-validation per una valutazione robusta
# scoring='f1_weighted': ottimizza per l'F1-score, una metrica bilanciata
# n_jobs=-1: usa tutti i core della CPU per velocizzare
grid_search = GridSearchCV(estimator=knn,
                           param_grid=param_grid,
                           scoring='f1_weighted',
                           cv=5,
                           n_jobs=-1,
                           verbose=1) # verbose=1 ti mostra l'avanzamento

grid_search.fit(X_train_scaled, y_train)

# 4. Mostra i risultati migliori trovati
print("\nMigliori iperparametri trovati per KNN:")
print(grid_search.best_params_)
print("\nMiglior F1-score ottenuto durante la cross-validation:")
print(f"{grid_search.best_score_:.4f}")


# 5. Visualizza i risultati (la parte più importante per capire)
# Convertiamo i risultati della grid search in un DataFrame per una facile manipolazione
results_df = pd.DataFrame(grid_search.cv_results_)

# Creiamo un grafico per visualizzare come cambia la performance al variare di 'n_neighbors'
# per ogni combinazione di 'weights' e 'metric'
plt.figure(figsize=(15, 8))
sns.lineplot(data=results_df,
             x='param_n_neighbors',
             y='mean_test_score',
             hue='param_weights',
             style='param_metric',
             marker='o')

plt.title('Performance di KNN al Variare degli Iperparametri')
plt.xlabel('Numero di Vicini (k)')
plt.ylabel('F1-Score Ponderato (Cross-Validation)')
plt.legend(title='Configurazione')
plt.grid(True)
plt.show()
