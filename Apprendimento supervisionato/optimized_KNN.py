import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from train_val import X_train_scaled, y_train, X_test_scaled, y_test

print("\n-----Ricerca dei migliori iperparametri per KNN-----")

knn = KNeighborsClassifier()

# imposto gli iperparametri da ottimizzare
param_grid = {
    'n_neighbors': list(range(1, 31))
}

# utilizzo GridSearchCV
grid_search = GridSearchCV(estimator=knn,
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=5,
                           n_jobs=-1,
                           verbose=1,
                           return_train_score=True)

grid_search.fit(X_train_scaled, y_train)

# stampo i risultati
print("\n-----Ricerca completata-----")
best_k = grid_search.best_params_['n_neighbors']
print(f"Il miglior valore di n_neighbors trovato è: {best_k}")
print(f"Il miglior valore per Accuracy è: {grid_search.best_score_:.3f}")

# valutazione finale
print("\n-----Valutazione finale del modello KNN ottimizzato-----")
best_knn_model = grid_search.best_estimator_
y_final_preds = best_knn_model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_final_preds)
print(f"\n Classification Report:\nAccuracy: {acc:.3f}\n", classification_report(y_test, y_final_preds, target_names=['Extrovert', 'Introvert']))

if __name__ == "__main__":
    # rappresentazione grafica dei risultati
    results_df = pd.DataFrame(grid_search.cv_results_)
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=results_df['param_n_neighbors'], 
                y=results_df['mean_test_score'], 
                marker='o', 
                color='royalblue', 
                label='Validation Score')
    sns.lineplot(x=results_df['param_n_neighbors'], 
                y=results_df['mean_train_score'], 
                marker='o', 
                color='darkorange', 
                label='Training Score')
    plt.title('Performance di KNN (Training vs. Validation)')
    plt.xlabel('Numero di Vicini (k)')
    plt.ylabel('Accuracy')
    plt.axvline(x=best_k, color='red', linestyle='--', label=f'Miglior valore n_neighbors = {best_k}')
    plt.legend()
    plt.grid(True)
    plt.show()

    # matrice di confusione
    cm = confusion_matrix(y_test, y_final_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Extrovert', 'Introvert'], 
                yticklabels=['Extrovert', 'Introvert'])
    plt.xlabel('Valore Previsto')
    plt.ylabel('Valore Reale')
    plt.title('Matrice di confusione per il modello KNN ottimizzato')
    plt.show()

    # curva di ROC
    from sklearn.metrics import roc_curve, roc_auc_score
    import matplotlib.pyplot as plt

    # Assumiamo che 'best_knn_model' sia il tuo modello ottimizzato
    # e che X_test_scaled e y_test siano pronti

    # 1. Ottieni le probabilità di previsione per la classe positiva (di solito la classe '1')
    # predict_proba restituisce le probabilità per ogni classe, [prob_classe_0, prob_classe_1]
    # A noi interessa la seconda colonna, quindi usiamo [:, 1]
    y_pred_proba = best_knn_model.predict_proba(X_test_scaled)[:, 1]

    # 2. Calcola i valori per la curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    # 3. Calcola l'area sotto la curva (AUC)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 4. Crea il grafico
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Classificatore Casuale') # Linea di riferimento
    plt.xlabel('Tasso di Falsi Positivi (FPR)')
    plt.ylabel('Tasso di Veri Positivi (TPR)')
    plt.title('Curva ROC per il Modello KNN Ottimizzato')
    plt.legend()
    plt.grid(True)
    plt.show()