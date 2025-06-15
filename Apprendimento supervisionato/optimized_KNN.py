import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
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

# utilizzo Grid Search CV
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
print(f"Il miglior valore di Accuracy per la CV è: {grid_search.best_score_:.3f}")

# valutazione finale
print("\n-----Valutazione finale del modello KNN ottimizzato-----")
best_knn_model = grid_search.best_estimator_
y_final_preds = best_knn_model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_final_preds)
print(f"\n Classification Report:\nAccuracy: {acc:.3f}\n", classification_report(y_test, y_final_preds, target_names=['Extrovert', 'Introvert']))

# rappresentazione grafica dei risultati
results_df = pd.DataFrame(grid_search.cv_results_)
plt.figure(num = "Ricerca Iperparametri", figsize=(12, 6))
plt.suptitle('PERFORMANCE DI KNN (TRAINING VS VALIDATION)', fontsize=16)
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
plt.xlabel('Numero di Vicini (k)')
plt.ylabel('Accuracy')
plt.axvline(x=best_k, color='red', linestyle='--', label=f'Miglior valore n_neighbors = {best_k}')
plt.legend()
plt.grid(True)
plt.show()

# matrice di confusione
cm = confusion_matrix(y_test, y_final_preds)

plt.figure(num = "Matrice di Confusione KNN Ottimizzato", figsize=(8, 6))
plt.suptitle('MATRICE DI CONFUSIONE PER KNN OTTIMIZZATO', fontsize=16)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Extrovert', 'Introvert'], 
                yticklabels=['Extrovert', 'Introvert'])
plt.xlabel('Valore Previsto')
plt.ylabel('Valore Reale')
plt.show()

# curva di ROC
y_pred_proba = best_knn_model.predict_proba(X_test_scaled)[:, 1]

# valori per la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# area sotto la curva (AUC)
auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(num = "Curva ROC KNN Ottimizzato", figsize=(8, 6))
plt.suptitle('CURVA ROC PER KNN OTTIMIZZATO', fontsize=16)
plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Classificatore Casuale')
plt.xlabel('Tasso di Falsi Positivi (FPR)')
plt.ylabel('Tasso di Veri Positivi (TPR)')
plt.legend()
plt.grid(True)
plt.show()