import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV
from train_val import X_train_scaled, y_train, X_test_scaled, y_test
from preprocessing import df

# nomi delle colonne necessarie per i grafici
feature_names = df.drop("Personality", axis=1).columns.tolist()

print("\n-----Ricerca dei migliori iperparametri per Random Forest-----")
print("Attenzione: questa operazione potrebbe richiedere del tempo")
rf = RandomForestClassifier(random_state=42)

# imposto gli iperparametri da ottimizzare
param_distributions = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, 40, 50, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True]
}

# utilizzo Randomized Search CV
random_search = RandomizedSearchCV(estimator=rf,
                                   param_distributions=param_distributions,
                                   n_iter=100,
                                   cv=5,
                                   scoring='accuracy',
                                   n_jobs=-1,
                                   random_state=42)

random_search.fit(X_train_scaled, y_train)

# salvataggio dei risultati
print("\n-----Ricerca completata-----")
print("-----Salvataggio dei risultati su file CSV-----")
results_df = pd.DataFrame(random_search.cv_results_)
interesting_columns = [
    'rank_test_score',
    'mean_test_score',
    'std_test_score',
    'param_n_estimators',
    'param_max_depth',
    'param_min_samples_split',
    'param_min_samples_leaf'
]
results_df = results_df[interesting_columns].sort_values(by='rank_test_score')
results_df.to_csv('tuning_results/random_forest_tuning_results.csv', index=False)
print("Risultati salvati con successo in 'tuning_results/random_forest_tuning_results.csv'")

# stampo i risultati
best_params = random_search.best_params_
print(f"\nI migliori iperparametri trovati sono: {best_params}")
print(f"Il miglior valore di Accuracy per la CV è: {random_search.best_score_:.3f}")

# valutazione finale
print("\n-----Valutazione finale del modello Random Forest ottimizzato-----")
best_rf_model = random_search.best_estimator_
y_final_preds = best_rf_model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_final_preds)
print(f"\n Classification Report:\nAccuracy: {acc:.3f}\n", classification_report(y_test, y_final_preds, target_names=['Extrovert', 'Introvert']))

# rappresentazione grafica dei risultati
importances = best_rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
}).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(15, 7))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Importanza delle Feature nel Modello Random Forest')
plt.xlabel('Punteggio di Importanza')
plt.ylabel('Feature')
plt.show()

# curva di ROC
y_pred_proba = best_rf_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Classificatore Casuale')
plt.xlabel('Tasso di Falsi Positivi (FPR)')
plt.ylabel('Tasso di Veri Positivi (TPR)')
plt.title('Curva ROC per il Modello Random Forest Ottimizzato')
plt.legend(loc='lower right')
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
plt.title('Matrice di confusione per il modello Random Forest ottimizzato')
plt.show()