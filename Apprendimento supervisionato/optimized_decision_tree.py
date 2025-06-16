import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from train_val import X_test_scaled, X_train_scaled, y_train, y_test, X
import matplotlib.pyplot as plt
import seaborn as sns

print("\n-----Ricerca dei migliori iperparametri per Decision Tree-----")

dt = DecisionTreeClassifier(random_state=42)

# iperparametri da ottimizzare
param_grid = {
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8]
}

grid_search_dt = GridSearchCV(estimator=dt,
                              param_grid=param_grid,
                              cv=5,
                              scoring='accuracy',
                              n_jobs=-1,
                              verbose=1)
grid_search_dt.fit(X_train_scaled, y_train)

# visualizzazione dei risultati
print("\n-----Ricerca completata-----")
print("Migliori iperparametri trovati: ", grid_search_dt.best_params_)
print(f"Miglior Accuracy (cross-validation): {grid_search_dt.best_score_:.3f}")

# valutazione finale
print("\n-----Valutazione finale del modello Decision Tree ottimizzato-----")
best_dt_model = grid_search_dt.best_estimator_
y_final_preds = best_dt_model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_final_preds)
print(f"\n Classification Report:\nAccuracy: {acc:.3f}\n", classification_report(y_test, y_final_preds, target_names=['Extrovert', 'Introvert']))

# rappresentazione grafica dei risultati

# curva di ROC
y_pred_proba = best_dt_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.figure(num = "Curva ROC Decision Tree Ottimizzato", figsize=(8, 6))
plt.suptitle('CURVA ROC PER IL MODELLO DECISION TREE OTTIMIZZATO', fontsize=14)
plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Classificatore Casuale')
plt.xlabel('Tasso di Falsi Positivi (FPR)')
plt.ylabel('Tasso di Veri Positivi (TPR)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# matrice di confusione
cm = confusion_matrix(y_test, y_final_preds)
plt.figure(num = "Matrice di Confusione Decision Tree Ottimizzato", figsize=(9, 6))
plt.suptitle('MATRICE DI CONFUSIONE PER IL MODELLO DECISION TREE OTTIMIZZATO', fontsize=14)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Extrovert', 'Introvert'], 
                yticklabels=['Extrovert', 'Introvert'])
plt.xlabel('Valore Previsto')
plt.ylabel('Valore Reale')
plt.show()

# importanza delle features
feature_names = X.columns.tolist()
importances = best_dt_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
plt.figure(num = "Importanza Features Decision Tree Ottimizzato", figsize=(15, 7))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='plasma')
plt.suptitle('IMPORTANZA DELLE FEATURES NEL MODELLO DECISION TREE OTTIMIZZATO', fontsize=14)
plt.xlabel('Punteggio di Importanza')
plt.ylabel('Feature')
plt.show()
