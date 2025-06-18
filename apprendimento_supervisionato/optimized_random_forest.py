import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from train_val import X_train_scaled, y_train, X_test_scaled, y_test, X
from preprocessing import df

print("\n-----FASE 1: Inizio ricerca rapida con Random Search CV-----")
rf = RandomForestClassifier(random_state=42)

# definisco gli iperparametri
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 20, 30, 40, 50, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# stabilisco di eseguire 25 iterate
random_search = RandomizedSearchCV(estimator=rf,
                                   param_distributions=param_dist,
                                   n_iter=25,
                                   cv=5,
                                   scoring='accuracy',
                                   n_jobs=-1,
                                   random_state=42,
                                   verbose=1)
random_search.fit(X_train_scaled, y_train)

# stampo i risultati della fase 1
print("\n-----Risultati FASE 1-----")
best_params_random = random_search.best_params_
print(f"Migliori parametri trovati nella ricerca casuale: {best_params_random}")

print("\n-----FASE 2: Inizio ricerca rirata con Grid Search CV-----")

param_grid_focused = {
    'n_estimators': [best_params_random['n_estimators'] - 50, 
                     best_params_random['n_estimators'], 
                     best_params_random['n_estimators'] + 50],
    'max_depth': [best_params_random['max_depth'] - 5 if best_params_random['max_depth'] is not None else 25, 
                  best_params_random['max_depth'], 
                  best_params_random['max_depth'] + 5 if best_params_random['max_depth'] is not None else 35],
    'min_samples_split': [best_params_random['min_samples_split']],
    'min_samples_leaf': [best_params_random['min_samples_leaf']]
}

grid_search = GridSearchCV(estimator=rf,
                           param_grid=param_grid_focused,
                           scoring='accuracy',
                           cv=5,
                           n_jobs=-1,
                           verbose=1)
grid_search.fit(X_train_scaled, y_train)

print("\n-----Risultati FASE 2-----")
print(f"Migliori iperparametri definitivi trovati:  {grid_search.best_params_}")
print(f"\nMiglior accuracy ottenuto durante la cross-validation finale: {grid_search.best_score_:.3f}")

print("\n-----Valutazione finale del modello Random Forest ottimizzato-----")
best_rf_final = grid_search.best_estimator_
y_final_preds = best_rf_final.predict(X_test_scaled)
acc = accuracy_score(y_test, y_final_preds)
print(f"\n Classification Report:\nAccuracy: {acc:.3f}\n", classification_report(y_test, y_final_preds, target_names=['Extrovert', 'Introvert']))

# rappresentazione grafica dei risultati
feature_names = X.columns.tolist()
importances = best_rf_final.feature_importances_
feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
}).sort_values(by='Importance', ascending=False)
plt.figure(num = "Importanza Features Random Forest Ottimizzato", figsize=(15, 7))
plt.suptitle('IMPORTANZA DELLE FEATURES NEL MODELLO RANDOM FOREST OTTIMIZZATO', fontsize=14)
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.xlabel('Punteggio di Importanza')
plt.ylabel('Feature')
plt.show()

# curva di ROC
y_pred_proba = best_rf_final.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.figure(num = "Curva ROC Random Forest Ottimizzato", figsize=(8, 6))
plt.suptitle('CURVA ROC PER IL MODELLO RANDOM FOREST OTTIMIZZATO', fontsize=14)
plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Classificatore Casuale')
plt.xlabel('Tasso di Falsi Positivi (FPR)')
plt.ylabel('Tasso di Veri Positivi (TPR)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# matrice di confusione
cm = confusion_matrix(y_test, y_final_preds)
plt.figure(num = "Matrice di Confusione Random Forest Ottimizzato", figsize=(9, 6))
plt.suptitle('MATRICE DI CONFUSIONE PER IL MODELLO RANDOM FOREST OTTIMIZZATO', fontsize=14)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Extrovert', 'Introvert'], 
                yticklabels=['Extrovert', 'Introvert'])
plt.xlabel('Valore Previsto')
plt.ylabel('Valore Reale')
plt.show()