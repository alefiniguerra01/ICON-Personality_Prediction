from preprocessing import df
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# Importazione del DataFrame preprocessato

# Suddivisione tra features e target
X = df.drop("Personality", axis=1)
y = df["Personality"]

# Suddivisione in training set e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

# Standardizzazione
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Addestramento
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(),
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_preds)
    print(f"\n-----{name}= Accuracy: {acc:.3f}-----")
    print("\n Classification Report:\n", classification_report(y_test, y_preds, target_names=['Extrovert', 'Introvert']))

    print("------------------")

    # DA ELIMINARE
    if __name__ == "__main__":
        cm = confusion_matrix(y_test, y_preds)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Extrovert', 'Introvert'], 
                    yticklabels=['Extrovert', 'Introvert'])
        plt.xlabel('Valore Previsto')
        plt.ylabel('Valore Reale')
        plt.title(f'Matrice di confusione per il modello {name}')
        plt.show()

        # curva di ROC
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        # valori per la curva ROC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

        # area sotto la curva (AUC)
        auc = roc_auc_score(y_test, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Classificatore Casuale')
        plt.xlabel('Tasso di Falsi Positivi (FPR)')
        plt.ylabel('Tasso di Veri Positivi (TPR)')
        plt.title(f'Curva ROC per il Modello {name}')
        plt.legend()
        plt.grid(True)
        plt.show()