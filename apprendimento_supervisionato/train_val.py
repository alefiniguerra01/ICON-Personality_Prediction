from preprocessing import df
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import warnings

warnings.simplefilter("ignore", category=FutureWarning)

# suddivisione tra features e target
X = df.drop("Personality", axis=1)
y = df["Personality"]

# suddivisione in training set e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

# standardizzazione
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# addestramento
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(),
}

accuracy_results = []

if __name__ == "__main__":
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_preds)

        print(f"\n{'-'*10} {name} {'-'*10}")
        print(f"Accuracy: {acc:.3f}")
        print("Classification Report:\n", classification_report(y_test, y_preds, target_names=['Extrovert', 'Introvert']))

        accuracy_results.append({"Modello": name, "Accuracy": acc})

    results = pd.DataFrame(accuracy_results)
    print("-"*30, "Riepilogo Accuracy", "-"*30)
    results = results.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
    results.index += 1
    print(results)