from preprocessing import df
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

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
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_preds)
    print(f"\n-----{name}: {acc:.3f}-----")
    mean = np.mean(y_preds)
    std = np.std(y_preds)
    print(f"Mean: {mean:.3f}, Standard Deviation: {std:.3f}")
    print("\n Classification Report:\n", classification_report(y_test, y_preds, target_names=['Extrovert', 'Introvert']))

    print("------------------")