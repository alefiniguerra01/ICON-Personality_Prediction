import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from preprocessing import df

warnings.simplefilter("ignore", category=FutureWarning)

# suddivisione tra features e target
X = df.drop("Personality", axis=1)
y = df["Personality"]

n_runs = 10
means_cv = []
stds_cv = []
best_params_all_runs = []

# parametri
param_dist = {
    'max_depth': [None, 3, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6]
}

print("\n----- Ricerca dei migliori iperparametri per Decision Tree -----")

for run in range(n_runs):
    print(f"\nRun {run+1}/{n_runs}")

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=run)

    dt = DecisionTreeClassifier(random_state=run)
    random_search = RandomizedSearchCV(
        dt,
        param_distributions=param_dist,
        n_iter=15,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=run
    )
    random_search.fit(X_train, y_train)

    mean_cv = random_search.cv_results_['mean_test_score'][random_search.best_index_]
    std_cv = random_search.cv_results_['std_test_score'][random_search.best_index_]

    means_cv.append(mean_cv)
    stds_cv.append(std_cv)
    best_params_all_runs.append(random_search.best_params_)

    # metriche per ogni run
    print(f"  Cross-Validation mean accuracy: {mean_cv:.4f}")
    print(f"  Cross-Validation std accuracy: {std_cv:.4f}")
    print(f"  Migliori iperparametri: {random_search.best_params_}")

# tabella riassuntiva delle run
summary_df = pd.DataFrame({
    'Run': list(range(1, n_runs+1)),
    'CV Accuracy': means_cv,
    'CV Std': stds_cv
})
print("\nTabella riassuntiva delle run:")
print(summary_df.to_string(index=False, float_format="%.4f"))

# grafico finale accuracy media e std su 10 run
plt.figure(figsize=(10, 6), num = "Decision Tree - CV Accuracy su 10 Run")
plt.errorbar(
    summary_df['Run'],
    summary_df['CV Accuracy'],
    yerr=summary_df['CV Std'],
    fmt='o--',
    capsize=4,
    elinewidth=2,
    markeredgewidth=2,
    color='purple',
    label='CV Accuracy (Media ± Deviazione Standard)'
)
plt.xlabel("Run")
plt.ylabel("CV Accuracy")
plt.title("DECISION TREE - ACCURACY MEDIA IN CROSS-VALIDATION (10 RUN)", fontsize=14)
plt.ylim(0.5, 1.0)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
