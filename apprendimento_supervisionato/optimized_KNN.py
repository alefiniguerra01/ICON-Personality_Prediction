from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from preprocessing import df
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import warnings

warnings.simplefilter("ignore", category=FutureWarning)

print("\n----- Ricerca dei migliori iperparametri per KNN -----")

# suddivisione tra features e target
X = df.drop("Personality", axis=1)
y = df["Personality"]

# parametri
n_runs = 10
n_neighbors = range(1, 31)

all_means = []  # Lista di liste, ogni sotto-lista media accuracies per ogni n_neighbors in una run
all_stds = []   # Lista di liste, deviazioni standard per ogni n_neighbors in una run

for run in range(n_runs):
    print(f"Run {run + 1}/{n_runs}")
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=run, stratify=y)

    # standardizzazione
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    mean_scores = []
    std_scores = []

    for neighbors in n_neighbors:
        knn = KNeighborsClassifier(n_neighbors=neighbors)
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
        mean_scores.append(scores.mean())
        std_scores.append(scores.std())

    all_means.append(mean_scores)
    all_stds.append(std_scores)

    # media e std delle accuracy di tutti i neighbors per la run corrente
    mean_run = np.mean(mean_scores)
    std_run = np.std(mean_scores)
    print(f"  Cross-Validation mean accuracy: {mean_run:.4f}") 
    print(f"  Cross-Validation std accuracy: {std_run:.4f}")

all_means_array = np.array(all_means)
all_stds_array = np.array(all_stds)

# accuracy media e deviazione standard per ogni neighbors su tutte le run
mean_accuracy_for_neighbors = np.mean(all_means_array, axis=0)
std_accuracy_for_neighbors = np.std(all_means_array, axis=0)

best_global_idx = np.argmax(mean_accuracy_for_neighbors)
best_global_neighbors = n_neighbors[best_global_idx]

print(f"\n Miglior valore globale di k: {best_global_neighbors}")
print(f"   Accuracy media: {mean_accuracy_for_neighbors[best_global_idx]:.4f}")
print(f"   Deviazione standard: {std_accuracy_for_neighbors[best_global_idx]:.4f}")

# grafico unico con accuracy media e deviazione standard per tutti i valori di n_neighbors
plt.figure(figsize=(10, 5), num = "Accuracy Media e Deviazione Standard per Valori di n_neighbors")
plt.errorbar(
    n_neighbors, 
    mean_accuracy_for_neighbors, 
    yerr=std_accuracy_for_neighbors,
    fmt='o--', 
    capsize=4,
    elinewidth=2,
    markeredgewidth=2, 
    color = 'blue',
    label='Accuracy media ± std'
)
plt.axvline(
    x=best_global_neighbors, 
    color='red', 
    linestyle='--', 
    label=f'Miglior neighbors = {best_global_neighbors}'
)
plt.title("ACCURACY MEDIA E DEVIAZIONE STANDARD (SU 10 RUN)", fontsize=14)
plt.xlabel("Valore di n_neighbors")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# tabella riassuntiva delle run
run_summary = []
for i in range(n_runs):
    mean_run = np.mean(all_means_array[i])
    std_run = np.std(all_means_array[i])
    run_summary.append({'Run': i+1, 'CV Accuracy': mean_run, 'CV Std': std_run})

summary_df = pd.DataFrame(run_summary)

print("\nTabella riassuntiva delle run:")
print(summary_df.to_string(index=False, float_format="%.4f"))

# grafico della media e deviazione standard delle accuracy per run
plt.figure(figsize=(10, 6), num = "KNN - CV Accuracy su 10 Run")
plt.errorbar(
    summary_df['Run'],
    summary_df['CV Accuracy'],
    yerr=summary_df['CV Std'],
    fmt='o--',
    capsize=4,
    elinewidth=2,
    markeredgewidth=2,
    color='blue',
    label ='CV Accuracy (Media ± Deviazione Standard)'
)
plt.xlabel("Run")
plt.ylabel("CV Accuracy")
plt.title("KNN - ACCURACY MEDIA IN CROSS-VALIDATION (10 RUN)", fontsize=14)
plt.ylim(0.5, 1.0)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()