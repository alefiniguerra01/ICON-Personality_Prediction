from data_exploration import df, numerical_columns, categorical_columns, sns, plt
from sklearn.impute import KNNImputer,SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Controllo della presenza di valori nulli
print("\n-----Controllo dei valori nulli-----")
print(df.isnull().sum())

# Rimozione valori nulli feature numeriche
knn = KNNImputer(n_neighbors=3)
df[numerical_columns] = knn.fit_transform(df[numerical_columns])

# Rimozione valori nulli feature categoriche
categorical_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])

print("\n-----Controllo dei valori nulli dopo l'imputazione-----")
print(df.isnull().sum())

# Conversione delle variabili categoriche in numeriche
binary_map = {'Yes': 1, 'No': 0}
df[categorical_columns] = df[categorical_columns].replace(binary_map)

# Conversione della variabile target in numerica
le = LabelEncoder()
df['Personality'] = le.fit_transform(df['Personality'])
print("\n-----Visualizzazione delle prime righe dopo la conversione-----")
print(df.head())

if __name__ == "__main__":
    # Heatmap per visualizzare la correlazione tra le variabili
    corr = df.corr()
    plt.figure(num = "Heatmap Correlazione Variabili", figsize=(10, 6))
    plt.suptitle('HEATMAP DELLA CORRELAZIONE TRA VARIABILI', fontsize=14)
    sns.heatmap(corr, annot=True, fmt=".2f", square=False, cmap='coolwarm')
    plt.tight_layout()
    plt.show()