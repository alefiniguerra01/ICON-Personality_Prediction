import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.simplefilter("ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)  # visualizza tutte le colonne del dataset

# Caricamento del dataset
df = pd.read_csv('./Dataset/personality_dataset.csv')

# Visualizzazione delle prime righe del dataset e delle informazioni generali
print("-----Prime righe del dataset-----")
print(df.head())

print(f"\nIl dataset contiene {df.shape[0]} righe e {df.shape[1]} colonne.")

print("\n-----Informazioni generali sul dataset-----")
print(df.info())

print("\n-----Rimozione dei valori duplicati-----")
df.drop_duplicates(inplace=True)
print(f"Il dataset ora contiene {df.shape[0]} righe e {df.shape[1]} colonne")

# separazione delle colonne numeriche e categoriche
numerical_columns = df.select_dtypes(include=['float64']).columns
categorical_columns = df.select_dtypes(include=['object']).columns
print("\n-----Colonne numeriche-----")
print(numerical_columns)
print("\n-----Colonne categoriche-----")
print(categorical_columns)

# visualizzazione delle feature numeriche
'''plt.figure(figsize=(10, 8))

for i, col in enumerate(numerical_columns, 1):
    plt.subplot(3, 2, i)
    sns.histplot(x=col, data=df, kde=False, bins=20, color='blue')
    plt.title(f'{col}')
    plt.xlabel('')
    plt.ylabel('Frequenza')

plt.tight_layout()
plt.show()'''

# visualizzazione features numeriche
plt.figure(figsize=(10, 7))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(3, 2, i)
    sns.histplot(data=df, x=col, bins=15, kde=True, color="skyblue")
    plt.title(f'{col}')
    plt.xlabel("")
    plt.ylabel("Frequenza")
plt.tight_layout(pad=3.0)
plt.show()

# visualizzazione delle features categoriche
plt.figure(figsize=(10, 7))
for i, col in enumerate(categorical_columns, 1):
    plt.subplot(2, 2, i)
    sns.countplot(data=df, x=col, palette="pastel")
    plt.title(f'{col}')
    plt.xlabel("")
    plt.ylabel("Conteggio")
    plt.xticks(rotation=45)
plt.tight_layout(pad=3.0)
plt.show()

