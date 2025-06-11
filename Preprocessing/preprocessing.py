import pandas as pd

pd.set_option('display.max_columns', None)  # visualizza tutte le colonne del dataset

# Caricamento del dataset
data = pd.read_csv('./Dataset/personality_dataset.csv')

# Visualizzazione delle prime righe del dataset e delle informazioni generali
print(data.head())