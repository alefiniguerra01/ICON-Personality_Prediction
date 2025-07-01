import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.calibration import LabelEncoder
from sklearn.impute import KNNImputer, SimpleImputer

warnings.simplefilter("ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)  # visualizza tutte le colonne del dataset

# caricamento del dataset
df = pd.read_csv('../dataset/personality_dataset.csv')

# visualizzazione delle prime righe del dataset
print("\n-----Visualizzazione delle prime righe del dataset-----")
print(df.head())

# rimozione dei valori duplicati
df.drop_duplicates(inplace=True)

# separazione delle colonne numeriche e categoriche
numerical_columns = df.select_dtypes(include=['float64']).columns
categorical_columns = df.select_dtypes(include=['object']).columns

# controllo della presenza di valori nulli
print("\n-----Controllo dei valori nulli-----")
print(df.isnull().sum())

# rimozione valori nulli feature numeriche
knn = KNNImputer(n_neighbors=3)
df[numerical_columns] = knn.fit_transform(df[numerical_columns])

# rimozione valori nulli feature categoriche
categorical_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])

print("\n-----Controllo dei valori nulli dopo l'imputazione-----")
print(df.isnull().sum())

# conversione delle variabili categoriche in numeriche
binary_map = {'Yes': 1, 'No': 0}
df[categorical_columns] = df[categorical_columns].replace(binary_map)

# conversione della variabile target in numerica
le = LabelEncoder()
df['Personality'] = le.fit_transform(df['Personality'])
print("\n-----Visualizzazione delle prime righe dopo la conversione-----")
print(df.head())

if __name__ == "__main__":
    # istogramma per vedere come sono distribuiti i dati
    print("\n-----Visualizzazione della distribuzione dei dati-----")

    columns = df.columns.drop('Personality')
    fig, axes = plt.subplots(nrows = 2, ncols = 4, figsize=(14, 7), num = "Distribuzione dei Dati")
    axes = axes.flatten()

    for i, col in enumerate(columns):
        df[col].hist(ax=axes[i], color = 'orange')
        axes[i].set_title(col)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("DISTRIBUZIONE DELLE VARIABILI NEL DATASET", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # visualizzazione delle variabili che incidono maggiornamente
    print("\n-----Visualizzazione delle variabili che incidono maggiormente-----")
    variables = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency']
    plt.figure(num = "Incidenza Variabili", figsize=(16, 7))
    
    for i, var in enumerate(variables, 1):
        plt.subplot(2, 4, i)
        plt.hist(df.loc[df['Personality']==0, var], bins=30, alpha=0.5, label='Extrovert', color='red')
        plt.hist(df.loc[df['Personality']==1, var], bins=30, alpha=0.5, label='Introvert', color='blue')
        
        plt.title(f'{var} based on Personality')
        plt.legend()
    
    plt.tight_layout(h_pad=3, w_pad=2)
    plt.show()