import pandas as pd
import os
import numpy as np
from pathlib import Path
#from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
import random
from schema_json import ReportData


TEST_SIZE = 0.2
VALIDATION_SIZE = 0.05
RANDOM_STATE = 2025
STRATIFY_COLUMNS = (
    'posizione',
    'infiltrazione_sfinteri',
    'infiltrazione_tessuto_adiposo',
    'depositi_tumorali',
    'emvi_esteso',
    'stadio_T',
    'stadio_N',
    'stadio_N1c',
    'metastasi'
)

random.seed(RANDOM_STATE)


base_dir = Path(__file__).parent.parent
print(base_dir)

# Get data
data_path = base_dir / "data" / "base.tumoreprimitivo.csv"
# Se il file non è presente, inserirlo manualmente prendendolo dalla cartella dropbox
data = pd.read_csv(data_path)


# Elimina righe doppie
# Intere righe duplicate. Keep last perchè i report hanno id decrescente
duplicati = data.iloc[:, 1:].duplicated(keep='last')
data_clean = data[duplicati == False]
# Rimuovi righe con report duplicati
duplicati = data_clean['report_text'].duplicated(keep='last')
data_clean = data_clean[duplicati == False]
# Resetta indici
data_clean.reset_index(inplace=True, drop=True)

print(f'{data_clean.shape = }')


# Keep omly report and target columns
target_columns = ReportData.model_fields.keys()
X = data_clean[['report_text'] + list(target_columns)].copy(deep=True)
print(f'{X.shape = }')


# Isola righe che hanno features appartenenti a classi poco numerose (minori di 2)

colonne = [
    'lesioni_ossee',
    'infiltrazione_sfinteri',
    'infiltrazione_sfinteri',
    'coinvolgimento_riflessione_peritoneale',
    'coinvolgimento_riflessione_peritoneale',
    'coinvolgimento_fascia_mesorettale',
    'depositi_tumorali',
    'stadio_N',
    'stadio_N',
    'mrf',
    'infiltrazione_tessuto_adiposo'
]
valori = [
    'si',
    'interno',
    'interno_piano_esterno',
    'rischio',
    'NaN',
    'NaN',
    'sospetto',
    'N1c',
    'NaN',
    'NaN',
    'NaN'
]

train_rows = []
test_rows = []
for col, val in zip(colonne, valori):
    if val == 'NaN':
        index = X[X[col].fillna('NaN') == val].index
    else:
        index = X[X[col] == val].index
    if len(index) == 0:
        continue
    elif len(index) == 1:
        train_rows.append(X.iloc[index])
        # tolgo riga dal dataset
        X.drop(index=index, inplace=True)
    elif len(index) == 2:
        x = random.randint(0, 1)
        y = 1 - x
        train_rows.append(X.iloc[index[x]].to_frame().T)
        test_rows.append(X.iloc[index[y]].to_frame().T)
        # Tolgo riga dal dataset
        X.drop(index=index, inplace=True)
    else:
        continue
    
print('OK')
print(f'{X.shape = }')

# Crea gli split
train_split = pd.concat(train_rows)
test_split = pd.concat(test_rows)

print(f'{train_split.shape = }')
print(f'{test_split.shape = }')

Y = X.drop(columns='report_text').copy(deep=True)
X = X['report_text'].copy(deep=True)

print(f'{X.shape = }')
print(f'{Y.shape = }')

# Selezioniamo solo le colonne su cui vogliamo stratificare
X_strat = X.values
Y_strat = Y[list(STRATIFY_COLUMNS)].values

X_train, Y_train, X_test, Y_test = iterative_train_test_split(X_strat, Y_strat, test_size=TEST_SIZE)



exit()

# Splitta i dati rimasti
for col in STRATIFY_COLUMNS:
    print(data_clean.fillna('NaN')[col].value_counts())
    print("\n")
    
train, test = train_test_split(data_clean,
                               test_size=TEST_SIZE,
                               random_state=RANDOM_STATE,
                               stratify=data_clean[list(STRATIFY_COLUMNS)].fillna('NaN'))



print(f'{train.shape = }')
print(f'{test.shape = }')
