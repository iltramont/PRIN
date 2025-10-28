import pandas as pd
from pathlib import Path
#from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split

import random
from schema_json import ReportData
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')


TEST_SIZE = 0.2
VALIDATION_SIZE = 0.05
RANDOM_STATE = 2025
DATA_FILE_NAME = "base.tumoreprimitivo.csv"

TRAIN_SPLIT_FILE_NAME = 'train_split.csv'
TEST_SPLIT_FILE_NAME = 'test_split.csv'
VALIDATION_SPLIT_FILE_NAME = 'validation_split.csv'

REPORT__COLUMN_NAME = 'report_text'
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

# Get data
data_path = base_dir / "data" / DATA_FILE_NAME
# Se il file non è presente, inserirlo manualmente prendendolo dalla cartella dropbox
data = pd.read_csv(data_path)


# Elimina righe doppie
# Intere righe duplicate. Keep last perchè i report hanno id decrescente
duplicati = data.iloc[:, 1:].duplicated(keep='last')
data_clean = data[duplicati == False]
# Rimuovi righe con report duplicati
duplicati = data_clean[REPORT__COLUMN_NAME].duplicated(keep='last')
data_clean = data_clean[duplicati == False]
# Resetta indici
data_clean.reset_index(inplace=True, drop=True)

print(f'\nRighe doppie eliminate\n{data_clean.shape = }\n')


# Keep only report and target columns
target_columns = ReportData.model_fields.keys()
X = data_clean[[REPORT__COLUMN_NAME] + list(target_columns)].copy(deep=True)
print(f'Selezionate solo colonne di interesse\n{X.shape = }\n')

# Create dummies to stratify (train - test)
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(X[list(STRATIFY_COLUMNS)])
Y_dummy = encoder.transform(X[list(STRATIFY_COLUMNS)])
print(f'Create dummies per stratificazione\n{Y_dummy.shape = }')




# Train test split with stratification
index_train, Y_train, index_test, Y_test = iterative_train_test_split(X.index.to_numpy().reshape(-1, 1),
                                                                      Y_dummy,
                                                                      test_size=TEST_SIZE)

index_train = index_train.reshape(1, -1)[0]
index_test = index_test.reshape(1, -1)[0]

X_train = X.iloc[index_train].copy(deep=True)
X_test = X.iloc[index_test].copy(deep=True)


# Create dummies to stratify (train - validation)
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(X_train[list(STRATIFY_COLUMNS)])
Y_dummy = encoder.transform(X_train[list(STRATIFY_COLUMNS)])
print(f'Create dummies per stratificazione\n{Y_dummy.shape = }')


# Train validation split with stratification
index_train, Y_train, index_validation, Y_validation = iterative_train_test_split(X_train.index.to_numpy().reshape(-1, 1),
                                                                                  Y_dummy,
                                                                                  test_size=VALIDATION_SIZE)

index_train = index_train.reshape(1, -1)[0]
index_validation = index_validation.reshape(1, -1)[0]


X_validation = X_train.loc[index_validation].copy(deep=True)
X_train = X_train.loc[index_train].copy(deep=True)


X_train['split'] = 'train'
X_test['split'] = 'test'
X_validation['split'] = 'validation'

print(f'\nSplitatto il dataset\n{X_train.shape = }\n{X_test.shape = }\n{X_validation.shape = }')


# Visualize stratification
n_columns = 3
n_rows, r = divmod(len(STRATIFY_COLUMNS), n_columns)
if r != 0:
    n_rows += 1

fig, axes = plt.subplots(n_rows, n_columns)
fig.suptitle("Stratification", fontsize='xx-large')

df = pd.concat([X_train, X_test, X_validation]).fillna('NaN')
for i in range(len(STRATIFY_COLUMNS)):
    sns.countplot(data=df, x=STRATIFY_COLUMNS[i], hue='split', ax=axes[i//n_columns][i%n_columns])

plt.show()

# Manual adjustments
# Stadio_N NaN
df_manual = X_validation[X_validation.stadio_N.fillna('NaN') == 'NaN'].iloc[0:1].copy(deep=True)
if len(df_manual) > 0:
    X_validation.drop(index=df_manual.index, inplace=True)
    X_train = pd.concat([X_train, df_manual])

# Infiltrazione sfinteri
df_manual = X_validation[X_validation.infiltrazione_sfinteri == 'interno_piano_esterno'].iloc[0:1].copy(deep=True)
if len(df_manual) > 0:
    X_validation.drop(index=df_manual.index, inplace=True)
    X_train = pd.concat([X_train, df_manual])
    
# Stadio_N N1c
df_manual = X_test[X_test.stadio_N == 'N1c'].iloc[0:1].copy(deep=True)
if len(df_manual) > 0 and len(X_train[X_train.stadio_N == 'N1c']) == 0:
    X_test.drop(index=df_manual.index, inplace=True)
    X_train = pd.concat([X_train, df_manual])

print(f'\nAggiustati i dataset\n{X_train.shape = }\n{X_test.shape = }\n{X_validation.shape = }')


X_train['split'] = 'train'
X_test['split'] = 'test'
X_validation['split'] = 'validation'

# Visualize stratification 2
n_columns = 3
n_rows, r = divmod(len(STRATIFY_COLUMNS), n_columns)
if r != 0:
    n_rows += 1

fig, axes = plt.subplots(n_rows, n_columns)
fig.suptitle("Stratification", fontsize='xx-large')

df = pd.concat([X_train, X_test, X_validation]).fillna('NaN')
for i in range(len(STRATIFY_COLUMNS)):
    sns.countplot(data=df, x=STRATIFY_COLUMNS[i], hue='split', ax=axes[i//n_columns][i%n_columns])

plt.show()


# Save data
if True:
    train_path = base_dir / "data" / TRAIN_SPLIT_FILE_NAME
    X_train.to_csv(train_path, index=False)
    test_path = base_dir / "data" / TEST_SPLIT_FILE_NAME
    X_test.to_csv(test_path, index=False)
    validation_path = base_dir / "data" / VALIDATION_SPLIT_FILE_NAME
    X_validation.to_csv(validation_path, index=False)
