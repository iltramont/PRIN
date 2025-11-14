import pandas as pd
from pathlib import Path
#from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split

import random
from schema_json import ReportData
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import ast

plt.style.use('ggplot')


TEST_SIZE = 0.2
VALIDATION_SIZE = 0.006
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
    'metastasi',
    'lesioni_ossee',
    'coinvolgimento_riflessione_peritoneale',
    'coinvolgimento_fascia_mesorettale'
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

# Sostituiamo valore -1 con 0 per la colonna numero depositi
data_clean.loc[data_clean['numero_depositi'] == -1.0, 'numero_depositi'] = 0.0

# Eliminiamo con combinazioni di valori incompatibili
data_clean.drop(
    index=data_clean[
        (data_clean['linfonodi_sospetti'] == 0) &
        (data_clean['numero_linfonodi_non_conosciuto'] == False) &
        (data_clean['sedi_locoregionali'] != '[]') &
        (data_clean['profile'] == 'GuidoImbemba')
    ].index
, inplace=True)
data_clean.reset_index(inplace=True, drop=True)

print(f'\nRighe con valori incompatibili eliminate\n{data_clean.shape = }\n')

# Sostituiamo i valore con None nella colonna dei dettagli degli organi coinvolti pechè altri valori non sono compatibili con "no" nella colonna infiltrazione_organi_extra
for i in range(data_clean.shape[0]):
    if (data_clean.loc[i, 'infiltrazione_organi_extra'] == 'no') and (data_clean.loc[i, 'infiltrazione_organi_dettagli'] is not None):
        data_clean.loc[i, 'infiltrazione_organi_dettagli'] = None

print(data_clean[(data_clean['infiltrazione_organi_dettagli'] > '') &
                 (data_clean['infiltrazione_organi_extra'] == 'no')
      ][
          ['profile', 'infiltrazione_organi_dettagli', 'infiltrazione_organi_extra']
      ])


# creazione nuova colonna "sedi_linfonodi_sospetti"
sedi_linfonodi = []
for s1, s2 in zip(data_clean.sedi_locoregionali, data_clean.sedi_non_locoregionali):
    sedi_linfonodi.append(str(ast.literal_eval(s1) + ast.literal_eval(s2)))
data_clean['sedi_linfonodi'] = sedi_linfonodi

print(f'Nuova colonna "sedi_linfonodi" creata\n{data_clean.shape = }')


# Teniamo solo le colonne di Guido
data_clean_guido = data_clean[data_clean['profile'] == 'GuidoImbemba']
data_clean_guido.reset_index(inplace=True, drop=True)
print(f'{data_clean_guido.shape = }')

# Aggregazione / modifica delle colonne
# Dettagli infiltrazione organi
# Teniamo solo NaN, pavimento_pelvico e altro
infiltrazione_organi_dettagli_new = []
for s in data_clean_guido.infiltrazione_organi_dettagli.fillna('NaN'):
    dettagli = []
    if s == 'NaN':
        infiltrazione_organi_dettagli_new.append(str(dettagli))
    else:
        d = ast.literal_eval(s)
        if 'pavimento_pelvico' in d:
            dettagli.append('pavimento_pelvico')
        if ('altro' in d) or ('utero' in d) or ('sacro' in d):
            dettagli.append('altro')
        infiltrazione_organi_dettagli_new.append(str(dettagli))
data_clean_guido.loc[:, 'infiltrazione_organi_dettagli'] = infiltrazione_organi_dettagli_new

#print(data_clean_guido.infiltrazione_organi_dettagli.value_counts())

# Sedi linfonodi
# Teniamo solo NaN, altro, mesorettali, rettali_superiori, otturatori, iliaci
sedi_linfonodi_new = []
for s in data_clean_guido.sedi_linfonodi:
    sedi = ast.literal_eval(s)
    sedi_new = set()
    for sede in sedi:
        if sede in ['mesorettali', 'rettali_superiori', 'otturatori']:
            sedi_new.add(sede)
        elif sede in ['iliaci_comuni', 'iliaci_interni', 'iliaci_esterni']:
            sedi_new.add('iliaci')
        else:
            sedi_new.add('altro')
    sedi_linfonodi_new.append(str(list(sedi_new)))
data_clean_guido.loc[:, 'sedi_linfonodi'] = sedi_linfonodi_new
#print(data_clean_guido.sedi_linfonodi.value_counts())

# Coinvolgimento fascia mesorettale. Trasformiamo rischio in si
data_clean_guido.loc[data_clean_guido['coinvolgimento_fascia_mesorettale'] == 'rischio', 'coinvolgimento_fascia_mesorettale'] = 'si'
#print(data_clean_guido.coinvolgimento_fascia_mesorettale.value_counts())

# Coinvolgimento riflessione peritoneale. Trasformiamo rischio in si
data_clean_guido.loc[data_clean_guido['coinvolgimento_riflessione_peritoneale'] == 'rischio', 'coinvolgimento_riflessione_peritoneale'] = 'si'
#print(data_clean_guido.coinvolgimento_riflessione_peritoneale.value_counts())

# Infiltrazione sfinteri. Trasformiamo la posizione in si. per otenere una classe (si/no/NaN)
data_clean_guido.loc[data_clean_guido['infiltrazione_sfinteri'] == 'interno_piano', 'infiltrazione_sfinteri'] = 'si'
data_clean_guido.loc[data_clean_guido['infiltrazione_sfinteri'] == 'interno', 'infiltrazione_sfinteri'] = 'si'
data_clean_guido.loc[data_clean_guido['infiltrazione_sfinteri'] == 'interno_piano_esterno', 'infiltrazione_sfinteri'] = 'si'
#print(data_clean_guido.infiltrazione_sfinteri.value_counts())

# Coinvolgimento riflessione peritoneale. Trasformiamo rischio in si
data_clean_guido.loc[data_clean_guido['emvi_esteso'] == 'sospetto', 'emvi_esteso'] = 'si'
#print(data_clean_guido.emvi_esteso.value_counts())


# PLOT
data_plot = data_clean_guido

columns_plot = ['morfologia', 'infiltrazione_tessuto_adiposo', 'coinvolgimento_fascia_mesorettale',
                'carcinosi_peritoneale', 'riflessione_peritoneale_anteriore', 'coinvolgimento_riflessione_peritoneale',
                'stadio_T', 'stadio_N', 'stadio_N1c',
                'mrf', 'emvi', 'metastasi',
                'infiltrazione_sfinteri', 'infiltrazione_organi_extra',
                'linfonodi_sospetti', 'numero_linfonodi_non_conosciuto', 'lesioni_ossee',
                'depositi_tumorali', 'numero_depositi', 'emvi_esteso']

# hue_column = None
# columns_plot = [hue_column] + columns_plot

n_columns = 3
n_rows, r = divmod(len(columns_plot), n_columns)
if r != 0:
    n_rows += 1

fig, axes = plt.subplots(n_rows, n_columns, figsize=(n_columns * 7, n_rows * 3))
fig.suptitle("Count values (Guido)", fontsize='xx-large')

for i, col in enumerate(columns_plot):
    ax = axes[i // n_columns][i % n_columns]
    sns.countplot(data=data_plot, x=col, order=data_plot[col].value_counts().index, ax=ax)
    # Add values on top of bars
    for p in ax.patches:
        y_text = int(p.get_height())
        x_text = p.get_x() + p.get_width() / 2
        ax.text(x=x_text, y=y_text, s=f'{y_text}', ha='center', va='bottom')

plt.tight_layout()
# plt.savefig("distribuzione_featurse_per_annotatore.png", dpi=300, bbox_inches='tight', transparent=False)
plt.show()

data_plot = data_clean_guido.fillna('NaN')

# Analisi posizione, sedi locoregionali, sedi non locoregionali
columns = ['posizione_multiple', 'sedi_locoregionali', 'sedi_non_locoregionali', 'sedi_linfonodi',
           'infiltrazione_organi_dettagli']

possible_values = {col: ['NaN'] for col in columns}

for col in columns:
    for s in data_plot[col].value_counts().index:
        possible_values[col] += ast.literal_eval(s)
    possible_values[col] = list(set(possible_values[col]))

counts = {
    col: {val: 0 for val in possible_values[col]}
    for col in columns
}
for col in columns:
    for s in data_plot[col]:
        value_list = ast.literal_eval(s)
        for value in possible_values[col]:
            if value in value_list:
                counts[col][value] += 1
        if value_list == []:
            counts[col]['NaN'] += 1

n_columns = 2
n_rows, r = divmod(len(columns), n_columns)
if r != 0:
    n_rows += 1

fig, axes = plt.subplots(n_rows, n_columns, figsize=(21, n_rows * 3))
orientation = 'h'
for i, col in enumerate(columns):
    ax = axes[i // n_columns][i % n_columns]
    series = pd.Series(counts[col], name=col).sort_values(ascending=False)
    sns.barplot(data=series, ax=ax, orient=orientation)
    # Add values on top of bars
    if orientation == 'v':
        for p in ax.patches:
            y_text = int(p.get_height())
            x_text = p.get_x() + p.get_width() / 2
            ax.text(x=x_text, y=y_text, s=f'{y_text}', ha='center', va='bottom')
    if orientation == 'h':
        for p in ax.patches:
            x_text = p.get_width()
            y_text = p.get_y() + p.get_height() / 2
            ax.text(x=x_text, y=y_text, s=f'{int(x_text)}', va='center', ha='left')

plt.tight_layout()
plt.show()


exit()

# Keep only report and target columns
target_columns = list(ReportData.model_fields.keys())

index = target_columns.index('sedi_linfonodi_locoregionali')
target_columns[index] = 'sedi_locoregionali'  # Correzione nome colonna

index = target_columns.index('sedi_linfonodi_non_locoregionali')
target_columns[index] = 'sedi_non_locoregionali'  # Correzione nome colonna


X = data_clean[[REPORT__COLUMN_NAME] + target_columns].copy(deep=True)
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
"""
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

"""

# Save data
if True:
    train_path = base_dir / "data" / TRAIN_SPLIT_FILE_NAME
    X_train.to_csv(train_path, index=False)
    test_path = base_dir / "data" / TEST_SPLIT_FILE_NAME
    X_test.to_csv(test_path, index=False)
    validation_path = base_dir / "data" / VALIDATION_SPLIT_FILE_NAME
    X_validation.to_csv(validation_path, index=False)
