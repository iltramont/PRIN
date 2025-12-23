import pandas as pd
from pathlib import Path

import random
import matplotlib.pyplot as plt
import seaborn as sns
import ast


plt.style.use('ggplot')

############
# Parameters
############
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
RANDOM_STATE = 2025
DATA_FILE_NAME = "base.tumoreprimitivo.csv"

TRAIN_SPLIT_FILE_NAME = 'train_split.csv'
TEST_SPLIT_FILE_NAME = 'test_split.csv'
VALIDATION_SPLIT_FILE_NAME = 'validation_split.csv'
ONLY_GUIDO = True

REPORT__COLUMN_NAME = 'report_text'
STRATIFY_COLUMNS = (
    'morfologia',
    'infiltrazione_sfinteri',
    'infiltrazione_tessuto_adiposo',
    'depositi_tumorali',
    'emvi_esteso',
    'stadio_T',
    'stadio_N',
    'stadio_N1c',
    'metastasi',
    'coinvolgimento_riflessione_peritoneale',
    'coinvolgimento_fascia_mesorettale',
    'infiltrazione_organi_extra'
)

#############
# Preliminari
#############
random.seed(RANDOM_STATE)
base_dir = Path(__file__).parent.parent


##############
# Get raw data
##############
data_path = base_dir / "data" / DATA_FILE_NAME
# Se il file non è presente, inserirlo manualmente prendendolo dalla cartella dropbox
data = pd.read_csv(data_path)


###################
# First adjustments
###################
# Elimina la colonna "posizione" in quanto obsoleta, rinominando la colonna "posizione_multiple" in "posizione"
data.drop(columns=['posizione'], inplace=True)
data.rename(columns={'posizione_multiple': 'posizione'}, inplace=True)
# Elimina colonne "carcinosi peritoneale" e "lesioni ossee" in quanto solo rispettivamente 1 e 2 record sono valorizzati.
data.drop(columns=['carcinosi_peritoneale', 'lesioni_ossee'], inplace=True)


###################
# Delete duplicates
###################
# Intere righe duplicate. Keep last perchè i report hanno id decrescente
duplicati = data.iloc[:, 1:].duplicated(keep='last')
data = data[duplicati == False]
# Rimuovi righe con report duplicati
duplicati = data[REPORT__COLUMN_NAME].duplicated(keep='last')
data = data[duplicati == False]
# Resetta indici
data.reset_index(inplace=True, drop=True)
print(f'\nRighe doppie eliminate\n{data.shape = }\n')


#########################
# Consistency adjustments
#########################
# Sostituiamo valore -1 con 0 per la colonna numero depositi
data.loc[data['numero_depositi'] == -1.0, 'numero_depositi'] = 0.0

# Eliminiamo con combinazioni di valori incompatibili
data.drop(
    index=data[
        (data['linfonodi_sospetti'] == 0) &
        (data['numero_linfonodi_non_conosciuto'] == False) &
        (data['sedi_locoregionali'] != '[]')
        ].index
    , inplace=True)
data.reset_index(inplace=True, drop=True)
print(f'\nRighe con valori incompatibili eliminate\n{data.shape = }\n')

# Sostituiamo i valore con None nella colonna dei dettagli degli organi coinvolti
# pechè altri valori non sono compatibili con "no" nella colonna infiltrazione_organi_extra
for i in range(data.shape[0]):
    if (data.loc[i, 'infiltrazione_organi_extra'] == 'no') and (data.loc[i, 'infiltrazione_organi_dettagli'] is not None):
        data.loc[i, 'infiltrazione_organi_dettagli'] = None

# creazione nuova colonna "sedi_linfonodi"
sedi_linfonodi = []
for s1, s2 in zip(data.sedi_locoregionali, data.sedi_non_locoregionali):
    sedi_linfonodi.append(ast.literal_eval(s1) + ast.literal_eval(s2))
data['sedi_linfonodi'] = sedi_linfonodi
print(f'Nuova colonna "sedi_linfonodi" creata\n{data.shape = }')

# Aggregazione / modifica delle colonne
# Dettagli infiltrazione organi -> il formato della colonna viene reso uguale a quello dei linfonodi
# Teniamo solo NaN, pavimento_pelvico e altro
infiltrazione_organi_dettagli_new = []
for s in data.infiltrazione_organi_dettagli.fillna('NaN'):
    dettagli = []
    if s == 'NaN':
        infiltrazione_organi_dettagli_new.append(dettagli)
    else:
        d = ast.literal_eval(s)
        if 'pavimento_pelvico' in d:
            dettagli.append('pavimento_pelvico')
        if ('altro' in d) or ('utero' in d) or ('sacro' in d):
            dettagli.append('altro')
        infiltrazione_organi_dettagli_new.append(dettagli)
data['infiltrazione_organi_dettagli'] = infiltrazione_organi_dettagli_new

# Sedi linfonodi
# Teniamo solo NaN, altro, mesorettali, rettali_superiori, otturatori, iliaci
sedi_linfonodi_new = []
for s in data.sedi_linfonodi:
    sedi = s
    sedi_new = set()
    for sede in sedi:
        if sede in ['mesorettali', 'rettali_superiori', 'otturatori']:
            sedi_new.add(sede)
        elif sede in ['iliaci_comuni', 'iliaci_interni', 'iliaci_esterni']:
            sedi_new.add('iliaci')
        else:
            sedi_new.add('altro')
    sedi_linfonodi_new.append(list(sedi_new))
data['sedi_linfonodi'] = sedi_linfonodi_new

# Coinvolgimento fascia mesorettale. Trasformiamo rischio in si
data.loc[data['coinvolgimento_fascia_mesorettale'] == 'rischio', 'coinvolgimento_fascia_mesorettale'] = 'si'

# Coinvolgimento riflessione peritoneale. Trasformiamo rischio in si
data.loc[data['coinvolgimento_riflessione_peritoneale'] == 'rischio', 'coinvolgimento_riflessione_peritoneale'] = 'si'

# Infiltrazione sfinteri. Trasformiamo la posizione in si per ottenere una classe (si/no/NaN)
data.loc[data['infiltrazione_sfinteri'] == 'interno_piano', 'infiltrazione_sfinteri'] = 'si'
data.loc[data['infiltrazione_sfinteri'] == 'interno', 'infiltrazione_sfinteri'] = 'si'
data.loc[data['infiltrazione_sfinteri'] == 'interno_piano_esterno', 'infiltrazione_sfinteri'] = 'si'

# Emvi. Trasformiamo sospetto in si
data.loc[data['emvi_esteso'] == 'sospetto', 'emvi_esteso'] = 'si'

# Depositi tumorali. Trasformiamo sospetto in si
data.loc[data['depositi_tumorali'] == 'sospetto', 'depositi_tumorali'] = 'si'


########
# PLOT 1
########
data_plot = data.fillna('NaN')
columns_plot = ['morfologia', 'infiltrazione_tessuto_adiposo', 'coinvolgimento_fascia_mesorettale',
                'riflessione_peritoneale_anteriore',
                'coinvolgimento_riflessione_peritoneale',
                'stadio_T', 'stadio_N', 'stadio_N1c',
                'mrf', 'emvi', 'metastasi',
                'infiltrazione_sfinteri', 'infiltrazione_organi_extra',
                'linfonodi_sospetti', 'numero_linfonodi_non_conosciuto',
                'depositi_tumorali', 'numero_depositi', 'emvi_esteso']
n_columns = 3
n_rows, r = divmod(len(columns_plot), n_columns)
if r != 0:
    n_rows += 1
fig, axes = plt.subplots(n_rows, n_columns, figsize=(n_columns * 7, n_rows * 3))
fig.suptitle("Count values", fontsize='xx-large')
for i, col in enumerate(columns_plot):
    ax = axes[i // n_columns][i % n_columns]
    sns.countplot(data=data_plot, x=col, order=data_plot[col].value_counts().index, ax=ax)
    # Add values on top of bars
    for p in ax.patches:
        y_text = int(p.get_height())
        x_text = p.get_x() + p.get_width() / 2
        ax.text(x=x_text, y=y_text, s=f'{y_text}', ha='center', va='bottom')
plt.tight_layout()


########
# PLOT 2
########
columns = ['posizione', 'sedi_locoregionali', 'sedi_non_locoregionali',
           'sedi_linfonodi', 'infiltrazione_organi_dettagli']
possible_values = {col: ['NaN'] for col in columns}
for col in columns:
    for s in data_plot[col].value_counts().index:
        if type(s) is list:
            possible_values[col] += s
        else:
            possible_values[col] += ast.literal_eval(s)
    possible_values[col] = list(set(possible_values[col]))
counts = {
    col: {val: 0 for val in possible_values[col]}
    for col in columns
}
for col in columns:
    for s in data_plot[col]:
        if type(s) is list:
            value_list = s
        else:
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


# Save data
if True:
    data.to_csv(base_dir / "data" / "tumoreprimitivo_clean.csv", index=False)
    