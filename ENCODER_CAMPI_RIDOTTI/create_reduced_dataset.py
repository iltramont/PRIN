from pathlib import Path

from model_utils import (create_label_to_id_map,
                         labels_to_bits,
                         bits_to_labels,
                         get_regression_fields,
                         get_multiple_choice_fields,
                         get_binary_classification_fields,
                         get_classification_fields)


from constants import (AnnotationsReduced,
                       StadioT,
                       RiflessionePeritonealeAnteriore,
                       InfiltrazioneTessutoAdiposo,
                       Morfologia,
                       NAN_VALUE)

import pandas as pd
from ast import literal_eval
import seaborn as sns
import matplotlib.pyplot as plt
import math

###########
# Parametri
###########
DATA_FILE_NAME = "tumoreprimitivo_clean.csv"
"""
TRAIN_FILE_NAME = "train_split.csv"
VALIDATION_FILE_NAME = "validation_split.csv"
TEST_FILE_NAME = "test_split.csv"
"""
#############
# Preliminari
#############
SEED = 2026
base_dir = Path(__file__).parent.parent

###########
# Load data
###########
data = pd.read_csv(base_dir / "data" / DATA_FILE_NAME)
# Get model fields
reg_fields = get_regression_fields(AnnotationsReduced)
cl_fields = get_classification_fields(AnnotationsReduced)
mc_fields = get_multiple_choice_fields(AnnotationsReduced)
bc_fields = get_binary_classification_fields(AnnotationsReduced)
print("Campi di regressione:", reg_fields)
print("Campi di classificazione:", cl_fields)
print("Campi di multi-scelta:", mc_fields)
print("Campi binari:", bc_fields)    
# Funzione per il parsing dei campi multi-scelta
def parse_mc_field(value: str) -> list[str] | None:
    return literal_eval(value)

for field in mc_fields:
    data[field] = data[field].apply(parse_mc_field)
# Check correct type (list)
print(data[mc_fields].head())

# Fill nan values with NAN_VALUE for easier processing
data = data.fillna(NAN_VALUE)


###################
# Class aggregation
###################
# Morfologia
data['morfologia'].replace(
    {
        NAN_VALUE: Morfologia.Altro.value,
        'mucinoso': Morfologia.Altro.value
    },
    inplace=True
)

# Riflessione peritoneale anteriore
data['riflessione_peritoneale_anteriore'].replace(
    {
        'sopra': RiflessionePeritonealeAnteriore.NaN.value,
        NAN_VALUE: RiflessionePeritonealeAnteriore.NaN.value
    },
    inplace=True
)

# Infiltrazione tessuto adiposo
data['infiltrazione_tessuto_adiposo'].replace(
    {
        NAN_VALUE: InfiltrazioneTessutoAdiposo.No.value,
        'sospetto': InfiltrazioneTessutoAdiposo.Si5mm.value
    },
    inplace=True
)

# Infiltrazione sfinteri
data['infiltrazione_sfinteri'].replace(
    {
        NAN_VALUE: False,
        'no': False,
        'si': True
    },
    inplace=True
)

# Infiltrazione organi extra
data['infiltrazione_organi_extra'].replace(
    {
        NAN_VALUE: False,
        'no': False,
        'sospetto': True,
        'si': True
    },
    inplace=True
)

# Coinvolgimento riflessione peritoneale
data['coinvolgimento_riflessione_peritoneale'].replace(
    {
        NAN_VALUE: False,
        'no': False,
        'si': True
    },
    inplace=True
)

# Coinvolgimento fascia mesorettale
data['coinvolgimento_fascia_mesorettale'].replace(
    {
        NAN_VALUE: False,
        'no': False,
        'si': True
    },
    inplace=True
)

# Depositi tumorali
data['depositi_tumorali'].replace(
    {
        NAN_VALUE: False,
        'no': False,
        'si': True
    },
    inplace=True
)

# EMVI esteso
data['emvi_esteso'].replace(
    {
        NAN_VALUE: False,
        'no': False,
        'si': True
    },
    inplace=True
)

# Stadio T
data['stadio_T'].replace(
    {   
        NAN_VALUE: StadioT.T1_2_NaN.value,
        'T1-2': StadioT.T1_2_NaN.value,
        'T4a': StadioT.T4.value,
        'T4b': StadioT.T4.value
    },
    inplace=True
)

# Stadio N
data['stadio_N'].replace(
    {
        NAN_VALUE: False,
        'N0': False,
        'N+': True,
        'N1a': True,
        'N1b': True,
        'N1c': True,
        'N2a': True,
        'N2b': True
    },
    inplace=True
)

# MRF
data['mrf'].replace(
    {
        NAN_VALUE: False,
        '-': False,
        '+': True
    },
    inplace=True
)

# EMVI
data['emvi'].replace(
    {
        NAN_VALUE: False,
        '-': False,
        '+': True
    },
    inplace=True
)

# Metastasi
data['metastasi'].replace(
    {
        NAN_VALUE: False,
        'MX': False,
        'M1': True
    },
    inplace=True
)

######################
# Check processed data
######################
plt.style.use('ggplot')
fields = ['profile'] + cl_fields + bc_fields
n_cols = 4
n_rows = math.ceil(len(fields) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 7*n_rows))
axes = axes.reshape(n_rows, n_cols)

df = []
for i, field in enumerate(fields):
    # PLot counts
    r = i // n_cols
    c = i % n_cols
    ax = axes[r, c]
    sns.countplot(x=field, data=data.fillna('NaN'),
                  ax=ax, order=data.fillna('NaN')[field].value_counts().index)
    ax.set_title(field)
    ax.set_xlabel("")
    ax.set_ylim(0, data.shape[0])
    for p in ax.patches:
        y_text = int(p.get_height())
        x_text = p.get_x() + p.get_width() / 2
        ax.text(x=x_text, y=y_text, s=f'{y_text}', ha='center', va='bottom')

for idx in range(len(fields), n_rows*n_cols):
    r = idx // n_cols
    c = idx % n_cols
    axes[r, c].axis("off")

plt.show()


# Save processed dataset
output_file = base_dir / "data" / "tumoreprimitivo_reduced_clean.csv"
data.to_csv(output_file, index=False)