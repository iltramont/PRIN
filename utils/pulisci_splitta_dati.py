import pandas as pd
import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


TEST_SIZE = 0.2
VALIDATION_SIZE = 0.05
RANDOM_STATE = 2025
STRATIFY_COLUMNS = (
    'morfologia',
    'infiltrazione_sfinteri'
)

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


# Isola righe che hanno valori di colonne appartenenti a classi poco numerose (minori di 2)
lesioni_ossee_si = data_clean[data_clean.lesioni_ossee == 'si']
"""
infiltrazione_sfinteri_interno = data_clean[data_clean.infiltrazione_sfinteri == 'interno']
infiltrazione_sfinteri_interno_piano_esterno = data_clean[data_clean.infiltrazione_sfinteri == 'interno_piano_esterno']
coinvolgimento_riflessione_peritoneale_rischio = data_clean[data_clean.coinvolgimento_riflessione_peritoneale == 'rischio']
coinvolgimento_riflessione_peritoneale_NAN = data_clean[data_clean.coinvolgimento_riflessione_peritoneale.fillna('NaN') == 'NaN']
coinvolgimento_fascia_mesorettale_NaN = data_clean[data_clean.coinvolgimento_fascia_mesorettale.fillna('NaN') == 'NaN']
depositi_tumorali_sospetto = data_clean[data_clean.depositi_tumorali == 'sospetto']
mrf_NaN = data_clean[data_clean.mrf.fillna('NaN') == 'NaN']
stadio_N_NaN = data_clean[data_clean.stadio_N.fillna('NaN') == 'NaN']
stadio_N_N1c = data_clean[data_clean.stadio_N == 'N1c']
"""
print(len(lesioni_ossee_si))
"""
print(len(infiltrazione_sfinteri_interno))
print(len(infiltrazione_sfinteri_interno_piano_esterno))
print(len(coinvolgimento_riflessione_peritoneale_rischio))
print(len(coinvolgimento_riflessione_peritoneale_NAN))
print(len(coinvolgimento_fascia_mesorettale_NaN))
print(len(depositi_tumorali_sospetto))
print(len(mrf_NaN))
print(len(stadio_N_NaN))
print(len(stadio_N_N1c))"""
print(lesioni_ossee_si.index)
"""
print(infiltrazione_sfinteri_interno.index)
print(infiltrazione_sfinteri_interno_piano_esterno.index)
print(coinvolgimento_riflessione_peritoneale_rischio.index)
print(coinvolgimento_riflessione_peritoneale_NAN.index)
print(coinvolgimento_fascia_mesorettale_NaN.index)
print(depositi_tumorali_sospetto.index)
print(mrf_NaN.index)
print(stadio_N_NaN.index)
print(stadio_N_N1c.index)"""

# Crea split di dati
"""train_split, test_split = train_test_split(data_clean,
                                           test_size=TEST_SIZE,
                                           random_state=RANDOM_STATE,
                                           stratify=data_clean[list(STRATIFY_COLUMNS)].fillna('NaN')
                                           )

print(train_split.morfologia.value_counts())
print(test_split.morfologia.value_counts())
"""
print(data_clean.iloc[127].mrf is np.nan)
