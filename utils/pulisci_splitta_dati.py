import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split


# Get data
data_path = Path('../data/base.tumoreprimitivo.csv/')
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