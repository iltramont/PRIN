import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import ast

import constants


############
# Parameters
############


###############
# Preliminaries
###############
# Set base dir
base_dir = Path(__file__).parent.parent
# Set plot style and colors
plt.style.use('ggplot')
hls_palette = sns.color_palette('hls')
finomnia_palette = sns.color_palette(('#2659ab',   # Blue
                                      '#db038a',   # Pink
                                      '#45c9f5',   # Ligth blue
                                      '#66218a',   # Violet
                                      '#081c36'))  # Dark blue
sns.set_palette(finomnia_palette)

##############
# Get raw data
##############
data = pd.read_csv(base_dir / "data" / constants.RAW_DATA_FILE_NAME)
data.sort_values(by='id', inplace=True, ignore_index=True)

###########
# Posizione
###########
# delete column "posizione" as it is obsolete, renaming "posizione_multiple" in "posizione"
data.drop(columns=['posizione'], inplace=True)
data.rename(columns={'posizione_multiple': 'posizione'}, inplace=True)


###################
# Not clear reports
###################
# Referti poco chiari. Sono quelli che hanno la stringa "escludere". Anche il referto 281 è da escludere.
for i, r in data[data['interpretazioni'].str.contains("escludere", case=False, na=False)].iterrows():
    print(f"*--- id referto: {r['id']}\ninterpretazione: {r['interpretazioni']}\n{100*'-'}")
    
r = data[data['id'] == 281].iloc[0]
print(f"*--- id referto: {r['id']}\ninterpretazione: {r['interpretazioni']}\n{100*'-'}")
rows_to_drop = data[data['id'].isin([171, 259, 281])]
data = data.drop(index=rows_to_drop.index)
data.reset_index(inplace=True, drop=True)


###################
# Delete duplicates
###################
# Rimuovi righe duplicate
print(f'Shape iniziale: {data.shape}')
# Intere righe duplicate. Keep last perchè i report hanno id crescente
duplicati = data.iloc[:, 1:].duplicated(keep='last')
print(f'Numero righe duplicate: {duplicati.sum()}')
data_clean = data[duplicati == False]
print('Righe doppie eliminate')

# Rimuovi righe con report duplicati
duplicati = data_clean[constants.REPORT_COLUMN_NAME].duplicated(keep='last')
righe_report_duplicato = data_clean[duplicati]
print(f'Numero righe con stesso referto: {duplicati.sum()}')
data_clean = data_clean[duplicati == False]
print('Righe eliminate')
data_clean.reset_index(inplace=True, drop=True)
print(f'{data_clean.shape = }')

# Cambia indice al dataset
data_clean.set_index('id', inplace=True)

#########################
# Consistency adjustments
#########################
n = 'linfonodi_sospetti'
sn = 'stadio_N'
flag_n = 'numero_linfonodi_non_conosciuto'
loco = 'sedi_locoregionali'
non_loco = 'sedi_non_locoregionali'
fascia = 'coinvolgimento_fascia_mesorettale'
mrf = 'mrf'
spessore = 'spessore_parietale'
estensione_cc= 'estensione_cranio_caudale'
ofine = 'ore_fine'
oinizio = 'ore_inizio'
oai = 'distanza_oai'
M = 'metastasi'
infiltrazione_tessuto = 'infiltrazione_tessuto_adiposo'
pos = 'posizione'
coin_peritoneo = 'coinvolgimento_riflessione_peritoneale'
relaz = 'riflessione_peritoneale_anteriore'
sfinteri = 'infiltrazione_sfinteri'

# --- Posizione --- #
data_clean.loc[44, 'posizione'] = "['basso', 'medio']"
data_clean.loc[120, 'posizione'] = "['medio', 'alto']" # Prima era ['medio']
data_clean.loc[126, 'posizione'] = "['alto', 'giunzione']" # Prima era ['giunzione']
data_clean.loc[140, 'posizione'] = "['basso', 'medio']" # Prima era ['basso']
data_clean.loc[184, 'posizione'] = "['medio', 'alto']" # Prima era ['alto']
data_clean.loc[186, 'posizione'] = "['medio', 'alto']" # Prima era ['alto']
data_clean.loc[204, 'posizione'] = "['basso', 'medio']" # Prima era ['medio']
data_clean.loc[209, 'posizione'] = "['medio', 'alto']" # Prima era ['medio']
data_clean.loc[221, 'posizione'] = "['medio', 'alto']" # Prima era ['medio']
data_clean.loc[316, 'posizione'] = "['basso', 'medio', 'alto']" # Prima era ['basso']
data_clean.loc[326, 'posizione'] = "['medio', 'alto']" # Prima era ['medio']

data_clean.loc[53, 'posizione'] = "['medio', 'alto']" # Prima era ['alto']
data_clean.loc[56, 'posizione'] = "['medio']" # Prima era ['basso']
data_clean.loc[78, 'posizione'] = "['basso', 'medio', 'alto']" # Prima era ['medio']
data_clean.loc[137, 'posizione'] = "['medio', 'alto', 'giunzione']" # Prima era ['giunzione']
data_clean.loc[180, 'posizione'] = "['medio', 'alto']" # Prima era ['alto']
data_clean.loc[183, 'posizione'] = "['basso', 'medio']" # Prima era ['medio']
data_clean.loc[189, 'posizione'] = "['alto', 'giunzione']" # Prima era ['alto'] # Controllare meglio
data_clean.loc[202, 'posizione'] = "['medio', 'alto']" # Prima era ['medio']
data_clean.loc[205, 'posizione'] = "['medio', 'alto']" # Prima era ['alto']
data_clean.loc[211, 'posizione'] = "['medio', 'alto']" # Prima era ['alto']
data_clean.loc[212, 'posizione'] = "['medio', 'alto']" # Prima era ['medio']
data_clean.loc[217, 'posizione'] = "['medio', 'alto']" # Prima era ['medio']
#### incongruuenza in referto data_clean.loc[220, 'posizione'] = "['basso', 'medio']" # Prima era ['medio']

data_clean.loc[77, 'posizione'] = "['medio', 'alto']" # Prima era ['medio']
data_clean.loc[127, 'posizione'] = "['medio', 'alto']" # Prima era ['alto']
data_clean.loc[128, 'posizione'] = "['medio', 'alto']" # Prima era ['medio']
data_clean.loc[134, 'posizione'] = "['basso', 'medio', 'alto']" # Prima era ['basso'] # Da controllare bene ma mi sembra giusto scrivere basso medio e alto
data_clean.loc[139, 'posizione'] = "['basso', 'medio']" # Prima era ['basso']
data_clean.loc[144, 'posizione'] = "['basso', 'medio']" # Prima era ['basso']
data_clean.loc[177, 'posizione'] = "['alto', 'giunzione']" # Prima era ['giunzione']
data_clean.loc[192, 'posizione'] = "['basso', 'medio']" # Prima era ['medio']
data_clean.loc[198, 'posizione'] = "['basso', 'medio']" # Prima era ['medio']
data_clean.loc[200, 'posizione'] = "['medio', 'alto']" # Prima era ['alto']
data_clean.loc[206, 'posizione'] = "['medio', 'alto']" # Prima era ['medio']
data_clean.loc[219, 'posizione'] = "['medio', 'alto']" # Prima era ['alto']
############## Secondo blocco
data_clean.loc[
    57, ['distanza_oai', 'linfonodi_sospetti', 'sedi_locoregionali']
    ] = [0, 3, "['mesorettali', 'otturatori']"]

data_clean.loc[
    106, ['distanza_oai', 'linfonodi_sospetti', 'sedi_locoregionali', 'sedi_non_locoregionali', 'metastasi']
    ] = [0, 0, "[]", "[]", "MX"]

data_clean.loc[160, ['distanza_oai', 'linfonodi_sospetti']] = [0, 3]

data_clean.loc[163, ['morfologia', 'ore_inizio', 'ore_fine']] = ['solido_polipoide', 6, 9]

data_clean.loc[
    185, ['distanza_oai', 'linfonodi_sospetti', 'numero_linfonodi_non_conosciuto']
    ] = [70, 4, False]

data_clean.loc[
    208, ['distanza_oai', 'linfonodi_sospetti', 'estensione_cranio_caudale']
    ] = [0, 2, 60]

data_clean.loc[232, ['ore_inizio', 'ore_fine']] = [8, 1]

data_clean.loc[234, ['distanza_oai', 'linfonodi_sospetti']] = [0, 4]

data_clean.loc[
    271, ['mrf', 'coinvolgimento_fascia_mesorettale', 'ore_inizio', 'ore_fine']
    ] = ['-', 'no', 11, 4]

data_clean.loc[
    307, ['distanza_oai', 'ore_inizio', 'ore_fine',
          'linfonodi_sospetti', 'numero_linfonodi_non_conosciuto', 'stadio_N', 'sedi_locoregionali', 'sedi_non_locoregionali']
    ] = [0, 2, 7, 0, False, 'N0', "[]", "[]"]

data_clean.loc[
    310, ['ore_inizio', 'ore_fine', 'linfonodi_sospetti', 'metastasi']
    ] = [12, 12, 9, 'M1']

data_clean.loc[
    313, ['ore_inizio', 'ore_fine', 'linfonodi_sospetti', 'distanza_oai', M]
    ] = [12, 12, 2, 0, 'MX']

data_clean.loc[314, ['ore_inizio', 'ore_fine', 'distanza_oai']] = [8, 1, 0]

data_clean.loc[
    322, ['distanza_oai', 'ore_inizio', 'ore_fine',
          'linfonodi_sospetti', 'numero_linfonodi_non_conosciuto', 'stadio_N']
    ] = [0, 12, 12, 1, True, 'N+']

data_clean.loc[
    323, ['distanza_oai', 'ore_inizio', 'ore_fine',
          'mrf', 'coinvolgimento_fascia_mesorettale', 'metastasi']
    ] = [0, 3, 9, '-', 'no', 'MX']

data_clean.loc[
    335, ['distanza_oai', 'morfologia',
          'linfonodi_sospetti', 'numero_linfonodi_non_conosciuto', 'stadio_N', 'sedi_locoregionali']
    ] = [0, 'mucinoso', 1, True, 'N+', "['sacrali']"]

data_clean.loc[336, ['distanza_oai', 'ore_inizio', 'ore_fine', 'linfonodi_sospetti']] = [0, 6, 12, 5]
data_clean.loc[340, ['distanza_oai', 'ore_inizio', 'ore_fine', 'mrf']] = [0, 12, 12, '-']
data_clean.loc[348, ['distanza_oai', 'spessore_parietale', 'sedi_locoregionali']] = [0, 20, "[]"]
data_clean.loc[354, ['linfonodi_sospetti']] = [4]
data_clean.loc[
    357, ['ore_inizio', 'ore_fine', 'linfonodi_sospetti', 'sedi_locoregionali']
    ] = [12, 12, 4, "['mesorettali', 'rettali_superiori']"]

data_clean.loc[
    359, ['distanza_oai', 'linfonodi_sospetti', 'numero_linfonodi_non_conosciuto', 'stadio_N', 'sedi_locoregionali']
    ] = [0, 0, False, 'N0', "[]"]

data_clean.loc[362, ['distanza_oai', 'linfonodi_sospetti', 'ore_inizio', 'ore_fine']] = [0, 1, 3, 9]
data_clean.loc[366, ['distanza_oai', 'mrf']] = [0, '-']

data_clean.loc[
    384,['distanza_oai', 'mrf', 'emvi', 'emvi_esteso', 'linfonodi_sospetti', 'sedi_locoregionali']
    ] = [0, '-', '-', 'no', 1, "['otturatori']"]

data_clean.loc[387,['sedi_non_locoregionali', 'metastasi']] = ["['iliaci_esterni']", 'M1']

data_clean.loc[
    388,['distanza_oai', 'ore_inizio', 'ore_fine','linfonodi_sospetti',
         'sedi_locoregionali', 'sedi_non_locoregionali', 'numero_linfonodi_non_conosciuto', 'stadio_N']
    ] = [0, 12, 12, 0, "[]", "[]", False, 'N0']

data_clean.loc[394, ['distanza_oai', 'ore_inizio', 'ore_fine']] = [0, 12, 12]
data_clean.loc[399,['distanza_oai', 'linfonodi_sospetti', 'emvi', 'emvi_esteso', M]] = [0, 4, '+', 'si', 'M1']

data_clean.loc[
    411, ['distanza_oai', 'linfonodi_sospetti', 'ore_inizio', 'ore_fine', 'numero_linfonodi_non_conosciuto']
    ] = [0, 2, 12, 9, True]

data_clean.loc[
    418, ['distanza_oai', 'linfonodi_sospetti', 'sedi_locoregionali', 'sedi_non_locoregionali',
         'numero_linfonodi_non_conosciuto', 'stadio_N', 'mrf', 'coinvolgimento_fascia_mesorettale']
    ] = [0, 0, "[]", "[]", False, 'N0', '-', 'no']

data_clean.loc[420, ['distanza_oai', 'linfonodi_sospetti', 'metastasi']] = [0, 1, 'MX']
data_clean.loc[67, [relaz]] = [None]

data_clean.loc[
    125, ['distanza_oai', 'posizione', 'infiltrazione_tessuto_adiposo', 'stadio_T', 'sedi_non_locoregionali',
         'mrf', 'coinvolgimento_fascia_mesorettale', 'sedi_locoregionali', 'linfonodi_sospetti']
    ] = [45, "['basso', 'medio']", 'si_5mm_plus', 'T3cd', "[]",
         '+', 'si', "['mesorettali', 'rettali_superiori']", 4]
    
data_clean.loc[
    166, ['morfologia', 'linfonodi_sospetti', 'numero_linfonodi_non_conosciuto', 'estensione_cranio_caudale']
    ] = ['solido_polipoide', 4, True, 40]

data_clean.loc[
    179, ['distanza_oai', 'ore_inizio', 'ore_fine',
          'sedi_non_locoregionali', 'numero_linfonodi_non_conosciuto', 'stadio_N', 'metastasi']
    ] = [0, 12, 6, "[]", True, 'N+', 'MX']

data_clean.loc[
    187, ['distanza_oai', 'posizione', 'emvi', 'emvi_esteso','linfonodi_sospetti',
          'numero_linfonodi_non_conosciuto', 'stadio_N']
    ] = [0, "['basso', 'medio']", '+', 'si', 4, True, 'N+']

data_clean.loc[213, ['distanza_oai', 'ore_inizio', 'ore_fine']] = [0, 3, 9]

# Terzo blocco
data_clean.loc[120, [spessore]] = [None]
data_clean.loc[120, [n, flag_n, loco, non_loco, sn]] = [0, False, "[]", "[]", 'N0']

data_clean.loc[140, [n, flag_n, loco, sn]] = [4, True, "['mesorettali', 'rettali_superiori']", 'N+']

data_clean.loc[184, [oinizio, ofine, oai, flag_n]] = [12, 12, 60, True]

data_clean.loc[186, [oinizio, ofine, n, flag_n, loco]] = [12, 12, 4, True, "['mesorettali']"]

data_clean.loc[204, [oinizio, ofine]] = [12, 9]
data_clean.loc[204, [n, flag_n, loco, non_loco, sn]] = [0, False, "[]", "[]", 'N0']

data_clean.loc[209, [oinizio, ofine, n]] = [6, 3, 4]

data_clean.loc[221, [oinizio, ofine]] = [12, 12]
data_clean.loc[221, [n, flag_n, loco, non_loco, sn]] = [0, False, "[]", "[]", 'N0']

data_clean.loc[316, [M, n, sn]] = ['M1', 4, 'N+']

data_clean.loc[326, [oinizio, ofine, n, non_loco, M]] = [12, 12, 4, "[]", 'MX']

data_clean.loc[53, [oinizio, ofine, spessore, estensione_cc]] = [12, 12, 27, 65]
data_clean.loc[53, [n, flag_n, loco, non_loco, sn]] = [0, False, "[]", "[]", 'N0']

data_clean.loc[56, [spessore, oai]] = [None, 87]
data_clean.loc[56, [n, flag_n, loco, non_loco, sn]] = [0, False, "[]", "[]", 'N0']

data_clean.loc[78, [oai, fascia, mrf, n, flag_n]] = [None, 'si', '+', 4, True]

data_clean.loc[137, [loco, non_loco, M]] = ["['mesorettali']", "['iliaci_comuni']", 'M1']

data_clean.loc[180, [oinizio, ofine, loco]] = [12, 9, "['mesorettali', 'rettali_superiori']"]

data_clean.loc[189, [spessore, oai, oinizio, ofine]] = [6, 70, 9, 3]
data_clean.loc[189, [n, flag_n, loco, non_loco, sn]] = [0, False, "[]", "[]", 'N0']

data_clean.loc[202, [oinizio, ofine]] = [6, 12]
data_clean.loc[202, [n, flag_n, loco, non_loco, sn]] = [0, False, "[]", "[]", 'N0']

data_clean.loc[211, [oinizio, ofine, estensione_cc]] = [12, 12, 40]
data_clean.loc[211, [n, flag_n, loco, non_loco, sn]] = [4, True, "['mesorettali']", "[]", 'N+']

data_clean.loc[212, [oinizio, ofine, oai]] = [5, 10, 70]
data_clean.loc[212, [n, flag_n, loco, non_loco, sn]] = [0, False, "[]", "[]", 'N0']

data_clean.loc[217, [oinizio, ofine]] = [12, 12]

data_clean.loc[220, [oinizio, ofine, flag_n, infiltrazione_tessuto]] = [2, 7, True, 'si_5mm_plus']

data_clean.loc[77, [fascia, mrf]] = ['no', '-']

data_clean.loc[127, [oinizio, ofine]] = [8, 4]
data_clean.loc[127, [n, flag_n, loco, non_loco, sn]] = [0, False, "[]", "[]", 'N0']

data_clean.loc[128, [oinizio, ofine]] = [3, 12]
data_clean.loc[128, [n, flag_n, sn]] = [4, True, 'N+']

data_clean.loc[134, [oinizio, ofine, n, loco, flag_n]] = [9, 3, 4, "['rettali_superiori']", True]

data_clean.loc[139, [oinizio, ofine, fascia, mrf, non_loco, n, M]] = [5, 1, 'si', '+', "[]", 4, 'MX']

data_clean.loc[144, [n, flag_n, loco, non_loco, sn]] = [0, False, "[]", "[]", 'N0']

data_clean.loc[177, [n, flag_n, sn, M]] = [4, True, 'N+', 'M1']

data_clean.loc[192, [oinizio, ofine]] = [8, 1]

data_clean.loc[198, [oinizio, ofine, n]] = [12, 12, 4]

data_clean.loc[200, [n]] = [4]

data_clean.loc[206, [n, flag_n, loco, non_loco, sn]] = [0, False, "[]", "[]", 'N+']


data_clean.loc[46, [n]] = [4]
data_clean.loc[66, [n, flag_n]] = [4, True]
data_clean.loc[74, [oinizio, ofine]] = [5, 10]
data_clean.loc[87, [oinizio, ofine]] = [9, 6]
data_clean.loc[88, [oinizio, ofine, n, flag_n, mrf, fascia]] = [6, 9, 3, True, '-', 'no']
data_clean.loc[92, [n, flag_n]] = [5, True]
data_clean.loc[98, [n, flag_n]] = [4, True]
data_clean.loc[116, [oinizio, ofine, n, flag_n]] = [4, 11, 4, True]
data_clean.loc[129, [n, flag_n]] = [4, True]
data_clean.loc[130, [pos, flag_n, loco, non_loco, M]] = [
    "['alto', 'giunzione']", True, "['mesorettali', 'rettali_superiori']", "[]", 'MX']
data_clean.loc[131, [n, flag_n, loco, non_loco, sn, estensione_cc, spessore]] = [
    0, False, "[]", "[]", 'N+', 40, None]
data_clean.loc[132, [oinizio, ofine, spessore, sn, loco, non_loco, n, flag_n]] = [
    12, 12, None, 'N+', "['mesorettali', 'rettali_superiori']", "[]", 4, True]
data_clean.loc[178, [oinizio, ofine, estensione_cc, loco, non_loco]] = [
    12, 9, 50, "['mesorettali', 'rettali_superiori']", "[]"]
data_clean.loc[181, [n]] = [6]
data_clean.loc[190, [oinizio, ofine, loco, non_loco, mrf, fascia]] = [
    9, 3, "['mesorettali']", "[]", "-", "no"]

data_clean.loc[193, [oinizio, ofine, oai, mrf, fascia]] = [12, 12, 54, "-", "no"]
data_clean.loc[193, [n, flag_n, loco, non_loco, sn]] = [0, False, "[]", "[]", 'N0']

data_clean.loc[195, [oinizio, ofine, n]] = [12, 12, 4]
data_clean.loc[196, [oinizio, ofine, mrf, fascia]] = [3, 12, "-", "no"]
data_clean.loc[199, [estensione_cc]] = [30]
data_clean.loc[207, [estensione_cc, n]] = [35, 4]
data_clean.loc[210, [estensione_cc, oai, fascia, mrf, 'emvi_esteso', 'emvi']] = [
    38, 100, "no", "-", "si", "+"]
data_clean.loc[214, [oinizio, ofine, M]] = [8, 4, 'MX']
data_clean.loc[227, ['morfologia', coin_peritoneo, fascia, mrf]] = ['solido_polipoide', "no", "no", "-"]
data_clean.loc[229, [oinizio, ofine, coin_peritoneo, fascia, mrf]] = [12, 6, "no", "no", "-"]
data_clean.loc[230, [oinizio, ofine, n, flag_n, M]] = [5, 10, 1, True, 'MX']
data_clean.loc[237, [pos]] = ["['medio', 'alto', 'giunzione']"]
data_clean.loc[242, [oinizio, ofine]] = [8, 1]
data_clean.loc[244, [n]] = [4]
data_clean.loc[250, [n, flag_n]] = [4, True]
data_clean.loc[251, [oinizio, ofine]] = [12, 6]
data_clean.loc[257, [oinizio, ofine, n, flag_n, fascia, mrf]] = [12, 12, 4, True, 'no', '-']
data_clean.loc[258, [n, flag_n]] = [4, True]
data_clean.loc[260, [n, flag_n, sfinteri, fascia, mrf, coin_peritoneo]] = [
    3, False, 'interno', 'no', '-', 'no']
data_clean.loc[262, [oinizio, ofine]] = [9, 3]
data_clean.loc[263, [oinizio, ofine]] = [5, 10]
data_clean.loc[270, [oinizio, ofine, fascia, mrf]] = [9, 3, 'no', '-']
data_clean.loc[273, [oinizio, ofine, fascia, mrf]] = [3, 12, 'no', '-']
data_clean.loc[275, [fascia, mrf]] = ['no', '-']
data_clean.loc[284, [oai, estensione_cc, fascia, mrf]] = [0, 45, 'no', '-']
data_clean.loc[288, [oinizio, ofine]] = [12, 12]
data_clean.loc[294, [oinizio, ofine]] = [8, 1]
data_clean.loc[295, [fascia, mrf]] = ['no', '-']
data_clean.loc[296, [oinizio, ofine]] = [9, 3]
data_clean.loc[298, [fascia, mrf]] = ['no', '-']
data_clean.loc[300, [fascia, mrf]] = ['no', '-']
data_clean.loc[302, [oinizio, ofine]] = [6, 3]
data_clean.loc[305, [fascia, mrf]] = ['no', '-']
data_clean.loc[311, [oinizio, ofine, n, flag_n]] = [3, 12, 4, True]
data_clean.loc[312, [oinizio, ofine]] = [12, 6]
data_clean.loc[315, [oinizio, ofine, fascia, mrf]] = [9, 3, 'no', '-']
data_clean.loc[318, [oinizio, ofine, fascia, mrf, n, flag_n]] = [3, 9, 'no', '-', 1, True]
data_clean.loc[320, [fascia, mrf]] = ['no', '-']

data_clean.loc[324, [oinizio, ofine, fascia, mrf, coin_peritoneo]] = [6, 12, 'no', '-', 'no']
data_clean.loc[324, [n, flag_n, loco, non_loco, sn]] = [0, False, "[]", "[]", 'N0']

data_clean.loc[325, [fascia, mrf]] = ['no', '-']
data_clean.loc[329, [oinizio, ofine, n, flag_n]] = [12, 12, 4, True]

data_clean.loc[331, [oinizio, ofine, fascia, mrf, M]] = [8, 10, 'no', '-', 'MX']
data_clean.loc[331, [n, flag_n, loco, non_loco, sn]] = [0, False, "[]", "[]", 'N0']

data_clean.loc[333, [oinizio, ofine]] = [12, 12]
data_clean.loc[333, [n, flag_n, loco, non_loco, sn]] = [0, False, "[]", "[]", 'N+']

data_clean.loc[334, [n, flag_n, sn]] = [4, True, 'N+']
data_clean.loc[338, [oinizio, ofine, M, non_loco]] = [12, 12, 'MX', "[]"]
data_clean.loc[342, [oinizio, ofine]] = [9, 12]
data_clean.loc[343, [n, flag_n, loco, non_loco]] = [3, False, "['mesorettali', 'iliaci_interni']", "[]"]
data_clean.loc[344, [oinizio, ofine]] = [5, 1]
data_clean.loc[345, [oinizio, ofine, n, flag_n]] = [9, 6, 4, True]
data_clean.loc[346, [oinizio, ofine, n, flag_n, sn]] = [5, 1, 4, True, 'N+']
data_clean.loc[347, [n, flag_n, loco, non_loco, sn, fascia, mrf]] = [0, False, "[]", "[]", 'N0', 'no', '-']
data_clean.loc[349, [n, flag_n, fascia, mrf]] = [1, True, 'no', '-']

data_clean.loc[350, [oinizio, ofine]] = [12, 12]
data_clean.loc[350, [n, flag_n, loco, non_loco, sn]] = [0, False, "[]", "[]", 'N0']

data_clean.loc[351, [oinizio, ofine]] = [12, 6]
data_clean.loc[352, [oinizio, ofine, n, flag_n]] = [6, 3, 1, True]
data_clean.loc[353, [oinizio, ofine, n]] = [3, 6, 1]
data_clean.loc[355, [oinizio, ofine, infiltrazione_tessuto, 'stadio_T', n, flag_n]] = [
    3, 9, 'no', 'T1-2', 2, True]
data_clean.loc[356, [oinizio, ofine, n, sn, fascia, mrf]] = [12, 12, 2, 'N+', 'no', '-']
data_clean.loc[358, [oinizio, ofine, fascia, mrf]] = [5, 7, 'no', '-']
data_clean.loc[360, [oinizio, ofine, 'emvi_esteso', 'emvi', fascia, mrf]] = [
    12, 6, 'no', '-', 'no', '-']
data_clean.loc[361, [oinizio, ofine, n, flag_n, fascia, mrf]] = [12, 12, 4, True, 'no', '-']

data_clean.loc[363, [oinizio, ofine, 'morfologia']] = [2, 7, 'solido_polipoide']
data_clean.loc[363, [n, flag_n, loco, non_loco, sn]] = [0, False, "[]", "[]", 'N0']

data_clean.loc[364, [oinizio, ofine]] = [6, 3]

data_clean.loc[365, [oinizio, ofine, M]] = [12, 12, 'MX']
data_clean.loc[365, [n, flag_n, loco, non_loco, sn]] = [0, False, "[]", "[]", 'N0']

data_clean.loc[367, [oinizio, ofine]] = [12, 12]
data_clean.loc[367, [n, flag_n, loco, non_loco, sn]] = [0, False, "[]", "[]", 'N+']

data_clean.loc[368, [oinizio, ofine, n, flag_n]] = [12, 12, 4, True]
data_clean.loc[369, [fascia, mrf]] = ['no', '-']

data_clean.loc[370, [oinizio, ofine]] = [12, 12]
data_clean.loc[370, [n, flag_n, loco, non_loco, sn]] = [0, False, "[]", "[]", 'N0']

data_clean.loc[371, [oinizio, ofine, n, flag_n, sn]] = [12, 12, 4, True, 'N+']
data_clean.loc[373, [oinizio, ofine, fascia, mrf]] = [9, 3, 'no', '-']
data_clean.loc[374, [oinizio, ofine, n, flag_n]] = [11, 4, 4, True]
data_clean.loc[375, [oinizio, ofine, n, sn, flag_n]] = [12, 12, 4, 'N+', True]
data_clean.loc[376, [fascia, mrf]] = ['no', '-']
data_clean.loc[377, [oinizio, ofine, relaz, n, flag_n]] = [5, 1, 'cavallo', 1, True]
data_clean.loc[379, [oinizio, ofine, n, flag_n]] = [2, 10, 4, True]
data_clean.loc[380, [oinizio, ofine]] = [11, 7]
data_clean.loc[381, [oinizio, ofine, n, flag_n]] = [8, 4, 4, True]

data_clean.loc[386, [oinizio, ofine]] = [12, 9]
data_clean.loc[386, [n, flag_n, loco, non_loco, sn]] = [0, False, "[]", "[]", 'N0']

data_clean.loc[389, [oinizio, ofine, M]] = [5, 10, 'M1']
data_clean.loc[389, [n, flag_n, loco, non_loco, sn]] = [1, False, "[]", "['iliaci_esterni']", 'N+']

data_clean.loc[392, [oinizio, ofine]] = [12, 12]
data_clean.loc[392, [n, flag_n, loco, non_loco, sn]] = [0, False, "[]", "[]", 'N0']

data_clean.loc[393, [n, flag_n, loco, non_loco, sn]] = [0, False, "[]", "[]", 'N0']

data_clean.loc[395, [oinizio, ofine]] = [12, 12]
data_clean.loc[395, [n, flag_n, loco, non_loco, sn]] = [0, False, "[]", "[]", 'N+']

data_clean.loc[396, [oinizio, ofine, n, flag_n]] = [6, 12, 1, False]
data_clean.loc[397, [oinizio, ofine, n, flag_n]] = [3, 12, 4, True]

data_clean.loc[400, [oinizio, ofine, pos]] = [2, 8, "['medio', 'alto', 'giunzione']"]
data_clean.loc[400, [n, flag_n, sn, M]] = [4, True, 'N+', 'M1']

data_clean.loc[401, [n, flag_n, sn, fascia, mrf]] = [1, True, 'N+', 'no', '-']

data_clean.loc[402, [n, flag_n, loco, non_loco, sn, fascia, mrf]] = [
    0, False, "[]", "[]", 'N0', 'no', '-']

data_clean.loc[403, [oinizio, ofine, M]] = [4, 11, 'MX']
data_clean.loc[403, [n, flag_n, loco, non_loco, sn]] = [0, False, "[]", "[]", 'N0']

data_clean.loc[407, [oinizio, ofine, n, sn, flag_n]] = [12, 12, 4, 'N+', False]
data_clean.loc[409, [oinizio, ofine, n, flag_n]] = [3, 9, 4, True]

data_clean.loc[410, [oinizio, ofine]] = [12, 12]
data_clean.loc[410, [n, flag_n, loco, non_loco, sn]] = [0, False, "[]", "[]", 'N+']

data_clean.loc[412, [oinizio, ofine]] = [12, 6]
data_clean.loc[412, [n, flag_n, loco, non_loco, sn]] = [0, False, "[]", "[]", 'N+']

data_clean.loc[413, [oinizio, ofine, n, flag_n, sn, M]] = [12, 12, 4, True, 'N+', 'M1']
data_clean.loc[415, [n, flag_n, sn, M]] = [4, True, 'N+', 'MX']
data_clean.loc[416, [n, flag_n, sn, M, loco, non_loco]] = [
    4, True, 'N+', 'MX', "['otturatori', 'iliaci_interni']", "[]"]
data_clean.loc[419, [M, n, flag_n]] = ['MX', 4, True]
data_clean.loc[421, [oinizio, ofine, estensione_cc, n, loco, non_loco, sn, flag_n, M]] = [
    3, 12, 68, 3, "['mesorettali']", "[]", 'N+', False, 'MX']
data_clean.loc[422, [oinizio, ofine, n, flag_n, sn]] = [12, 12, 6, True, 'N+']
data_clean.loc[75, [oinizio, ofine]] = [5, 7]
data_clean.loc[79, [oinizio, ofine]] = [5, 1]
data_clean.loc[100, [oinizio, ofine, n, flag_n]] = [2, 10, 4, True]
data_clean.loc[104, ['morfologia']] = ['solido_polipoide']
data_clean.loc[115, [n, flag_n]] = [4, True]
data_clean.loc[135, [oinizio, ofine, pos, n, flag_n]] = [
    12, 12, "['medio', 'alto', 'giunzione']", 4, True]
data_clean.loc[136, [n, flag_n]] = [4, True]

data_clean.loc[141, [oinizio, ofine, fascia, mrf, M]] = [6, 9, 'no', '-', 'MX']
data_clean.loc[141, [n, flag_n, loco, non_loco, sn]] = [0, False, "[]", "[]", 'N0']

data_clean.loc[142, [oinizio, ofine, fascia, mrf, n, flag_n]] = [6, 3, 'si', '+', 4, True]
data_clean.loc[162, [n, flag_n]] = [4, True]
data_clean.loc[170, [n, flag_n, relaz, fascia, mrf]] = [3, True, 'cavallo', 'no', '-']
data_clean.loc[173, [n, flag_n]] = [4, True]
data_clean.loc[175, [n, flag_n]] = [4, True]
data_clean.loc[203, [oinizio, ofine, pos, estensione_cc, fascia, mrf]] = [
    12, 12, "['medio', 'alto']", 55, 'si', '+']
data_clean.loc[215, [oinizio, ofine, infiltrazione_tessuto, coin_peritoneo, 'stadio_T', M]] = [
    6, 9, 'si_5mm', 'no', 'T3ab', 'MX']
data_clean.loc[226, [n, flag_n]] = [10, True]
data_clean.loc[228, [oinizio, ofine]] = [12, 12]
data_clean.loc[238, [n, flag_n]] = [4, True]
data_clean.loc[245, [n, flag_n, spessore]] = [1, True, 20]
data_clean.loc[247, [oinizio, ofine, n, flag_n]] = [11, 7, 4, True]
data_clean.loc[255, [oinizio, ofine]] = [12, 12]
data_clean.loc[261, [fascia, mrf]] = ['no', '-']
data_clean.loc[269, [oinizio, ofine]] = [5, 10]
data_clean.loc[272, [fascia, mrf]] = ['no', '-']
data_clean.loc[283, [oinizio, ofine]] = [12, 6]
data_clean.loc[293, [oinizio, ofine]] = [2, 7]
data_clean.loc[299, [oinizio, ofine, loco, non_loco, n, flag_n]] = [8, 4, "['mesorettali']", "[]", 4, True]
data_clean.loc[304, [oinizio, ofine, n, flag_n, fascia, mrf]] = [12, 9, 6, True, 'no', '-']
data_clean.loc[308, [oinizio, ofine, n, flag_n]] = [12, 12, 1, True]
data_clean.loc[332, [n, flag_n]] = [4, True]
data_clean.loc[372, [oinizio, ofine, n, flag_n, sn]] = [12, 12, 1, True, 'N+']
data_clean.loc[52, [oinizio, ofine, M]] = [5, 10, 'MX']
data_clean.loc[62, [estensione_cc]] = [29]
data_clean.loc[63, [oinizio, ofine]] = [1, 5]
data_clean.loc[71, [n, flag_n, loco, non_loco, sn]] = [0, False, "[]", "[]", 'N0']

data_clean.loc[72, [oinizio, ofine]] = [12, 12]
data_clean.loc[72, [n, flag_n, loco, non_loco, sn]] = [0, False, "[]", "[]", 'N+']

data_clean.loc[73, [oinizio, ofine]] = [5, 1]
data_clean.loc[81, [oinizio, ofine, n, sn, flag_n]] = [10, 2, 1, 'N+', False]
data_clean.loc[83, [n, sn, flag_n]] = [4, 'N+', True]
data_clean.loc[91, [oinizio, ofine]] = [3, 9]
data_clean.loc[94, [oinizio, ofine]] = [5, 10]
data_clean.loc[95, [oinizio, ofine, estensione_cc, oai]] = [12, 6, 20, 0]
data_clean.loc[101, [oinizio, ofine, fascia, mrf]] = [9, 3, 'no', '-']
data_clean.loc[103, [oinizio, ofine, fascia, mrf]] = [2, 7, 'no', '-']
data_clean.loc[105, [oinizio, ofine, n, flag_n, loco, non_loco, sn, fascia, mrf]] = [
    11, 7, 1, False, "['otturatori']", "[]", 'N+', 'no', '-']

data_clean.loc[107, [oinizio, ofine, spessore, fascia, mrf]] = [None, None, None, 'no', '-']
data_clean.loc[107, [n, flag_n, loco, non_loco, sn]] = [0, False, "[]", "[]", 'N0']

data_clean.loc[117, [n, flag_n, fascia, mrf]] = [4, True, 'no', '-']
data_clean.loc[118, [oinizio, ofine, n, flag_n]] = [9, 6, 4, True]

data_clean.loc[133, [oinizio, ofine]] = [2, 8]
data_clean.loc[133, [n, flag_n, loco, non_loco, sn]] = [0, False, "[]", "[]", 'N0']

data_clean.loc[138, [oinizio, ofine, fascia, mrf]] = [5, 1, 'no', '-']

data_clean.loc[143, [oinizio, ofine, oai, n, flag_n, loco, non_loco, M, fascia, mrf]] = [
    12, 12, 100, 4, True, "['mesorettali']", "[]", 'MX', 'no', '-']
data_clean.loc[157, [oinizio, ofine, n, flag_n]] = [5, 7, 4, True]
data_clean.loc[159, ['morfologia', n, flag_n, pos]] = ["solido_polipoide", 13, True, "['basso', 'alto']"]
data_clean.loc[164, [oinizio, ofine, fascia, mrf]] = [9, 12, 'no', '-']
data_clean.loc[168, [oinizio, ofine, n , flag_n]] = [6, 3, 4, True]
data_clean.loc[169, [oinizio, ofine, fascia, mrf]] = [9, 3, 'no', '-']
data_clean.loc[182, [oinizio, ofine]] = [11, 7]
data_clean.loc[197, [n, flag_n]] = [6, True]
data_clean.loc[201, [oai]] = [None]
data_clean.loc[203, [n, flag_n, fascia, mrf]] = [5, True, 'no', '-']
data_clean.loc[218, [oinizio, ofine, non_loco, M, fascia, mrf]] = [9, 3, "[]", 'MX', 'no', '-']
data_clean.loc[219, [oinizio, ofine, pos, flag_n]] = [12, 12, "['medio', 'alto']", True]
data_clean.loc[233, [oinizio, ofine]] = [8, 4]
data_clean.loc[240, [oinizio, ofine, n, flag_n]] = [11, 4, 4, True]
data_clean.loc[249, [oinizio, ofine, n, loco, sn, flag_n]] = [
    8, 1, 4, "['mesorettali', 'iliaci_interni']", "N+", False]
data_clean.loc[252, [n, flag_n, fascia, mrf, oai]] = [4, True, 'no', '-', None]
data_clean.loc[317, [oinizio, ofine]] = [9, 3]
data_clean.loc[330, ['morfologia', n, flag_n]] = ['solido_polipoide', 4, True]
data_clean.loc[382, [n, flag_n, fascia, mrf]] = [2, False, 'no', '-']
data_clean.loc[383, [oinizio, ofine]] = [12, 12]
data_clean.loc[121, [fascia, mrf]] = ['no', '-']


# --- Stadio T --- #
# Correggiamo l'anomalia (Corretta con intervento di Ilaria)
"""
data_clean.loc[
    (data_clean['stadio_T'].isna()) &
    (data_clean['coinvolgimento_riflessione_peritoneale'] == 'rischio')
    , 'stadio_T'
    ] = 'T4a'

data_clean.loc[
    (data_clean['stadio_T'].isna()) &
    (data_clean['infiltrazione_tessuto_adiposo'] == 'si_5mm_plus')
    , 'stadio_T'
    ] = 'T3cd'

data_clean.loc[
    (data_clean['stadio_T'].isna()) &
    (data_clean['infiltrazione_tessuto_adiposo'] == 'si_5mm')
    , 'stadio_T'
    ] = 'T3ab'

data_clean.loc[
    (data_clean['stadio_T'].isna()) &
    (data_clean['infiltrazione_tessuto_adiposo'] == 'no')
    , 'stadio_T'
    ] = 'T1-2'
"""

# --- Stadio N and lymph nodes --- #
# Stadio N valorizzato con "N1c". Anomalo solo il referto 44.
data_clean.loc[44, 'stadio_N1c'] = True
data_clean.loc[44, 'stadio_N'] = 'N+'
"""
# Adjusted linfonodi sospetti, stadio N and numero linfonodi non conosciuto. DO NOT CHANGE ORDER!
data_clean.loc[
    (data_clean['linfonodi_sospetti'].isna()) &
    (data_clean['stadio_N'] == 'N0') & 
    (data_clean['sedi_locoregionali'] == '[]') &
    (data_clean['sedi_non_locoregionali'] == '[]')
    , 'linfonodi_sospetti'
    ] = 0

data_clean.loc[
    (data_clean['linfonodi_sospetti'].isna()) &
    (data_clean['stadio_N'] == 'N0')
    , 'stadio_N'
    ] = 'N+'

data_clean.loc[
    (data_clean['linfonodi_sospetti'].isna()) &
    (data_clean['stadio_N'] != 'N0')
    , 'numero_linfonodi_non_conosciuto'
    ] = True

data_clean.loc[
    (data_clean['linfonodi_sospetti'].isna()) &
    (data_clean['stadio_N'] != 'N0')
    , 'linfonodi_sospetti'
    ] = 0
"""

# Anomalia. Se i linfonodi sospetti sono 0, non posso avere le sedi valorizzate.
# Dando un'occhiata ai referti si osserva come il modo corretto per risolvere l'anomalia è quello di valorizzare a True il campo numero_linfonodi_non_conosciuto.
data_clean.loc[
    (data_clean['linfonodi_sospetti'] == 0) &
    (data_clean['numero_linfonodi_non_conosciuto'] == False) &
    ((data_clean['sedi_locoregionali'] != '[]') | (data_clean['sedi_non_locoregionali'] != '[]'))
    , 'stadio_N'
    ] = constants.StadioN.N_PLUS.value

data_clean.loc[
    (data_clean['linfonodi_sospetti'] == 0) &
    (data_clean['numero_linfonodi_non_conosciuto'] == False) &
    ((data_clean['sedi_locoregionali'] != '[]') | (data_clean['sedi_non_locoregionali'] != '[]'))
    , 'numero_linfonodi_non_conosciuto'
    ] = True

# --- EMVI --- #
data_clean.loc[
    (data_clean['emvi'] == '+') &
    (data_clean['emvi_esteso'] == 'no')
    , 'emvi'
    ] = '-'

data_clean.loc[
    (data_clean['emvi'].isna()) &
    (data_clean['emvi_esteso'] == 'sospetto')
    , 'emvi'
    ] = '+'

# --- MRF --- #
data_clean.loc[
    (data_clean['coinvolgimento_fascia_mesorettale'].isna())
    , ['mrf', 'coinvolgimento_fascia_mesorettale']
    ] = ['-', 'no'] 

# --- Deposits number --- #
data_clean.loc[data_clean['numero_depositi'] == -1.0, 'numero_depositi'] = 0.0
           
# --- Extra organs involved --- #
# Sostituiamo i valore con None nella colonna dei dettagli degli organi coinvolti pechè altri valori non sono compatibili
# con "no" nella colonna infiltrazione_organi_extra.
data_clean.loc[
    (data_clean['infiltrazione_organi_dettagli'] > '') &
    (data_clean['infiltrazione_organi_extra'] == 'no')
    , 'infiltrazione_organi_dettagli'
    ] = None

# --- Metastasi --- #
data_clean.loc[
    ((data_clean['metastasi'] == 'MX') | (data_clean['metastasi'].isna())) &
    ((data_clean['lesioni_ossee'] == 'si') | (data_clean['sedi_non_locoregionali'] != '[]'))
    , 'metastasi'] = 'M1'

# --- Stadio N+ per i depositi --- #
data_clean.loc[(data_clean['depositi_tumorali'] == 'si', 'stadio_N')] = 'N+'

# --- Binary N stage --- #
data_clean.loc[(data_clean['stadio_N'] != 'N0', 'stadio_N')] = 'N+'

#############################
# New column "sedi_linfonodi"
#############################
sedi_linfonodi = []
for s1, s2 in zip(data_clean.sedi_locoregionali, data_clean.sedi_non_locoregionali):
    sedi_linfonodi.append(str(ast.literal_eval(s1) + ast.literal_eval(s2)))
data_clean['sedi_linfonodi'] = sedi_linfonodi
print(f'Nuova colonna "sedi_linfonodi" creata\n{data_clean.shape = }')


###############################
# Cleaning column "Radiologist"
###############################
def sub_radiologist_name(x: str) -> str:
    if not isinstance(x, str):
        return x
    if 'barbaro' in x.lower():
        return 'Barbaro Brunella'
    if 'brizi' in x.lower():
        return 'Brizi Maria Gabriella'
    if 'de gaetano' in x.lower():
        return 'De Gaetano Anna Maria'
    if 'macis' in x.lower():
        return 'Macis Giuseppe'
    if 'genco' in x.lower():
        return 'Genco Enza'
    if 'rodolfino' in x.lower():
        return 'Rodolfino Elena'
    if 'manfredi' in x.lower():
        return 'Manfredi Riccardo'
    if 'minordi' in x.lower():
        return 'Minordi Laura Maria'
    if 'avesani' in x.lower():
        return 'Avesani Giacomo'
    if 'panico' in x.lower():
        return 'Panico Camilla'
    if 'farchione' in x.lower():
        return 'Farchione Alessandra'
    if 'micco' in x.lower():
        return "Micco' Maura"
    if 'gui' in x.lower() and 'benedetta' in x.lower():
        return 'Gui Benedetta'
    if 'ambra' in x.lower() and 'giulia' in x.lower():
        return "D'Ambra Giulia"
    if 'bonomo' in x.lower() and 'lorenzo' in x.lower():
        return "Bonomo Lorenzo"
    if 'natale' in x.lower() and 'luigi' in x.lower():
        return "Natale Luigi"
    if 'margo' in x.lower():
        return "Margo' Di Marco"
    return x

data_clean['radiologist'] = data_clean['radiologist'].apply(sub_radiologist_name)


#####################
# Columns aggregation
#####################
# Dettagli infiltrazione organi
infiltrazione_organi_dettagli_new = []
for s in data_clean.infiltrazione_organi_dettagli.fillna('NaN'):
    dettagli = []
    if s == 'NaN':
        infiltrazione_organi_dettagli_new.append(str(dettagli))
    else:
        d = ast.literal_eval(s)
        if 'pavimento_pelvico' in d:
            dettagli.append('pavimento_pelvico')
        if ('altro' in d) or ('utero' in d) or ('sacro' in d):
            if 'altro' in d:
                if 'muscolo elevatore' in d['altro']:
                    dettagli.append('muscolo_elevatore')
                else:
                    dettagli.append('altro')
            else:
                dettagli.append('altro')
        infiltrazione_organi_dettagli_new.append(str(dettagli))
data_clean.loc[:, 'infiltrazione_organi_dettagli'] = infiltrazione_organi_dettagli_new

# Sedi linfonodi
sedi_linfonodi_new = []
for s in data_clean.sedi_linfonodi:
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
data_clean.loc[:, 'sedi_linfonodi'] = sedi_linfonodi_new

# infiltrazione tessuto adiposo
data_clean['infiltrazione_tessuto_adiposo'].replace({'sospetto': constants.InfiltrazioneTessutoAdiposo.SI_5MM.value}, inplace=True)
data_clean['infiltrazione_tessuto_adiposo'].fillna(constants.InfiltrazioneTessutoAdiposo.NO.value, inplace=True)

# Coinvolgimento fascia mesorettale
#data_clean.loc[(data_clean['coinvolgimento_fascia_mesorettale'] == 'rischio'), 'coinvolgimento_fascia_mesorettale'] = constants.CoinvolgimentoFasciaMesorettale.SI.value
#data_clean.loc[(data_clean['coinvolgimento_fascia_mesorettale'].isna()) & (data_clean['mrf'] == '+'), 'coinvolgimento_fascia_mesorettale'] = constants.CoinvolgimentoFasciaMesorettale.SI.value
#data_clean.loc[(data_clean['coinvolgimento_fascia_mesorettale'].isna()) & (data_clean['mrf'] == '-'), 'coinvolgimento_fascia_mesorettale'] = constants.CoinvolgimentoFasciaMesorettale.NO.value

# Coinvolgimento riflessione peritoneale. Trasformiamo rischio in si
data_clean['coinvolgimento_riflessione_peritoneale'].replace({'rischio': constants.CoinvolgimentoRiflessionePeritoneale.SI.value}, inplace=True)
data_clean['coinvolgimento_riflessione_peritoneale'].fillna(constants.CoinvolgimentoRiflessionePeritoneale.NO.value, inplace=True)

# Infiltrazione sfinteri. Trasformiamo la posizione in si. Per ottenere una classe (si/no/NaN)
data_clean['infiltrazione_sfinteri'].replace({
    'interno_piano': constants.InfiltrazioneSfinteri.SI.value,
    'interno': constants.InfiltrazioneSfinteri.SI.value,
    'interno_piano_esterno': constants.InfiltrazioneSfinteri.SI.value
}, inplace=True)
data_clean['infiltrazione_sfinteri'].fillna(constants.InfiltrazioneSfinteri.NO.value, inplace=True)


# Stadio N
data_clean['stadio_N'].replace({
    'N1a': constants.StadioN.N_PLUS.value,
    'N1b': constants.StadioN.N_PLUS.value,
    'N2a': constants.StadioN.N_PLUS.value,
    'N2b': constants.StadioN.N_PLUS.value
}, inplace=True)

# Stadio N1c
data_clean['stadio_N1c'].replace({
    True: constants.StadioN1c.SI.value,
    False: constants.StadioN1c.NO.value
}, inplace=True)

# Numero linfonodi non conosciuto
data_clean['numero_linfonodi_non_conosciuto'].replace({
    True: constants.NumeroLinfonodiNonConosciuto.NON_CONOSCIUTO.value,
    False: constants.NumeroLinfonodiNonConosciuto.CONOSCIUTO.value
}, inplace=True)

# Emvi
data_clean['emvi_esteso'].replace({'sospetto': 'si'}, inplace=True)
data_clean['emvi'].fillna(constants.EMVI.MINUS.value, inplace=True)
data_clean['emvi'].replace({
    '-': constants.EMVI.MINUS.value,
    '+': constants.EMVI.PLUS.value
}, inplace=True)

data_clean['mrf'].replace({
    '-': constants.MRF.MINUS.value,
    '+': constants.MRF.PLUS.value
}, inplace=True)

# Metastasi
data_clean['metastasi'].fillna(constants.Metastasi.MX.value, inplace=True)

# Numero depositi
data_clean['numero_depositi'].fillna(0, inplace=True)

# Infiltrazione organi extra
data_clean['infiltrazione_organi_extra'].replace({'sospetto': constants.InfiltrazioneOrganiExtra.SI.value}, inplace=True)
data_clean['infiltrazione_organi_extra'].fillna(constants.InfiltrazioneOrganiExtra.NO.value, inplace=True)

# Lesioni ossee
data_clean['lesioni_ossee'].fillna('no', inplace=True)

# Carcinosi peritoneale
data_clean['carcinosi_peritoneale'].fillna('no', inplace=True)

# Depositi tumorali
data_clean['depositi_tumorali'].replace({'sospetto': constants.DepositiTumorali.SI.value}, inplace=True)
data_clean['depositi_tumorali'].fillna(constants.DepositiTumorali.NO.value, inplace=True)

# Riflessione peritoneale
data_clean['riflessione_peritoneale_anteriore'].fillna(constants.RiflessionePeritonealeAnteriore.NON_DESCRITTO.value, inplace=True)


##########
# Plotting
##########
data_plot = data_clean.fillna(constants.NAN_VALUE)
data_ilaria = data_plot[data_plot['profile'] == 'IlariaNacci']
data_plot = data_plot.drop(index=data_plot[data_plot['profile'] == 'IlariaNacci'].index)
data_x = pd.concat([data_plot, data_ilaria])


columns_plot = ['morfologia', 'infiltrazione_tessuto_adiposo', 'coinvolgimento_fascia_mesorettale',
                'carcinosi_peritoneale', 'riflessione_peritoneale_anteriore', 'coinvolgimento_riflessione_peritoneale',
                'stadio_T', 'stadio_N', 'stadio_N1c',
                'mrf', 'emvi', 'metastasi',
                'infiltrazione_sfinteri', 'infiltrazione_organi_extra',
                'linfonodi_sospetti', 'numero_linfonodi_non_conosciuto', 'lesioni_ossee',
                'depositi_tumorali', 'numero_depositi', 'emvi_esteso']


n_columns = 3
n_rows, r = divmod(len(columns_plot), n_columns)
if r != 0:
    n_rows += 1
    
fig, axes = plt.subplots(n_rows, n_columns, figsize=(n_columns*7, n_rows*3))
axes = axes.flatten()
fig.suptitle("Count values", fontsize='xx-large', y=1)

for i, col in enumerate(columns_plot):
    ax=axes[i]
    sns.countplot(data=data_x, x=col, order=data_x[col].value_counts().index, ax=ax)
    ax.set_title(col)
    # Add values on top of bars
    for p in ax.patches:
        y_text = int(p.get_height())
        x_text = p.get_x() + p.get_width() / 2
        ax.text(x=x_text, y=y_text, s=f'{y_text}', ha='center', va='bottom')

# Rimuove eventuali assi vuoti
for j in range(len(columns_plot), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


########
# PLOT 2
########
# Analisi posizione, sedi locoregionali, sedi non locoregionali
columns = ['posizione', 'sedi_locoregionali', 'sedi_non_locoregionali', 'sedi_linfonodi', 'infiltrazione_organi_dettagli']


n_columns = 2
n_rows, r = divmod(len(columns), n_columns)
if r != 0:
    n_rows += 1

fig, axes = plt.subplots(n_rows, n_columns, figsize=(21, n_rows * 3))
axes = axes.flatten()

for i, col in enumerate(columns):
    df_plot = pd.DataFrame(columns=[col, 'profile'])
    for _, row in data_x.iterrows():
        l = ast.literal_eval(row[col])
        if len(l) == 0:
            continue
        else:
            split = row['profile']
            for s in l:
                df_plot.loc[len(df_plot)] = [s, split]
    sns.countplot(data=df_plot, ax=axes[i], y=col)
    axes[i].set_title(col)
    axes[i].set_xlabel(None)
    axes[i].set_ylabel(None)
    
    for p in axes[i].patches:
        x_text = p.get_width()
        y_text = p.get_y() + p.get_height() / 2
        axes[i].text(x=x_text, y=y_text, s=f'{int(x_text)}', va='center', ha='left')

# Rimuove eventuali assi vuoti
for j in range(len(columns), len(axes)):
    fig.delaxes(axes[j])

plt.show()

#########################
# Create new flag columns
#########################
for pos in constants.Posizione:
    new_col_name = f'posizione_{pos.value}'
    print(new_col_name)
    data_clean[new_col_name] = data_clean['posizione'].str.contains(pos.value)
    data_clean.loc[data_clean[new_col_name] == True, new_col_name] = constants.Flag.SI.value
    data_clean.loc[data_clean[new_col_name] == False, new_col_name] = constants.Flag.NO.value

for pos in constants.InfiltrazioneOrganiDettagli:
    new_col_name = f'infiltrazione_organi_dettagli_{pos.value}'
    print(new_col_name)
    data_clean[new_col_name] = data_clean['infiltrazione_organi_dettagli'].str.contains(pos.value)
    data_clean.loc[data_clean[new_col_name] == True, new_col_name] = constants.Flag.SI.value
    data_clean.loc[data_clean[new_col_name] == False, new_col_name] = constants.Flag.NO.value

for pos in constants.SediLinfonodi:
    new_col_name = f'sedi_linfonodi_{pos.value}'
    print(new_col_name)
    data_clean[new_col_name] = data_clean['sedi_linfonodi'].str.contains(pos.value)
    data_clean.loc[data_clean[new_col_name] == True, new_col_name] = constants.Flag.SI.value
    data_clean.loc[data_clean[new_col_name] == False, new_col_name] = constants.Flag.NO.value


##################################
# Dropping no significance columns
##################################
#data_clean.drop(columns=list(constants.LOW_SIGNIFICANCE_COLUMNS), inplace=True)
#print(f'{data_clean.shape =}')


# !!!Segnalare a Ilaria che il referto 44 è da rivedere
data_clean.drop(44, inplace=True)


################################
# Convert float columns to Int64
################################
float_cols = data_clean.select_dtypes("float").columns
for col in float_cols:
    data_clean[col] = data_clean[col].round().astype("Int64")


# Save data
if True:
    data_clean.to_csv(base_dir / "data" / constants.CLEAN_DATA_FILE_NAME, index=True)
    print('Data saved')
    