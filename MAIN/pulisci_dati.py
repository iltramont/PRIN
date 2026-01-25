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
finomnia_palette = sns.color_palette(('#2659ab',
                                      '#db038a',
                                      '#45c9f5',
                                      '#66218a',
                                      '#081c36',))
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

#########################
# Consistency adjustments
#########################
# --- Stadio T --- #
# Correggiamo l'anomalia
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

# --- Stadio N and lymph nodes --- #
# Stadio N valorizzato con "N1c". Anomalo solo il referto 44.
data_clean.loc[(data_clean['id'] == 44), 'stadio_N1c'] = True
data_clean.loc[(data_clean['id'] == 44), 'stadio_N'] = 'N2a'

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

# Anomalia. Se i linfonodi sospetti sono 0, non posso avere le sedi valorizzate.
# Dando un'occhiata ai referti si osserva come il modo corretto per risolvere l'anomalia è quello di valorizzare a True il campo numero_linfonodi_non_conosciuto.
data_clean.loc[
    (data_clean['linfonodi_sospetti'] == 0) &
    (data_clean['numero_linfonodi_non_conosciuto'] == False) &
    ((data_clean['sedi_locoregionali'] != '[]') | (data_clean['sedi_non_locoregionali'] != '[]'))
    , 'numero_linfonodi_non_conosciuto'
    ] = True

# --- EMVI --- #
data_clean.loc[(data_clean['emvi'] == '+') & (data_clean['emvi_esteso'] == 'no'), 'emvi'] = '-'

data_clean.loc[(data_clean['emvi'].isna()) & (data_clean['emvi_esteso'] == 'sospetto'), 'emvi'] = '+'

# --- MRF --- #
data_clean.loc[
    (data_clean['mrf'].isna()) &
    (data_clean['coinvolgimento_fascia_mesorettale'] == 'rischio')
    , 'mrf'
    ] = '+'

data_clean.loc[
    (data_clean['mrf'].isna()) &
    (data_clean['coinvolgimento_fascia_mesorettale'].isna()) & 
    (data_clean['stadio_T'] == 'T4b')
    , 'mrf'
    ] = '+'

data_clean.loc[
    (data_clean['mrf'].isna()) &
    (data_clean['coinvolgimento_fascia_mesorettale'].isna()) & 
    (data_clean['stadio_T'] == 'T4a')
    , 'mrf'
    ] = '+'

data_clean.loc[
    (data_clean['mrf'].isna()) &
    (data_clean['coinvolgimento_fascia_mesorettale'].isna())
    , 'mrf'
    ] = '-'

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
for s in data_clean.infiltrazione_organi_dettagli.fillna(constants.NAN_VALUE):
    dettagli = []
    if s == constants.NAN_VALUE:
        infiltrazione_organi_dettagli_new.append(str(dettagli))
    else:
        d = ast.literal_eval(s)
        if constants.InfiltrazioneOrganiDettagli.PavimentoPelvico.value in d:
            dettagli.append(constants.InfiltrazioneOrganiDettagli.PavimentoPelvico.value)
        if ('altro' in d) or ('utero' in d) or ('sacro' in d):
            dettagli.append(constants.InfiltrazioneOrganiDettagli.Altro.value)
        infiltrazione_organi_dettagli_new.append(str(dettagli))
data_clean.loc[:, 'infiltrazione_organi_dettagli'] = infiltrazione_organi_dettagli_new

# Sedi linfonodi
sedi_linfonodi_new = []
for s in data_clean.sedi_linfonodi:
    sedi = ast.literal_eval(s)
    sedi_new = set()
    for sede in sedi:
        if sede in [constants.SediLinfonodi.Mesorettali.value,
                    constants.SediLinfonodi.RettaliSuperiori.value,
                    constants.SediLinfonodi.Otturatori.value]:
            sedi_new.add(sede)
        elif sede in ['iliaci_comuni',
                      'iliaci_interni',
                      'iliaci_esterni']:
            sedi_new.add(constants.SediLinfonodi.Iliaci.value)
        else:
            sedi_new.add(constants.SediLinfonodi.Altro.value)
    sedi_linfonodi_new.append(str(list(sedi_new)))
data_clean.loc[:, 'sedi_linfonodi'] = sedi_linfonodi_new

# infiltrazione tessuto adiposo
data_clean.loc[data_clean['infiltrazione_tessuto_adiposo'] == 'sospetto', 'infiltrazione_tessuto_adiposo'] = constants.InfiltrazioneTessutoAdiposo.Si5mm.value
data_clean.loc[data_clean['infiltrazione_tessuto_adiposo'].isna(), 'infiltrazione_tessuto_adiposo'] = constants.InfiltrazioneTessutoAdiposo.No.value

# Coinvolgimento fascia mesorettale
data_clean.loc[(data_clean['coinvolgimento_fascia_mesorettale'] == 'rischio'), 'coinvolgimento_fascia_mesorettale'] = constants.CoinvolgimentoFasciaMesorettale.Si.value
data_clean.loc[(data_clean['coinvolgimento_fascia_mesorettale'].isna()) & (data_clean['mrf'] == '+'), 'coinvolgimento_fascia_mesorettale'] = constants.CoinvolgimentoFasciaMesorettale.Si.value
data_clean.loc[(data_clean['coinvolgimento_fascia_mesorettale'].isna()) & (data_clean['mrf'] == '-'), 'coinvolgimento_fascia_mesorettale'] = constants.CoinvolgimentoFasciaMesorettale.No.value

# Coinvolgimento riflessione peritoneale. Trasformiamo rischio in si
data_clean.loc[data_clean['coinvolgimento_riflessione_peritoneale'] == 'rischio', 'coinvolgimento_riflessione_peritoneale'] = constants.CoinvolgimentoRiflessionePeritoneale.Si.value
data_clean.loc[data_clean['coinvolgimento_riflessione_peritoneale'].isna(), 'coinvolgimento_riflessione_peritoneale'] = constants.CoinvolgimentoRiflessionePeritoneale.No.value

# Infiltrazione sfinteri. Trasformiamo la posizione in si. Per ottenere una classe (si/no/NaN)
data_clean.loc[data_clean['infiltrazione_sfinteri'] == 'interno_piano', 'infiltrazione_sfinteri'] = constants.InfiltrazioneSfinteri.Si.value
data_clean.loc[data_clean['infiltrazione_sfinteri'] == 'interno', 'infiltrazione_sfinteri'] = constants.InfiltrazioneSfinteri.Si.value
data_clean.loc[data_clean['infiltrazione_sfinteri'] == 'interno_piano_esterno', 'infiltrazione_sfinteri'] = constants.InfiltrazioneSfinteri.Si.value
data_clean.loc[data_clean['infiltrazione_sfinteri'].isna(), 'infiltrazione_sfinteri'] = constants.InfiltrazioneSfinteri.No.value

# Stadio N
data_clean.loc[data_clean['stadio_N'] == 'N1a', 'stadio_N'] = constants.StadioN.N1.value
data_clean.loc[data_clean['stadio_N'] == 'N1b', 'stadio_N'] = constants.StadioN.N1.value
data_clean.loc[data_clean['stadio_N'] == 'N2a', 'stadio_N'] = constants.StadioN.N2.value
data_clean.loc[data_clean['stadio_N'] == 'N2b', 'stadio_N'] = constants.StadioN.N2.value

# Emvi
data_clean.loc[data_clean['emvi_esteso'] == 'sospetto', 'emvi_esteso'] = 'si'
data_clean.loc[data_clean['emvi'].isna(), 'emvi'] = constants.EMVI.Minus.value

# Metastasi
data_clean.loc[data_clean['metastasi'].isna(), 'metastasi'] = constants.Metastasi.MX.value

# Numero depositi
data_clean.loc[data_clean['numero_depositi'].isna(), 'numero_depositi'] = 0

# Infiltrazione organi extra
data_clean.loc[data_clean['infiltrazione_organi_extra'].isna(), 'infiltrazione_organi_extra'] = constants.InfiltrazioneOrganiExtra.No.value
data_clean.loc[data_clean['infiltrazione_organi_extra'] == 'sospetto', 'infiltrazione_organi_extra'] = constants.InfiltrazioneOrganiExtra.Si.value

# Lesioni ossee
data_clean.loc[data_clean['lesioni_ossee'].isna(), 'lesioni_ossee'] = 'no'

# Carcinosi peritoneale
data_clean.loc[data_clean['carcinosi_peritoneale'].isna(), 'carcinosi_peritoneale'] = 'no'

# Depositi tumorali
data_clean.loc[data_clean['depositi_tumorali'].isna(), 'depositi_tumorali'] = constants.DepositiTumorali.No.value
data_clean.loc[data_clean['depositi_tumorali'] == 'sospetto', 'depositi_tumorali'] = constants.DepositiTumorali.Si.value


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

possible_values = {col: [constants.NAN_VALUE] for col in columns}

for col in columns:
    for s in data_x[col].value_counts().index:
        possible_values[col] += ast.literal_eval(s)
    possible_values[col] = list(set(possible_values[col]))    


counts = {
    col: {val: 0 for val in possible_values[col]}
    for col in columns
}
for col in columns:
    for s in data_x[col]:
        value_list = ast.literal_eval(s)
        for value in possible_values[col]:
            if value in value_list:
                counts[col][value] += 1
        if value_list == []:
            counts[col][constants.NAN_VALUE] += 1
              
n_columns = 2
n_rows, r = divmod(len(columns), n_columns)
if r != 0:
    n_rows += 1
            
fig, axes = plt.subplots(n_rows, n_columns, figsize=(21, n_rows*3))
orientation = 'h'
for i, col in enumerate(columns):
    ax=axes[i//n_columns][i%n_columns]
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


##################################
# Dropping no significance columns
##################################
#data_clean.drop(columns=list(constants.LOW_SIGNIFICANCE_COLUMNS), inplace=True)
#print(f'{data_clean.shape =}')


# !!!Segnalare a Ilaria che il referto 44 è da rivedere
data_clean.drop(index=data_clean[data_clean['id'] == 44].index, inplace=True)


# Save data
if True:
    data_clean.to_csv(base_dir / "data" / constants.CLEAN_DATA_FILE_NAME, index=False)
    