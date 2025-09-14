# Le righe del file di trainini devono essere ncessariamente formattate in questo modo:
"""
{
    "messages": [
        {"role": "system", "content": <system_text>},
        {"role": "user", "content": <user_text>},
        {"role": "assistant", "content": <assistant_text>}
    ] 
}
"""
# Nella versione compatta, senza indentazione:
"""
{"messages": [{"role": "system", "content": <system_text>}, {"role": "user", "content": <user_text>}, {"role": "assistant", "content": <assistant_text>}]}
"""
# System text sarà sempre lo stesso per tutti gli esempi di training.
# User text sarà il report.
# Assistant text sarà l'output del modello in formato json.

import os
import pandas as pd
import json

from pprint import pprint

### Step 1-convertire i record del file dei referti annotati in formato json, ovvero formattati come vorremmo che il modello risponda.

# Get data
current_path = os.getcwd().split("\\")
file_name = "base.tumoreprimitivo.csv"
path = "\\".join(current_path[:len(current_path)]) + "\\data\\" + file_name
data = pd.read_csv(path)
#print(data.head(3))

# Seleziona solo le colonne target
target_columns = ['morfologia', 'ore_inizio', 'ore_fine', 'dimensione_dll',
       'dimensione_dap', 'spessore_parietale', 'estensione_cranio_caudale',
       'distanza_oai', 'posizione', 'carcinosi_peritoneale', 'lesioni_ossee',
       'riflessione_peritoneale_anteriore',
       'infiltrazione_tessuto_adiposo', 'infiltrazione_sfinteri',
       'infiltrazione_organi_extra', 'infiltrazione_organi_dettagli',
       'coinvolgimento_riflessione_peritoneale',
       'coinvolgimento_fascia_mesorettale', 'distanza_minima_fascia_ore',
       'linfonodi_sospetti', 'sedi_locoregionali', 'sedi_non_locoregionali',
       'depositi_tumorali', 'numero_depositi', 'emvi_esteso'
]

#print(data[target_columns].head(3))
#print(data.info())

# Dato che molte colonne contengono valori nulli, le escludo
selected_columns = ['morfologia', 'spessore_parietale', 'estensione_cranio_caudale',
       'distanza_oai', 'posizione', 'carcinosi_peritoneale', 'lesioni_ossee',
       'riflessione_peritoneale_anteriore',
       'infiltrazione_tessuto_adiposo', 'infiltrazione_sfinteri',
       'infiltrazione_organi_extra', 'infiltrazione_organi_dettagli',
       'coinvolgimento_riflessione_peritoneale',
       'coinvolgimento_fascia_mesorettale',
       'linfonodi_sospetti', 'sedi_locoregionali', 'sedi_non_locoregionali',
       'depositi_tumorali', 'numero_depositi', 'emvi_esteso'
]

#print(data[selected_columns].info())

# !!! Questo prompt di sistema non è definitivo
SYSTEM_CONTENT = "Sei un modello di intelligenza artificiale specializzato nell'estrazione di informazioni cliniche dai referti radiologici."

def convert_row_to_json(row: pd.Series, columns: list[str] = selected_columns) -> dict:
    user_content = row["report_text"]
    # Crea l'output desiderato
    assistant_content = dict()
    for column in columns:
        if pd.notna(row[column]):
            if column in ['sedi_non_locoregionali', 'sedi_locoregionali']:
                value = row[column]  # Contenuto grezzo della clonna
                if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                    value = value[1:-1]  # Rimuove le parentesi quadre
                    if value.strip() == '':  # Controlla se non è una stringa vuota dopo aver rimosso le parentesi
                        assistant_content[column] = []
                    else:
                        assistant_content[column] = [item.strip()[1:-1] for item in value.split(',')]
            elif column == 'infiltrazione_organi_dettagli':
                #TODO
                pass
            else:
                assistant_content[column] = row[column]
        else:
            assistant_content[column] = None
    # Crea il dizionario JSON
    json_dict = {
        "messages": [
            {"role": "system", "content": SYSTEM_CONTENT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }
    return json_dict 

#riga = data.iloc[0]
#print('*--- RIGA:')
#print(riga[selected_columns])

#print('*--- RIGA come DIZIONARIO:')
#riga_json_dict = convert_row_to_json(riga)
#print(riga_json_dict)

#print('*--- RIGA come JSON string:')
#riga_json_str = json.dumps(riga_json_dict, indent=4)
#print(riga_json_str)

# Crea i dizionari
json_list = []
for index, row in data.iterrows():
    row_json_dict = convert_row_to_json(row)
    json_list.append(row_json_dict)


# Crea il file di training in formato json
output_file = "data_simple_training.json"
current_path = os.getcwd().split("\\")
path = "\\".join(current_path[:len(current_path)]) + "\\data\\" + output_file

CREATE_FILE = True
if CREATE_FILE:
    with open(path, 'w', encoding='utf-8') as f:
        for json_dict in json_list[:-1]:  # Evita di aggiungere una nuova linea dopo l'ultimo record
            f.write(json.dumps(json_dict) + '\n')
        f.write(json.dumps(json_list[-1]))  # Scrive l'ultimo record senza nuova linea