import os
import pandas as pd
import json
import utils

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


# VARIABILI
NOME_FILE_GENERATO = "data_training.json"
INDICE_SYSTEM_PROMPT = 0  # Indice del prompt di sistema da caricare dal file "system_prompts.xlsx"

# Carica il prompt di sistema
system_content = utils.carica_system_prompts(INDICE_SYSTEM_PROMPT)

# Get data
current_path = os.getcwd().split("\\")
file_name = "base.tumoreprimitivo.csv"
path = "\\".join(current_path[:len(current_path)]) + "\\data\\" + file_name
data = pd.read_csv(path)

# Dato che molte colonne contengono valori nulli, le escludo
selected_columns = [
    'morfologia',
    'spessore_parietale',
    'estensione_cranio_caudale',
    'distanza_oai',
    'posizione',
    'carcinosi_peritoneale',
    'lesioni_ossee',
    'riflessione_peritoneale_anteriore',
    'infiltrazione_tessuto_adiposo',
    'infiltrazione_sfinteri',
    'infiltrazione_organi_extra',
    'infiltrazione_organi_dettagli',
    'coinvolgimento_riflessione_peritoneale',
    'coinvolgimento_fascia_mesorettale',
    'linfonodi_sospetti',
    'sedi_locoregionali',
    'sedi_non_locoregionali',
    'depositi_tumorali',
    'numero_depositi',
    'emvi_esteso'
]


def convert_row_to_json(row: pd.Series, columns: list[str] = selected_columns) -> dict:
    user_content = row["report_text"]
    # Crea l'output desiderato
    assistant_content = dict()
    for column in columns:
        if pd.notna(row[column]):
            value = row[column]  # Contenuto grezzo della colonna
            if column in ['sedi_non_locoregionali', 'sedi_locoregionali', 'infiltrazione_organi_dettagli']:
                if value == '[object Object]':
                    assistant_content[column] = None
                else:
                    assistant_content[column] = utils.trasforma_string_in_lista_o_dizionario(value)
            else:
                assistant_content[column] = value
        else:
            assistant_content[column] = None
    # Crea il dizionario JSON
    json_dict = {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }
    return json_dict 


# Crea i dizionari
json_list = []
for index, row in data.iterrows():
    row_json_dict = convert_row_to_json(row)
    json_list.append(row_json_dict)

# Salva il file in formato json
current_path = os.getcwd().split("\\")
base_path = "\\".join(current_path[:len(current_path)]) + "\\data\\"

# Aggiungi suffisso se il file esiste già
filename, ext = os.path.splitext(NOME_FILE_GENERATO)
final_path = os.path.join(base_path, NOME_FILE_GENERATO)
counter = 1
while os.path.exists(final_path):
    final_path = os.path.join(base_path, f"{filename}_{counter}{ext}")
    counter += 1

print(f"File will be saved to: {final_path}")

CREATE_FILE = True
if CREATE_FILE:
    with open(final_path, 'w', encoding='utf-8') as f:
        for json_dict in json_list[:-1]:  # Evita di aggiungere una nuova linea dopo l'ultimo record
            f.write(json.dumps(json_dict) + '\n')
        f.write(json.dumps(json_list[-1]))  # Scrive l'ultimo record senza nuova linea