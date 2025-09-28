import os
import pandas as pd
import json
import utils
from pathlib import Path
import ast

# Le righe del file di training devono essere ncessariamente formattate in questo modo:
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
CREATE_FILE = True  # Impostare a False se non si vuole creare il file json
NOME_FILE_GENERATO = "data_training_luca_1.json"
SYSTEM_PROMPT_FILE_NAME = "system_prompt_1.txt"
STILE_FILE = 'HUGGINGFACE'  # 'OPENAI' o 'HUGGINGFACE'
# Dato che molte colonne contengono valori nulli, le escludo
SELECTED_COLUMNS = (
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
#    'infiltrazione_organi_dettagli',
    'coinvolgimento_riflessione_peritoneale',
    'coinvolgimento_fascia_mesorettale',
    'linfonodi_sospetti',
    'sedi_locoregionali',
    'sedi_non_locoregionali',
    'depositi_tumorali',
    'numero_depositi',
    'emvi_esteso'
)


# --- Define possible values for boolean conversion ---
# Based on the user's request for Linfonodi_non_regionali
NON_REGIONALI_POSSIBLE_VALUES = [
    "inguinali", 
    "iliaci_esterni", 
    "iliaci_comuni", 
    "paraortici", 
    "altri"
]

# Based on the user's request for Linfonodi_regionali.Sedi
SEDI_REGIONALI_POSSIBLE_VALUES = [
    "mesorettali", 
    "rettali_superiori", 
    "mesenterici_inferiori", 
    "iliaci_interni", 
    "otturatori", 
    "sacrali", 
    "inguinali_sotto_dentata"
]

# --- Safe parsing helpers ---
def safe_list_parse(val):
    if isinstance(val, str):
        try:
            # Safely evaluate string literal into a Python list
            parsed_list = ast.literal_eval(val)
            # Ensure the result is actually a list (in case of empty string or other literals)
            return parsed_list if isinstance(parsed_list, list) else []
        except Exception:
            return []
    elif isinstance(val, list):
        return val
    else:
        return []


# --- Helper for boolean conversion ---
def convert_list_to_boolean_dict(data_list, possible_values) -> dict:
    """Converts a list of strings into a dictionary of boolean flags."""
    # Ensure data_list is a list, even if empty
    data_list = safe_list_parse(data_list)
    
    # Create a dictionary with a boolean for each possible value
    boolean_dict = {
        value: (value in data_list) 
        for value in possible_values
    }
    return boolean_dict


def convert_row_to_json(row: pd.Series, system_content: str, columns: tuple[str] = SELECTED_COLUMNS) -> dict:
    """
    Data una riga del dataset del gemelli, crea un dizionario json con le colonne specificate.
    Il dizionario è nella forma:
    {
        "messages": [
            {"role": "system", "content": <system_content>},
            {"role": "user", "content": <user_content>},
            {"role": "assistant", "content": <assistant_content>}
        ]
    }
    """
    # Crea l'output desiderato
    assistant_content = dict()
    for column in columns:
        if pd.notna(row[column]):
            value = row[column]  # Contenuto grezzo della colonna
            if column == 'sedi_non_locoregionali':
                assistant_content[column] = convert_list_to_boolean_dict(value, NON_REGIONALI_POSSIBLE_VALUES)
            elif column == 'sedi_locoregionali':
                assistant_content[column] = convert_list_to_boolean_dict(value, SEDI_REGIONALI_POSSIBLE_VALUES)
            else:
                assistant_content[column] = value
        else:
            assistant_content[column] = None
    # Crea il dizionario JSON
    json_dict = {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": row["report_text"]},
            {"role": "assistant", "content": assistant_content}
        ]
    }
    return json_dict 

def main():
    base_dir = Path(__file__).parent.parent
    
    # Carica il prompt di sistema
    prompt_path = base_dir / "models" / "prompts" / SYSTEM_PROMPT_FILE_NAME
    print(prompt_path)
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_content = f.read().strip()

    # Get data
    data_path = base_dir / "data" / "base.tumoreprimitivo.csv"
    data = pd.read_csv(data_path)

    # Crea un dizionario per ogni riga
    json_list = []
    for index, row in data.iterrows():
        row_json_dict = convert_row_to_json(row, system_content)
        json_list.append(row_json_dict)

    # Crea il percorso per il nuovo file
    base_path = base_dir / "data" / "ft-dataset"
    # Aggiungi suffisso se il file esiste già
    filename, ext = os.path.splitext(NOME_FILE_GENERATO)
    new_file_path = os.path.join(base_path, NOME_FILE_GENERATO)
    counter = 1
    while os.path.exists(new_file_path):
        new_file_path = os.path.join(base_path, f"{filename}_{counter}{ext}")
        counter += 1

    # Stampa percorso per controllo
    print(f"File will be saved to: {new_file_path}")

    # Crea il nuovo file
    if CREATE_FILE:
        if STILE_FILE == 'OPENAI':
            with open(new_file_path, 'w', encoding='utf-8') as f:
                for json_dict in json_list[:-1]:  # Evita di aggiungere una nuova linea dopo l'ultimo record
                    f.write(json.dumps(json_dict) + '\n')
                f.write(json.dumps(json_list[-1]))  # Scrive l'ultimo record senza nuova linea
        elif STILE_FILE == 'HUGGINGFACE':
            # creo un solo dizionario con chiave 'data' e valore la lista di esempi
            huggingface_dict = {'conversations': json_list}
            json.dump(huggingface_dict, open(new_file_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()