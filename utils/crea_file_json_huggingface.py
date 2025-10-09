import os
import pandas as pd
import json
import utils
from pathlib import Path
from sklearn.model_selection import train_test_split


# Per OpenAI le righe del file di training devono essere ncessariamente formattate in questo modo:
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
NOME_FILE_GENERATO = "data_luca"
SYSTEM_PROMPT_FILE_NAME = "system_prompt_2.txt"
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
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
NON_REGIONALI_POSSIBLE_VALUES = [
    "inguinali", 
    "iliaci_esterni", 
    "iliaci_comuni", 
    "paraortici", 
    "altri"
]
SEDI_REGIONALI_POSSIBLE_VALUES = [
    "mesorettali", 
    "rettali_superiori", 
    "mesenterici_inferiori", 
    "iliaci_interni", 
    "otturatori", 
    "sacrali", 
    "inguinali_sotto_dentata"
]

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
                flags = utils.convert_list_to_boolean_dict(value, NON_REGIONALI_POSSIBLE_VALUES)
                for key in flags.keys():
                    assistant_key = 'linfonodi_' + key
                    assistant_content[assistant_key] = flags[key]
                #assistant_content[column] = utils.convert_list_to_boolean_dict(value, NON_REGIONALI_POSSIBLE_VALUES)
            elif column == 'sedi_locoregionali':
                flags = utils.convert_list_to_boolean_dict(value, SEDI_REGIONALI_POSSIBLE_VALUES)
                for key in flags.keys():
                    assistant_key = 'linfonodi_' + key
                    assistant_content[assistant_key] = flags[key]
                #assistant_content[column] = utils.convert_list_to_boolean_dict(value, SEDI_REGIONALI_POSSIBLE_VALUES)
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


def crea_lista_di_dizionari_da_dataframe(dataframe: pd.DataFrame, system_content: str) -> list[dict]:
    json_list = []
    for index, row in dataframe.iterrows():
        row_json_dict = convert_row_to_json(row, system_content)
        json_list.append(row_json_dict)
    return json_list


def split_dataframe(dataframe: pd.DataFrame, test_size: float = 0.2, validation_size: float = 0.1, random_state: int = 2025) -> tuple[pd.DataFrame]:
    train_split, test_split = train_test_split(dataframe, test_size=test_size, random_state=random_state)
    if validation_size > 0.0:
        relative_val_size = validation_size / (1 - test_size)
        train_split, val_split = train_test_split(train_split, test_size=relative_val_size, random_state=random_state)
        return train_split, test_split, val_split
    return train_split, test_split


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
    
    # Split data
    train_data, test_data, val_data = split_dataframe(data, test_size=0.2, validation_size=0.1, random_state=42)

    # Crea lista di dizionari per ogni split
    train_dicts = crea_lista_di_dizionari_da_dataframe(train_data, system_content)
    test_dicts = crea_lista_di_dizionari_da_dataframe(test_data, system_content)
    if VALIDATION_SIZE > 0.0:
        val_dicts = crea_lista_di_dizionari_da_dataframe(val_data, system_content)    
    
    # Crea il percorso per il nuovo file
    base_path = base_dir / "data" / "ft-dataset"
    # Aggiungi suffisso se il file esiste già
    ext = ".jsonl"
    train_file_path = os.path.join(base_path, NOME_FILE_GENERATO + "_train" + ext)
    test_file_path = os.path.join(base_path, NOME_FILE_GENERATO + "_test" + ext)
    val_file_path = os.path.join(base_path, NOME_FILE_GENERATO + "_val" + ext)
    counter = 1
    while os.path.exists(train_file_path):
        train_file_path = os.path.join(base_path, f"{NOME_FILE_GENERATO}_train-v{counter}{ext}")
        test_file_path = os.path.join(base_path, f"{NOME_FILE_GENERATO}_test-v{counter}{ext}")
        if VALIDATION_SIZE > 0.0:
            val_file_path = os.path.join(base_path, f"{NOME_FILE_GENERATO}_val-v{counter}{ext}")
        counter += 1

    # Stampa percorso per controllo
    print(f"File will be saved to: {train_file_path}, {test_file_path}, {val_file_path}")

    for f_path, dict_list in zip(
        [train_file_path, test_file_path] + ([val_file_path] if VALIDATION_SIZE > 0.0 else []),
        [train_dicts, test_dicts] + ([val_dicts] if VALIDATION_SIZE > 0.0 else [])
    ):
        with open(f_path, 'w', encoding='utf-8') as f:
                for json_dict in dict_list:
                    f.write(json.dumps(json_dict) + '\n')


if __name__ == '__main__':
    main()