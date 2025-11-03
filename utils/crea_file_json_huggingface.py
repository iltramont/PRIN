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
# System text sarà sempre lo stesso per tutti gli esempi di training.
# User text sarà il report.
# Assistant text sarà l'output del modello in formato json.


# VARIABILI
CREATE_FILE = True  # Impostare a False se non si vuole creare il file json
NOME_FILE_GENERATO = "data_finetune_hints"
SYSTEM_PROMPT_FILE_NAME = "system_prompt_with_hints_1.txt"

TRAIN_SPLIT_NAME: str = 'train_split.csv'
TEST_SPLIT_NAME: str = 'test_split.csv'
VALIDATION_SPLIT_NAME: str | None = 'validation_split.csv'

TIPO = 'huggingface'
if TIPO != 'openai':
    TIPO = 'huggingface'



def crea_lista_di_dizionari_da_dataframe(dataframe: pd.DataFrame, system_content: str) -> list[dict]:
    json_list = []
    for index, row in dataframe.iterrows():
        row_json_dict = utils.convert_row_to_json(row, system_content, style=TIPO)
        json_list.append(row_json_dict)
    return json_list


def main():
    base_dir = Path(__file__).parent.parent
    
    # Carica il prompt di sistema
    prompt_path = base_dir / "models" / "prompts" / SYSTEM_PROMPT_FILE_NAME
    print(prompt_path)
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_content = f.read().strip()

    # Get data
    train_path = base_dir / "data" / TRAIN_SPLIT_NAME
    test_path = base_dir / "data" / TEST_SPLIT_NAME
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Crea lista di dizionari per ogni split
    train_dicts = crea_lista_di_dizionari_da_dataframe(train_data, system_content)
    test_dicts = crea_lista_di_dizionari_da_dataframe(test_data, system_content)
    
    # Validation
    if VALIDATION_SPLIT_NAME is not None:
        validation_path = base_dir / "data" / VALIDATION_SPLIT_NAME
        validation_data = pd.read_csv(validation_path)
        validation_dicts = crea_lista_di_dizionari_da_dataframe(validation_data, system_content)    
    
    # Crea il percorso per i nuovi file
    base_path = base_dir / "data" / "ft-dataset"
    # Aggiungi suffisso se il file esiste già
    ext = ".jsonl"
    train_file_path = os.path.join(base_path, NOME_FILE_GENERATO + '_' + TIPO + "_train" + ext)
    test_file_path = os.path.join(base_path, NOME_FILE_GENERATO + '_' + TIPO+ "_test" + ext)
    val_file_path = os.path.join(base_path, NOME_FILE_GENERATO + '_' + TIPO + "_val" + ext)
    counter = 1
    while os.path.exists(train_file_path):
        train_file_path = os.path.join(base_path, f"{NOME_FILE_GENERATO + '_' + TIPO}_train-v{counter}{ext}")
        test_file_path = os.path.join(base_path, f"{NOME_FILE_GENERATO + '_' + TIPO}_test-v{counter}{ext}")
        if VALIDATION_SPLIT_NAME is not None:
            val_file_path = os.path.join(base_path, f"{NOME_FILE_GENERATO + '_' + TIPO}_val-v{counter}{ext}")
        counter += 1

    # Stampa percorso per controllo
    print(f"File will be saved to: {train_file_path}, {test_file_path}, {val_file_path}")
    # Salva
    for f_path, dict_list in zip(
        [train_file_path, test_file_path] + ([val_file_path] if VALIDATION_SPLIT_NAME is not None else []),
        [train_dicts, test_dicts] + ([validation_dicts] if VALIDATION_SPLIT_NAME is not None else [])
    ):
        with open(f_path, 'w', encoding='utf-8') as f:
                for json_dict in dict_list:
                    f.write(json.dumps(json_dict) + '\n')


if __name__ == '__main__':
    main()