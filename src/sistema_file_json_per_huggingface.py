import os
import json

FILE_JSON_ORIGINE = 'data_training.json'
NOME_FILE_GENERATO = 'data_training_huggingface.json'

def main():
    # Get data
    current_path = os.getcwd().split("\\")
    path = "\\".join(current_path[:len(current_path)]) + "\\data\\" + FILE_JSON_ORIGINE
    print(path)
    print('pipo')
    data_json = json.load(open(path, 'r', encoding='utf-8'))
    print(data_json)

if __name__ == "__main__":
    main()
    