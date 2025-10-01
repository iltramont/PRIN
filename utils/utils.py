# Utility generali
import random
import json
import os
import pandas as pd
from pathlib import Path
import ast


def da_lista_a_elenco_virgole(lista):
    # Trasforma una lista in un elenco separato da virgole
    # Esempio: ['a', 'b', 'c'] -> 'a, b, c'
    result = ''
    for item in lista:
        result += f'{item}, '
    return result[:-2]


def random_DD(options: list):
    # Sceglie un elemento a caso da una lista
    return random.choice(options)


def prepara_stringa_per_json(stringa):
    result = stringa.replace("{'", '{"').replace("'}", '"}').replace("',", '",').replace(" '", ' "')
    result = result.replace("['", '["').replace("']", '"]')
    result = result.replace(":'", ':"').replace("':", '":')
    return result.replace("True", "true").replace("False", "false")


def trasforma_string_in_lista_o_dizionario(stringa):
    return json.loads(prepara_stringa_per_json(stringa))


def load_prompt(file_name: str, strip=True) -> str:
    """
    Carica il prompt da un file di testo nella cartella models/prompts.
    """
    base_dir = Path(__file__).parent.parent / "models" / "prompts"
    path = base_dir / file_name
    with open(path, "r", encoding="utf-8") as f:
        if strip:
            return f.read().strip()
        else:
            return f.read()
        
        
def safe_list_parse(val):
    """
    Trasforma in modo sicuro una stringa che rappresenta una lista in una lista Python.
    """
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