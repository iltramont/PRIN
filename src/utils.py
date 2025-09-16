# Utility generali
import random
import json
import os
import pandas as pd

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

def carica_system_prompts(indice: int) -> str:
    path="\\data\\system_prompts.xlsx"
    l = os.getcwd().split("\\")
    complete_path = "\\".join(l[:len(l)]) + path    
    return pd.read_excel(complete_path).loc[indice, "system_prompt"]