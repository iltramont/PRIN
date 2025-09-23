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


def load_prompt(file_name: str, strip=True) -> str:
    path="\\data\\prompts\\" + file_name
    l = os.getcwd().split("\\")
    path = "\\".join(l[:len(l)]) + path
    with open(path, "r", encoding="utf-8") as f:
        if strip:
            return f.read().strip()
        else:
            return f.read()