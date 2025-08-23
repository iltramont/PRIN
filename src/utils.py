# Utility generali
import random

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