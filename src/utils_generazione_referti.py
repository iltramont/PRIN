import pandas as pd
import random
from utils import da_lista_a_elenco_virgole, random_DD



def carica_referti(path="data\Referti Gemelli 1-10 con annotazioni.xlsx"):
    # Questa funzione serve solo a caricare i referti
    # Restituisce una lista di stringhe, ognuna delle quali è un referto
    print('pippo')
    row_data = pd.read_excel(path, index_col=0)
    referti = row_data.T['TESTO:'].tolist()
    return referti
    
   
def anni():
    # Restituisce un dizionario con età (casuale) in anni e stringa corrispondente
    x = random.randint(18, 90)
    return {'v': x, 's': f'{x} anni'}

def genere():
    # Restituisce un dizionario con genere (casuale) e stringa corrispondente
    values = ['uomo', 'donna']
    strings = ['un paziente uomo', 'una paziente donna']
    i = random.randint(0, len(values)-1)
    return {'v': values[i], 's': strings[i]}

def morfologia():
    # Restituisce un dizionario con morfologia (casuale) e stringa corrispondente
    values = ['solido - polipoide', 'solido - semi-anulare', 'mucinoso']
    v = random_DD(values)
    s = f'morfologia: {v};\n'
    return {'v': v, 's': s}

def posizione():
    # Restituisce un dizionario con posizione (casuale) e stringa corrispondente
    values = ['retto basso', 'retto medio', 'retto alto', 'giunzione sigma-retto']
    v = random_DD(values)
    s = f'posizione: {v};\n'
    return {'v': v, 's': s}

def relazione_con_riflessione_peritonale_anteriore():
    # Restituisce un dizionario con relazione con riflessione peritoneale anteriore (casuale) e stringa corrispondente
    values = ['al di sotto', 'a cavallo', 'al di sopra']
    v = random_DD(values)
    s = f'relazione con riflessione peritoneale anteriore: {v};\n'
    return {'v': v, 's': s}

def infiltrazione_tessuto_adiposo_mesorettale():
    # Restituisce un dizionario con infiltrazione tessuto adiposo mesorettale (casuale) e stringa corrispondente
    values = ['no', 'si, < 5mm', 'si, >= 5mm', 'sospetto']
    v = random_DD(values)
    s = 'infiltrazione tessuto adiposo mesorettale: '
    if v == 'no':
        s += 'assente;\n'
    elif v == 'sospetto':
        s += 'sospetta;\n'
    elif v == 'si, < 5mm':
        x = str(random.randint(1, 4)) + ' millimetri;\n'
        s += x
    else:
        x = str(random.randint(5, 8)) + ' millimetri;\n'
        s += x
    return {'v': v, 's': s}

def infiltrazione_sfinteri():
    # Restituisce un dizionario con infiltrazione sfinteri (casuale) e stringa corrispondente
    values = [
            'no',
            'sfintere interno',
            'sfintere interno, piano intersfinterico',
            'sfintere interno, piano intersfinterico, sfintere esterno'
        ]
    strings = [
            'assente',
            'presente nello sfintere interno',
            'presente nello sfintere interno e nel piano intersfinterico',
            'presente nello sfintere interno, nel piano intersfinterico e nello sfintere esterno'
        ]
    i = random.randint(0, len(values)-1)
    v = values[i]
    s = f'infiltrazione sfinteri: {strings[i]};\n'
    return {'v': values[i], 's': s}

def altri_organi_involti(numero_organi: int):
    # Restituisce un dizionario con altri organi coinvolti (casuale) e stringa corrispondente
    if numero_organi == 0:
        return {'v': [''], 's': ''}
    organi = [
        'Utero',
        'Sacro',
        'Pavimento pelvico',
        'Vasi extra mesorettali',
        'Anse intestinali',
        'Tessuto adiposo Otturatorio',
        'Tessuto adiposo Para Iliaco',
        'Fossa ischio rettale'
    ]
    organi_scelti = random.sample(organi, numero_organi)
    result = ''
    for organ in organi_scelti:
        result += f"{organ}, "
    result = result[:-2]
    result = result + ';\n'
    return {'v': organi_scelti, 's': result}

def infiltrazione_organi_extra_mesorettali():
    if random.randint(0, 1) == 0:
        return ({'v': 'no', 's': 'infiltrazione degli organi extra mesorettali: assente;\n'}, 0)
    else:
        s = 'infiltrazione organi extra mesorettali: '
        values = ['si', 'sospetta']
        v = random_DD(values)
        if v == 'si':
            s += 'presente in '
            numero_organi = random.randint(1, 4)
        else:
            s += 'sospetta in '
            numero_organi = random.randint(1, 2)
        return ({'v': v, 's': s}, numero_organi)

def coinvolgimento_riflessione_peritoneale():
    values = ['si', 'no', 'a rischio']
    v = random_DD(values)
    if v == 'no':
        x = 2 + round(2*random.random(), 1)
    elif v == 'a rischio':
        x = 1 + round(random.random(), 1)
    else:
        x = round(random.random(), 1)
    s = f'distanza da riflessione peritoneale/peritoneo: {x} millimetri;\n'
    return {'v': v, 's': s}

def numero_depositi_tumorali_mesorettali():
    if random.randint(0, 1) == 0:
        v = 0
        s = ''
    else:
        opt = [1, 2, 3, 4, 5]
        v = random_DD(opt)
        s = f'numero depositi tumorali mesorettali: {v};\n'
    return ({'v': v, 's': s}, v)

def depositi_tumorali_mesorettali(numero_depositi):
    if numero_depositi == 0:
        return {'v': 'no', 's': 'depositi tumorali mesorettali: assenti;\n'}
    else:
        return {'v': 'si', 's': 'depositi tumorali mesorettali: presenti;\n'}

def EMVI():
    if random.randint(0, 1) == 0:
        v = 'no'
        s = 'extra mural vascular invasion (EMVI): assente;\n'
    else:
        opt = ['sospetta', 'presente']
        v = random_DD(opt)
        s = f'extra mural vascular invasion (EMVI): {v};\n'
    return {'v': v, 's': s}

def segni_carcinosi_peritoneale():
    opt = ['presente', 'assente']
    v = random_DD(opt)
    s = f'segni carcinosi peritoneale: {v};\n'
    return {'v': v, 's': s}

def lesioni_ossee_focali_sospette():
    opt = ['presenti', 'assenti']
    v = random_DD(opt)
    s = f'lesioni ossee focali sospette: {v};\n'
    return {'v': v, 's': s}

def linfonodi(n_loco, v_loco, n_non_loco, v_non_loco):
    n = n_loco + n_non_loco
    if n == 0:
        s =  'linfonodi sospetti: nessuno;\n'
    elif n == 1:
        s =  f'linfonodi sospetti: un linfonodo sospetto, nella regione dei linfonodi '
        if n_loco == 1:
            s += f'{v_loco[0]};\n'
        else:
            s += f'{v_non_loco[0]};\n'
    else:
        s =  f'linfonodi sospetti: {n} linfonodi, tra cui '
        if n_loco > 0:
            s += f'{da_lista_a_elenco_virgole(v_loco)} '
            if n_non_loco > 0:
                s += f'e {da_lista_a_elenco_virgole(v_non_loco)}'
        else:
            s += f'{da_lista_a_elenco_virgole(v_non_loco)}'
        s += ';\n'
    return {'v': n, 's': s}

def linfonodi_locoregionali():
    s = ''
    if random.randint(0, 1) == 0:
        v = []
        n = 0
    else:
        opt = [
            'mesorettali',
            'rettali superiori',
            'mesenterici superiori',
            'iliaci interni',
            'otturatori',
            'sacrali laterali',
            'presacrali',
            'promontorio'
        ]
        n = random.randint(1, len(opt))
        v = random.sample(opt, n)
    return ({'v': v, 's': s}, n)

def linfonodi_non_locoregionali():
    s = ''
    if random.randint(0, 1) == 0:
        v = []
        n = 0
    else:
        opt = [
            'inguinali',
            'iliaci interni',
            'iliaci esterni',
            'iliaci comuni',
            'paraortici'
        ]
        n = random.randint(1, len(opt))
        v = random.sample(opt, n)
    return ({'v': v, 's': s}, n)

