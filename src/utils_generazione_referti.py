import pandas as pd
import random
from utils import da_lista_a_elenco_virgole, random_DD
from google.genai import types

def carica_referti(path="data\Referti Gemelli 1-10 con annotazioni.xlsx"):
    # Questa funzione serve solo a caricare i referti
    # Restituisce una lista di stringhe, ognuna delle quali è un referto')
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
    # Restituisce un dizionario con infiltrazione organi extra mesorettali (casuale) e stringa corrispondente
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
    # Restituisce un dizionario con coinvolgimento riflessione peritoneale (casuale) e stringa corrispondente
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
    # Restituisce un dizionario con numero depositi tumorali mesorettali (casuale) e stringa corrispondente
    if random.randint(0, 1) == 0:
        v = 0
        s = ''
    else:
        opt = [1, 2, 3, 4, 5]
        v = random_DD(opt)
        s = f'numero depositi tumorali mesorettali: {v};\n'
    return ({'v': v, 's': s}, v)

def depositi_tumorali_mesorettali(numero_depositi):
    # Restituisce un dizionario con depositi tumorali mesorettali (casuale) e stringa corrispondente
    if numero_depositi == 0:
        return {'v': 'no', 's': 'depositi tumorali mesorettali: assenti;\n'}
    else:
        return {'v': 'si', 's': 'depositi tumorali mesorettali: presenti;\n'}

def EMVI():
    # Restituisce un dizionario con extra mural vascular invasion (EMVI) (casuale) e stringa corrispondente
    if random.randint(0, 1) == 0:
        v = 'no'
        s = 'extra mural vascular invasion (EMVI): assente;\n'
    else:
        opt = ['sospetta', 'presente']
        v = random_DD(opt)
        s = f'extra mural vascular invasion (EMVI): {v};\n'
    return {'v': v, 's': s}

def segni_carcinosi_peritoneale():
    # Restituisce un dizionario con segni di carcinosi peritoneale (casuale) e stringa corrispondente
    opt = ['presente', 'assente']
    v = random_DD(opt)
    s = f'segni carcinosi peritoneale: {v};\n'
    return {'v': v, 's': s}

def lesioni_ossee_focali_sospette():
    # Restituisce un dizionario con lesioni ossee focali sospette (casuale) e stringa corrispondente
    opt = ['presenti', 'assenti']
    v = random_DD(opt)
    s = f'lesioni ossee focali sospette: {v};\n'
    return {'v': v, 's': s}

def linfonodi(n_loco, v_loco, n_non_loco, v_non_loco):
    # Restituisce un dizionario con linfonodi sospetti (casuale) e stringa corrispondente
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
    # Restituisce un dizionario con il numero di linfonodi locoregionali sospetti (casuale) e stringa corrispondente
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
    # Restituisce un dizionario con il numero di linfonodi non locoregionali sospetti (casuale) e stringa corrispondente
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


def has_conclusions(modalita = 'no'):
    # Questa funzione serve per decidere se il prompt deve specificare
    # che il referto generato contenga le conclusioni oppure no
    if modalita == 'no':
        v = False
    elif modalita == 'yes':
        v = True
    else:
        v = random.choice([True, False])
    if v:
        s = ';\nConclusioni.'
    else:
        s = '.'
    return {'v': v, 's': s}


def specifica_assenze():
    # Questa funzione serve per specificare nel prompt l'assenza o meno di certi parametri
    if random.randint(0, 1) == 0:
        return {'v': True, 's': ''}
    else:
        return {'v': False, 's': '\nNel caso alcuni parametri siano marcati come "assenti", non specificarlo nel referto.'}


def generate_generation_dict():
    # Questa funzione serve a creare il dizionario di generazione
    gen_dict = dict()
    gen_dict['anni'] = anni()
    gen_dict['genere'] = genere()
    gen_dict['morfologia'] = morfologia()
    gen_dict['posizione'] = posizione()
    gen_dict['relazione con riflessione peritonale anteriore'] = relazione_con_riflessione_peritonale_anteriore()
    gen_dict['infiltrazione tessuto adiposo mesorettale'] = infiltrazione_tessuto_adiposo_mesorettale()
    gen_dict['infiltrazione sfinteri'] = infiltrazione_sfinteri()
    gen_dict['infiltrazione organi extra mesorettali'], n_organi = infiltrazione_organi_extra_mesorettali()
    gen_dict['altri organi coinvolti'] = altri_organi_involti(n_organi)
    gen_dict['coinvolgimento riflessione peritoneale'] = coinvolgimento_riflessione_peritoneale()
    gen_dict['linfonodi locoregionali sospetti'], n_loco = linfonodi_locoregionali()
    gen_dict['linfonodi non locoregionali sospetti'], n_non_loco = linfonodi_non_locoregionali()
    v_loco = gen_dict['linfonodi locoregionali sospetti']['v']
    v_non_loco = gen_dict['linfonodi non locoregionali sospetti']['v']
    gen_dict['linfonodi sospetti'] = linfonodi(n_loco, v_loco, n_non_loco, v_non_loco)
    gen_dict['numero depositi tumorali mesorettali'], numero_depositi = numero_depositi_tumorali_mesorettali()
    gen_dict['depositi tumorali mesorettali'] = depositi_tumorali_mesorettali(numero_depositi)
    gen_dict['EMVI'] = EMVI()
    gen_dict['segni carcinosi peritoneale'] = segni_carcinosi_peritoneale()
    gen_dict['lesioni ossee focali sospette'] = lesioni_ossee_focali_sospette()
    gen_dict['has_conclusions'] = has_conclusions()
    gen_dict['specifica assenze'] = specifica_assenze()
    #gen_dict['indice referti esempio'] = indice_referti_esempio(lista_indici_esempio)
    #gen_dict['temperature'] = temperature()
    #gen_dict['topK'] = topK()
    #gen_dict['topP'] = topP()
    #gen_dict['repetition penalty'] = repetition_penalty()
    #gen_dict['frequency penalty'] = frequency_penalty()
    return gen_dict


def generate_system_text(reports: list, gendict: dict, indici_esempio: list[int]):
    # Funzione generatrice del prompt di sistema
    # Reports è una lista di referti
    # gendict è il dizionario di generazione
    # indici_esempio è una lista di indici dei referti che devono essere usati come esempio
    # Write text
    result = "Agisci come un medico radiologo esperto in diagnostica per immagini dell’apparato digerente.\n"\
             "Il tuo compito è generare referti di risonanza magnetica del colon-retto per pazienti con neoplasie rettali.\n"\
             "I referti devono seguire la struttura e il linguaggio tecnico dei referti radiologici reali, mantenendo coerenza, precisione e plausibilità clinica. "\
             "Il contenuto deve essere completamente inventato, ma realistico e compatibile con la pratica clinica. "\
             "\nSalvo diversa indicazione, il testo dovrà contenere:"\
             "\nTecnica utilizzata;"\
             "\nCaratterizazione del tumore;"\
             "\nRiferimenti a Linfonodi sospetti;\n"\
             f"Eventuali altri reperti{gendict['has_conclusions']['s']}"\
             "\nDi seguito alcuni esempi reali (per ispirazione di stile e tono, non da copiare):"
    # Add examples
    #random_indexes = gendict['indice referti esempio']
    for i in indici_esempio:
        result += f"\n\nEsempio:\n{reports[i]}"
    return result

def sections_to_be_included(has_conclusions: bool):
    # Questa funzione serve a generare le sezioni che devono essere incluse nel prompt utente e quindi nel referto
    # Se has_conclusions è True, le conclusioni sono incluse
    sections = [
        "Tecnica utilizzata",
        "Caratterizzazione del tumore",
        "Linfonodi sospetti",
        "Eventuali altri reperti",
        "Conclusioni"
    ]
    result = ''
    if has_conclusions:
        for section in sections[:-1]:
            result += f"{section}, "
        result = result[:-2]
        result = result + ' e ' + sections[-1] + '.'
    else:
        for section in sections[:-2]:
            result += f"{section}, "
        result = result[:-2]
        result = result + ' e ' + sections[-2] + '.'
    return result

def generate_user_text(gendict: dict):
    # Funzione generatrice del prompt dell'utente
    # gendict è il dizionario di generazione
    result = f"Genera un referto di risonanza magnetica pelvica per {gendict['genere']['s']} di {gendict['anni']['s']}, con neoplasia rettale."\
             f"\nI dati devono essere completamente inventati ma clinicamente plausibili."\
             "\nUtilizza lo stile e la struttura definiti nel prompt di sistema. "\
             f"Includi tutte le sezioni: {sections_to_be_included(gendict['has_conclusions']['v'])}"\
             "\nAggiungi anche dettagli su eventuale infiltrazione mesorettale, coinvolgimento dei linfonodi e presenza/assenza di metastasi locali."\
             "\nIl testo deve essere continuo, senza che vi sia una dichiarazione esplicita delle sezioni.\n"\
             "Per aiutarti a scrivere il referto, utilizza i seguenti dati:\n"\
             f"{gendict['morfologia']['s']}"\
             f"{gendict['posizione']['s']}"\
             f"{gendict['relazione con riflessione peritonale anteriore']['s']}"\
             f"{gendict['infiltrazione tessuto adiposo mesorettale']['s']}"\
             f"{gendict['infiltrazione sfinteri']['s']}"\
             f"{gendict['infiltrazione organi extra mesorettali']['s']}"\
             f"{gendict['altri organi coinvolti']['s']}"\
             f"{gendict['coinvolgimento riflessione peritoneale']['s']}"\
             f"{gendict['linfonodi sospetti']['s']}"\
             f"{gendict['linfonodi locoregionali sospetti']['s']}"\
             f"{gendict['linfonodi non locoregionali sospetti']['s']}"\
             f"{gendict['depositi tumorali mesorettali']['s']}"\
             f"{gendict['numero depositi tumorali mesorettali']['s']}"\
             f"{gendict['EMVI']['s']}"\
             f"{gendict['segni carcinosi peritoneale']['s']}"\
             f"{gendict['lesioni ossee focali sospette']['s']}"\
             f"{gendict['segni carcinosi peritoneale']['s']}"
    result = result[:-2]
    result += '.'
    result += f"{gendict['specifica assenze']['s']}"
    return result

def indice_referti_esempio(lista_indici: list[int], numero_esempi = 3):
    # Questa funzione serve a generare gli indici dei referti che devono essere usati come esempio
    if numero_esempi > len(lista_indici):
        numero_esempi = len(lista_indici)
    indexes = random.sample(lista_indici, numero_esempi)
    return indexes

# Funzioni per generazione massiva
def genera_referto_artificiale(client, referti_esempio: list[str], indici_referti_esempio: list[str], gendict: dict,
                               temperatura: float, model_name: str = "gemini-2.0-flash"):
    """
    Dati un set di referti esempio, degli indici di questi referti e un dizionario con i parametri di generazione,
    crea un prompt di sistema e un prompt utente, che vengono poi dati in input al modello per la generazione.
    I modelli accettati sono quelli Gemini, di default viene usato il modello base, Gemini-2.0-flash.
    TODO: aggiungere un modo per generare anche i parametri di generazione come Temperatura e TopK.
    """
    # Crea prompt di sistema
    sys_text = generate_system_text(referti_esempio, gendict, indici_referti_esempio)
    # Crea prompt utente
    user_text = generate_user_text(gendict)
    # Generazione tramite chiamata API
    response = client.models.generate_content(
    model=model_name,
    config=types.GenerateContentConfig(
        system_instruction=sys_text,
        seed=25,
        temperature=temperatura,
        topK=40,
        topP=0.95
        #presence_penalty=0.0,
        #frequency_penalty=0.0,
        ),
    contents=user_text
    )
    return response, sys_text, user_text


def genera_multipli_referti_con_uguale_gendict(client, referti_esempio: list[str], numero_referti_generati: int = 3 ,
                                               numero_esempi: int = 3, temperatura_iniziale=1.0, model_name: str = "gemini-2.0-flash",
                                               incremento_temperatura=0.2):
    # Inizializza variabili di uscita
    referti_generati = []
    indici_utilizzati = []
    temp_utilizzate = []
    # Crea dizionario di generazione
    gendict = generate_generation_dict()
    # Genera referti
    indici_disponibili = list(range(len(referti_esempio)))
    temperatura = temperatura_iniziale
    for i in range(numero_referti_generati):
        if len(indici_disponibili) < numero_esempi:
            break
        indici_scelti = random.sample(indici_disponibili, numero_esempi)
        indici_utilizzati.append(indici_scelti)
        # Rimuovi gli indici scelti da quelli disponibili
        for indice in indici_scelti:
            indici_disponibili.remove(indice)
        # Genera il referto
        referto, _, _ = genera_referto_artificiale(client, referti_esempio, indici_scelti, gendict, temperatura, model_name)
        temp_utilizzate.append(temperatura)
        temperatura += incremento_temperatura
        referti_generati.append(referto)
    return referti_generati, gendict, indici_utilizzati, temp_utilizzate
