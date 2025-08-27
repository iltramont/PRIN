# In questo file verranno generati dei referti artificiali partendo da 10 referti reali
# Questo sorgente è utilizzato esclusivamente per testare le funzioni
import random
import pandas as pd
import time
import os

from utils_generazione_referti import generate_generation_dict, carica_referti
from utils_generazione_referti import genera_multipli_referti_con_uguale_gendict
from dotenv import load_dotenv
from google import genai


NUMERO_ITERAZIONI = 50  # Verranno generati multipli referti (di base 3) per ogni iterazione
PAUSA = 12  # Secondi di pausa tra una chiamata e l'altra necessari per non fare troppe chiamate all'API
REFERTI_PER_ITERAZIONE = 3  # Numero di referti da generare per ogni iterazione
MODELLO_GEMINI = "gemini-2.0-flash"  # Modello da utilizzare per la generazione dei referti
NOME_FILE_REFERTI_GENERATI = 'referti_generati_5.xlsx'  # File excel con i referti generati
CARTELLA_OUTPUT = 'data'  # Cartella di output per il file excel

def main():
    """
    Generazione massiva di referti sintetici
    """
    # Accesso API google
    load_dotenv()
    google_api_key = os.getenv('GOOGLE_API_KEY')
    client = genai.Client(api_key=google_api_key)
    
    # Carica referti reali di esempio
    referti = carica_referti()
    
    # Creiamo le colonne del dataframe
    gendict = generate_generation_dict()
    columns = [
    'report',
    'indice referti esempio',
    'temperature'
    ]
    columns += list(gendict.keys())
    
    # Creiamo il dataframe
    df = pd.DataFrame(columns=columns)
    
    # Generazione
    i = 0
    for iterazione in range(NUMERO_ITERAZIONI):
        print(f'*--- Iterazione {iterazione+1}')  # Tieni traccia dell'avanzamento
        
        # Effettiva generazione
        referti_generati, gendict, indici_utilizzati, temp_utilizzate = genera_multipli_referti_con_uguale_gendict(client, referti, REFERTI_PER_ITERAZIONE, temperatura_iniziale=1.0, model_name=MODELLO_GEMINI)
        for j in range(len(referti_generati)):
            df.loc[i, 'report'] = referti_generati[j].text
            df.loc[i, 'indice referti esempio'] = indici_utilizzati[j]
            df.loc[i, 'temperature'] = temp_utilizzate[j]
            for k in gendict.keys():
                df.loc[i, k] = gendict[k]['v']
            i += 1
            
        print(f'*--- Iterazione completata')  # Tieni traccia dell'avanzamento
        time.sleep(PAUSA)  # Pausa per non fare troppe chiamate all'API
    
    # Visualizzazione risultati
    print(df.head())
    
    # Salvataggio risultati
    print('*--- Salvataggio risultati...')
    os.makedirs(CARTELLA_OUTPUT, exist_ok=True)  # Crea la cartella se non esiste
    output_path = os.path.join(CARTELLA_OUTPUT, NOME_FILE_REFERTI_GENERATI)
    df.to_excel(output_path)
    print(f'*--- File "{NOME_FILE_REFERTI_GENERATI}" salvato nella cartella "{CARTELLA_OUTPUT}"')
    
if __name__ == "__main__":
    main()