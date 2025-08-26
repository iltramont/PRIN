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
    numero_iterazioni = 50  # Verranno generati multipli referti (di base 3) per ogni iterazione
    pausa = 12  # Secondi di pausa tra una chiamata e l'altra necessari per non fare troppe chiamate all'API
    i = 0
    for iterazione in range(numero_iterazioni):
        referti_generati, gendict, indici_utilizzati, temp_utilizzate = genera_multipli_referti_con_uguale_gendict(client, referti,
                                                                                                                    temperatura_iniziale=1.0)
        for j in range(len(referti_generati)):
            df.loc[i, 'report'] = referti_generati[j].text
            df.loc[i, 'indice referti esempio'] = indici_utilizzati[j]
            df.loc[i, 'temperature'] = temp_utilizzate[j]
            for k in gendict.keys():
                df.loc[i, k] = gendict[k]['v']
            i += 1
        print(f'Iterazione {iterazione+1} completata')
        time.sleep(pausa)
    
    # Visualizzazionerisultati
    print(df.head())
    
    # Salvataggio risultati
    output_path = os.path.join('data', 'referti_generati_4.xlsx')
    df.to_excel(output_path)
    
if __name__ == "__main__":
    main()