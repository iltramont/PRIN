# Questo sorgente è utilizzato esclusivamente per testare le funzioni
from utils_generazione_referti import generate_generation_dict, carica_referti, indice_referti_esempio
from utils_generazione_referti import generate_system_text, generate_user_text                                    

def main():
    # Carica referti
    referti = carica_referti()
    
    # Genera il dizionario di generazione
    generation_dict = generate_generation_dict()

    # Stampa il dizionario di generazione
    for key, value in generation_dict.items():
        print(f"{key}: {value}")
        
    # Seleziona gli indici da utilizzare per selezionare gli esempi    
    indici_esempio = indice_referti_esempio(list(range(len(referti))))
    print(f"Indici esempio selezionati: {indici_esempio}")
    
    # Genera i prompt di sistema e utente
    sys_text = generate_system_text(referti, generation_dict, indici_esempio)
    user_text = generate_user_text(generation_dict)
    
    # Stampa i prompt generati
    print('### SYSTEM TEXT ###')
    print(sys_text)
    print('\n### USER TEXT ###')
    print(user_text)
    
    
if __name__ == "__main__":
    main()