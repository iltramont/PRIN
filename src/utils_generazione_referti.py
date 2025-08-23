import pandas as pd



def carica_referti(path="Referti Gemelli 1-10 con annotazioni.xlsx"):
    print('pippo')
    row_data = pd.read_excel(path, index_col=0)
    referti = row_data.T['TESTO:'].tolist()
    for i in range(len(referti)):
        print(f'### REFERTO {i + 1}')
   
    

carica_referti()
#print(referti)              
        