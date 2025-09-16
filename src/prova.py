# Questo sorgente è utilizzato esclusivamente per testare le funzioni                        
import json 
def main():
    stringa = "{'altro': 'collo uterino', 'utero': True}"
    print(stringa)
    print(type(stringa))
    diz = json.loads(stringa.replace("'", '"').replace("True", "true").replace("False", "false"))

    print(diz)
    print(type(diz))
if __name__ == "__main__":
    main()