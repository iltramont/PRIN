# Questo sorgente è utilizzato esclusivamente per testare le funzioni                        
import json 
import os
import pandas as pd

def main():
    from utils import carica_system_prompts

    # Test della funzione carica_system_prompt
    print(carica_system_prompts(0))
    
if __name__ == "__main__":
    main()