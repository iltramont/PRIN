# Questo sorgente è utilizzato esclusivamente per testare le funzioni                        
import json 
import os
#import pandas as pd
import utils

def main():

    x = utils.load_prompt('system_prompt_1.txt')
    print(x)

if __name__ == "__main__":
    main()