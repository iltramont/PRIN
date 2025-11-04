import pandas as pd
from schema_json import ReportData
from utils import safe_list_parse



def convert_row_to_json_mistral(row: pd.Series) -> dict:
    
    """
    Data una riga del dataset del gemelli, crea un dizionario json con le colonne specificate.
    Il dizionario è nella forma:
    {
        "text": "The report",
        "labels": {
            "label_1": "value_1",
            "label_2": "value_2",
            "label_multi": [
                "value_multi_1",
                "value_multi_2",
                ...
            ],
            "label_n": "value_n",
            ...    
        }  
    } 
    """
    labels_content = dict()
    fields = ReportData.model_fields.keys()
    # Crea l'output desiderato
    for field in fields:
        f = field
        if f == 'sedi_linfonodi_locoregionali':
            f = 'sedi_locoregionali'
        if f == 'sedi_linfonodi_non_locoregionali':
            f = 'sedi_non_locoregionali'
        if pd.notna(row[f]):
            value = row[f]  # Contenuto grezzo della colonna
            if f == 'sedi_locoregionali' or f == 'sedi_non_locoregionali':
                labels_content[field] = safe_list_parse(value)
            else:
                labels_content[field] = value
        else:
            labels_content[f] = None
    
    json_dict = {
        "text": row["report_text"],
        "labels": labels_content
    }
    return json_dict
    

if __name__ == "__main__":
    pass