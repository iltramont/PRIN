import pandas as pd
import json
import ast
from sklearn.model_selection import train_test_split

# --- Define possible values for boolean conversion ---
# Based on the user's request for Linfonodi_non_regionali
NON_REGIONALI_POSSIBLE_VALUES = [
    "inguinali", 
    "iliaci_esterni", 
    "iliaci_comuni", 
    "paraortici", 
    "altri"
]

# Based on the user's request for Linfonodi_regionali.Sedi
SEDI_REGIONALI_POSSIBLE_VALUES = [
    "mesorettali", 
    "rettali_superiori", 
    "mesenterici_inferiori", 
    "iliaci_interni", 
    "otturatori", 
    "sacrali", 
    "inguinali_sotto_dentata"
]

# --- Safe parsing helpers ---
def safe_list_parse(val):
    if isinstance(val, str):
        try:
            # Safely evaluate string literal into a Python list
            parsed_list = ast.literal_eval(val)
            # Ensure the result is actually a list (in case of empty string or other literals)
            return parsed_list if isinstance(parsed_list, list) else []
        except Exception:
            return []
    elif isinstance(val, list):
        return val
    else:
        return []

def get_val(row, col):
    return row[col] if col in row and pd.notna(row[col]) else None

# --- Helper for boolean conversion ---
def convert_list_to_boolean_dict(data_list, possible_values):
    """Converts a list of strings into a dictionary of boolean flags."""
    # Ensure data_list is a list, even if empty
    data_list = safe_list_parse(data_list)
    
    # Create a dictionary with a boolean for each possible value
    boolean_dict = {
        value: (value in data_list) 
        for value in possible_values
    }
    return boolean_dict

# --- Convert row to structured JSON ---
def row_to_structured_json(row):
    # Process Linfonodi_regionali.Sedi list into a boolean dictionary
    sedi_regionali_list = get_val(row, "sedi_locoregionali")
    sedi_regionali_boolean_dict = convert_list_to_boolean_dict(
        sedi_regionali_list, 
        SEDI_REGIONALI_POSSIBLE_VALUES
    )

    # Process Linfonodi_non_regionali list into a boolean dictionary
    non_regionali_list = get_val(row, "sedi_non_locoregionali")
    non_regionali_boolean_dict = convert_list_to_boolean_dict(
        non_regionali_list, 
        NON_REGIONALI_POSSIBLE_VALUES
    )
    
    return {
        "Morfologia": get_val(row, "morfologia"),
        "Posizione": get_val(row, "posizione"),
        "Dimensioni_Misure": {
            "Estensione_angolare": {
                "Da_ore": get_val(row, "ore_inizio"),
                "A_ore": get_val(row, "ore_fine")
            },
            "Diametro_LL_mm": get_val(row, "dimensione_dll"),
            "Diametro_AP_mm": get_val(row, "dimensione_dap"),
            "Spessore_parietale_mm": get_val(row, "spessore_parietale"),
            "Estensione_CC_mm": get_val(row, "estensione_cranio_caudale"),
            "Distanza_OAI_mm": get_val(row, "distanza_oai")
        },
        "Relazione_rifl_peritoneale": get_val(row, "riflessione_peritoneale_anteriore"),
        "Infiltrazione_mesorettale": get_val(row, "infiltrazione_tessuto_adiposo"),
        "Infiltrazione_sfinteri": get_val(row, "infiltrazione_sfinteri"),
        "Infiltrazione_extra_mesorettale": get_val(row, "infiltrazione_organi_extra"),
        "Coinvolgimento_peritoneo": get_val(row, "coinvolgimento_riflessione_peritoneale"),
        "CRM_coinvolgimento": {
            "Stato": get_val(row, "coinvolgimento_fascia_mesorettale"),
            "Distanza_minima_ore": get_val(row, "distanza_minima_fascia_ore")
        },
        "Linfonodi_regionali": {
            "Numero_sospetti": get_val(row, "linfonodi_sospetti"),
            # Replaced safe_list_parse with the new boolean dictionary
            "Sedi": sedi_regionali_boolean_dict 
        },
        # Replaced safe_list_parse with the new boolean dictionary
        "Linfonodi_non_regionali": non_regionali_boolean_dict, 
        "Depositi_tumorali_mesorettali": get_val(row, "depositi_tumorali"),
        "EMVI": get_val(row, "emvi"),
        "Altri_reperti": {
            "Carcinosi_peritoneale": get_val(row, "carcinosi_peritoneale"),
            "Lesioni_ossee": get_val(row, "lesioni_ossee")
        }
    }

# --- Load system prompt ---
# Assuming 'prompt.txt' exists in the current directory
with open("src/prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

# --- Load Dataset ---
referti = pd.read_excel("data/dataset.xlsx")
referti

# --- Split dataset ---
try:
    # Assuming 'referti' DataFrame is available from the context where this code will be executed
    train_df, val_df = train_test_split(referti, test_size=0.2, random_state=42)
except NameError:
    print("\n ERROR: DataFrame 'referti' not found. Cannot proceed with splitting and writing JSONL files.")
 

# --- Write JSONL ---
def write_jsonl(df, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": row["report_text"]},
                    {"role": "assistant", "content": json.dumps(row_to_structured_json(row), ensure_ascii=False)}
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

# --- Create training and validation JSONL files ---
# Only run if 'referti' was successfully split
try:
    write_jsonl(train_df, "referti_finetune_train.jsonl")
    write_jsonl(val_df, "referti_finetune_val.jsonl")

    print("\n✅ JSONL creati:")
    print("  • referti_finetune_train.jsonl (training set)")
    print("  • referti_finetune_val.jsonl (validation set)")
except NameError:
    pass # Error message already printed above