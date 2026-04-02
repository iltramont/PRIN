#####################
# --- Constants --- #
#####################

from pydantic import BaseModel, ConfigDict, field_validator, Field
from enum import Enum
from typing import List, Optional, Any

NAN_VALUE = 'NaN'
SEED = 2026

BERT_ENCODER_CHECKPOINT = "bert-base-multilingual-cased"
XLM_ROBERTA_ENCODER_CHECKPOINT = "FacebookAI/xlm-roberta-base"
XLM_ROBERTA_LARGE_ENCODER_CHECKPOINT = "FacebookAI/xlm-roberta-large"
BIOBERT_ITALIAN_ENCODER = "IVN-RIN/bioBIT"

############
# Embeddings
############
EMBEDDING_FILE_NAME_MISTRAL = "mistral_embeddings.jsonl"
MISTRAL_EMBED = "mistral-embed-2312"

#########
# Mistral
#########
MISTRAL_TRAIN_FILE_NAME = "train_mistral.jsonl"
MISTRAL_VALIDATION_FILE_NAME = "validation_mistral.jsonl"
MISTRAL_TEST_FILE_NAME = "test_mistral.jsonl"

MISTRAL_LARGE_3 = "mistral-large-2512"

########
# OpenAI
########
OPENAI_GPT_4_1_NANO = "gpt-4.1-nano-2025-04-14"
OPENAI_GPT_4_1_MINI = "gpt-4.1-mini-2025-04-14"
OPENAI_GPT_4_1      = "gpt-4.1-2025-04-14"
OPENAI_GPT_5_2      = "gpt-5.2-2025-12-11"
OPENAI_GPT_5_4      = "gpt-5.4-2026-03-05"

# Tuned models
TUNED_GPT_4_1_NANO = "ft:gpt-4.1-nano-2025-04-14:luca-tramonti:report-extractor:DNDCPvfn"
TUNED_GPT_4_1_NANO_OVERSAMPLE = "ft:gpt-4.1-nano-2025-04-14:luca-tramonti:report-extractor-oversampling:DPqaFyZI:ckpt-step-486"
TUNED_GPT_4_1 = "ft:gpt-4.1-2025-04-14:luca-tramonti:report-extractor:DOcFh6T3"

OPENAI_TRAIN_FILE_NAME      = "openai_train.jsonl"
OPENAI_VALIDATION_FILE_NAME = "openai_validation.jsonl"
OPENAI_TRAIN_OVERSAMPLING_FILE_NAME = "openai_train_oversampling.jsonl"


###########
# Anthropic
###########
CLAUDE_OPUS_4_6 = "claude-opus-4-6"

#######
# Llama
#######
LLAMA_3_2_3B_TUNED = "iltramont/llama-3b-report-extractor"
LLAMA_3_2_1B_TUNED = "iltramont/llama-1b-report-extractor"

############
# File names
############
RAW_DATA_FILE_NAME = "base.tumoreprimitivo_febbraio.csv"
CLEAN_DATA_FILE_NAME = "tumoreprimitivo_clean.csv"
TRAIN_SPLIT_FILE_NAME = 'train_split.csv'
TEST_SPLIT_FILE_NAME = 'test_split.csv'
VALIDATION_SPLIT_FILE_NAME = 'validation_split.csv'
# Prompts
SYSTEM_PROMPT = "system_prompt.txt"
SYSTEM_PROMPT_2 = "system_prompt_2.txt"
SYSTEM_PROMPT_3 = "system_prompt_3.txt"
SYSTEM_PROMPT_4 = "system_prompt_4.txt"
SYSTEM_PROMPT_5 = "system_prompt_5.txt"
SYSTEM_PROMPT_6 = "system_prompt_6.txt"
SYSTEM_PROMPT_SHORT = "system_prompt_short.txt"

#################
# Raw data fields
#################
REPORT_COLUMN_NAME = "report_text"
ANNOTATOR_COLUMN_NAME = "profile"
ID_COMULM_NAME = "id"
LOW_SIGNIFICANCE_COLUMNS = (
    'carcinosi_peritoneale',
    'lesioni_ossee',
    'numero_depositi',
    'dimensione_dll',
    'dimensione_dap'
)


######################
# Splitting parameters
######################
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
STRATIFY_COLUMNS = (
    'morfologia',
    'infiltrazione_organi_dettagli',
    'stadio_T',
    'depositi_tumorali',
    'emvi',
    'metastasi',
    'infiltrazione_sfinteri'
)


##################
# Campi multilabel
##################
class Flag(str, Enum):
    NO = "no"
    SI = "si"
    
class Posizione(str, Enum):
    BASSO = "basso"
    MEDIO = "medio"
    ALTO = "alto"
    GIUNZIONE = "giunzione"
    
class PosizioneFlags(BaseModel):
    basso: Flag
    medio: Flag
    alto: Flag
    giunzione: Flag
    
    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
    )
    
class InfiltrazioneOrganiDettagli(str, Enum):
    MUSCOLO_ELEVATORE = "muscolo_elevatore"
    PAVIMENTO_PELVICO =  "pavimento_pelvico"
    ALTRO = "altro"

class InfiltrazioneOrganiDettagliFlags(BaseModel):
    muscolo_elevatore: Flag
    pavimento_pelvico: Flag
    altro: Flag
    
    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
    )
    
class SediLinfonodi(str, Enum):
    MESORETTALI = "mesorettali"
    RETTALI_SUPERIORI = "rettali_superiori"
    OTTURATORI = "otturatori"
    ILIACI = "iliaci"
    ALTRO = "altro"
    
class SediLinfonodiFlags(BaseModel):
    mesorettali: Flag
    rettali_superiori: Flag
    otturatori: Flag
    iliaci: Flag
    altro: Flag
    
    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
    )

##################
# Campi multiclass
##################
class Morfologia(str, Enum):
    SOLIDO_POLIPOIDE = "solido_polipoide"
    SOLIDO_ANULARE = "solido_anulare"
    MUCINOSO = "mucinoso"

class InfiltrazioneTessutoAdiposo(str, Enum):      
    NO = "no" 
    SI_5MM = "si_5mm"
    SI_5MM_PLUS = "si_5mm_plus"

class RiflessionePeritonealeAnteriore(str, Enum):
    SOTTO = "sotto"
    CAVALLO = "cavallo"
    #NON_VALUTABILE = "non_valutabile"
    NON_DESCRITTO = "non_descritto"

class StadioT(str, Enum):
    T1_2 = "T1-2"
    T3AB = "T3ab"
    T3CD = "T3cd"
    T4A = "T4a"
    T4B = "T4b"
    

##############    
# Campi binari
##############
class CoinvolgimentoFasciaMesorettale(str, Enum):
    NO = "no"
    SI = "si"
    
class CoinvolgimentoRiflessionePeritoneale(str, Enum):
    NO = "no"
    SI = "si"
    
class StadioN(str, Enum):
    N0 = "N0"
    #N1 = "N1"
    #N2 = "N2"
    N_PLUS = "N+"
    
class StadioN1c(str, Enum):
    NO = "no"
    SI = "si"
    
class MRF(str, Enum):
    MINUS = "no"
    PLUS = "si"
    #MINUS = "-"
    #PLUS = "+"

class EMVI(str, Enum):
    MINUS = "no"
    PLUS = "si"
    #MINUS = "-"
    #PLUS = "+"

class Metastasi(str, Enum):
    MX = "MX"
    M1 = "M1"
    
class InfiltrazioneSfinteri(str, Enum):
    NO = "no"
    SI = "si"
    
class InfiltrazioneOrganiExtra(str, Enum):
    NO = "no"
    SI = "si"
    
class NumeroLinfonodiNonConosciuto(str, Enum):
    CONOSCIUTO = "conosciuto"
    NON_CONOSCIUTO = "non_conosciuto"
    
class DepositiTumorali(str, Enum):
    NO = "no"
    SI = "si"
    
    
   
########################################################################
# Modello completo con anche campi numerici. Campi multilabel come flag.
########################################################################
class RectalCancerStagingData(BaseModel):
    # Tumore primitivo
    morfologia: Morfologia = Field(description='Morfologia del tumore')
    ore_inizio: Optional[int] = None
    ore_fine: Optional[int] = None
    spessore_parietale: Optional[int] = Field(default=None, description="Spessore parietale massimo (mm)")
    estensione_cranio_caudale: Optional[int] = Field(default=None, description='Estensione cranio-caudale (mm)')
    distanza_oai: Optional[int] = Field(default=None, description="Distanza tra il margine inferiore del tumore e l'Orifizio Anale Interno (giunzione ano-rettale) (mm)")
    posizione: PosizioneFlags = Field(description="Posizione del tumore")
    riflessione_peritoneale_anteriore: RiflessionePeritonealeAnteriore = Field(description="Relazione con la riflessione peritoneale anteriore")
    infiltrazione_tessuto_adiposo: InfiltrazioneTessutoAdiposo = Field(description="Infiltrazione del tessuto adiposo")
    infiltrazione_sfinteri: InfiltrazioneSfinteri = Field(description="Infiltrazione degli sfinteri")
    infiltrazione_organi_extra: InfiltrazioneOrganiExtra = Field(description="Infiltrazione altri organi / strutture extra-mesorettali")
    infiltrazione_organi_dettagli: InfiltrazioneOrganiDettagliFlags = Field(description="Altri organi coinvolti")
    coinvolgimento_riflessione_peritoneale: CoinvolgimentoRiflessionePeritoneale = Field(description="Coinvolgimento riflessione peritoneale/peritoneo")
    coinvolgimento_fascia_mesorettale: CoinvolgimentoFasciaMesorettale = Field(description="Coinvolgimento fascia mesorettale")
    # Linfonodi Sospetti
    numero_linfonodi_non_conosciuto: NumeroLinfonodiNonConosciuto = Field(description="Indica se il numero di linfonodi sospetti è conosciuto o non conosciuto")
    linfonodi_sospetti: int = Field(description="Numero di linfonodi sospetti")
    sedi_linfonodi: SediLinfonodiFlags = Field(description="Sedi dei linfonodi sospetti")
    depositi_tumorali: DepositiTumorali = Field(description="Presenza di depositi tumorali")
    # Conclusioni
    emvi: EMVI = Field(description="Extra Mural Vascular Invasion, infiltrazione diretta dei vasi mesorettali")
    stadio_T: StadioT = Field(description="Stadiazione T secondo TNM (invasione locale del tumore)")
    stadio_N: StadioN = Field(description="Stadiazione linfonodale secondo TNM")
    #stadio_N1c: StadioN1c
    mrf: MRF = Field(description="Infiltrazione della fascia mesorettale")
    metastasi: Metastasi = Field(description="Stadiazione M secondo TNM (metastasi a distanza)")

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        extra="forbid"
    )

class AnnotatedRectalCancerReport(BaseModel):
    report_text: str
    report_data: RectalCancerStagingData

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        extra="forbid"
    )    


# Feature weights (importance)
FEATURE_WEIGHTS = {
    "morfologia": 1,
    #"ore_inizio": 1,
    #"ore_fine": 1,
    'ore': 0.2,  # meno importante
    "spessore_parietale": 0.5, # Meno importante
    "estensione_cranio_caudale": 1, 
    "distanza_oai": 1,
    "posizione": 1, 
    "riflessione_peritoneale_anteriore": 1, 
    "infiltrazione_tessuto_adiposo": 1,
    "infiltrazione_sfinteri": 1,
    "infiltrazione_organi_extra": 1, 
    "infiltrazione_organi_dettagli": 0.5, 
    "coinvolgimento_riflessione_peritoneale": 0.5, 
    "coinvolgimento_fascia_mesorettale": 0.5, 
    "numero_linfonodi_non_conosciuto": 0.5, # meno importante
    "linfonodi_sospetti": 0.5,
    "sedi_linfonodi": 1, 
    "depositi_tumorali": 1,
    "emvi": 1, 
    "stadio_T": 1,
    "stadio_N": 1,
    "mrf": 0.5,
    "metastasi": 1, 
}



if __name__ == '__main__':
    from openai.lib._pydantic import to_strict_json_schema
    from pprint import pprint
    schema = to_strict_json_schema(RectalCancerStagingData)
    pprint(schema)