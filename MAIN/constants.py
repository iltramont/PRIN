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


#########
# Mistral
#########
MISTRAL_MODEL_30_EPOCHS = "ft:classifier:ministral-3b-latest:6f88df17:20260201:63a1a8ac"
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
OPENAI_GPT_5_NANO   = "gpt-5-nano-2025-08-07"
OPENAI_GPT_5_2      = "gpt-5.2-2025-12-11"

# Tuned models
TUNED_GPT_4_1_NANO = "ft:gpt-4.1-nano-2025-04-14:luca-tramonti:report-extractor:D8F5Ak8z"
TUNED_GPT_4_1_NANO_OVERSAMPLING = "ft:gpt-4.1-nano-2025-04-14:luca-tramonti:report-extractor:D8k71VLO"
TUNED_GPT_4_1_OVERSAMPLING = "ft:gpt-4.1-2025-04-14:luca-tramonti:report-extractor:D8nTVX9B"

OPENAI_TRAIN_FILE_NAME      = "openai_train.jsonl"
OPENAI_VALIDATION_FILE_NAME = "openai_validation.jsonl"
OPENAI_TRAIN_OVERSAMPLING_FILE_NAME = "openai_train_oversampling.jsonl"


###########
# Anthropic
###########
CLAUDE_OPUS_4_6 = "claude-opus-4-6"



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
    'infiltrazione_tessuto_adiposo',
    'coinvolgimento_riflessione_peritoneale',
    'stadio_T',
    'stadio_N1c',
    'emvi',
    'metastasi',
    'infiltrazione_sfinteri',
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
    PAVIMENTO_PELVICO =  "pavimento_pelvico"
    ALTRO = "altro"

class InfiltrazioneOrganiDettagliFlags(BaseModel):
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
    NON_VALUTABILE = "non_valutabile"

class StadioT(str, Enum):
    T1_2 = "T1-2"
    T3AB = "T3ab"
    T3CD = "T3cd"
    T4A = "T4a"
    T4B = "T4b"
    
class StadioN(str, Enum):
    N0 = "N0"
    N1 = "N1"
    N2 = "N2"
    N_PLUS = "N+"

##############    
# Campi binari
##############
class CoinvolgimentoFasciaMesorettale(str, Enum):
    NO = "no"
    SI = "si"
    
class CoinvolgimentoRiflessionePeritoneale(str, Enum):
    NO = "no"
    SI = "si"
    
class StadioN1c(str, Enum):
    NO = "no"
    SI = "si"
    
class MRF(str, Enum):
    MINUS = "-"
    PLUS = "+"

class EMVI(str, Enum):
    MINUS = "-"
    PLUS = "+"

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
    stadio_N1c: StadioN1c = Field(description="Presenza di depositi tumorali")
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
    
    
    
    

########################################################################
# Modello completo con anche campi numerici. Campi multilabel come liste
########################################################################
class Annotations(BaseModel):
    # Tumore primitivo
    morfologia: Morfologia
    ore_inizio: Optional[int] = None
    ore_fine: Optional[int] = None
    spessore_parietale: Optional[int] = None
    estensione_cranio_caudale: Optional[int] = None
    distanza_oai: Optional[int] = None
    posizione: List[Posizione] = Field(default_factory=list)
    riflessione_peritoneale_anteriore: RiflessionePeritonealeAnteriore
    infiltrazione_tessuto_adiposo: InfiltrazioneTessutoAdiposo
    infiltrazione_sfinteri: InfiltrazioneSfinteri
    infiltrazione_organi_extra: InfiltrazioneOrganiExtra
    infiltrazione_organi_dettagli: List[InfiltrazioneOrganiDettagli] = Field(default_factory=list)
    coinvolgimento_riflessione_peritoneale: CoinvolgimentoRiflessionePeritoneale
    coinvolgimento_fascia_mesorettale: CoinvolgimentoFasciaMesorettale
    # Linfonodi Sospetti
    linfonodi_sospetti: int
    numero_linfonodi_non_conosciuto: NumeroLinfonodiNonConosciuto
    sedi_linfonodi: List[SediLinfonodi] = Field(default_factory=list)
    depositi_tumorali: DepositiTumorali
    # Conclusioni
    emvi: EMVI  
    stadio_T: StadioT
    stadio_N: StadioN  
    stadio_N1c: StadioN1c
    mrf: MRF
    metastasi: Metastasi


    @field_validator(
        "posizione",
        "infiltrazione_organi_dettagli",
        "sedi_linfonodi",
        mode="before",
    )
    @classmethod
    def unique_list(cls, v: Any):
        if not isinstance(v, list):
            return v
        return list(dict.fromkeys(v))


    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
    )
    
    
class AnnotatedReport(BaseModel):
    report_text: str
    report_data: Annotations

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
    )
    

##############################
# Modello senza campi numerici
##############################
class AnnotationsReduced(BaseModel):
    # Tumore primitivo
    morfologia: Morfologia
    posizione: List[Posizione] = Field(default_factory=list)
    riflessione_peritoneale_anteriore: RiflessionePeritonealeAnteriore
    infiltrazione_tessuto_adiposo: InfiltrazioneTessutoAdiposo
    infiltrazione_sfinteri: InfiltrazioneSfinteri
    infiltrazione_organi_extra: InfiltrazioneOrganiExtra
    infiltrazione_organi_dettagli: List[InfiltrazioneOrganiDettagli] = Field(default_factory=list)
    coinvolgimento_riflessione_peritoneale: CoinvolgimentoRiflessionePeritoneale
    coinvolgimento_fascia_mesorettale: CoinvolgimentoFasciaMesorettale
    # Linfonodi Sospetti
    #linfonodi_sospetti: int  # linfonodi sospetti si potrebbe discretizzare e far diventare categorico
    numero_linfonodi_non_conosciuto: NumeroLinfonodiNonConosciuto
    sedi_linfonodi: List[SediLinfonodi] = Field(default_factory=list)
    depositi_tumorali: DepositiTumorali
    # Conclusioni
    emvi: EMVI  
    stadio_T: StadioT
    stadio_N: StadioN  
    stadio_N1c: StadioN1c
    mrf: MRF
    metastasi: Metastasi
    
    @field_validator(
        "posizione",
        "infiltrazione_organi_dettagli",
        "sedi_linfonodi",
        mode="before",
    )
    @classmethod
    def unique_list(cls, v: Any):
        if not isinstance(v, list):
            return v
        return list(dict.fromkeys(v))    

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
    )

class AnnotatedReportReduced(BaseModel):
    report_text: str
    report_data: AnnotationsReduced

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
    )
    



if __name__ == '__main__':
    from openai.lib._pydantic import to_strict_json_schema
    from pprint import pprint
    schema = to_strict_json_schema(RectalCancerStagingData)
    pprint(schema)