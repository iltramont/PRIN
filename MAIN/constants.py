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
# File names
############
RAW_DATA_FILE_NAME = "base.tumoreprimitivo_finale.csv"
CLEAN_DATA_FILE_NAME = "tumoreprimitivo_clean.csv"
TRAIN_SPLIT_FILE_NAME = 'train_split_reduced'
TEST_SPLIT_FILE_NAME = 'test_split_reduced'
VALIDATION_SPLIT_FILE_NAME = 'validation_split_reduced'

#################
# Raw data fields
#################
REPORT_COLUMN_NAME = "report_text"
ANNOTATOR_COLUMN_NAME = "report_text"
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
    
class PosizioneFlag(BaseModel):
    basso: Flag
    medio: Flag
    alto: Flag
    giunzione: Flag   
    
class InfiltrazioneOrganiDettagli(str, Enum):
    PAVIMENTO_PELVICO =  "pavimento_pelvico"
    ALTRO = "altro"

class InfiltrazioneOrganiDettagliFlag(BaseModel):
    pavimento_pelvico: Flag
    altro: Flag
    
class SediLinfonodi(str, Enum):
    MESORETTALI = "mesorettali"
    RETTALI_SUPERIORI = "rettali_superiori"
    OTTURATORI = "otturatori"
    ILIACI = "iliaci"
    ALTRO = "altro"
    
class SediLinfonodiFlag(BaseModel):
    mesorettali: Flag
    rettali_superiori: Flag
    otturatori: Flag
    iliaci: Flag
    altro: Flag

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
    NO = "no"
    SI = "si"
    
class DepositiTumorali(str, Enum):
    NO = "no"
    SI = "si"

########################################################################
# Modello completo con anche campi numerici. Campi multilabel come liste
########################################################################
class Annotations(BaseModel):
    # Tumore primitivo
    morfologia: Morfologia
    ore_inizio: Optional[int] = None
    ore_fine: Optional[int] = None
    spessore_parietale: Optional[float] = None
    estensione_cranio_caudale: Optional[float] = None
    distanza_oai: Optional[float] = None
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
    
    
########################################################################
# Modello completo con anche campi numerici. Campi multilabel come flag.
########################################################################
class AnnotationsExtended(BaseModel):
    # Tumore primitivo
    morfologia: Morfologia
    ore_inizio: Optional[int] = None
    ore_fine: Optional[int] = None
    spessore_parietale: Optional[float] = None
    estensione_cranio_caudale: Optional[float] = None
    distanza_oai: Optional[float] = None
    posizione: PosizioneFlag
    riflessione_peritoneale_anteriore: RiflessionePeritonealeAnteriore
    infiltrazione_tessuto_adiposo: InfiltrazioneTessutoAdiposo
    infiltrazione_sfinteri: InfiltrazioneSfinteri
    infiltrazione_organi_extra: InfiltrazioneOrganiExtra
    infiltrazione_organi_dettagli: InfiltrazioneOrganiDettagliFlag
    coinvolgimento_riflessione_peritoneale: CoinvolgimentoRiflessionePeritoneale
    coinvolgimento_fascia_mesorettale: CoinvolgimentoFasciaMesorettale
    # Linfonodi Sospetti
    linfonodi_sospetti: int
    sedi_linfonodi: SediLinfonodiFlag
    depositi_tumorali: DepositiTumorali
    # Conclusioni
    emvi: EMVI  
    stadio_T: StadioT
    stadio_N: StadioN  
    stadio_N1c: StadioN1c
    mrf: MRF
    metastasi: Metastasi

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
    )

class AnnotatedReportExtended(BaseModel):
    report_text: str
    report_data: AnnotationsExtended

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
    )