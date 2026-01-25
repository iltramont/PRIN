#####################
# --- Constants --- #
#####################

from pydantic import BaseModel, ConfigDict
from enum import Enum
from typing import List, Optional

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


# Raw data fields
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
class Posizione(str, Enum):
    Basso = 'basso'
    Medio = 'medio'
    Alto = 'alto'
    Giunzione = 'giunzione'
    
class InfiltrazioneOrganiDettagli(str, Enum):
    PavimentoPelvico =  'pavimento_pelvico'
    Altro = 'altro'
    
class SediLinfonodi(str, Enum):
    Mesorettali = 'mesorettali'
    RettaliSuperiori = 'rettali_superiori'
    Otturatori = 'otturatori'
    Iliaci = 'iliaci'
    Altro = 'altro'

##################
# Campi multiclass
##################
class Morfologia(str, Enum):
    SolidoPolipoide = "solido_polipoide"
    SolidoAnulare = "solido_anulare"
    Mucinoso = "mucinoso"

class InfiltrazioneTessutoAdiposo(str, Enum):      
    No = "no" 
    Si5mm = "si_5mm"
    Si5mmPlus = "si_5mm_plus"

class RiflessionePeritonealeAnteriore(str, Enum):
    Sotto = "sotto"
    Cavallo = "cavallo"
    NaN = NAN_VALUE

class StadioT(str, Enum):
    T1_2 = "T1-2"
    T3ab = "T3ab"
    T3cd = "T3cd"
    T4a = "T4a"
    T4b = "T4b"
    
class StadioN(str, Enum):
    N0 = "N0"
    N1 = "N1"
    N2 = "N2"
    N_plus = "N+"

##############    
# Campi binari
##############
class CoinvolgimentoFasciaMesorettale(str, Enum):
    No = "no"
    Si = "si"
    
class CoinvolgimentoRiflessionePeritoneale(str, Enum):
    No = "no"
    Si = "si"
    
class StadioN1c(str, Enum):
    No = "no"
    Si = "si"
    
class MRF(str, Enum):
    Minus = "-"
    Plus = "+"

class EMVI(str, Enum):
    Minus = "-"
    Plus = "+"

class Metastasi(str, Enum):
    MX = "MX"
    M1 = "M1"
    
class InfiltrazioneSfinteri(str, Enum):
    No = "no"
    Si = "si"
    
class InfiltrazioneOrganiExtra(str, Enum):
    No = "no"
    Si = "si"
    
class NumeroLinfonodiNonConosciuto(str, Enum):
    No = "no"
    Si = "si"
    
class DepositiTumorali(str, Enum):
    No = "no"
    Si = "si"

###########################################
# Modello completo con anche campi numerici
###########################################
class Annotations(BaseModel):
    # Tumore primitivo
    morfologia: Morfologia
    ore_inizio: Optional[int]
    ore_fine: Optional[int]
    spessore_parietale: Optional[float]
    estensione_cranio_caudale: Optional[float]
    distanza_oai: Optional[float]
    posizione: List[Posizione]
    riflessione_peritoneale_anteriore: RiflessionePeritonealeAnteriore
    infiltrazione_tessuto_adiposo: InfiltrazioneTessutoAdiposo
    infiltrazione_sfinteri: InfiltrazioneSfinteri
    infiltrazione_organi_extra: InfiltrazioneOrganiExtra
    infiltrazione_organi_dettagli: List[InfiltrazioneOrganiDettagli]
    coinvolgimento_riflessione_peritoneale: CoinvolgimentoRiflessionePeritoneale
    coinvolgimento_fascia_mesorettale: CoinvolgimentoFasciaMesorettale
    # Linfonodi Sospetti
    linfonodi_sospetti: int
    sedi_linfonodi: List[SediLinfonodi]
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
    posizione: List[Posizione]
    riflessione_peritoneale_anteriore: RiflessionePeritonealeAnteriore
    infiltrazione_tessuto_adiposo: InfiltrazioneTessutoAdiposo
    infiltrazione_sfinteri: InfiltrazioneSfinteri
    infiltrazione_organi_extra: InfiltrazioneOrganiExtra
    infiltrazione_organi_dettagli: List[InfiltrazioneOrganiDettagli]
    coinvolgimento_riflessione_peritoneale: CoinvolgimentoRiflessionePeritoneale
    coinvolgimento_fascia_mesorettale: CoinvolgimentoFasciaMesorettale
    # Linfonodi Sospetti
    #linfonodi_sospetti: int  # linfonodi sospetti si potrebbe discretizzare e far diventare categorico
    sedi_linfonodi: List[SediLinfonodi]
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

class AnnotatedReportReduced(BaseModel):
    report_text: str
    report_data: AnnotationsReduced

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
    )