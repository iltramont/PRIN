#####################
# --- Constants --- #
#####################

from pydantic import BaseModel, ConfigDict
from enum import Enum, Optional
from typing import List

NAN_VALUE = 'NaN'
SEED = 2026

BERT_ENCODER_CHECKPOINT = "bert-base-multilingual-cased"
XLM_ROBERTA_ENCODER_CHECKPOINT = "FacebookAI/xlm-roberta-base"
XLM_ROBERTA_LARGE_ENCODER_CHECKPOINT = "FacebookAI/xlm-roberta-large"
BIOBERT_ITALIAN_ENCODER = "IVN-RIN/bioBIT"

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

#########
# Modello
#########
class Annotations(BaseModel):
    # Tumore primitivo
    morfologia: Morfologia
    #ore_inizio: Optional[int]
    #ore_fine: Optional[int]
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
    