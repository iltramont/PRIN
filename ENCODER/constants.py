from pydantic import BaseModel, ConfigDict
from enum import Enum
from typing import Optional, List

NAN_VALUE = 'NaN'
SEED = 2026

BERT_ENCODER_CHECKPOINT = "bert-base-multilingual-cased"
XLM_ROBERTA_ENCODER_CHECKPOINT = "FacebookAI/xlm-roberta-base"
XLM_ROBERTA_LARGE_ENCODER_CHECKPOINT = "FacebookAI/xlm-roberta-large"
BIOBERT_ITALIAN_ENCODER = "IVN-RIN/bioBIT"


class Morfologia(str, Enum):
    SolidoPolipoide = "solido_polipoide"
    SolidoAnulare = "solido_anulare"
    Mucinoso = "mucinoso"
    
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

class RiflessionePeritonealeAnteriore(str, Enum):
    Sotto = "sotto"
    Cavallo = "cavallo"
    Sopra = "sopra"

class InfiltrazioneTessutoAdiposo(str, Enum):      
    No = "no"
    Sospetto = "sospetto"
    #Si = "si"
    Si5mm = "si_5mm"
    Si5mmPlus = "si_5mm_plus"

class SiNo(str, Enum):
    No = "no"
    Si = "si"
    
class SiNoSospetto(str, Enum):
    No = "no"
    Sospetto = "sospetto"
    Si = "si"

class StadioT(str, Enum):
    T1_2 = "T1-2"
    T3ab = "T3ab"
    T3cd = "T3cd"
    T4a = "T4a"
    T4b = "T4b"
    
class StadioN(str, Enum):
    N0 = "N0"
    N1a = "N1a"
    N1b = "N1b"
    N1c = "N1c"
    N2a = "N2a"
    N2b = "N2b"
    NPlus = "N+"
     
class PlusMinus(str, Enum):
    Minus = "-"
    Plus = "+"

class Metastasi(str, Enum):
    MX = "MX"
    M1 = "M1"

class Annotations(BaseModel):
    morfologia: Optional[Morfologia]
    posizione: List[Posizione]
    ore_inizio: Optional[int]
    ore_inizio_is_nan: bool
    ore_fine: Optional[int]
    ore_fine_is_nan: bool
    spessore_parietale: Optional[int]
    spessore_parietale_is_nan: bool
    estensione_cranio_caudale: Optional[int]
    estensione_cranio_caudale_is_nan: bool
    distanza_oai: Optional[float]
    distanza_oai_is_nan: bool
    riflessione_peritoneale_anteriore: Optional[RiflessionePeritonealeAnteriore]
    infiltrazione_tessuto_adiposo: Optional[InfiltrazioneTessutoAdiposo]
    infiltrazione_sfinteri: Optional[SiNo]
    infiltrazione_organi_extra: Optional[SiNoSospetto]
    infiltrazione_organi_dettagli: List[InfiltrazioneOrganiDettagli]
    coinvolgimento_riflessione_peritoneale: Optional[SiNo]
    coinvolgimento_fascia_mesorettale: Optional[SiNo]
    linfonodi_sospetti: Optional[int]
    numero_linfonodi_non_conosciuto: bool
    sedi_linfonodi: List[SediLinfonodi]
    depositi_tumorali: Optional[SiNoSospetto]
    numero_depositi: Optional[int]
    emvi_esteso: Optional[SiNo]
    stadio_T: Optional[StadioT]
    stadio_N: Optional[StadioN]
    stadio_N1c: bool
    mrf: Optional[PlusMinus]
    emvi: Optional[PlusMinus]
    metastasi: Optional[Metastasi]

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
    