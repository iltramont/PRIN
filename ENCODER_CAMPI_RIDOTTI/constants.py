#############################
# --- Reduced constants --- #
#############################

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
    Altro = "altro"  # Comprende anche mucinoso
    
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
    NaN = NAN_VALUE
    """
    # Sopra non è mai presente nel dataset, quindi la escludiamo
    Sopra = "sopra"
    """

# TODO verificare eventuali aggregazioni da fare eventualmente chiedere a Ilaria
class InfiltrazioneTessutoAdiposo(str, Enum):      
    No = "no"  # Comprende anche NaN
    Si5mm = "si_5mm"  # Comprende anche sospetto
    Si5mmPlus = "si_5mm_plus"

class StadioT(str, Enum):
    T1_2_NaN = "T1-2/NaN"
    T3ab = "T3ab"
    T3cd = "T3cd"
    T4 = "T4"
    """
    # Aggreghiamo le sottoclassi di T4
    T4a = "T4a"
    T4b = "T4b"
    """


class AnnotationsReduced(BaseModel):
    morfologia: Morfologia  # solido_polipoide, solido_anulare, altro. Mucinoso viene incluso in altro
    posizione: List[Posizione]
    riflessione_peritoneale_anteriore: RiflessionePeritonealeAnteriore  # sotto, cavallo, NaN
    infiltrazione_tessuto_adiposo: InfiltrazioneTessutoAdiposo  # no, si_5mm, si_5mm_plus
    infiltrazione_sfinteri: bool  # True: evidenza. False: no evidenza
    infiltrazione_organi_extra: bool  # True: evidenza. False: no evidenza
    infiltrazione_organi_dettagli: List[InfiltrazioneOrganiDettagli]
    coinvolgimento_riflessione_peritoneale: bool  # True: evidenza. False: no evidenza
    coinvolgimento_fascia_mesorettale: bool  # True: evidenza. False: no evidenza
    sedi_linfonodi: List[SediLinfonodi]
    depositi_tumorali: bool  # True: evidenza. False: no evidenza
    emvi_esteso: bool  # True: evidenza. False: no evidenza
    stadio_T: StadioT
    stadio_N: bool  # True: N+ (comprende N1, N2). False: N0 (no evidenza di linfonodi intaccati)
    stadio_N1c: bool  # True: evidenza. False: no evidenza
    mrf: bool  # True: evidenza. False: no evidenza
    emvi: bool  # True: evidenza. False: no evidenza
    metastasi: bool  # True: M1. False: no evidenza di metastasi o MX

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
    