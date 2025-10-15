from pydantic import BaseModel
from enum import Enum


class Morfologia(str, Enum):
    solido_polipoide = "solido_polipoide"
    solido_anulare = "solido_anulare"
    mucinoso = "mucinoso"

class Posizione(str, Enum):
    basso = "basso"
    medio = "medio"
    alto = "alto"
    giunzione = "giunzione"

class RiflessionePeritonealeAnteriore(str, Enum):
    sotto = "sotto"
    cavallo = "cavallo"
    sopra = "sopra"

class InfiltrazioneTessutoAdiposo(str, Enum):      
    no = "no"
    si_5mm = "si_5mm"
    si_5mm_plus = "si_5mm_plus"
    sospetto = "sospetto"

class InfiltrazioneSfinteri(str, Enum):
    no = "no"
    interno = "interno"
    interno_piano = "interno_piano"
    interno_piano_esterno = "interno_piano_esterno"

class InfiltrazioneOrganiExtra(str, Enum):
    no = "no"
    sospetto = "sospetto"
    si = "si"

class CoinvolgimentoRiflessionePeritoneale(str, Enum):
    no = "no"
    rischio = "rischio"
    si = "si"

class CoinvolgimentoFasciaMesorettale(str, Enum):
    no = "no"
    rischio = "rischio"
    si = "si"   

class DepositiTumorali(str, Enum):
    no = "no"
    si = "si"
    sospetto = "sospetto"

class EMVIEsteso(str, Enum):
    no = "no"
    si = "si"
    sospetto = "sospetto"

class CarcinosiPeritoneale(str, Enum):
    no = "no"
    si = "si"

class LesioniOssee(str, Enum):
    no = "no"
    si = "si"

class SediLocoregionali(BaseModel):
    mesorettali: bool
    rettali_superiori: bool 
    mesenterici_inferiori: bool 
    iliaci_interni: bool 
    otturatori: bool 
    sacrali: bool 
    inguinali_sotto_dentata: bool 

class SediNonLocoregionali(BaseModel):
    inguinali: bool
    iliaci_esterni: bool 
    iliaci_comuni: bool 
    paraortici: bool 
    altri: bool 

class ReportData(BaseModel):
    morfologia: Morfologia | None
    posizione: Posizione | None
    spessore_parietale: int | None
    estensione_cranio_caudale: int | None
    distanza_oai: int | None
    riflessione_peritoneale_anteriore: RiflessionePeritonealeAnteriore | None
    infiltrazione_tessuto_adiposo: InfiltrazioneTessutoAdiposo | None
    infiltrazione_sfinteri: InfiltrazioneSfinteri | None
    infiltrazione_organi_extra: InfiltrazioneOrganiExtra | None
    coinvolgimento_riflessione_peritoneale: CoinvolgimentoRiflessionePeritoneale | None
    coinvolgimento_fascia_mesorettale: CoinvolgimentoFasciaMesorettale | None
    linfonodi_sospetti: int
    sedi_locoregionali: SediLocoregionali
    sedi_non_locoregionali: SediNonLocoregionali
    depositi_tumorali: DepositiTumorali | None
    numero_depositi: int | None
    emvi_esteso: EMVIEsteso | None
    carcinosi_peritoneale: CarcinosiPeritoneale | None
    lesioni_ossee: LesioniOssee | None
