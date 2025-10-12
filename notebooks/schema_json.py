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

class SediLinfonodiLocoregionali(BaseModel):
    mesorettali: bool
    rettali_superiori: bool 
    mesenterici_inferiori: bool 
    iliaci_interni: bool 
    otturatori: bool 
    sacrali: bool 
    inguinali_sotto_dentata: bool 

class SediLinfonodiNonLocoregionali(BaseModel):
    inguinali: bool
    iliaci_esterni: bool 
    iliaci_comuni: bool 
    paraortici: bool 
    altri: bool 

class ReportData(BaseModel):
    morfologia: Morfologia
    posizione: Posizione
    spessore_parietale: int | None
    estensione_cranio_caudale: int | None
    distanza_oai: int | None
    riflessione_peritoneale_anteriore: RiflessionePeritonealeAnteriore
    infiltrazione_tessuto_adiposo: InfiltrazioneTessutoAdiposo
    infiltrazione_sfinteri: InfiltrazioneSfinteri
    infiltrazione_organi_extra: InfiltrazioneOrganiExtra
    coinvolgimento_riflessione_peritoneale: CoinvolgimentoRiflessionePeritoneale
    coinvolgimento_fascia_mesorettale: CoinvolgimentoFasciaMesorettale
    linfonodi_sospetti: int
    sedi_linfonodi_locoregionali: SediLinfonodiLocoregionali
    sedi_linfonodi_non_locoregionali: SediLinfonodiNonLocoregionali
    depositi_tumorali: DepositiTumorali
    emvi_esteso: EMVIEsteso
    carcinosi_peritoneale: CarcinosiPeritoneale
    lesioni_ossee: LesioniOssee