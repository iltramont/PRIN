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

class StadioT(str, Enum):
    T3cd = "T3cd"
    T4b = "T4b"
    T3ab = "T3ab"
    T4a = "T4a"
    T1_2 = "T1-2"
    
class StadioN(str, Enum):
    N2a = "N2a"
    N1b = "N1b"
    N0 = "N0"
    N_plus = "N+"
    N2b = "N2b"
    N1a = "N1a"
    N1c = "N1c"
    
class Mrf(str, Enum):
    plus = "+"
    minus = "-"
    
class Emvi(str, Enum):
    plus = "+"
    minus = "-"

class Metastasi(str, Enum):
    MX = "MX"
    M1 = "M1"
    

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
    stadio_T: StadioT | None
    stadio_N: StadioN | None
    stadio_N1c: bool
    mrf: Mrf | None
    emvi: Emvi | None
    metastasi: Metastasi | None


class ReportDataOneLevel(BaseModel):
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
    linfonodi_mesorettali: bool
    linfonodi_rettali_superiori: bool
    linfonodi_mesenterici_inferiori: bool
    linfonodi_iliaci_interni: bool
    linfonodi_otturatori: bool
    linfonodi_sacrali: bool
    linfonodi_inguinali_sotto_dentata: bool
    linfonodi_inguinali: bool
    linfonodi_iliaci_esterni: bool 
    linfonodi_iliaci_comuni: bool 
    linfonodi_paraortici: bool 
    linfonodi_altri: bool     
    depositi_tumorali: DepositiTumorali | None
    numero_depositi: int | None
    emvi_esteso: EMVIEsteso | None
    carcinosi_peritoneale: CarcinosiPeritoneale | None
    lesioni_ossee: LesioniOssee | None    
    stadio_T: StadioT | None
    stadio_N: StadioN | None
    stadio_N1c: bool
    mrf: Mrf | None
    emvi: Emvi | None
    metastasi: Metastasi | None