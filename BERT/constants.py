from pydantic import BaseModel, ConfigDict
from enum import Enum
from typing import Optional, List, Union, get_type_hints, get_origin, get_args


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
    Si5mm = "si_5mm"
    Si5mmPlus = "si_5mm_plus"
    Sospetto = "sospetto"

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
    Plus = "+"
    Minus = "-"

class Metastasi(str, Enum):
    MX = "MX"
    M1 = "M1"

class Annotations(BaseModel):
    morfologia: Optional[Morfologia]
    posizione: List[Posizione]
    ore_inizio: Optional[int]
    ore_fine: Optional[int]
    spessore_parietale: Optional[int]
    estensione_cranio_caudale: Optional[int]
    distanza_oai: Optional[float]
    riflessione_peritoneale_anteriore: Optional[RiflessionePeritonealeAnteriore]
    infiltrazione_tessuto_adiposo: Optional[InfiltrazioneTessutoAdiposo]
    infiltrazione_sfinteri: Optional[SiNo]
    infiltrazione_organi_extra: Optional[SiNoSospetto]
    infiltrazione_organi_dettagli: List[InfiltrazioneOrganiDettagli]
    coinvolgimento_riflessione_peritoneale: Optional[SiNo]
    coinvolgimento_fascia_mesorettale: Optional[SiNo]
    numero_linfonodi_non_conosciuto: bool
    linfonodi_sospetti: Optional[int]
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


def get_field_values(model: type[BaseModel]) -> dict[str, list[str]]:
    """
    Restituisce un dizionario con i campi Enum e bool del modello
    e i valori possibili per ciascuno.
    """
    field_values = {}
    hints = get_type_hints(model)

    for field_name, field_type in hints.items():
        # Gestione Optional
        if get_origin(field_type) is Union:
            args = [t for t in get_args(field_type) if t is not type(None)]
            if args:
                field_type = args[0]

        # Gestione List
        if get_origin(field_type) is list:
            args = get_args(field_type)
            if args:
                field_type = args[0]

        # Se è un Enum
        if isinstance(field_type, type) and issubclass(field_type, Enum):
            field_values[field_name] = [e.value for e in field_type]

        # Se è un bool
        elif field_type is bool:
            field_values[field_name] = [True, False]

    return field_values

        
if __name__ == "__main__":
    from pprint import pprint

    pprint(get_field_values(Annotations))