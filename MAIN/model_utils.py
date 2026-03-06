from pydantic import BaseModel
from enum import Enum
from typing import Union, get_type_hints, get_origin, get_args
from constants import NAN_VALUE, Flag
import json
from ast import literal_eval
import pandas as pd

from pathlib import Path


def get_regression_fields(model: type[BaseModel]) -> list[str]:
    """
    Analizza il modello Pydantic e restituisce i campi numerici
    (int o float) per i quali serve regressione.
    """
    regression_fields = []
    for name, field in model.model_fields.items():
        field_type = field.annotation
        origin = get_origin(field_type) or field_type

        # Consideriamo int e float come regressione
        if origin in (int, float):
            regression_fields.append(name)
        # Consideriamo anche Optional[int]/Optional[float]
        elif origin is Union:
            args = get_args(field_type)
            if any(arg in (int, float) for arg in args):
                regression_fields.append(name)
    return regression_fields

def get_optional_regression_fields(model: type[BaseModel]) -> list[str]:
    """
    Restituisce i campi numerici opzionali (Optional[int] o Optional[float]).
    """
    result = []

    for name, field in model.model_fields.items():
        annotation = field.annotation
        origin = get_origin(annotation)

        # Caso Optional[...] → Union[T, NoneType]
        if origin is Union:
            args = get_args(annotation)

            # Controlla se uno degli argomenti è numerico
            has_numeric = any(a in (int, float) for a in args)
            has_none = type(None) in args

            if has_numeric and has_none:
                result.append(name)

    return result

def is_multilabel_model(cls: type[BaseModel]) -> bool:
    if not issubclass(cls, BaseModel):
        return False
    for f in cls.model_fields.values():
        if not (isinstance(f.annotation, type) and issubclass(f.annotation, Enum)):
            return False
        if f.annotation.__name__ != "Flag":
            return False
    return True

def get_multiple_choice_fields(model: type[BaseModel]) -> list[str]:
    fields = []
    for name, field in model.model_fields.items():
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            if is_multilabel_model(annotation):
                fields.append(name)
    return fields

def get_binary_classification_fields(model: type[BaseModel]) -> list[str]:
    fields = []
    for name, field in model.model_fields.items():
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, Enum):
            if len(annotation.__members__) == 2:
                fields.append(name)
    return fields

def get_classification_fields(model: type[BaseModel]) -> list[str]:
    """
    Restituisce i campi di classificazione multiclass (Enum con >= 3 valori).
    Esclude automaticamente i campi binari.
    """
    fields = []
    for name, field in model.model_fields.items():
        annotation = field.annotation

        # Deve essere un Enum
        if isinstance(annotation, type) and issubclass(annotation, Enum):
            # Escludi i binari (Enum con 2 membri)
            if len(annotation.__members__) >= 3:
                fields.append(name)

    return fields

def get_field_values(model: type[BaseModel]) -> dict[str, list[str]]:
    """Restituisce un dizionario: campo -> lista di valori possibili (solo stringhe)."""
    result = {}

    for name, field in model.model_fields.items():
        ann = field.annotation

        # Caso 1: Enum (multiclass o binario)
        if isinstance(ann, type) and issubclass(ann, Enum):
            result[name] = [m.value for m in ann.__members__.values()]
            continue

        # Caso 2: multilabel → lista dei nomi dei flag
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            if is_multilabel_model(ann):
                result[name] = list(ann.model_fields.keys())
                continue

        # Caso 3: Optional numerico → lista vuota
        origin = get_origin(ann)
        if origin is Union:
            args = get_args(ann)
            if any(a in (int, float) for a in args):
                result[name] = []
                continue

        # Caso 4: numerico non opzionale → lista vuota
        if ann in (int, float):
            result[name] = []
            continue

        # Default
        result[name] = []

    return result


def get_number_of_classes(model: type[BaseModel]) -> dict[str, int]:
    """
    Restituisce il numero di classi per ogni campo del modello.
    """
    field_values = get_field_values(model)
    result = dict()
    for field_name, values in field_values.items():
        result[field_name] = len(values)
    return result

def create_label_to_id_map(model: type[BaseModel]) -> dict[str, dict[str, dict[str, int]]]:
    """
    Crea due mappe per ogni campo categorico di model: label to ID e la corrispondente ID to label.
    {
        field_name: {
            "label_to_id": {label: id, ...},
            "id_to_label": {id: label, ...}
        },
    }
    """
    field_values = get_field_values(model)
    result = dict()
    for field_name, values in field_values.items():
        result[field_name] = {
            'label_to_id': {label: idx for idx, label in enumerate(values)},
            'id_to_label': {idx: label for idx, label in enumerate(values)},
        }
    return result

def labels_to_bits(labels: list[str], label_to_id_map: dict[str, int]) -> list[int]:
    result = [0] * len(label_to_id_map)
    for label in labels:
        if label not in label_to_id_map:
            raise KeyError(f"Label sconosciuta: {label}")
        result[label_to_id_map[label]] = 1
    return result

def bits_to_labels(bits: list[int], id_to_label_map: dict[int, str]) -> list[str]:
    result = []
    for i, bit in enumerate(bits):
        if bit == 1:
            result.append(id_to_label_map[i])
    return result


def from_list_to_flags(possible_values: list[str], values: list[str]) -> dict[str, str]:
    """
    Converte una lista di valori in un dizionario di flag per un FlagModel.
    """
    for v in values:
        if v not in possible_values:
            raise ValueError(f"Valore non valido per multilabel: {v}")
    result = dict()
    for p in possible_values:
        if p in values:
            result[p] = Flag.SI.value
        else:
            result[p] = Flag.NO.value
    return result

def from_series_to_basemodel(series: pd.Series, ann_model: type[BaseModel]) -> BaseModel:
    """
    Converte una pd.Series in un'istanza del modello Pydantic ann_model.
    Gestisce automaticamente:
    - campi numerici
    - campi multiclass (Enum)
    - campi binari
    - campi multilabel (FlagModel)
    """
    data = {}
    field_values = get_field_values(ann_model)

    for field_name, field_info in ann_model.model_fields.items():
        ann = field_info.annotation
        raw_value = series.get(field_name)

        # Caso 1: multilabel → lista di flag attivi
        if isinstance(ann, type) and issubclass(ann, BaseModel) and is_multilabel_model(ann):
            possible_flags = field_values[field_name]  # es. ["basso", "medio", ...]
            
            # raw_value può essere:
            # - una lista
            # - una stringa tipo "['basso','alto']"
            # - NaN → nessun flag attivo
            if pd.isna(raw_value):
                active_flags = []
            elif isinstance(raw_value, list):
                active_flags = raw_value
            else:
                active_flags = literal_eval(raw_value)

            data[field_name] = from_list_to_flags(possible_flags, active_flags)
            continue

        # Caso 2: Enum (multiclass o binario)
        if isinstance(ann, type) and issubclass(ann, Enum):
            # raw_value deve essere una stringa valida
            data[field_name] = None if pd.isna(raw_value) else raw_value
            continue

        # Caso 3: numerico (int/float o Optional)
        if pd.isna(raw_value):
            data[field_name] = None
        else:
            data[field_name] = raw_value

    return ann_model(**data)


def from_basemodel_to_series(model_instance: BaseModel) -> pd.Series:
    """
    Converte un'istanza Pydantic in una pd.Series.
    Gestisce automaticamente:
    - campi numerici
    - campi multiclass (Enum)
    - campi binari
    - campi multilabel (FlagModel)
    """
    data = {}
    model_cls = type(model_instance)

    for field_name, field_info in model_cls.model_fields.items():
        ann = field_info.annotation
        value = getattr(model_instance, field_name)

        # Caso 1: multilabel → lista di flag attivi
        if isinstance(ann, type) and issubclass(ann, BaseModel) and is_multilabel_model(ann):
            active_flags = []
            for flag_name, flag_value in value.items():
                if flag_value == Flag.SI.value:
                    active_flags.append(flag_name)
            data[field_name] = active_flags
            continue

        # Caso 2: Enum (multiclass o binario)
        if isinstance(ann, type) and issubclass(ann, Enum):
            data[field_name] = value
            continue

        # Caso 3: numerico o Optional
        data[field_name] = value

    return pd.Series(data)

                
def flags_to_bits(flags: dict[str, str]) -> list[int]:
    """
    Converte un dizionario di flag ("si"/"no") in una lista di bit (1/0).
    L'ordine dei bit segue l'ordine delle chiavi nel dizionario.
    """
    result = []
    for key, value in flags.items():
        if value == Flag.SI.value:
            result.append(1)
        elif value == Flag.NO.value:
            result.append(0)
        else:
            raise ValueError(f"Valore non valido per flag '{key}': {value}")
    return result
                
                
if __name__ == "__main__":
    import constants
    from pprint import pprint
    model_type = constants.RectalCancerStagingData
    reg_fields = get_regression_fields(model_type)
    opt_reg_fields = get_optional_regression_fields(model_type)
    cl_fields = get_classification_fields(model_type)
    mc_fields = get_multiple_choice_fields(model_type)
    bc_fields = get_binary_classification_fields(model_type)
    label_to_id_map = create_label_to_id_map(model_type)
    print(f'regression:\n{reg_fields}')
    print(f'opt regression:\n{opt_reg_fields}')
    print(f'classification:\n{cl_fields}')
    print(f'binary:\n{bc_fields}')
    print(f'multilabel:\n{mc_fields}')
    field_values = get_field_values(model_type)
    pprint(field_values)
    n_classes = get_number_of_classes(model_type)
    pprint(n_classes)
    label_to_id_map = create_label_to_id_map(model_type)
    pprint(label_to_id_map)
