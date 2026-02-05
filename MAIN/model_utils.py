from pydantic import BaseModel
from enum import Enum
from typing import Union, get_type_hints, get_origin, get_args
from constants import NAN_VALUE, Flag, AnnotatedReportReduced
import json
from ast import literal_eval
import pandas as pd

from pathlib import Path


def unwrap_type(t):
    """Rimuove Optional, Union e List, restituendo il tipo base."""
    origin = get_origin(t)
    if origin is Union:
        args = [a for a in get_args(t) if a is not type(None)]
        return unwrap_type(args[0]) if args else t
    if origin is list:
        args = get_args(t)
        return unwrap_type(args[0]) if args else t
    return t

def is_optional_type(t):
    return get_origin(t) is Union and type(None) in get_args(t)

def is_flag_model(t):
    """Riconosce un FlagModel: BaseModel i cui campi sono tutti Enum."""
    if not (isinstance(t, type) and issubclass(t, BaseModel)):
        return False
    hints = t.__annotations__
    return all(
        isinstance(unwrap_type(ft), type) and issubclass(unwrap_type(ft), Enum)
        for ft in hints.values()
    )

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
    Analizza il modello Pydantic e restituisce i campi numerici che possono anche essere nulli.
    """
    regression_fields = get_regression_fields(model)
    result = []
    for field in regression_fields:
        field_type = model.model_fields[field].annotation
        origin = get_origin(field_type) or field_type
        if origin is Union:
            args = get_args(field_type)
            if type(None) in args:
                result.append(field)
    return result

def get_multiple_choice_fields(model: type[BaseModel]) -> list[str]:
    """
    Analizza il modello Pydantic e restituisce i campi per i quali è possibile selezionare più di una classe.
    """
    result = []
    for name, field in model.model_fields.items():
        field_type = field.annotation
        origin = get_origin(field_type) or field_type
        if origin is list:
            result.append(name)
    return result

def get_binary_classification_fields(model: type[BaseModel]) -> list[str]:
    """
    Restituisce i campi di classificazione binaria (due classi),
    includendo anche i sotto-campi dei FlagModel.
    """
    regression_fields = get_regression_fields(model)
    multiple_choice_fields = get_multiple_choice_fields(model)
    num_classes = get_number_of_classes(model)

    binary_fields = []

    for name, field in model.model_fields.items():
        field_type = unwrap_type(field.annotation)

        # Escludi regressione
        if name in regression_fields:
            continue

        # Escludi multilabel (liste)
        if name in multiple_choice_fields:
            continue

        # Caso 1: campo normale → binario se num_classes == 2
        if num_classes.get(name, 0) == 2:
            binary_fields.append(name)
            continue

        # Caso 2: FlagModel → includi i sotto-campi
        if is_flag_model(field_type):
            sub_hints = field_type.__annotations__
            for sub_name in sub_hints.keys():
                full_name = f"{name}_{sub_name}"
                # Ogni flag ha sempre 2 classi
                binary_fields.append(full_name)

    return binary_fields

def get_classification_fields(model: type[BaseModel]) -> list[str]:
    """
    Restituisce i campi di classificazione (non numerici, non multi-scelta,
    non FlagModel, e con numero di classi > 2).
    """
    regression_fields = get_regression_fields(model)
    multiple_choice_fields = get_multiple_choice_fields(model)
    num_classes = get_number_of_classes(model)

    classification_fields = []

    for name, field in model.model_fields.items():
        field_type = unwrap_type(field.annotation)
        # Escludi regressione
        if name in regression_fields:
            continue
        # Escludi multilabel (liste)
        if name in multiple_choice_fields:
            continue
        # Escludi FlagModel
        if is_flag_model(field_type):
            continue
        # Escludi binari (num_classes <= 2)
        if num_classes.get(name, 0) <= 2:
            continue
        # Se arrivi qui → è classificazione multiclass
        classification_fields.append(name)

    return classification_fields

def get_field_values(model: type[BaseModel]) -> dict[str, list[str]]:
    field_values = {}
    hints = model.__annotations__

    for field_name, field_type in hints.items():
        optional = is_optional_type(field_type)
        base_type = unwrap_type(field_type)

        # --- Caso 1: Enum ---
        if isinstance(base_type, type) and issubclass(base_type, Enum):
            values = [e.value for e in base_type]
            if optional:
                values.append(NAN_VALUE)
            field_values[field_name] = values
            continue

        # --- Caso 2: bool ---
        if base_type is bool:
            values = [False, True]
            if optional:
                values.append(NAN_VALUE)
            field_values[field_name] = values
            continue

        # --- Caso 3: FlagModel (BaseModel con campi Enum/Flag) ---
        if isinstance(base_type, type) and issubclass(base_type, BaseModel):
            # Controllo che sia davvero un FlagModel (tutti i campi Enum o Flag)
            sub_hints = base_type.__annotations__
            is_flag_model = all(
                isinstance(unwrap_type(t), type) and issubclass(unwrap_type(t), Enum)
                for t in sub_hints.values()
            )

            if is_flag_model:
                for sub_name, sub_type in sub_hints.items():
                    sub_base = unwrap_type(sub_type)
                    values = [e.value for e in sub_base]
                    if optional:
                        values.append(NAN_VALUE)
                    field_values[f"{field_name}_{sub_name}"] = values

    return field_values

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
        result[label_to_id_map[label]] = 1
    return result

def bits_to_labels(bits: list[int], id_to_label_map: dict[int, str]) -> list[str]:
    result = []
    for i, bit in enumerate(bits):
        if bit == 1:
            result.append(id_to_label_map[i])
    return result

def genera_schema_json_per_prompt(model: type[BaseModel]) -> dict:
    """
    Genera lo schema JSON per il prompt, basato sui campi del modello Pydantic.
    """
    field_values = get_field_values(model)
    schema = {}
    for field_name, values in field_values.items():
        schema[field_name] = {
            "type": "string" if len(values) <= 10 else "array",
            "enum": values
        }
    return schema

def field_is_flag_model(field: str, model: type[BaseModel]) -> bool:
    field_type = unwrap_type(model.model_fields[field].annotation)
    return is_flag_model(field_type)

def from_list_to_flags(possible_values: list[str], values: list[str]) -> dict:
    """
    Converte una lista di valori in un dizionario di flag per un FlagModel.
    """
    result = dict()
    for p in possible_values:
        if p in values:
            result[p] = Flag.SI.value
        else:
            result[p] = Flag.NO.value
    return result

def from_series_to_basemodel(series: pd.Series, ann_model: type[BaseModel]) -> BaseModel:
    data_dict = dict()
    reg_fields = get_regression_fields(ann_model)
    mc_fields = get_multiple_choice_fields(ann_model)
    for field in ann_model.model_fields.keys():
        v = series[field]
        if field in reg_fields:
            if pd.isna(v):
                data_dict[field] = None
            else:
                data_dict[field] = v
        elif field_is_flag_model(field, ann_model):
            v = literal_eval(v)
            possible_values = list(ann_model.model_fields[field].annotation.model_fields.keys())
            data_dict[field] = from_list_to_flags(possible_values, v)
        elif field in mc_fields:
            data_dict[field] = literal_eval(v)
        else:
            data_dict[field] = v
    return ann_model(**data_dict)

def annotated_report_to_mistral_dict(annotated_report: type[BaseModel]) -> dict:
    assert isinstance(annotated_report, AnnotatedReportReduced)
    result = {
        'text': annotated_report.report_text,
        'labels': dict()
    }
    for k, v in annotated_report.report_data.model_dump().items():
            result['labels'][k] = v
    return result
  
                
if __name__ == "__main__":
    pass