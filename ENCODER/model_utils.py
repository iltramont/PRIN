from pydantic import BaseModel
from enum import Enum
from typing import Union, get_type_hints, get_origin, get_args
from constants import NAN_VALUE
import torch


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
    Restituisce i campi di classificazione binaria (due classi).
    """
    regression_fields = get_regression_fields(model)
    multiple_choice_fields = get_multiple_choice_fields(model)
    num_classes = get_number_of_classes(model)
    binary_fields = []
    for name in model.model_fields.keys():
        if (name not in regression_fields) and (name not in multiple_choice_fields) and (num_classes[name] == 2):
            binary_fields.append(name)
    return binary_fields

def get_classification_fields(model: type[BaseModel]) -> list[str]:
    """
    Restituisce i campi di classificazione (non numerici e non multi-scelta).
    """
    regression_fields = get_regression_fields(model)
    multiple_choice_fields = get_multiple_choice_fields(model)
    num_classes = get_number_of_classes(model)
    classification_fields = []
    for name in model.model_fields.keys():
        if (name not in regression_fields) and (name not in multiple_choice_fields) and (num_classes[name] > 2):
            classification_fields.append(name)
    return classification_fields

def get_field_values(model: type[BaseModel]) -> dict[str, list[str]]:
    """
    Restituisce un dizionario con i campi Enum e bool del modello
    e i valori possibili per ciascuno.
    """
    field_values = {}
    hints = get_type_hints(model)
    for field_name, field_type in hints.items():
        # Gestione Optional
        is_optional = (get_origin(field_type) is Union) and (type(None) in get_args(field_type))
        if is_optional:
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
            if is_optional:
                field_values[field_name].append(NAN_VALUE)
        # Se è un bool
        elif field_type is bool:
            field_values[field_name] = [False, True]
            if is_optional:
                field_values[field_name].append(NAN_VALUE)
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


def from_output_to_labels(model_output: dict[torch.Tensor],
                          regression_fields: list[str],
                          binary_fields: list[str],
                          classification_fields: list[str],
                          multiple_choice_fields: list[str],
                          label_to_id_map: dict[str, dict[str, dict[str, int]]],
                          normalization_stats: dict[str, tuple[float]] | None = None) -> dict:
    with torch.no_grad():
        result = dict()
        for field in regression_fields:
            mu, std = normalization_stats[field]
            tensor = (model_output[field] * std) + mu
            result[field] = tensor.reshape(-1).cpu()
        for field in binary_fields:
            tensor = torch.nn.functional.sigmoid(model_output[field])
            #tensor = tensor > 0.5
            result[field] = tensor.reshape(-1).cpu()
        for field in multiple_choice_fields:
            tensor = torch.nn.functional.sigmoid(model_output[field])
            result[field] = tensor.cpu()
        for field in classification_fields:
            tensor = torch.nn.functional.softmax(model_output[field])
            result[field] = tensor.cpu()
    return result