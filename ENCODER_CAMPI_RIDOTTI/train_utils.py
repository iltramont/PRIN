from datasets import Dataset, DatasetDict
from constants import AnnotatedReport, Annotations, NAN_VALUE
from model_utils import get_optional_regression_fields, get_multiple_choice_fields, labels_to_bits

import numpy as np
import pandas as pd
import torch

from ast import literal_eval


def get_device():
    print(f'{torch.cuda.is_available() = }')  # True se la GPU è disponibile
    print(f'{torch.cuda.device_count() = }')  # Numero di GPU disponibili
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_hugging_face_dataset(annotated_reports: list[AnnotatedReport]) -> Dataset:
    text = [report.report_text for report in annotated_reports]
    return Dataset.from_dict({'text': text})


def add_nan_flag_to_df(df: pd.DataFrame) -> None:
    for col in get_optional_regression_fields(Annotations):
        new_name = f'{col}_is_nan'
        df[new_name] = df[col].isna()
        
        
def create_list_of_annotated_reports(data: pd.DataFrame) -> list[AnnotatedReport]:
    result = []
    mc_fields = get_multiple_choice_fields(Annotations)
    df = data.fillna(NAN_VALUE)
    for _, row in df.iterrows():
        annotations_dict = dict()
        for field in Annotations.model_fields.keys():
            v = row[field]
            if v == NAN_VALUE:
                v = None
            if field in mc_fields:
                v = literal_eval(v)
            annotations_dict[field] = v
        result.append(AnnotatedReport(report_text=row['report_text'], report_data=annotations_dict))
    return result    

def get_normalization_stats(annotated_reports: list[AnnotatedReport], reg_fields: list[str]) -> dict[str, tuple[float]]:
    result = dict()
    for f in reg_fields:
        values = []
        for r in annotated_reports:
            v = getattr(r.report_data, f)
            if v is not None:
                values.append(float(v))
        result[f] = (np.mean(values), np.std(values))
    return result
        
        
def add_target_columns_to_dataset(dataset: Dataset,
                                  annotated_reports: list[AnnotatedReport],
                                  label_to_id_map: dict[str, dict[str, dict[str, int]]],
                                  classification_columns: list[str],
                                  binary_classification_columns: list[str],
                                  multiple_choice_columns: list[str],
                                  regression_columns: list[str],
                                  normalization_stats: dict[str, tuple[float]]) -> Dataset:
    result = dataset
    # Classification fields
    for f in classification_columns:
        target: list[int] = []
        for r in annotated_reports:
            label = getattr(r.report_data, f)
            if label is None:
                label = NAN_VALUE
            id = label_to_id_map[f]['label_to_id'][label]
            target.append(id)
        result = result.add_column(f, target)
    # Binary classification fields
    for f in binary_classification_columns:
        target: list[int] = []
        for r in annotated_reports:
            label = getattr(r.report_data, f)
            id = label_to_id_map[f]['label_to_id'][label]
            target.append(id)
        result = result.add_column(f, target)
    # Regression fields
    for f in regression_columns:
        target: list[float] = []
        for r in annotated_reports:
            value = getattr(r.report_data, f)
            if value is None:
                value = 0
            mu, std = normalization_stats[f]
            value = (float(value) - mu) / std
            target.append(value)
        result = result.add_column(f, target)
    # Multiple choice fields
    for f in multiple_choice_columns:
        target: list[list[int]] = []
        for r in annotated_reports:
            values = getattr(r.report_data, f)
            bits = labels_to_bits(values, label_to_id_map[f]['label_to_id'])
            target.append(bits)
        result = result.add_column(f, target)
    return result
    
    
def model_parameters_info(model: torch.nn.Module):
    total_params = sum(p.numel() for p in model.parameters())  # Total parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # Parametri allenabili (quelli con gradiente)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    # Calcolo parametri totali delle heads
    head_params = 0
    for head in model.heads.values():
        in_features = head.in_features
        out_features = head.out_features
        head_params += in_features * out_features + out_features  # Linear layer params (weights + bias)
    print(f"Extraction heads parameters (weights+bias): {head_params:,}")
    # Calcolo parametri totali dei task adapters
    adapter_params = 0
    for adapter in model.task_adapters.values():
        for layer in adapter:
            if isinstance(layer, torch.nn.Linear):
                adapter_params += layer.in_features * layer.out_features + layer.out_features
    print(f"Task adapters parameters (weights+bias): {adapter_params:,}")

    
def compute_classification_weights(column: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Compute class weights for imbalanced classification.
    Args:
        column: numpy array of shape (num_samples,) with class labels.
        num_classes: total number of classes.
    Returns:
        weights: torch tensor of shape (num_classes,) with weights for each class.
    """
    class_counts = np.zeros(num_classes)
    for i in range(num_classes):
        class_counts[i] = np.sum(column == i)
    # evita divisioni per zero
    class_counts = np.clip(class_counts, a_min=1, a_max=None)
    return class_counts.sum() / (num_classes * class_counts)


def compute_multilabel_pos_weights(matrix: np.ndarray) -> np.ndarray:
    """
    Compute class weights for imbalanced multilabel classification.
    Args:
        matrix: numpy array of shape (num_samples, num_classes) with binary labels (0 or 1).
    Returns:
        weights: torch tensor of shape (num_classes,) with weights for each class.
    """
    num_samples, num_classes = matrix.shape
    positives = matrix.sum(axis=0)
    # evita divisioni per zero
    positives = np.clip(positives, a_min=1, a_max=None)
    pos_weights = (num_samples - positives) / positives
    return pos_weights


def compute_binary_pos_weight(column: np.ndarray) -> np.ndarray:
    """
    Compute positive class weight for imbalanced binary classification.
    Args:
        column: numpy array of shape (num_samples,) with binary class labels (0 or 1).
    Returns:
        weight: float, weight for the positive class.
    """
    positives = np.sum(column == 1)
    negatives = np.sum(column == 0)
    # evita divisioni per zero
    positives = max(positives, 1)
    pos_weight = negatives / positives
    return min(pos_weight, 10.0)  # limita il peso massimo a 10.0


def compute_loss_weights(
    dataset: Dataset,
    classification_columns: list[str],
    binary_classification_columns: list[str],
    multiple_choice_columns: list[str],
    num_classes_dict: dict[str, int],
    device: torch.device | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """
    Compute loss weights for each field, aligned with PyTorch losses.
    """
    weights = dict()

    # Multiclass classification → CrossEntropyLoss(weight=...)
    for f in classification_columns:
        column = np.array(dataset[f])
        weights[f] = {
            "type": "cross_entropy",
            "weight": torch.tensor(compute_classification_weights(column, num_classes_dict[f]), device=device, dtype=torch.float)
        }

    # Binary classification → BCEWithLogitsLoss(pos_weight=...)
    for f in binary_classification_columns:
        column = np.array(dataset[f])
        weights[f] = {
            "type": "binary_bce",
            "pos_weight": torch.tensor(compute_binary_pos_weight(column), device=device, dtype=torch.float)
        }

    # Multilabel → BCEWithLogitsLoss(pos_weight=...)
    for f in multiple_choice_columns:
        matrix = np.array(dataset[f])
        weights[f] = {
            "type": "multilabel_bce",
            "pos_weight": torch.tensor(compute_multilabel_pos_weights(matrix), device=device, dtype=torch.float)
        }
    return weights

