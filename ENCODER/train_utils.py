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
            mu = normalization_stats[f][0]
            std = normalization_stats[f][1]
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
    # Total parameters
    total_params = sum(p.numel() for p in model.parameters())
    # Parametri allenabili (quelli con gradiente)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Print parameters info
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    out_feat = 0
    for head in model.heads.values():
        out_feat += head.out_features
    print(f'Extraction heads parameters: {model.encoder.config.hidden_size * out_feat + out_feat:,}')

    