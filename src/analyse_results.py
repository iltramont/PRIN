from pathlib import Path
import pandas as pd
import json
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
from pprint import pprint
from sklearn.metrics import (ConfusionMatrixDisplay,
                             f1_score,
                             recall_score,
                             precision_score,
                             matthews_corrcoef,
                             confusion_matrix,
                             mean_absolute_error,
                             mean_absolute_percentage_error,
                             precision_recall_curve)
import seaborn as sns
import math
import constants
import model_utils
from performance_utils import ore_score


###############
# Preliminaries
###############
base_dir = Path(__file__).parent.parent
# Parameters
SAVE_RESULTS = True 
RESULTS_FILE = "new_results_opus-4.6.jsonl"
SAVING_FILE = "new_metrics_opus-4.6.csv"
ANN_MODEL = constants.RectalCancerStagingData

SPLIT_CONFIGS = {
    'total': None,                          # nessun filtro
    'val-test': ('validation', 'test'),
    'train': ('train',),
    'validation': ('validation',),
    'test': ('test',),
}

with open(base_dir / "data" / "inference" / RESULTS_FILE, "r") as f:
    results = [json.loads(line) for line in f]

reg_fields = model_utils.get_regression_fields(ANN_MODEL)
clas_fields = model_utils.get_classification_fields(ANN_MODEL)
multi_fields = model_utils.get_multiple_choice_fields(ANN_MODEL)
bin_fields = model_utils.get_binary_classification_fields(ANN_MODEL)
label_to_id_map = model_utils.create_label_to_id_map(ANN_MODEL)

all_totals = {}

for split_name, split_filter in SPLIT_CONFIGS.items():
    if split_filter is not None:
        filtered = [r for r in results if r['split'] in split_filter and r['prediction'] != 'no output']
    else:
        filtered = [r for r in results if r['prediction'] != 'no output']

    # Regression
    rows = []
    for field in reg_fields:
        actual, predicted = [], []
        for row in filtered:
            actual.append(row['actual'][field])
            predicted.append(row['prediction'][field])
        if len(predicted) == 0:
            continue
        act, pred = [], []
        act_missing, pred_missing = [], []
        for a, p in zip(actual, predicted):
            a_missing = (a is None)
            p_missing = (p is None)
            act_missing.append(1 if a_missing else 0)
            pred_missing.append(1 if p_missing else 0)
            if not a_missing and not p_missing:
                act.append(a)
                pred.append(p)
        act_mape = [x for x in act if x != 0]
        pred_mape = [p for x, p in zip(act, pred) if x != 0]
        m = {
            'field': field,
            'split': split_name,
            'mae': mean_absolute_error(act, pred) if len(act) > 0 else None,
            'mape': mean_absolute_percentage_error(act_mape, pred_mape) if len(act_mape) > 0 else None,
            'f1_missing': f1_score(act_missing, pred_missing, average='macro', zero_division=0),
            'precision_missing': precision_score(act_missing, pred_missing, zero_division=0),
            'recall_missing': recall_score(act_missing, pred_missing, zero_division=0)
        }
        rows.append(pd.Series(m))

    # Ore (Clock IoU)
    ore_actual, ore_pred = [], []
    for row in filtered:
        ore_actual.append((row['actual']['ore_inizio'], row['actual']['ore_fine']))
        ore_pred.append((row['prediction']['ore_inizio'], row['prediction']['ore_fine']))

    iou_scores = [ore_score(a[0], a[1], p[0], p[1]) for a, p in zip(ore_actual, ore_pred)]
    ore_row = {
        'field': 'ore_iou',
        'split': split_name,
        'iou_mean': np.mean(iou_scores),
        'iou_median': np.median(iou_scores),
    }
    rows.append(pd.Series(ore_row))

    df_reg = pd.DataFrame(rows)

    # Binary Classification
    df = []
    for i, field in enumerate(bin_fields):
        if field not in ANN_MODEL.model_fields.keys():
            continue
        actual, predicted = [], []
        for row in filtered:
            actual.append(label_to_id_map[field]['label_to_id'][row['actual'][field]])
            predicted.append(label_to_id_map[field]['label_to_id'][row['prediction'][field]])
        if len(predicted) == 0:
            continue
        m = {
            'field': field,
            'split': split_name,
            'f1_macro': f1_score(actual, predicted, average='macro', zero_division=0),
            'f1': f1_score(actual, predicted, zero_division=0),
            'mcc': matthews_corrcoef(actual, predicted),
        }
        df.append(pd.Series(m))
    df_binary = pd.DataFrame(df)

    # Classification
    df = []
    for i, field in enumerate(clas_fields):
        if field not in ANN_MODEL.model_fields.keys():
            continue
        actual, predicted = [], []
        for row in filtered:
            actual.append(label_to_id_map[field]['label_to_id'][row['actual'][field]])
            predicted.append(label_to_id_map[field]['label_to_id'][row['prediction'][field]])
        if len(predicted) == 0:
            continue
        m = {
            'field': field,
            'split': split_name,
            'f1_macro': f1_score(actual, predicted, average='macro', zero_division=0),
            'mcc': matthews_corrcoef(actual, predicted)
        }
        df.append(pd.Series(m))
    df_classification = pd.DataFrame(df)

    # Multilabel
    df = []
    for field in multi_fields:
        if field not in ANN_MODEL.model_fields.keys():
            continue
        actual, predicted = [], []
        for row in filtered:
            actual.append(model_utils.flags_to_bits(row['actual'][field]))
            predicted.append(model_utils.flags_to_bits(row['prediction'][field]))
        actual = np.array(actual)
        predicted = np.array(predicted)
        if len(predicted) == 0:
            continue
        m = {
            'field': field,
            'split': split_name,
            'f1_macro': f1_score(actual, predicted, average='macro', zero_division=0),
            'f1_samples': f1_score(actual, predicted, average='samples', zero_division=0)
        }
        df.append(pd.Series(m))
        for label, i in label_to_id_map[field]['label_to_id'].items():
            s = f'{field}_{label}'
            m = {
                'field': s,
                'split': split_name,
                'f1_macro': f1_score(actual[:, i], predicted[:, i], average='macro', zero_division=0),
                'f1': f1_score(actual[:, i], predicted[:, i], zero_division=0),
            }
            df.append(pd.Series(m))
    df_multilabel = pd.DataFrame(df)

    total = pd.concat([df_reg, df_binary, df_classification, df_multilabel], ignore_index=True)
    all_totals[split_name] = total

# Combina tutto
final = pd.concat(all_totals.values(), ignore_index=True)
final.set_index(['field', 'split'], inplace=True)
print(final)

if SAVE_RESULTS:
    output_path = base_dir / "data" / "metrics"
    output_path.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_path / SAVING_FILE)