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
                             matthews_corrcoef,
                             confusion_matrix,
                             mean_absolute_error,
                             mean_absolute_percentage_error,
                             precision_recall_curve)
import seaborn as sns
import math
import constants
import model_utils


###############
# Preliminaries
###############
base_dir = Path(__file__).parent.parent
# Parameters
SAVE_RESULTS = True
RESULTS_FILE = "results_gpt-4.1-tuned-similar_examples.jsonl"
SAVING_FILE = "metrics_gpt-4.1-tuned-similar_examples.csv"
USE_SCORES = False  # If True, use scores instead of hard predictions
USE_JSONL = True
ANN_MODEL = constants.RectalCancerStagingData

# Set plot style
plt.style.use('ggplot')
#sns.set_palette('hls')
# Colors
finomnia_palette = sns.color_palette(('#db038a',   # Pink
                                      '#66218a',   # Violet
                                      '#2659ab',   # Blue
                                      '#081c36',   # Dark blue
                                      '#45c9f5'))  # Light blue
sns.set_palette(finomnia_palette)

with open(base_dir / "data" / "inference" / RESULTS_FILE, "r") as f:
    if USE_JSONL:
        results = [json.loads(line) for line in f]
        reg_fields = model_utils.get_regression_fields(ANN_MODEL)
        clas_fields = model_utils.get_classification_fields(ANN_MODEL)
        multi_fields = model_utils.get_multiple_choice_fields(ANN_MODEL)
        bin_fields = model_utils.get_binary_classification_fields(ANN_MODEL)
        label_to_id_map = model_utils.create_label_to_id_map(ANN_MODEL)
    else:
        results = json.load(f)
        for field, d in results['info']['label_to_id_map'].items():
            d['id_to_label'] = {int(k): v for k, v in d['id_to_label'].items()}
            reg_fields = results['info']['regression_fields']
            clas_fields = results['info']['classification_fields']
            multi_fields = results['info']['multiple_choice_fields']
            bin_fields = results['info']['binary_classification_fields']
            label_to_id_map = results['info']['label_to_id_map']


############
# Regression
############
rows = []
for field in reg_fields:
    for split in ('validation', 'test'):
        if USE_JSONL:
            actual, predicted = [], []
            for row in results:
                if row['split'] == split:
                    actual.append(row['actual'][field])
                    predicted.append(row['prediction'][field])
        else:
            predicted = np.array(results[split]['predicted'][field], dtype=object)
            actual = np.array(results[split]['actual'][field], dtype=object)
        if len(predicted) == 0:
            continue
        # Liste pulite
        act, pred = [], []
        act_missing, pred_missing = [], []
        for a, p in zip(actual, predicted):
            # Missing = None
            a_missing = (a is None)
            p_missing = (p is None)
            act_missing.append(1 if a_missing else 0)
            pred_missing.append(1 if p_missing else 0)
            # Valori validi
            if not a_missing and not p_missing:
                act.append(a)
                pred.append(p)
        # MAPE: rimuovi gli zeri da y_true
        act_mape = [x for x in act if x != 0]
        pred_mape = [p for x, p in zip(act, pred) if x != 0]
        m = {
            'field': field,
            'split': split,
            'mae': mean_absolute_error(act, pred) if len(act) > 0 else None,
            'mape': mean_absolute_percentage_error(act_mape, pred_mape) if len(act_mape) > 0 else None,
            'f1_missing': f1_score(act_missing, pred_missing, average='macro', zero_division=0),
            'precision_missing': f1_score(act_missing, pred_missing, zero_division=0),
            'recall_missing': f1_score(act_missing, pred_missing, zero_division=0)
        }
        rows.append(pd.Series(m))
df_reg = pd.DataFrame(rows)


#######################
# Binary Classification
#######################
def get_best_threshold_binary(y_true: np.ndarray, pred_prob: np.ndarray) -> float:
    """
    Calcola la soglia ottimale per classificazione binaria
    massimizzando l'F1 score.

    Args:
        y_true: array delle etichette vere (0 o 1)
        pred_prob: array delle probabilità previste dal modello (float tra 0 e 1)

    Returns:
        best_threshold: soglia ottimale (float)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, pred_prob)
    # Calcola F1 score
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    # thresholds ha lunghezza len(f1_scores) - 1 → consideriamo solo i primi f1_scores
    best_index = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_index]
    return float(best_threshold)


df = []
for i, field in enumerate(bin_fields):
    best_threshold = None
    if USE_SCORES:
        # Trova soglia ottimale
        best_threshold = get_best_threshold_binary(np.array(results['validation']['actual'][field]), np.array(results['validation']['predicted'][field]))
    for split in ('validation', 'test'):
        if USE_JSONL:
            if field not in ANN_MODEL.model_fields.keys():
                continue
            actual, predicted = [], []
            for row in results:
                if row['split'] == split:
                    actual.append(label_to_id_map[field]['label_to_id'][row['actual'][field]])
                    predicted.append(label_to_id_map[field]['label_to_id'][row['prediction'][field]])
        else:
            predicted = np.array(results[split]['predicted'][field])
            actual = np.array(results[split]['actual'][field])
        if len(predicted) == 0:
            continue
        if best_threshold is not None:
            # Applica soglia
            predicted = (predicted >= best_threshold).astype(int)
        m = {
            'field': field,
            'split': split, 
            'f1_macro': f1_score(actual, predicted, average='macro', zero_division=0),
            'f1': f1_score(actual, predicted, zero_division=0),
            'mcc': matthews_corrcoef(actual, predicted),
            'best_threshold': best_threshold
        }
        df.append(pd.Series(m))
df_binary = pd.DataFrame(df)


################
# Classification
################
df = []
for i, field in enumerate(clas_fields):
    for split in ('validation', 'test'):
        if USE_JSONL:
            if field not in ANN_MODEL.model_fields.keys():
                continue
            actual, predicted = [], []
            for row in results:
                if row['split'] == split:
                    actual.append(label_to_id_map[field]['label_to_id'][row['actual'][field]])
                    predicted.append(label_to_id_map[field]['label_to_id'][row['prediction'][field]])
        else:
            predicted = np.array(results[split]['predicted'][field])
            actual = np.array(results[split]['actual'][field])
        if len(predicted) == 0:
            continue
        if USE_SCORES:
            # Convert scores to hard predictions
            predicted = np.argmax(predicted, axis=-1)
        m = {
            'field': field,
            'split': split,
            'f1_macro': f1_score(actual, predicted, average='macro', zero_division=0),
            'mcc': matthews_corrcoef(actual, predicted)
        }
        df.append(pd.Series(m))
df_classification = pd.DataFrame(df)


############
# Multilabel
############
def get_best_thresholds_multilabel(y_true: np.ndarray, y_pred_probs: np.ndarray) -> np.ndarray:
    """
    Calcola la soglia ottimale per ciascuna classe in un problema multilabel.
    Args:
        y_true: array shape (num_samples, num_classes), valori 0/1
        y_pred_probs: array shape (num_samples, num_classes), probabilità predette
        
    Returns:
        thresholds: array shape (num_classes,), soglia ottimale per ciascuna classe
    """
    num_classes = y_true.shape[1]
    thresholds = []

    for i in range(num_classes):
        precisions, recalls, ths = precision_recall_curve(y_true[:, i], y_pred_probs[:, i])
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        if len(ths) > 0:
            thresholds.append(ths[np.argmax(f1_scores)])
        else:
            thresholds.append(0.5)  # fallback se la classe è assente
    return thresholds


df = []
for field in multi_fields:
    thresholds = None
    if USE_SCORES:
        # Optimal thresholds
        thresholds = get_best_thresholds_multilabel(np.array(results['validation']['actual'][field]),
                                                    np.array(results['validation']['predicted'][field]))
    for split in ('validation', 'test'):
        if USE_JSONL:
            if field not in ANN_MODEL.model_fields.keys():
                continue
            actual, predicted = [], []
            for row in results:
                if row['split'] == split:
                    actual.append(model_utils.flags_to_bits(row['actual'][field]))
                    predicted.append(model_utils.flags_to_bits(row['prediction'][field]))
            actual = np.array(actual)
            predicted = np.array(predicted)
        else:
            predicted = np.array(results[split]['predicted'][field])
            actual = np.array(results[split]['actual'][field])
        if len(predicted) == 0:
            continue
        if thresholds is not None:
            thresholds = np.array(thresholds)
            predicted = (predicted > thresholds).astype(int)
        m = {
            'field': field,
            'split': split,
            'f1_macro': f1_score(actual, predicted, average='macro', zero_division=0),
            'f1_samples': f1_score(actual, predicted, average='samples', zero_division=0)
        }
        df.append(pd.Series(m))
        for label, i in label_to_id_map[field]['label_to_id'].items():
            s = f'{field}_{label}'
            m = {
                'field': s,
                'split': split,
                'f1_macro': f1_score(actual[:, i], predicted[:, i], average='macro', zero_division=0),
                'f1': f1_score(actual[:, i], predicted[:, i], zero_division=0),
                'best_threshold': thresholds[i] if thresholds is not None else None
            }
            df.append(pd.Series(m))
df_multilabel = pd.DataFrame(df)


######
# Save
######
total = pd.concat([df_reg, df_binary, df_classification, df_multilabel], ignore_index=True)
total.set_index(['field', 'split'], inplace=True)
print(total)

if SAVE_RESULTS:
    output_path = base_dir / "data" / "metrics"
    output_path.mkdir(parents=True, exist_ok=True)
    total.to_csv(output_path / SAVING_FILE)