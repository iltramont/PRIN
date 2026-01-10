from pathlib import Path
import pandas as pd
import json
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
from pprint import pprint
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, matthews_corrcoef, confusion_matrix
import seaborn as sns
import math



###############
# Preliminaries
###############
base_dir = Path.cwd()
matplotlib.use("QtAgg")
# Parameters
SPLIT = 'validation'


with open(base_dir / "BASELINE_RESULTS" / "results_baseline.json", "r") as f:
    results = json.load(f)

for field, d in results['info']['label_to_id_map'].items():
    d['id_to_label'] = {int(k): v for k, v in d['id_to_label'].items()}

"""
############
# Regression
############
reg_fields = results['info']['regression_fields']
df = []
for field in reg_fields:
    predicted = np.array(results[SPLIT]['predicted'][field])
    actual = np.array(results[SPLIT]['actual'][field])
    is_nan_field = field + "_is_nan"
    print(len(predicted), len(actual))
    if is_nan_field in results[SPLIT]['actual']:
        is_nan = np.array(results[SPLIT]['actual'][is_nan_field], dtype=bool)
        predicted = predicted[~is_nan]
        actual = actual[~is_nan]
    print(len(predicted), len(actual))
    m = metric_utils.metrics_regression(actual, predicted, plot=True, field_name=field)
    df.append(pd.Series(m, name=field))
df = pd.DataFrame(df)
"""

#######################
# Binary Classification
#######################
bin_fields = results['info']['binary_classification_fields']
n_cols = 4
n_rows = math.ceil(len(bin_fields) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
axes = axes.reshape(n_rows, n_cols)

df = []
for i, field in enumerate(bin_fields):
    predicted = np.array(results[SPLIT]['predicted'][field])
    actual = np.array(results[SPLIT]['actual'][field])
    m = {
        'f1_macro': f1_score(actual, predicted, average='macro', zero_division=0),
        'f1': f1_score(actual, predicted, zero_division=0),
        'mcc': matthews_corrcoef(actual, predicted)
    }
    cm = confusion_matrix(actual, predicted)
    df.append(pd.Series(m, name=field))
    # PLot confusion matrix
    r = i // n_cols
    c = i % n_cols
    ax = axes[r, c]
    sns.heatmap(cm, annot=True, cmap='Blues', ax=ax, cbar=False, square=True)
    ax.grid(False)
    ax.set_title(field)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
df_binary = pd.DataFrame(df)
for idx in range(len(bin_fields), n_rows*n_cols):
    r = idx // n_cols
    c = idx % n_cols
    axes[r, c].axis("off")

#plt.show()
print(df_binary)


################
# Classification
################
clas_fields = results['info']['classification_fields']
n_cols = 4
n_rows = math.ceil(len(clas_fields) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
axes = axes.reshape(n_rows, n_cols)
df = []
for i, field in enumerate(clas_fields):
    predicted = np.array(results[SPLIT]['predicted'][field])
    actual = np.array(results[SPLIT]['actual'][field])    
    m = {
        'f1_macro': f1_score(actual, predicted, average='macro', zero_division=0),
        'mcc': matthews_corrcoef(actual, predicted)
    }
    labels = list(results['info']['label_to_id_map'][field]['id_to_label'].values())
    y_pred = [results['info']['label_to_id_map'][field]['id_to_label'][i] for i in predicted]
    y_true = [results['info']['label_to_id_map'][field]['id_to_label'][i] for i in actual] 
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df.append(pd.Series(m, name=field))
    # PLot confusion matrix
    r = i // n_cols
    c = i % n_cols
    ax = axes[r, c]
    sns.heatmap(cm, annot=True, cmap='Blues', ax=ax, cbar=False, square=True, xticklabels=labels, yticklabels=labels)
    ax.grid(False)
    ax.set_title(field)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
df_classification = pd.DataFrame(df)
for idx in range(len(clas_fields), n_rows*n_cols):
    r = idx // n_cols
    c = idx % n_cols
    axes[r, c].axis("off")

#plt.show()
print(df_classification)


############
# Multilabel
############
multi_fields = results['info']['multiple_choice_fields']
df = []
for field in multi_fields:
    predicted = np.array(results[SPLIT]['predicted'][field])
    actual = np.array(results[SPLIT]['actual'][field])
    m = {
        'f1_macro': f1_score(actual, predicted, average='macro', zero_division=0),
        'f1_samples': f1_score(actual, predicted, average='samples', zero_division=0)
    }
    df.append(pd.Series(m, name=field))
    n_cols = 3
    n_rows = math.ceil(len(results['info']['label_to_id_map'][field]['label_to_id']) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    axes = axes.reshape(n_rows, n_cols)
    for label, i in results['info']['label_to_id_map'][field]['label_to_id'].items():
        s = f'{field}_{label}'
        m = {
            'f1_macro': f1_score(actual[:, i], predicted[:, i], average='macro', zero_division=0),
            'f1': f1_score(actual[:, i], predicted[:, i], zero_division=0),
        }
        df.append(pd.Series(m, name=s))
        cm = confusion_matrix(actual[:, i], predicted[:, i])
        # PLot confusion matrix
        r = i // n_cols
        c = i % n_cols
        ax = axes[r, c]
        sns.heatmap(cm, annot=True, cmap='Blues', ax=ax, cbar=False, square=True)
        ax.grid(False)
        ax.set_title(f"{label}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    for idx in range(len(results['info']['label_to_id_map'][field]['label_to_id']), n_rows*n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r, c].axis("off")
    fig.suptitle(field, fontsize="xx-large")
    #plt.show()
df_multilabel = pd.DataFrame(df)
print(df_multilabel)

total = pd.concat([df_binary, df_classification, df_multilabel])
total.to_csv(base_dir / "BASELINE_RESULTS" / 'baseline_random_equal_metrics.csv')

#TODO dummyclassifier