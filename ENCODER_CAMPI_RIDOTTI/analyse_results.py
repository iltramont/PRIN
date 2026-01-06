from pathlib import Path
import pandas as pd
import json
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
from pprint import pprint
import metric_utils
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
import math



###############
# Preliminaries
###############
base_dir = Path.cwd()
matplotlib.use("QtAgg")
# Parameters
SPLIT = 'validation'


with open(base_dir / "ENCODER_CAMPI_RIDOTTI" / "results.json", "r") as f:
    results = json.load(f)

for field, d in results['info']['label_to_id_map'].items():
    d['id_to_label'] = {int(k): v for k, v in d['id_to_label'].items()}


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
    best_threshold = metric_utils.get_best_threshold_binary(np.array(results['validation']['actual'][field]),
                                                            np.array(results['validation']['predicted'][field]))
    print(f"Best threshold for field {field}: {best_threshold}")
    m, cm = metric_utils.metrics_binary(actual, predicted, plot=False, field_name=field, threshold=best_threshold)
    df.append(pd.Series(m, name=field))
    
    # PLot confusion matrix
    r = i // n_cols
    c = i % n_cols
    ax = axes[r, c]
    sns.heatmap(cm, annot=True, cmap='viridis', ax=ax, cbar=False, square=True)
    ax.grid(False)
    ax.set_title(field)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
df = pd.DataFrame(df)

for idx in range(len(bin_fields), n_rows*n_cols):
    r = idx // n_cols
    c = idx % n_cols
    axes[r, c].axis("off")

plt.show()


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
    m, cm = metric_utils.metrics_classification(actual, predicted, plot=False, field_name=field, id_to_label_dict=results['info']['label_to_id_map'][field]['id_to_label'])
    df.append(pd.Series(m, name=field))
    # PLot confusion matrix
    r = i // n_cols
    c = i % n_cols
    ax = axes[r, c]
    labels = list(results['info']['label_to_id_map'][field]['id_to_label'].values())
    sns.heatmap(cm, annot=True, cmap='viridis', ax=ax, cbar=False, square=True, xticklabels=labels, yticklabels=labels)
    ax.grid(False)
    ax.set_title(field)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
df = pd.DataFrame(df)

for idx in range(len(clas_fields), n_rows*n_cols):
    r = idx // n_cols
    c = idx % n_cols
    axes[r, c].axis("off")

plt.show()


############
# Multilabel
############
multi_fields = results['info']['multiple_choice_fields']
df = []
for field in multi_fields:
    predicted = np.array(results[SPLIT]['predicted'][field])
    actual = np.array(results[SPLIT]['actual'][field])
    best_thresholds = metric_utils.get_best_thresholds_multilabel(np.array(results['validation']['actual'][field]),
                                                                  np.array(results['validation']['predicted'][field]))
    print(f"Best thresholds for field {field}: {best_thresholds}")
    m, cms = metric_utils.metrics_multilabel(actual,
                                             predicted,
                                             thresholds=best_thresholds,
                                             plot=False,
                                             field_name=field,
                                             id_to_label_dict=results['info']['label_to_id_map'][field]['id_to_label'])
    df.append(pd.Series(m, name=field))
    # PLot confusion matrix
    n_cols = 3
    n_rows = math.ceil(len(cms) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    axes = axes.reshape(n_rows, n_cols)
    for i, cm in enumerate(cms):
        r = i // n_cols
        c = i % n_cols
        ax = axes[r, c]
        labels = list(results['info']['label_to_id_map'][field]['id_to_label'].values())
        sns.heatmap(cm, annot=True, cmap='viridis', ax=ax, cbar=False, square=True)
        ax.grid(False)
        ax.set_title(f"{labels[i]}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    for idx in range(len(labels), n_rows*n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r, c].axis("off")
    fig.suptitle(field, fontsize="xx-large")
    plt.show()
df = pd.DataFrame(df)