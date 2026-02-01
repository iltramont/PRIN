from pathlib import Path
import pandas as pd
import json
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
from pprint import pprint
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, matthews_corrcoef, confusion_matrix, mean_absolute_error, mean_absolute_percentage_error
import seaborn as sns
import math



###############
# Preliminaries
###############
base_dir = Path(__file__).parent.parent
#matplotlib.use("QtAgg")
# Parameters
SAVE_RESULTS = True
RESULTS_FILE = "results_mistral_30.json"
SAVING_FILE = "metrics_mistral_30.csv"

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


with open(base_dir / "data" / RESULTS_FILE, "r") as f:
    results = json.load(f)


for field, d in results['info']['label_to_id_map'].items():
    d['id_to_label'] = {int(k): v for k, v in d['id_to_label'].items()}


############
# Regression
############
reg_fields = results['info']['regression_fields']
rows = []
for field in reg_fields:
    for split in ('validation', 'test'):
        predicted = np.array(results[split]['predicted'][field], dtype=object)
        actual = np.array(results[split]['actual'][field], dtype=object)
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
bin_fields = results['info']['binary_classification_fields']
#n_cols = 4
#n_rows = math.ceil(len(bin_fields) / n_cols)

#fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
#axes = axes.reshape(n_rows, n_cols)

df = []
for i, field in enumerate(bin_fields):
    for split in ('validation', 'test'):
        predicted = np.array(results[split]['predicted'][field])
        actual = np.array(results[split]['actual'][field])
        print(actual)
        print(predicted)
        m = {
            'field': field,
            'split': split, 
            'f1_macro': f1_score(actual, predicted, average='macro', zero_division=0),
            'f1': f1_score(actual, predicted, zero_division=0),
            'mcc': matthews_corrcoef(actual, predicted)
        }
        #cm = confusion_matrix(actual, predicted)
        df.append(pd.Series(m))
        # PLot confusion matrix
        #r = i // n_cols
        #c = i % n_cols
        #ax = axes[r, c]
        #sns.heatmap(cm, annot=True, cmap='Blues', ax=ax, cbar=False, square=True)
        #ax.grid(False)
        #ax.set_title(field)
        #ax.set_xlabel("Predicted")
        #ax.set_ylabel("Actual")
df_binary = pd.DataFrame(df)
    #for idx in range(len(bin_fields), n_rows*n_cols):
    #    r = idx // n_cols
    #    c = idx % n_cols
    #    axes[r, c].axis("off")

#plt.show()


################
# Classification
################
clas_fields = results['info']['classification_fields']
"""
n_cols = 4
n_rows = math.ceil(len(clas_fields) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
axes = axes.reshape(n_rows, n_cols)
"""
df = []
for i, field in enumerate(clas_fields):
    for split in ('validation', 'test'):
        predicted = np.array(results[split]['predicted'][field])
        actual = np.array(results[split]['actual'][field])    
        m = {
            'field': field,
            'split': split,
            'f1_macro': f1_score(actual, predicted, average='macro', zero_division=0),
            'mcc': matthews_corrcoef(actual, predicted)
        }
        df.append(pd.Series(m))
    """
    labels = list(results['info']['label_to_id_map'][field]['id_to_label'].values())
    y_pred = [results['info']['label_to_id_map'][field]['id_to_label'][i] for i in predicted]
    y_true = [results['info']['label_to_id_map'][field]['id_to_label'][i] for i in actual] 
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # PLot confusion matrix
    r = i // n_cols
    c = i % n_cols
    ax = axes[r, c]
    sns.heatmap(cm, annot=True, cmap='Blues', ax=ax, cbar=False, square=True, xticklabels=labels, yticklabels=labels)
    ax.grid(False)
    ax.set_title(field)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    """
df_classification = pd.DataFrame(df)
"""
for idx in range(len(clas_fields), n_rows*n_cols):
    r = idx // n_cols
    c = idx % n_cols
    axes[r, c].axis("off")
"""
#plt.show()


############
# Multilabel
############
multi_fields = results['info']['multiple_choice_fields']
df = []
for field in multi_fields:
    for split in ('validation', 'test'):
        predicted = np.array(results[split]['predicted'][field])
        actual = np.array(results[split]['actual'][field])
        m = {
            'field': field,
            'split': split,
            'f1_macro': f1_score(actual, predicted, average='macro', zero_division=0),
            'f1_samples': f1_score(actual, predicted, average='samples', zero_division=0)
        }
        df.append(pd.Series(m))
        """n_cols = 3
        n_rows = math.ceil(len(results['info']['label_to_id_map'][field]['label_to_id']) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        axes = axes.reshape(n_rows, n_cols)
        """
        for label, i in results['info']['label_to_id_map'][field]['label_to_id'].items():
            s = f'{field}_{label}'
            m = {
                'field': s,
                'split': split,
                'f1_macro': f1_score(actual[:, i], predicted[:, i], average='macro', zero_division=0),
                'f1': f1_score(actual[:, i], predicted[:, i], zero_division=0),
            }
            df.append(pd.Series(m))
            """
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
        plt.show()
        """
df_multilabel = pd.DataFrame(df)


######
# Save
######
total = pd.concat([df_reg, df_binary, df_classification, df_multilabel], ignore_index=True)
total.set_index(['field', 'split'], inplace=True)
print(total)

if SAVE_RESULTS:
    total.to_csv(base_dir / "data" / SAVING_FILE)