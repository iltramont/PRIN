from pathlib import Path
import pandas as pd
import json
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec


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
import os
from scipy import stats


###############
# Preliminaries
###############
base_dir = Path(__file__).parent.parent
os.makedirs(base_dir / "immagini", exist_ok=True)
image_dir = base_dir / "immagini"
#matplotlib.use("QtAgg")
# Parameters
SAVE_RESULTS = True
RESULTS_FILE = "results_gpt-4.1-nano-2025-04-14.jsonl"
SAVING_FILE = "metrics_gpt-4.1-nano.csv"
USE_SCORES = False  # If True, use scores instead of hard predictions
USE_JSONL = True
ANN_MODEL = constants.AnnotationsExtended

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

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return np.array(tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)))

def hex_to_rgb_norm(hex_color):
    return hex_to_rgb(hex_color) / 255

def plot_cm_seaborn(ax, y_true, y_pred, labels=None, normalize=False):
    if labels is None:
        labels = sorted(set(y_true))

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize='true' if normalize else None
    )

    # Colori base
    green = np.array([107/255, 203/255, 119/255])   # #6BCB77
    red   = np.array([255/255, 107/255, 107/255])   # #FF6B6B
    pink        = finomnia_palette[0]
    light_blue  = finomnia_palette[-1]   
    white = np.array([1, 1, 1])

    # Matrice colori: verde sulla diagonale, rosso altrove
    base_color_matrix = np.where(
        np.eye(len(labels), dtype=bool)[..., None],
        light_blue,
        pink
    )

    # Se valore = 0 → schiarisco verso bianco
    color_matrix = np.zeros((*cm.shape, 3))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            base = base_color_matrix[i, j]
            if cm[i, j] == 0:
                # blend 85% verso bianco
                color_matrix[i, j] = 0.85 * white + 0.15 * base
            else:
                color_matrix[i, j] = base

    # Heatmap "vuota" (useremo i nostri colori)
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        square=True,
        linewidths=0,
        cmap=ListedColormap(["white"]),  # placeholder
        ax=ax
    )

    # Applica i colori cella-per-cella
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.add_patch(plt.Rectangle(
                (j, i), 1, 1,
                facecolor=color_matrix[i, j],
                edgecolor='none'
            ))

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))


############
# Regression
############
rows = []
for field in reg_fields:
    # Plotting
    fig = plt.figure(figsize=(14, 7))
    axes = []

    # Titolo globale della figura
    fig.suptitle("Titolo della Figura Principale", fontsize=16)

    # Griglia principale: 2 righe, 1 colonna
    outer = gridspec.GridSpec(2, 1, hspace=0.35)

    # ============================================================
    # RIGA 1 — contiene 5 subsubplot
    # ============================================================

    # Asse contenitore invisibile per il titolo della riga
    ax_top_container = fig.add_subplot(outer[0])
    ax_top_container.set_title("Titolo della Riga Superiore", fontsize=13)
    ax_top_container.axis("off")

    # Griglia interna 1×5
    inner_top = gridspec.GridSpecFromSubplotSpec(
        1, 5, subplot_spec=outer[0], wspace=0.3
    )

    for i in range(5):
        ax = fig.add_subplot(inner_top[0, i])
        axes.append(ax)

    # ============================================================
    # RIGA 2 — contiene 5 subsubplot
    # ============================================================

    ax_bottom_container = fig.add_subplot(outer[1])
    ax_bottom_container.set_title("Titolo della Riga Inferiore", fontsize=13)
    ax_bottom_container.axis("off")

    inner_bottom = gridspec.GridSpecFromSubplotSpec(
        1, 5, subplot_spec=outer[1], wspace=0.3
    )

    for i in range(5):
        ax = fig.add_subplot(inner_bottom[0, i])
        axes.append(ax)
            
    i_plot = 0
        
    for split in ('validation', 'test'):
        if USE_JSONL:
            actual, predicted = [], []
            for row in results:
                if row['split'] == split:
                    actual.append(row['actual'][field])
                    predicted.append(row['prediction'][field])
            actual = np.array(actual)
            predicted = np.array(predicted)
        else:
            predicted = np.array(results[split]['predicted'][field], dtype=object)
            actual = np.array(results[split]['actual'][field], dtype=object)
        if len(predicted) == 0:
            continue
        # Liste stringhe
        act_str = [str(x) for x in actual]
        pred_str = [str(x) for x in predicted]
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
        act = np.array(act)
        pred = np.array(pred)
        # MAPE: rimuovi gli zeri da y_true
        act_mape = np.array([x for x in act if x != 0])
        pred_mape = np.array([p for x, p in zip(act, pred) if x != 0])
        
        # Plotting
        
        title = f'{field} - {split}'
        fig.suptitle(title, fontsize="xx-large")
        residuals = act - pred

        ax = axes[i_plot]
        labels = list(set(list(int(x) for x in act_str + pred_str if x != 'None')))
        labels.sort()
        labels.insert(0, None)
        labels = [str(x) for x in labels]
        
        plot_cm_seaborn(ax, act_str, pred_str, labels)
        i_plot += 1
        if len(act) <= 0:
            i_plot += 4
        else:
            colors = [
                finomnia_palette[-1] if a == p else finomnia_palette[0]
                for a, p in zip(act, pred)
            ]
            # 1. Scatter y_true vs y_pred
            jitter_strength = 0.02 * (act.max() - act.min())
            ax = axes[i_plot]
            act_j = act + np.random.normal(0, jitter_strength, size=len(act))
            pred_j = pred + np.random.normal(0, jitter_strength, size=len(pred))
            ax.scatter(act_j, pred_j, alpha=0.6, c=colors)
            import matplotlib.patches as mpatches

            correct_patch = mpatches.Patch(color=finomnia_palette[-1], label="Correct")
            wrong_patch   = mpatches.Patch(color=finomnia_palette[0],  label="Wrong")
            ax.legend(handles=[correct_patch, wrong_patch])
            
            ax.plot([act.min(), act.max()], [act.min(), act.max()], color=finomnia_palette[-1], linestyle="--")
            
            ax.set_title("y_true vs y_pred")
            ax.set_xlabel("y_true")
            ax.set_ylabel("y_pred")
            i_plot += 1
            
            # 2. Residual plot
            jitter_strength = 0.02 * (residuals.max() - residuals.min())
            res_j = residuals + np.random.normal(0, jitter_strength, size=len(residuals))
            pred_j = pred + np.random.normal(0, jitter_strength, size=len(pred))
            ax = axes[i_plot]
            ax.scatter(pred_j, res_j, alpha=0.6, c=colors)
            correct_patch = mpatches.Patch(color=finomnia_palette[-1], label="Correct")
            wrong_patch   = mpatches.Patch(color=finomnia_palette[0],  label="Wrong")
            ax.legend(handles=[correct_patch, wrong_patch])
            
            ax.axhline(0, color=finomnia_palette[-1], linestyle="--")
            ax.set_title("Residual plot")
            ax.set_xlabel("y_pred")
            ax.set_ylabel("Residuals")
            i_plot += 1
            
            # 3. Histogram of residuals
            ax = axes[i_plot]
            ax.hist(residuals, bins=30, alpha=0.7)
            ax.set_title("Residual distribution")
            ax.set_xlabel("Residual")
            ax.set_ylabel("Count")
            i_plot += 1
            
            # 4. QQ-plot
            ax = axes[i_plot]
            # Ordina residui e ottieni indici
            order = np.argsort(residuals)
            res_sorted = residuals[order]
            # Quantili teorici
            n = len(residuals)
            theoretical = stats.norm.ppf((np.arange(n) + 0.5) / n)
            # Colori ordinati nello stesso ordine dei residui
            colors_sorted = np.array(colors)[order]
            # Scatter
            ax.scatter(theoretical, res_sorted, alpha=0.6, c=colors_sorted)
            # Linea teorica
            mu, sigma = residuals.mean(), residuals.std()
            ax.plot(theoretical, mu + sigma * theoretical, linestyle="--", color=finomnia_palette[-1])
            # Titoli
            ax.set_title("QQ-plot of residuals")
            ax.set_xlabel("Theoretical quantiles")
            ax.set_ylabel("Ordered residuals")
            # Legenda
            correct_patch = mpatches.Patch(color=finomnia_palette[-1], label="Correct")
            wrong_patch   = mpatches.Patch(color=finomnia_palette[0],  label="Wrong")
            ax.legend(handles=[correct_patch, wrong_patch])
            i_plot += 1
            

    plt.tight_layout()
    plt.savefig(image_dir / f'{field}')
    #plt.show()
        


exit()

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



#n_cols = 4
#n_rows = math.ceil(len(bin_fields) / n_cols)

#fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
#axes = axes.reshape(n_rows, n_cols)

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
"""
n_cols = 4
n_rows = math.ceil(len(clas_fields) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
axes = axes.reshape(n_rows, n_cols)
"""
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
        """n_cols = 3
        n_rows = math.ceil(len(results['info']['label_to_id_map'][field]['label_to_id']) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        axes = axes.reshape(n_rows, n_cols)
        """
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
    output_path = base_dir / "data" / "metrics"
    output_path.mkdir(parents=True, exist_ok=True)

    total.to_csv(output_path / SAVING_FILE)