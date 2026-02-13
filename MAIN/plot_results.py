from pathlib import Path
import pandas as pd
import json
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
from PIL import Image


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
#matplotlib.use("QtAgg")
# Parameters
RESULTS_FILE = "results_opus-4.6.jsonl"
MODEL_NAME_PLOT = "Claude Opus 4.6"

os.makedirs(base_dir / "immagini" / "opus-4.6", exist_ok=True)
image_dir = base_dir / "immagini" / "opus-4.6"

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
        bin_fields = [x for x in bin_fields if x in ANN_MODEL.model_fields.keys()]
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
            v = cm[i, j]
            bland_color = 0.95 * white + 0.05 * base
            if v == 0:
                color_matrix[i, j] = bland_color
            else:
                sum_row    = cm[i, :].sum()
                sum_column = cm[:, j].sum()
                ratio = v/(sum_row + sum_column - v)
                color_matrix[i, j] = ratio * base + (1-ratio) * white

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

    ax.set_xlabel("Predicted", fontsize='xx-small')
    ax.set_ylabel("True", fontsize='xx-small')
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""), fontsize='large')


saved_paths = []

############
# Regression
############
rows = []

for field in reg_fields:
    # Plotting
    figsize = np.array((21, 9))
    top    = 0.8
    bottom = 0.1
    left   = 0.02
    hspace = 0.6
    wspace = 0.1
    
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("#beb088")      # o qualsiasi colore
    
    # Griglia principale: 2 righe, 1 colonna
    
    fig.subplots_adjust(top=top, bottom=bottom, hspace=hspace, left=left, right=1-left)
    outer        = gridspec.GridSpec(2, 1)
    inner_top    = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=outer[0], wspace=wspace)
    inner_bottom = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=outer[1], wspace=wspace)
    
    axes = []
    for inner in [inner_top, inner_bottom]:
        for i in range(6):
            ax = fig.add_subplot(inner[0, i])
            ax.set_box_aspect(1)
            axes.append(ax)
            
    i_plot = 0
    figtitle = f'{field} - {MODEL_NAME_PLOT}'
    fig.suptitle(figtitle, fontsize="xx-large", y=1-0.05, fontweight="bold")
    fig.text(0.5, top+0.05, "Validation split",
            ha="center", va="center", fontsize='x-large', fontweight="bold")

    fig.text(0.5, (bottom + (top-bottom)/(2+hspace) + 0.05), "Test split",
            ha="center", va="center", fontsize='x-large', fontweight="bold")

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
        act_missing = ['missing' if x==1 else 'present' for x in act_missing]
        pred_missing = ['missing' if x==1 else 'present' for x in pred_missing]
        
        # Plotting
        residuals = act - pred

        # Inference confusion matrix
        ax = axes[i_plot]
        labels = list(set(list(int(x) for x in act_str + pred_str if x != 'None')))
        labels.sort()
        labels.insert(0, None)
        labels = [str(x) for x in labels]
        if len(labels) <= 15:
            plot_cm_seaborn(ax, act_str, pred_str, labels)
        i_plot += 1
        
        # Missing confusion matrix
        ax = axes[i_plot]
        plot_cm_seaborn(ax, act_missing, pred_missing, labels=['present', 'missing'])
        ax.set_title('Missing detection', fontsize='large')
        i_plot += 1
        
        if len(act) <= 0:
            i_plot += 4
        else:
            colors = [
                finomnia_palette[-1] if a == p else finomnia_palette[0]
                for a, p in zip(act, pred)
            ]
            # 1. Scatter y_true vs y_pred
            rng = act.max() - act.min()
            jitter_strength = 0.01 * rng if rng > 0 else 1e-3

            ax = axes[i_plot]
            act_j = act + np.random.normal(0, jitter_strength, size=len(act))
            pred_j = pred + np.random.normal(0, jitter_strength, size=len(pred))
            ax.scatter(act_j, pred_j, alpha=0.6, c=colors)
            import matplotlib.patches as mpatches

            correct_patch = mpatches.Patch(color=finomnia_palette[-1], label="Correct")
            wrong_patch   = mpatches.Patch(color=finomnia_palette[0],  label="Wrong")
            ax.legend(handles=[correct_patch, wrong_patch])
            
            ax.plot([act.min(), act.max()], [act.min(), act.max()], color=finomnia_palette[-1], linestyle="--")
            
            ax.set_title("y_true vs y_pred", fontsize='large')
            ax.set_xlabel("y_true", fontsize='small')
            ax.set_ylabel("y_pred", fontsize='small')
            i_plot += 1
            
            # 2. Residual plot
            jitter_strength = 0.01 * (residuals.max() - residuals.min())
            res_j = residuals + np.random.normal(0, jitter_strength, size=len(residuals))
            pred_j = pred + np.random.normal(0, jitter_strength, size=len(pred))
            ax = axes[i_plot]
            ax.scatter(pred_j, res_j, alpha=0.6, c=colors)
            correct_patch = mpatches.Patch(color=finomnia_palette[-1], label="Correct")
            wrong_patch   = mpatches.Patch(color=finomnia_palette[0],  label="Wrong")
            ax.legend(handles=[correct_patch, wrong_patch])
            
            ax.axhline(0, color=finomnia_palette[-1], linestyle="--")
            ax.set_title("Residual plot", fontsize='large')
            ax.set_xlabel("y_pred", fontsize='small')
            ax.set_ylabel("Residuals", fontsize='small')
            i_plot += 1
            
            # 3. Histogram of residuals
            ax = axes[i_plot]
            ax.hist(residuals, bins=30, alpha=0.7)
            ax.set_title("Residual distribution", fontsize='large')
            ax.set_xlabel("Residual", fontsize='small')
            ax.set_ylabel("Count", fontsize='small')
            ax.grid(axis='x')
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
            ax.set_title("QQ-plot of residuals", fontsize='large')
            ax.set_xlabel("Theoretical quantiles", fontsize='small')
            ax.set_ylabel("Ordered residuals", fontsize='small')
            # Legenda
            correct_patch = mpatches.Patch(color=finomnia_palette[-1], label="Correct")
            wrong_patch   = mpatches.Patch(color=finomnia_palette[0],  label="Wrong")
            ax.legend(handles=[correct_patch, wrong_patch])
            i_plot += 1
            

    out_path = image_dir / f"{field}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    saved_paths.append(out_path)
    
if len(saved_paths) > 0:
    imgs = [Image.open(p) for p in saved_paths]

    # Altezza totale
    extra_h = int(imgs[0].height / 20)
    total_h = int(sum(img.height + extra_h for img in imgs) + extra_h)
    # Larghezza massima
    max_w = max(img.width for img in imgs) + 2 * extra_h
    combined = Image.new("RGB", (max_w, total_h), color="white")

    y_offset = extra_h
    for img in imgs:
        combined.paste(img, (extra_h, y_offset))
        y_offset += img.height + extra_h

    combined.save(image_dir / "regression_fields.png")
    for p in saved_paths:
        p.unlink(missing_ok=True)


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

n_cols = 5
n_rows = math.ceil(len(bin_fields) / n_cols)

# Plotting
saved_paths = []

figsize = np.array((21, 9))
top    = 0.8
bottom = 0.1
left   = 0.02
hspace = 0.6
wspace = 0.1

fig = plt.figure(figsize=figsize)
fig.patch.set_facecolor("#beb088")      # o qualsiasi colore

# Griglia principale: 

fig.subplots_adjust(top=top, bottom=bottom, hspace=hspace, left=left, right=1-left)
outer        = gridspec.GridSpec(2, 1)
inner_top    = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=outer[0], wspace=wspace, hspace=hspace)
inner_bottom = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=outer[1], wspace=wspace, hspace=hspace)

axes = []
for inner in [inner_top, inner_bottom]:
    for i in range(n_rows):
        for j in range(n_cols):
            ax = fig.add_subplot(inner[i, j])
            ax.set_box_aspect(1)
            axes.append(ax)
        
figtitle = f'Binary fields - {MODEL_NAME_PLOT}'
fig.suptitle(figtitle, fontsize="xx-large", y=1-0.05, fontweight="bold")
fig.text(0.5, top+0.05, "Validation split",
        ha="center", va="center", fontsize='x-large', fontweight="bold")

fig.text(0.5, (bottom + (top-bottom)/(2+hspace) + 0.05), "Test split",
        ha="center", va="center", fontsize='x-large', fontweight="bold")

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
            
        actual    = [label_to_id_map[field]['id_to_label'][x] for x in actual]
        predicted = [label_to_id_map[field]['id_to_label'][x] for x in predicted]
        
        if split == 'validation':
            i_plot = i
        else:
            i_plot = i + n_cols * n_rows
        ax = axes[i_plot]
        labels = [label_to_id_map[field]['id_to_label'][0], label_to_id_map[field]['id_to_label'][1]]
        plot_cm_seaborn(ax, actual, predicted, labels=labels)
        ax.set_title(field, fontsize='small')
        ax.set_xlabel("Predicted", fontsize='xx-small')
        ax.set_ylabel("True", fontsize='xx-small')
    
out_path = image_dir / f"bin_fields.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
saved_paths.append(out_path)

imgs = [Image.open(p) for p in saved_paths]

# Altezza totale
extra_h = int(imgs[0].height / 20)
total_h = int(sum(img.height + extra_h for img in imgs) + extra_h)
# Larghezza massima
max_w = max(img.width for img in imgs) + 2 * extra_h
combined = Image.new("RGB", (max_w, total_h), color="white")

y_offset = extra_h
for img in imgs:
    combined.paste(img, (extra_h, y_offset))
    y_offset += img.height + extra_h

combined.save(image_dir / "binary_fields.png")
for p in saved_paths:
    p.unlink(missing_ok=True)


################
# Classification
################
n_cols = min(5, len(clas_fields))
n_rows = math.ceil(len(clas_fields) / n_cols)

# Plotting
saved_paths = []

figsize = np.array((21, 9))
top    = 0.8
bottom = 0.1
left   = 0.02
hspace = 0.6
wspace = 0.1

fig = plt.figure(figsize=figsize)
fig.patch.set_facecolor("#beb088")      # o qualsiasi colore

# Griglia principale: 

fig.subplots_adjust(top=top, bottom=bottom, hspace=hspace, left=left, right=1-left)
outer        = gridspec.GridSpec(2, 1)
inner_top    = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=outer[0], wspace=wspace, hspace=hspace)
inner_bottom = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=outer[1], wspace=wspace, hspace=hspace)

axes = []
for inner in [inner_top, inner_bottom]:
    for i in range(n_rows):
        for j in range(n_cols):
            ax = fig.add_subplot(inner[i, j])
            ax.set_box_aspect(1)
            axes.append(ax)
        
figtitle = f'Multi-class fields - {MODEL_NAME_PLOT}'
fig.suptitle(figtitle, fontsize="xx-large", y=1-0.05, fontweight="bold")
fig.text(0.5, top+0.05, "Validation split",
        ha="center", va="center", fontsize='x-large', fontweight="bold")

fig.text(0.5, (bottom + (top-bottom)/(2+hspace) + 0.05), "Test split",
        ha="center", va="center", fontsize='x-large', fontweight="bold")

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
        
        # Convert to strings
        actual    = [label_to_id_map[field]['id_to_label'][x] for x in actual]
        predicted = [label_to_id_map[field]['id_to_label'][x] for x in predicted]
        
        if split == 'validation':
            i_plot = i
        else:
            i_plot = i + n_cols * n_rows
        ax = axes[i_plot]
        labels = [l for l in label_to_id_map[field]['label_to_id'].keys()]
        plot_cm_seaborn(ax, actual, predicted, labels=labels)
        ax.set_title(field, fontsize='small')
    
out_path = image_dir / f"clas_fields.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
saved_paths.append(out_path)

imgs = [Image.open(p) for p in saved_paths]

# Altezza totale
extra_h = int(imgs[0].height / 20)
total_h = int(sum(img.height + extra_h for img in imgs) + extra_h)
# Larghezza massima
max_w = max(img.width for img in imgs) + 2 * extra_h
combined = Image.new("RGB", (max_w, total_h), color="white")

y_offset = extra_h
for img in imgs:
    combined.paste(img, (extra_h, y_offset))
    y_offset += img.height + extra_h

combined.save(image_dir / "classification_fields.png")
for p in saved_paths:
    p.unlink(missing_ok=True)



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


saved_paths = []
for field in multi_fields:
    # Plotting
    
    n_cols = len(label_to_id_map[field]['label_to_id'])
    n_rows = 1
    
    figsize = np.array((21, 9))
    top    = 0.8
    bottom = 0.1
    left   = 0.02
    hspace = 0.6
    wspace = 0.1
    
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("#beb088")      # o qualsiasi colore
    
    # Griglia principale: 2 righe, 1 colonna
    
    fig.subplots_adjust(top=top, bottom=bottom, hspace=hspace, left=left, right=1-left)
    outer        = gridspec.GridSpec(2, 1)
    inner_top    = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=outer[0], wspace=wspace)
    inner_bottom = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=outer[1], wspace=wspace)
    
    axes = []
    for inner in [inner_top, inner_bottom]:
        for i in range(n_rows):
            for j in range(n_cols):
                ax = fig.add_subplot(inner[i, j])
                ax.set_box_aspect(1)
                axes.append(ax)
            
    i_plot = 0
    figtitle = f'{field} - {MODEL_NAME_PLOT}'
    fig.suptitle(figtitle, fontsize="xx-large", y=1-0.05, fontweight="bold")
    fig.text(0.5, top+0.05, "Validation split",
            ha="center", va="center", fontsize='x-large', fontweight="bold")

    fig.text(0.5, (bottom + (top-bottom)/(2+hspace) + 0.05), "Test split",
            ha="center", va="center", fontsize='x-large', fontweight="bold")    

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
            
        for label, i in label_to_id_map[field]['label_to_id'].items():
            s = f'{field}_{label}'
            ax = axes[i_plot]
            plot_cm_seaborn(ax, actual[:, i], predicted[:, i])
            i_plot += 1
            ax.set_title(f'{label}', fontsize='large')
    
    out_path = image_dir / f"{field}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    saved_paths.append(out_path)
        
imgs = [Image.open(p) for p in saved_paths]

# Altezza totale
extra_h = int(imgs[0].height / 20)
total_h = int(sum(img.height + extra_h for img in imgs) + extra_h)
# Larghezza massima
max_w = max(img.width for img in imgs) + 2 * extra_h
combined = Image.new("RGB", (max_w, total_h), color="white")

y_offset = extra_h
for img in imgs:
    combined.paste(img, (extra_h, y_offset))
    y_offset += img.height + extra_h

combined.save(image_dir / "multilabel_fields.png")
for p in saved_paths:
    p.unlink(missing_ok=True)

