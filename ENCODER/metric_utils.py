import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from scipy import stats


def metrics_binary(y_true, pred_prob, threshold=0.5, plot=True, field_name=None):
    y_pred = (pred_prob > threshold).astype(int)
    precision_score = metrics.precision_score(y_true, y_pred, zero_division=0.0)
    recall_score = metrics.recall_score(y_true, y_pred, zero_division=0.0)
    acc = metrics.accuracy_score(y_true, y_pred)
    bal_acc = metrics.balanced_accuracy_score(y_true, y_pred, adjusted=False)
    bal_acc_adj = metrics.balanced_accuracy_score(y_true, y_pred, adjusted=True)
    roc_auc_score = metrics.roc_auc_score(y_true, pred_prob)
    f1 = metrics.f1_score(y_true, y_pred)
    cohen_kappa = metrics.cohen_kappa_score(y_true, y_pred)
    average_precision = metrics.average_precision_score(y_true, pred_prob)
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, pred_prob)
    results = {
           'precision score': precision_score,
           'recall score': recall_score,
           'accuracy score': acc,
           'balanced accuracy score': bal_acc,
           'balanced accuracy score adjusted': bal_acc_adj,
           'roc auc score': roc_auc_score,    
           'f1 score': f1,    
           'cohen kappa': cohen_kappa,
           'average precision': average_precision
    }
    cm = metrics.confusion_matrix(y_true, y_pred)
    if plot:
        fig, axes = plt.subplots(1, 4, figsize=(25,5))
        if field_name is not None:
            fig.suptitle(field_name, fontsize='xx-large')
        metrics.PrecisionRecallDisplay.from_predictions(y_true, pred_prob, ax=axes[0], plot_chance_level=True)
        axes[0].set_title('Precision-Recall Curve')
        
        metrics.RocCurveDisplay.from_predictions(y_true, pred_prob, ax=axes[1], plot_chance_level=True)
        axes[1].set_title('ROC AUC Curve')
        
        metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=axes[2])
        axes[2].grid(False)
        axes[2].set_title(f'Confusion Matrix - threshold = {threshold}')
        
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        thresholds = np.append(thresholds, 1.0)
        best_threshold = thresholds[np.argmax(f1_scores)]
        best_f1 = np.max(f1_scores)
        print(f'{best_threshold = }')
        axes[3].plot(thresholds, f1_scores, label="F1 score")
        axes[3].set_xlabel("Threshold")
        axes[3].set_ylabel("F1 score")
        axes[3].set_title("F1 score vs Threshold")
        plt.show()
    return results, cm


def metrics_classification(y_true, pred_prob, id_to_label_dict: dict[int, str], plot=True, field_name=None):
    y_pred = pred_prob.argmax(axis=-1)
       
    result = {
        'accuracy': metrics.accuracy_score(y_true, y_pred),
        'cohen_kappa': metrics.cohen_kappa_score(y_true, y_pred),
        'f1_micro': metrics.f1_score(y_true, y_pred, average="micro", zero_division=0.0),
        'f1_macro': metrics.f1_score(y_true, y_pred, average="macro", zero_division=0.0),
        'f1_weighted': metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0.0),
        'precision_micro': metrics.precision_score(y_true, y_pred, average="micro", zero_division=0.0),
        'precision_macro': metrics.precision_score(y_true, y_pred, average="macro", zero_division=0.0),
        'precision_weighted': metrics.precision_score(y_true, y_pred, average="weighted", zero_division=0.0),
        'recall_micro': metrics.recall_score(y_true, y_pred, average="micro", zero_division=0.0),
        'recall_macro': metrics.recall_score(y_true, y_pred, average="macro", zero_division=0.0),
        'recall_weighted': metrics.recall_score(y_true, y_pred, average="weighted", zero_division=0.0),
    }
    cm = metrics.confusion_matrix(y_true, y_pred, labels=list(id_to_label_dict.keys()))
    if plot:
        y_pred = [id_to_label_dict[i] for i in y_pred]
        y_true = [id_to_label_dict[i] for i in y_true] 
        fig, axes = plt.subplots(1, 2, figsize=(20,5))
        if field_name is not None:
            fig.suptitle(field_name, fontsize='xx-large')
        metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=axes[0], xticks_rotation=45, cmap='Blues')
        axes[0].grid(False)
        axes[0].set_title(f'Confusion Matrix')
        metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=axes[1], xticks_rotation=45, normalize='all', cmap='Blues')
        axes[1].grid(False)
        axes[1].set_title(f'Confusion Matrix - percentage')
        plt.show()
    return result, cm


def metrics_regression(y_true, y_pred, plot=True, field_name=None):
    # --- Metriche principali ---
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = metrics.root_mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    mape = metrics.mean_absolute_percentage_error(y_true, y_pred)
    medae = metrics.median_absolute_error(y_true, y_pred)
    explained_var = metrics.explained_variance_score(y_true, y_pred)

    # Correlazioni
    #pearson_corr, _ = stats.pearsonr(y_true, y_pred)
    #spearman_corr, _ = stats.spearmanr(y_true, y_pred)

    results = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "Median AE": medae,
        "Explained variance": explained_var,
        #"Pearson correlation": pearson_corr,
        #"Spearman correlation": spearman_corr
    }

    # --- Plot ---
    if plot:
        fig, axes = plt.subplots(1, 4, figsize=(25, 5))

        if field_name is not None:
            fig.suptitle(field_name, fontsize="xx-large")

        residuals = y_true - y_pred

        # 1. Scatter y_true vs y_pred
        axes[0].scatter(y_true, y_pred, alpha=0.6)
        axes[0].plot([y_true.min(), y_true.max()],
                     [y_true.min(), y_true.max()],
                     color="red", linestyle="--")
        axes[0].set_title("y_true vs y_pred")
        axes[0].set_xlabel("y_true")
        axes[0].set_ylabel("y_pred")

        # 2. Residual plot
        axes[1].scatter(y_pred, residuals, alpha=0.6)
        axes[1].axhline(0, color="red", linestyle="--")
        axes[1].set_title("Residual plot")
        axes[1].set_xlabel("y_pred")
        axes[1].set_ylabel("Residuals")

        # 3. Histogram of residuals
        axes[2].hist(residuals, bins=30, alpha=0.7)
        axes[2].set_title("Residual distribution")
        axes[2].set_xlabel("Residual")
        axes[2].set_ylabel("Count")

        # 4. QQ-plot
        stats.probplot(residuals, dist="norm", plot=axes[3])
        axes[3].set_title("QQ-plot of residuals")

        plt.tight_layout()
        plt.show()

    return results


def metrics_multilabel(y_true: np.ndarray, pred_prob: np.ndarray,
                       id_to_label_dict: dict[int, str],
                       thresholds: float | list[float] = 0.5,
                       plot=True,
                       field_name=None):

    # --- Predizioni ---
    if type(thresholds) is list:
        thresholds = np.array(thresholds)
    y_pred = (pred_prob > thresholds).astype(int)

    labels = list(id_to_label_dict.values())
    result = {
        'exact_match_acc': (y_pred == y_true).all(axis=1).mean(),
        'hamming_loss': metrics.hamming_loss(y_true, y_pred),  # Lower is better
        'f1_micro': metrics.f1_score(y_true, y_pred, average="micro", zero_division=0.0),
        'f1_macro': metrics.f1_score(y_true, y_pred, average="macro", zero_division=0.0),
        'f1_weighted': metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0.0),
        'f1_samples': metrics.f1_score(y_true, y_pred, average="samples", zero_division=0.0),
        'jaccard_micro': metrics.jaccard_score(y_true, y_pred, average="micro", zero_division=0.0),
        'jaccard_macro': metrics.jaccard_score(y_true, y_pred, average="macro", zero_division=0.0),
        'jaccard_weighted': metrics.jaccard_score(y_true, y_pred, average="weighted", zero_division=0.0),
        'jaccard_samples': metrics.jaccard_score(y_true, y_pred, average="samples", zero_division=0.0)
    }
    cms = metrics.multilabel_confusion_matrix(y_true, y_pred)
    # --- Plot ---
    if plot:
        
        fig, axes = plt.subplots(1, len(labels), figsize=(5*len(labels), 5))
        if len(labels) == 1:
            axes = [axes]

        if field_name is not None:
            fig.suptitle(field_name, fontsize="xx-large")

        for i, m in enumerate(cms):
            metrics.ConfusionMatrixDisplay(confusion_matrix=m).plot(ax=axes[i])
            axes[i].grid(False)
            axes[i].set_title(labels[i])
        plt.show()

    return result, cms



