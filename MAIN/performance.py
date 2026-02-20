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
from pydantic import BaseModel
from sklearn.metrics import jaccard_score


###########
# Parametri
###########
base_dir = Path(__file__).parent.parent
ANN_MODEL = constants.RectalCancerStagingData

RESULT_FILES = (
    'results_gpt-4.1-nano.jsonl',
    'results_gpt-4.1-nano-tuned.jsonl',
    'results_gpt-4.1-nano-tuned-oversampling.jsonl',
    'results_gpt-4.1-mini.jsonl',
    'results_gpt-4.1.jsonl',
    'results_gpt-4.1-tuned-oversampling.jsonl',
    'results_gpt-5-nano.jsonl',
    'results_gpt-5.2.jsonl',
    'results_gpt-5.2-reasoning.jsonl',
    'results_gpt-5.2-low-reasoning.jsonl',
    'results_mistral_large_3.jsonl',
    'results_opus-4.6.jsonl',
    'results_opus-4.6_similar_examples.jsonl',
    'results_opus-4.6_MMR.jsonl'
)

# Set plot style
plt.style.use('ggplot')
#sns.set_palette('hls')
# Colors
finomnia_palette = sns.color_palette(('#db038a',   # Pink
                                      '#66218a',   # Violet
                                      '#2659ab',   # Blue
                                      '#081c36',   # Dark blue
                                      '#45c9f5'))  # Light blue


def load_results_data(file_name: str,
                      results_directory: Path = base_dir/"data"/"inference",
                      ann_model: type[BaseModel] = ANN_MODEL) -> list:
    with open(results_directory / file_name, "r") as file:
        dicts = [json.loads(line) for line in file]
        results = []
        for d in dicts:
            results.append({
                "model": d["model"],
                "split": d["split"],
                "id": d["id"],
                "actual": ann_model.model_validate(d["actual"]),
                "prediction": ann_model.model_validate(d["prediction"])
            })
        return results

def reg_score(actual, prediction) -> float:
    if actual is None and prediction is None:
        return 1.0
    elif actual is not None and prediction is not None:
        if actual == prediction:
            return 1.0
        else:
            if actual == 0:
                return 0.0
            else:
                return max(0.0, 1.0 - (np.abs((prediction - actual) / actual)))
    else:
        return 0.0

def stadio_n_score(actual, prediction) -> float:
    if actual == prediction:
        return 1.0
    if actual == constants.StadioN.N0.value or actual == constants.StadioN.N0.value:
        return 0.0
    else:
        return 0.5

def stadio_t_score(actual, prediction) -> float:
    t3 = [constants.StadioT.T3AB.value, constants.StadioT.T3CD.value]
    t4 = [constants.StadioT.T4A.value, constants.StadioT.T4B.value]
    if actual == prediction:
        return 1.0
    elif actual in t3 and prediction in t3:
        return 0.5
    elif actual in t4 and prediction in t4:
        return 0.5
    else:
        return 0.0

def infiltrazione_tessuto_adiposo_score(actual, prediction) -> float:
    if actual == prediction:
        return 1.0
    elif actual != constants.InfiltrazioneTessutoAdiposo.NO and prediction != constants.InfiltrazioneTessutoAdiposo.NO:
        return 0.5
    else:
        return 0.0

def compare_prediction(actual: ANN_MODEL, prediction: ANN_MODEL) -> pd.DataFrame:
    reg_fields = model_utils.get_regression_fields(ANN_MODEL)
    multilabel_fields = model_utils.get_multiple_choice_fields(ANN_MODEL)
    rows = []
    for f in ANN_MODEL.model_fields:
        a = getattr(actual, f)
        p = getattr(prediction, f)
        if f in reg_fields:
            row = [f, a, p, reg_score(a, p)]
        elif f == "stadio_N":
            row = [f, a, p, stadio_n_score(a, p)]
        elif f == "stadio_T":
            row = [f, a, p, stadio_t_score(a, p)]
        elif f == "infiltrazione_tessuto_adiposo":
            row = [f, a, p, infiltrazione_tessuto_adiposo_score(a, p)]
        elif f in multilabel_fields:
            row = [
                f,
                [x[0] for x in a if x[1] == constants.Flag.SI.value],
                [x[0] for x in p if x[1] == constants.Flag.SI.value]
            ]
            abits = [int(x[1] == constants.Flag.SI.value) for x in a]
            pbits = [int(x[1] == constants.Flag.SI.value) for x in p]
            if abits == pbits:
                row.append(1.0)
            else:
                row.append(jaccard_score(abits, pbits))
        else:
            row = [f, a, p, float(int(a == p))]
        rows.append(row)
    return pd.DataFrame.from_records(rows, columns=['field', 'actual', 'perdicted', 'score'])

def prediction_score(actual: ANN_MODEL, prediction: ANN_MODEL) -> float:
    comp_df = compare_prediction(actual, prediction)
    return comp_df['score'].mean()


result_dfs = []
for f in RESULT_FILES:
    data = load_results_data(f)
    scores = []
    for r in data:
        row = [
            r['id'],
            r['split'],
            r['model'],
            prediction_score(r['actual'], r['prediction'])
        ]
        scores.append(row)
    df = pd.DataFrame.from_records(scores, columns=['id', 'split', 'model', 'score'])
    result_dfs.append(df)

result_df: pd.DataFrame = pd.concat(result_dfs)
    
result_df = result_df.pivot(index=['id', 'split'], columns='model', values='score')
result_df.sort_values(by=['split', 'id'], ascending=[False, True], inplace=True)

result_df.to_csv(base_dir/"data"/"metrics"/"scores.csv")

print(result_df)




























