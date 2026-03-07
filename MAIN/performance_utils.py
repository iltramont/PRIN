from pathlib import Path
import pandas as pd
import json
import numpy as np
import json
import constants
import model_utils
from pydantic import BaseModel
from sklearn.metrics import jaccard_score


def load_results_data(file_name: str,
                      results_directory: Path,
                      ann_model: type[BaseModel]) -> list:
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
                # The score is 1 - MAPE, with a minimum of zero
                return max(0.0, 1.0 - (np.abs((prediction - actual) / actual)))
    else:
        return 0.0

def stadio_n_score(actual, prediction) -> float:
    positive_states = [constants.StadioN.N1.value, constants.StadioN.N2.value, constants.StadioN.N_PLUS.value]
    if actual == prediction:
        # Copre anche il caso N0, N0
        return 1.0
    if (actual in positive_states) and (prediction in positive_states):
        # Copre i casi (N1, N2), (N1, N+), (N2, N+)
        return 1.0
    else:
        return 0.0

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
        # Caso si_5mm, si_5mm_plus
        return 0.5
    else:
        return 0.0

def compare_prediction(actual: BaseModel, prediction: BaseModel) -> pd.DataFrame:
    reg_fields = model_utils.get_regression_fields(type(actual))
    multilabel_fields = model_utils.get_multiple_choice_fields(type(actual))
    rows = []
    for f in type(actual).model_fields:
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


def prediction_score(actual: BaseModel, prediction: BaseModel) -> float:
    comp_df = compare_prediction(actual, prediction)
    return comp_df['score'].mean()





