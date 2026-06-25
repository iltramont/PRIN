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


def converti_ore_in_archi(ore_inizio: int, ore_fine: int) -> list[tuple[int]]:
    if ore_inizio == ore_fine:
        # Arco completo
        return [(0, 12)]
    elif ore_inizio < ore_fine:
        return [(ore_inizio, ore_fine)]
    else:
        return [(ore_inizio, 12), (0, ore_fine)]

def converti_archi_in_lunghezza(archi: list[tuple[int]]) -> float:
    result = 0.0
    for arco in archi:
        result = result + arco[1] - arco[0]
    return result

def sovrapposizione_tra_2_archi(arco1: tuple[int], arco2: tuple[int]) -> float:
    return max(0.0, min(arco1[1], arco2[1]) - max(arco1[0], arco2[0]))

def ore_score(actual_inizio: int, actual_fine: int, pred_inizio: int, pred_fine: int) -> float:
    if actual_inizio is None and pred_inizio is None and actual_fine is None and pred_fine is None:
        return 1.0
    if actual_inizio is not None and pred_inizio is not None and actual_fine is not None and pred_fine is not None:
        if actual_inizio == pred_inizio and actual_fine == pred_fine:
            return 1.0
        else:
            # lunghezza totale = lunghezza actual + lunghezza pred - overlap
            archi_actual = converti_ore_in_archi(actual_inizio, actual_fine)
            archi_pred = converti_ore_in_archi(pred_inizio, pred_fine)
            lunghezza_actual = converti_archi_in_lunghezza(archi_actual)
            lunghezza_pred = converti_archi_in_lunghezza(archi_pred)
            overlap = 0.0
            for x in archi_actual:
                for y in archi_pred:
                    overlap = overlap + sovrapposizione_tra_2_archi(x, y)
            union = lunghezza_actual + lunghezza_pred - overlap
            if union == 0:
                return 0.0
            return overlap / union
    else:
        return 0.0


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
    if actual == prediction:
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

def morfologia_score(actual, prediction) -> float:
    solid = [constants.Morfologia.SOLIDO_POLIPOIDE.value, constants.Morfologia.SOLIDO_ANULARE.value]
    if actual == prediction:
        return 1.0
    elif actual in solid and prediction in solid:
        return 0.5
    else:
        return 0.0

def compare_prediction(actual: BaseModel, prediction: BaseModel, weights=constants.FEATURE_WEIGHTS) -> pd.DataFrame:
    reg_fields = model_utils.get_regression_fields(type(actual))
    multilabel_fields = model_utils.get_multiple_choice_fields(type(actual))
    rows = []
    for f in type(actual).model_fields:
        if f == 'ore_fine':
            continue
        w = 999
        a = getattr(actual, f)
        p = getattr(prediction, f)
        if (f in reg_fields) and (f not in ['ore_fine', 'ore_inizio']):
            w = weights[f]
            row = [f, a, p, reg_score(a, p), w]
        elif f == 'ore_inizio':
            w = weights['ore']
            a_inizio = a
            p_inizio = p
            a_fine = getattr(actual, 'ore_fine')
            p_fine = getattr(prediction, 'ore_fine')
            row = [(f, 'ore_fine'), (a_inizio, a_fine), (p_inizio, p_fine), ore_score(a_inizio, a_fine, p_inizio, p_fine), w]
        elif f == "stadio_N":
            w = weights[f]
            row = [f, a, p, stadio_n_score(a, p), w]
        elif f == "stadio_T":
            w = weights[f]
            row = [f, a, p, stadio_t_score(a, p), w]
        elif f == "morfologia":
            w = weights[f]
            row = [f, a, p, morfologia_score(a, p), w]
        elif f == "infiltrazione_tessuto_adiposo":
            w = weights[f]
            row = [f, a, p, infiltrazione_tessuto_adiposo_score(a, p), w]
        elif f in multilabel_fields:
            w = weights[f]
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
            row.append(w)
        else:
            w = weights[f]
            row = [f, a, p, float(int(a == p)), w]
        rows.append(row)
    return pd.DataFrame.from_records(rows, columns=['field', 'actual', 'prediction', 'score', 'weight']).set_index('field')


def add_penalties(comp_df: pd.DataFrame) -> None:
    if comp_df.loc['depositi_tumorali', 'prediction'] == 'si' and comp_df.loc['stadio_N', 'prediction'] == constants.StadioN.N0.value:
        comp_df.loc[['depositi_tumorali', 'stadio_N'], 'score'] = [0, 0]
    
    if ((comp_df.loc['coinvolgimento_fascia_mesorettale', 'prediction'] == 'si' and
         comp_df.loc['mrf', 'prediction'] == constants.MRF.MINUS.value)
         or (comp_df.loc['coinvolgimento_fascia_mesorettale', 'prediction'] == 'no' and
             comp_df.loc['mrf', 'prediction'] == constants.MRF.PLUS.value)):
        comp_df.loc[['coinvolgimento_fascia_mesorettale', 'mrf'], 'score'] = [0, 0]
    
    if ((comp_df.loc['infiltrazione_organi_extra', 'prediction'] == 'si' and
         comp_df.loc['infiltrazione_organi_dettagli', 'prediction'] == [])
         or (comp_df.loc['infiltrazione_organi_extra', 'prediction'] == 'no' and
             comp_df.loc['infiltrazione_organi_dettagli', 'prediction'] != [])):
        comp_df.loc[['infiltrazione_organi_extra', 'infiltrazione_organi_dettagli'], 'score'] = [0, 0]


def prediction_score(actual: BaseModel, prediction: BaseModel, weights=constants.FEATURE_WEIGHTS) -> float:
    if prediction is None:
        return 0.0
    comp_df = compare_prediction(actual, prediction, weights=weights)
    add_penalties(comp_df)
    return (comp_df['score'] * comp_df['weight']).sum() / comp_df['weight'].sum() 


if __name__ == '__main__':
    from pprint import pprint
    base_dir = Path(__file__).parent.parent
    ANN_MODEL = constants.RectalCancerStagingData
    data = load_results_data('new_results_gpt-4.1-mini.jsonl', base_dir/"data"/"inference", ANN_MODEL)
    comp_df = compare_prediction(data[0]['actual'], data[0]['prediction'])
    score = prediction_score(data[0]['actual'], data[0]['prediction'])
    #print(comp_df)
    #print(comp_df.loc[['depositi_tumorali', 'stadio_N'], 'score'])
    comp_df.loc[['depositi_tumorali', 'stadio_N'], 'score'] = [0, 0]
    #print(comp_df.loc[['depositi_tumorali', 'stadio_N'], 'score'])

