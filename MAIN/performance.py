from pathlib import Path
import pandas as pd
import json
import constants
from performance_utils import *


###########
# Parametri
###########
base_dir = Path(__file__).parent.parent
ANN_MODEL = constants.RectalCancerStagingData

RESULT_FILES = (
    'results_baseline_most_frequent.jsonl',
    'results_baseline_uniform.jsonl',
    'results_baseline_stratified.jsonl',
    'results_gpt-4.1-nano.jsonl',
    'results_gpt-4.1-nano-tuned.jsonl',
    'results_gpt-4.1-nano-tuned-oversampling.jsonl',
    'results_gpt-4.1-mini.jsonl',
    'results_gpt-4.1.jsonl',
    'results_gpt-4.1-tuned-oversampling.jsonl',
    'results_gpt-4.1-tuned-MMR.jsonl',
    'results_gpt-4.1-tuned-similar_examples.jsonl',
    'results_gpt-5-nano.jsonl',
    'results_gpt-5.2.jsonl',
    'results_gpt-5.2-reasoning.jsonl',
    'results_gpt-5.2-low-reasoning.jsonl',
    'results_mistral_large_3.jsonl',
    'results_opus-4.6.jsonl',
    'results_opus-4.6_similar_examples.jsonl',
    'results_opus-4.6_MMR.jsonl'
)


result_dfs = []
for f in RESULT_FILES:
    data = load_results_data(f, base_dir/"data"/"inference", ANN_MODEL)
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




























