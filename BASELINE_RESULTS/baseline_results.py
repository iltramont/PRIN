import pandas as pd
import matplotlib.pyplot as plt
import random
import json

from ast import literal_eval
from pathlib import Path
from constants import AnnotationsReduced
from model_utils import *


#################
# Reproducibility
#################
random.seed(2026)


############
# Parameters
############
# Data parameters
TRAIN_FILE_NAME = "train_split_reduced.csv"
VALIDATION_FILE_NAME = "validation_split_reduced.csv"
TEST_FILE_NAME = "test_split_reduced.csv"


#######
# Utils
#######
base_dir = Path(__file__).parent.parent
# Set plotting style
plt.style.use('ggplot')


###########
# Load data
###########
file_names = {
    'train': TRAIN_FILE_NAME,
    'validation': VALIDATION_FILE_NAME,
    'test': TEST_FILE_NAME
}
paths = {split: base_dir / "data" / file_name
         for split, file_name in file_names.items()}
data = {split: pd.read_csv(path).fillna(NAN_VALUE) for split, path in paths.items()}
train_data, validation_data, test_data = data['train'], data['validation'], data['test']
# Log
print(f"{len(train_data) = }")
print(f"{len(validation_data) = }")
print(f"{len(test_data) = }")



target_columns = AnnotationsReduced.model_fields.keys()

reg_fields = get_regression_fields(AnnotationsReduced)
cl_fields = get_classification_fields(AnnotationsReduced)
mc_fields = get_multiple_choice_fields(AnnotationsReduced)
bc_fields = get_binary_classification_fields(AnnotationsReduced)
label_to_id_map = create_label_to_id_map(AnnotationsReduced)

results = {
    'validation': {
        'predicted': {field: [] for field in (cl_fields + mc_fields + bc_fields + reg_fields)},
        'actual': {field: [] for field in (cl_fields + mc_fields + bc_fields + reg_fields)}
    },
    'test': {
        'predicted': {field: [] for field in (cl_fields + mc_fields + bc_fields + reg_fields)},
        'actual': {field: [] for field in (cl_fields + mc_fields + bc_fields + reg_fields)}
    },
    'info':{
        'regression_fields': reg_fields,
        'classification_fields': cl_fields,
        'binary_classification_fields': bc_fields,
        'multiple_choice_fields': mc_fields,
        'normalization_stats': {},
        'label_to_id_map': label_to_id_map,
    }
}

def save_actual_row(res_dict: dict, split: str, actual_row: pd.Series, label_to_id_map):
    for field in (cl_fields + bc_fields):
        label = actual_row[field]
        idx = label_to_id_map[field]['label_to_id'][label]
        res_dict[split]['actual'][field].append(idx)
    for field in mc_fields:
        l = literal_eval(actual_row[field])
        bits = labels_to_bits(l, label_to_id_map[field]['label_to_id'])
        res_dict[split]['actual'][field].append(bits)

def save_random_row(res_dict: dict, split: str, actual_row: pd.Series, label_to_id_map):
    for field in (cl_fields + bc_fields):
        idx = random.choice(list(label_to_id_map[field]['label_to_id'].values()))
        res_dict[split]['predicted'][field].append(idx)
    for field in mc_fields:
        bits = []
        for _ in label_to_id_map[field]['label_to_id'].values():
            bits.append(random.randint(0, 1))
        res_dict[split]['predicted'][field].append(bits)



for _, row in validation_data.iterrows():
    # Save actual labels
    save_actual_row(results,
                    'validation',
                    row,
                    label_to_id_map)
    # Save random predictions
    save_random_row(results,
                    'validation',
                    row,
                    label_to_id_map)

for _, row in test_data.iterrows():
    # Save actual labels
    save_actual_row(results,
                    'test',
                    row,
                    label_to_id_map)
    # Save random predictions
    save_random_row(results,
                    'test',
                    row,
                    label_to_id_map)


##############
# Save results
##############
with open('results_baseline.json', "w") as f:
    json.dump(results, f, indent=4)
print(f"Results saved")



























