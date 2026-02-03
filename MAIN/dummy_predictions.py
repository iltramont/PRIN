import pandas as pd
import matplotlib.pyplot as plt
import random
import json

from ast import literal_eval
from pathlib import Path

from pprint import pprint

import constants

from sklearn.dummy import DummyClassifier, DummyRegressor
import numpy as np

from model_utils import (
    get_binary_classification_fields,
    get_classification_fields,
    get_multiple_choice_fields,
    get_regression_fields,
    create_label_to_id_map,
    labels_to_bits,
    get_field_values,
    get_number_of_classes,
    get_optional_regression_fields
)


#################
# Reproducibility
#################
random.seed(constants.SEED)


############
# Parameters
############
ann_model = constants.Annotations
base_dir = Path(__file__).parent.parent
DUMMY_REG_STRATEGY = 'mean'  # Options: 'mean', 'median', 'quantile', 'constant'

# Carichiamo i nostri file csv
file_names = {
    'train': constants.TRAIN_SPLIT_FILE_NAME,
    'validation': constants.VALIDATION_SPLIT_FILE_NAME,
    'test': constants.TEST_SPLIT_FILE_NAME  
}

paths = {
    split: base_dir / "data" / file_name for split, file_name in file_names.items()
}

data = dict()
for split, path in paths.items():
    print(path)
    data[split] = pd.read_csv(path)

train_data, validation_data, test_data = data['train'], data['validation'], data['test']

split_to_data = {
    'train': train_data,
    'validation': validation_data,
    'test': test_data
}

print(f"{len(train_data) = }")
print(f"{len(validation_data) = }")
print(f"{len(test_data) = }")


target_columns = ann_model.model_fields.keys()

reg_fields = get_regression_fields(ann_model)
cl_fields = get_classification_fields(ann_model)
mc_fields = get_multiple_choice_fields(ann_model)
bc_fields = get_binary_classification_fields(ann_model)

label_to_id_map = create_label_to_id_map(ann_model)

################################
# Convert float columns to Int64
################################
float_cols = train_data.select_dtypes("float").columns
for col in float_cols:
    train_data[col] = train_data[col].astype("Int64")
    test_data[col] = test_data[col].astype("Int64")
    validation_data[col] = validation_data[col].astype("Int64")


###########
# Functions
###########
def save_actual_row(res_dict: dict, split: str, actual_row: pd.Series, label_to_id_map):
    for field in (cl_fields + bc_fields):
        label = actual_row[field]
        idx = label_to_id_map[field]['label_to_id'][label]
        res_dict[split]['actual'][field].append(idx)
    for field in mc_fields:
        l = literal_eval(actual_row[field])
        bits = labels_to_bits(l, label_to_id_map[field]['label_to_id'])
        res_dict[split]['actual'][field].append(bits)
    for field in reg_fields:
        v = actual_row[field]
        if pd.isna(v):
            res_dict[split]['actual'][field].append(None)
        else:
            res_dict[split]['actual'][field].append(v)
          
        
def dummy_classification(y_train: np.ndarray | list,
                         strategy: str,
                         len_validation: int,
                         len_test: int,
                         random_state=constants.SEED) -> tuple[list[int]]:
    x_train = np.zeros(len(y_train))
    classifier = DummyClassifier(strategy=strategy, random_state=random_state)
    classifier.fit(x_train, y_train)
    x_validation = np.zeros(len_validation)
    y_validation = classifier.predict(x_validation)
    x_test = np.zeros(len_test)
    y_test = classifier.predict(x_test)
    return y_validation.tolist(), y_test.tolist()


def dummy_regression(y_train: np.ndarray | list,
                     strategy: str,
                     len_validation: int,
                     len_test: int) -> tuple[list[int]]:
    x_train = np.zeros(len(y_train))
    regressor = DummyRegressor(strategy=strategy)
    regressor.fit(x_train, y_train)
    x_validation = np.zeros(len_validation)
    y_validation = regressor.predict(x_validation)
    x_test = np.zeros(len_test)
    y_test = regressor.predict(x_test)
    return y_validation.round().astype(int).tolist(), y_test.round().astype(int).tolist()


def create_res_dict():
    return {
        'train': {
            'predicted': {field: [] for field in (cl_fields + mc_fields + bc_fields + reg_fields)},
            'actual': {field: [] for field in (cl_fields + mc_fields + bc_fields + reg_fields)}
        },
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


results_most_frequent = create_res_dict()
results_uniform = create_res_dict()
results_stratified = create_res_dict()


len_validation = len(validation_data)
len_test = len(test_data)

strategies = ('most_frequent', 'uniform', 'stratified')
res_dicts = (results_most_frequent, results_uniform, results_stratified)

for strategy, res_dict in zip(strategies, res_dicts):
    for split in ['train', 'validation', 'test']:
        # Save actual labels
        for _, row in split_to_data[split].iterrows():
            save_actual_row(res_dict,
                            split,
                            row,
                            label_to_id_map)
        # Save dummy predictions
        for field in bc_fields + cl_fields:
            y_train = res_dict['train']['actual'][field]
            pred_validation, pred_test = dummy_classification(
                y_train,
                strategy,
                len_validation,
                len_test
            )
            res_dict['validation']['predicted'][field] = pred_validation
            res_dict['test']['predicted'][field] = pred_test
        for field in mc_fields:
            matrix_train = np.array(res_dict['train']['actual'][field])
            matrix_validation = []
            matrix_test = []
            for i in range(matrix_train.shape[1]):
                y_validation, y_test = dummy_classification(
                    y_train=matrix_train[:, i],
                    strategy=strategy,
                    len_validation=len_validation,
                    len_test=len_test
                )
                matrix_validation.append(y_validation)
                matrix_test.append(y_test)
            res_dict['validation']['predicted'][field] = np.array(matrix_validation).T.tolist()
            res_dict['test']['predicted'][field] = np.array(matrix_test).T.tolist()
        for field in reg_fields:
            y_train = res_dict['train']['actual'][field]
            pred_validation, pred_test = dummy_regression(
                pd.Series(np.array(y_train)).dropna().tolist(),
                DUMMY_REG_STRATEGY,
                len_validation,
                len_test
            )
            res_dict['validation']['predicted'][field] = pred_validation
            res_dict['test']['predicted'][field] = pred_test
                  

##############
# Save results
##############      
results_most_frequent.pop('train')
results_uniform.pop('train')
results_stratified.pop('train')

output_path = base_dir / "data" / "inference"
output_path.mkdir(parents=True, exist_ok=True)

with open(output_path / 'results_baseline_most_frequent.json', "w") as f:
    json.dump(results_most_frequent, f, indent=4)
    
with open(output_path / 'results_baseline_uniform.json', "w") as f:
    json.dump(results_uniform, f, indent=4)
    
with open(output_path / 'results_baseline_stratified.json', "w") as f:
    json.dump(results_stratified, f, indent=4)

print(f"Results saved")