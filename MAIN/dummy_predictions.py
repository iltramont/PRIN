import pandas as pd
import matplotlib.pyplot as plt
import random
import json

from ast import literal_eval
from pathlib import Path

from pprint import pprint

import constants

from sklearn.dummy import DummyClassifier
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

pprint(train_data.dtypes)
pprint([train_data[reg_fields]])


for f in reg_fields:
    print(validation_data[f].value_counts())

exit()


def save_actual_row(res_dict: dict, split: str, actual_row: pd.Series, label_to_id_map):
    for field in (cl_fields + bc_fields):
        label = actual_row[field]
        idx = label_to_id_map[field]['label_to_id'][label]
        res_dict[split]['actual'][field].append(idx)
    for field in mc_fields:
        l = literal_eval(actual_row[field])
        bits = labels_to_bits(l, label_to_id_map[field]['label_to_id'])
        res_dict[split]['actual'][field].append(bits)
    for field in get_regression_fields:
        v = actual_row[field]
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


######################
# Most frequent method
######################
strategy = 'most_frequent'
len_validation = len(validation_data)
len_test = len(test_data)

results_most_frequent = {
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

for split in ['train', 'validation', 'test']:
    # Save actual labels
    for _, row in split_to_data[split].iterrows():
        save_actual_row(results_most_frequent,
                        split,
                        row,
                        label_to_id_map)
    # Save dummy predictions
    for field in bc_fields + cl_fields:
        y_train = results_most_frequent['train']['actual'][field]
        pred_validation, pred_test = dummy_classification(
            y_train,
            strategy,
            len_validation,
            len_test
        )
        results_most_frequent['validation']['predicted'][field] = pred_validation
        results_most_frequent['test']['predicted'][field] = pred_test
    for field in mc_fields:
        matrix_train = np.array(results_most_frequent['train']['actual'][field])
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
        results_most_frequent['validation']['predicted'][field] = np.array(matrix_validation).T.tolist()
        results_most_frequent['test']['predicted'][field] = np.array(matrix_test).T.tolist()
        
        
        
################
# Uniform method
################
strategy = 'uniform'
len_validation = len(validation_data)
len_test = len(test_data)

results_uniform = {
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

for split in ['train', 'validation', 'test']:
    # Save actual labels
    for _, row in split_to_data[split].iterrows():
        save_actual_row(results_uniform,
                        split,
                        row,
                        label_to_id_map)
    # Save dummy predictions
    for field in bc_fields + cl_fields:
        y_train = results_uniform['train']['actual'][field]
        pred_validation, pred_test = dummy_classification(
            y_train,
            strategy,
            len_validation,
            len_test
        )
        results_uniform['validation']['predicted'][field] = pred_validation
        results_uniform['test']['predicted'][field] = pred_test
    for field in mc_fields:
        matrix_train = np.array(results_uniform['train']['actual'][field])
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
        results_uniform['validation']['predicted'][field] = np.array(matrix_validation).T.tolist()
        results_uniform['test']['predicted'][field] = np.array(matrix_test).T.tolist()
        
        
###################
# Stratified method
###################
strategy = 'stratified'
len_validation = len(validation_data)
len_test = len(test_data)

results_stratified = {
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

for split in ['train', 'validation', 'test']:
    # Save actual labels
    for _, row in split_to_data[split].iterrows():
        save_actual_row(results_stratified,
                        split,
                        row,
                        label_to_id_map)
    # Save dummy predictions
    for field in bc_fields + cl_fields:
        y_train = results_stratified['train']['actual'][field]
        pred_validation, pred_test = dummy_classification(
            y_train,
            strategy,
            len_validation,
            len_test
        )
        results_stratified['validation']['predicted'][field] = pred_validation
        results_stratified['test']['predicted'][field] = pred_test
    for field in mc_fields:
        matrix_train = np.array(results_stratified['train']['actual'][field])
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
        results_stratified['validation']['predicted'][field] = np.array(matrix_validation).T.tolist()
        results_stratified['test']['predicted'][field] = np.array(matrix_test).T.tolist()
        
        

##############
# Save results
##############      
results_most_frequent.pop('train')
results_uniform.pop('train')
results_stratified.pop('train')

with open(base_dir / 'BASELINE_RESULTS' / 'results_baseline_most_frequent.json', "w") as f:
    json.dump(results_most_frequent, f, indent=4)
    
with open(base_dir / 'BASELINE_RESULTS' / 'results_baseline_uniform.json', "w") as f:
    json.dump(results_uniform, f, indent=4)
    
with open(base_dir / 'BASELINE_RESULTS' / 'results_baseline_stratified.json', "w") as f:
    json.dump(results_stratified, f, indent=4)

print(f"Results saved")