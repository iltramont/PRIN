from pathlib import Path

from model_utils import (create_label_to_id_map,
                         labels_to_bits,
                         bits_to_labels,
                         NAN_VALUE,
                         get_regression_fields,
                         get_multiple_choice_fields,
                         get_binary_classification_fields,
                         get_classification_fields)


#from IPython.display import clear_output

#from huggingface_hub import login

#import torch
#import torch.nn as nn
#from torch.utils.data import DataLoader

from transformers import AutoTokenizer, DefaultDataCollator
from datasets import load_dataset, Dataset, DatasetDict

from constants import AnnotatedReportReduced, AnnotationsReduced

#from classifiers import ReportExtractor
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from ast import literal_eval
#import loop
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint
import math
import random

DATA_FILE_NAME = "tumoreprimitivo_clean.csv"
"""
TRAIN_FILE_NAME = "train_split.csv"
VALIDATION_FILE_NAME = "validation_split.csv"
TEST_FILE_NAME = "test_split.csv"
"""


SEED = 2026
base_dir = Path(__file__).parent.parent
random.seed(SEED)

df = pd.read_csv(base_dir / "data" / DATA_FILE_NAME)

reg_fields = get_regression_fields(AnnotationsReduced)
cl_fields = get_classification_fields(AnnotationsReduced)
mc_fields = get_multiple_choice_fields(AnnotationsReduced)
binary_fields = get_binary_classification_fields(AnnotationsReduced)

print("Campi di regressione:", reg_fields)
print("Campi di classificazione:", cl_fields)
print("Campi di multi-scelta:", mc_fields)
print("Campi binari:", binary_fields)    