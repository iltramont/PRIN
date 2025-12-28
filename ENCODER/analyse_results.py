from pathlib import Path
import pandas as pd
import json
import torch
import numpy as np
import json

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import DatasetDict
from train_utils import get_device, add_nan_flag_to_df, create_list_of_annotated_reports, create_hugging_face_dataset, add_target_columns_to_dataset
from classifiers import ReportExtractor
from pprint import pprint

base_dir = Path.cwd()
print(base_dir)

with open(base_dir / "ENCODER" / "results.json", "r") as f:
    results = json.load(f)

############
# Regression
############
