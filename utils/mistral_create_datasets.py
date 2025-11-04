# We want he dataset to be in the following format:
# {
#   "text": "The report",
#   "labels": {
#       "label_1": "value_1",
#       "label_2": "value_2",
#       "label_multi": [
#           "value_multi_1",
#           "value_multi_2",
#           ...
#       ],
#       "label_n": "value_n",
#       ...    
#   }  
# } 

# In particular we can handle single value labels (where the value is a string) and multi value labels (where the value is a list of strings)

import json
import pandas as pd
import numpy as np
from pathlib import Path
from mistral_utils import convert_row_to_json_mistral

# Parameters
TRAIN_SPLIT_NAME: str = 'train_split.csv'
TEST_SPLIT_NAME: str = 'test_split.csv'
VALIDATION_SPLIT_NAME: str | None = 'validation_split.csv'


# Load data
base_dir = Path(__file__).parent.parent

train_path = base_dir / "data" / TRAIN_SPLIT_NAME
test_path = base_dir / "data" / TEST_SPLIT_NAME
validation_path = base_dir / "data" / VALIDATION_SPLIT_NAME

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
validation_data = pd.read_csv(validation_path)

row = train_data.iloc[0]
json_dict = convert_row_to_json_mistral(row)

for k, v in json_dict['labels'].items():
    print(f"{k}: {v}\t{type(v)}")


def _to_python_types(obj):
    """Ricorsivamente converte numpy/pandas types in tipi Python serializzabili da json."""
    if isinstance(obj, dict):
        return {k: _to_python_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python_types(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_python_types(v) for v in obj)
    # numpy scalars
    if isinstance(obj, (np.bool_, )):
        return bool(obj)
    if isinstance(obj, (np.integer, )):
        return int(obj)
    if isinstance(obj, (np.floating, )):
        return float(obj)
    # numpy arrays
    if isinstance(obj, np.ndarray):
        return _to_python_types(obj.tolist())
    # pandas NA / NaN / pd.Timestamp -> fallback to str or native conversion
    try:
        import pandas as pd
        if obj is pd.NA:
            return None
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
    except Exception:
        pass
    return obj

py_labels = _to_python_types(json_dict['labels'])

for k, v in py_labels.items():
    print(f"{k}: {v}\t{type(v)}")

print(json_dict['labels']['sedi_linfonodi_locoregionali'])
print(json_dict['labels']['sedi_linfonodi_non_locoregionali'])
json_dict['labels']['sedi_linfonodi_non_locoregionali'] = ['Nan']
print(json.dumps(json_dict['labels'], indent=4))
