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


############
# Parameters
############
# Data parameters
VALIDATION_FILE_NAME = "validation_split.csv"
TEST_FILE_NAME = "test_split.csv"


#############
# Preliminari
#############
base_dir = Path.cwd()
device = get_device()
print(f'{device = }')


############
# Load model
############
model = ReportExtractor.from_pretrained(base_dir / "saved_models" / "report_extractor", device=device)
model.to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(base_dir / "saved_models" / "report_extractor")


###########
# Load data
###########
file_names = {
    'validation': VALIDATION_FILE_NAME,
    'test': TEST_FILE_NAME
}
paths = {split: base_dir / "data" / file_name
         for split, file_name in file_names.items()}
data = {split: pd.read_csv(path) for split, path in paths.items()}
validation_data, test_data = data['validation'], data['test']
# Log
print(f"{len(validation_data) = }")
print(f"{len(test_data) = }")
# Create nan-flag columns
for split, df in data.items():
    add_nan_flag_to_df(df)
# Create lists of Annotated reports
annotated_reports =  {split: create_list_of_annotated_reports(data[split]) for split in file_names}
# Check the maximum number of tokens for each report
print('model context length:', model.encoder.config.max_position_embeddings)
for split in annotated_reports:
    print(f'{split}: {len(annotated_reports[split])} reports')
    max_n_tokens = 0
    del_list = []
    for i, report in enumerate(annotated_reports[split]):
        x = tokenizer(report.report_text, return_tensors='pt')['input_ids'].shape[1]
        max_n_tokens = max(max_n_tokens, x)
        if x > model.encoder.config.max_position_embeddings:
            del_list.append(i)
    print(del_list)
    print(f'{max_n_tokens = }')
    # Delete long reports
    for i in del_list[::-1]:
        annotated_reports[split].pop(i)
    print(f'After deletion: {len(annotated_reports[split])} reports')
    
    
##############################
# Create Hugging Face Datasets
##############################
dataset = DatasetDict({
    split: create_hugging_face_dataset(annotated_reports[split])
    for split in annotated_reports
})
# Tokenize text and set format to torch
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="longest", max_length=model.encoder.config.max_position_embeddings)
dataset = dataset.map(tokenize_function, batched=True)
cols_to_remove = [col for col in ["token_type_ids"] if col in dataset['validation'].column_names]
dataset = dataset.remove_columns(cols_to_remove)
dataset.set_format('torch')
# Add annotation fields to the dataset
for split, reports in annotated_reports.items():
    dataset[split] = add_target_columns_to_dataset(dataset=dataset[split],
                                                   annotated_reports=reports,
                                                   label_to_id_map=model.label_to_id_map,
                                                   classification_columns=model.classification_fields,
                                                   binary_classification_columns=model.binary_classification_fields,
                                                   multiple_choice_columns=model.multiple_choice_fields,
                                                   regression_columns=model.regression_fields,
                                                   normalization_stats=model.normalization_stats)


###########
# Inference
###########
results = {split: dict() for split in file_names}
dataset.set_format(type="torch")
model.eval()
for split in results:
    loader = DataLoader(dataset[split], batch_size=16, shuffle=False)
    batches_results = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            model_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            reworked_output = model.rework_output(model_output)

            batches_results.append(reworked_output)

        predictions = {
            field: np.concatenate([r[field] for r in batches_results]).tolist()
            for field in model.all_fields()
        }
        results[split]['predicted'] = predictions

dataset.set_format(type="numpy")
for split in results:
    actual = dict()
    for f in model.all_fields():
        if f in model.regression_fields:
            mu, std = model.normalization_stats[f]
            actual[f] = np.round(np.abs((dataset[split][f] * std) + mu), 2).tolist()
        else:
            actual[f] = dataset[split][f].tolist()
    results[split]['actual'] = actual
    
    
    
print(model.label_to_id_map)    
exit()
    
results['info'] = {
    'regression_fields': model.regression_fields,
    'classification_fields': model.classification_fields,
    'binary_classification_fields': model.binary_classification_fields,
    'multiple_choice_fields': model.multiple_choice_fields,
    'normalization_stats': model.normalization_stats,
    'label_to_id_map': model.label_to_id_map
}
    

##############
# Save results
##############
with open(base_dir / "ENCODER" / "results.json", "w") as f:
    json.dump(results, f, indent=4)
