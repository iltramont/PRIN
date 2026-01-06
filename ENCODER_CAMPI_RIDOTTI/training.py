import os
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

from dotenv import load_dotenv
from pathlib import Path
from huggingface_hub import login
from transformers import AutoTokenizer
from datasets import DatasetDict

import loop
from constants import (SEED,
                       NAN_VALUE,
                       BERT_ENCODER_CHECKPOINT,
                       XLM_ROBERTA_ENCODER_CHECKPOINT,
                       XLM_ROBERTA_LARGE_ENCODER_CHECKPOINT,
                       BIOBERT_ITALIAN_ENCODER)
from classifiers import ReportExtractor
from model_utils import create_label_to_id_map

from train_utils import (create_hugging_face_dataset,
                         get_device,
                         add_nan_flag_to_df,
                         create_list_of_annotated_reports,
                         get_normalization_stats,
                         add_target_columns_to_dataset,
                         model_parameters_info)


#################
# Reproducibility
#################
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


############
# Parameters
############
# Data parameters
TRAIN_FILE_NAME = "train_split_reduced.csv"
VALIDATION_FILE_NAME = "validation_split_reduced.csv"
TEST_FILE_NAME = "test_split_reduced.csv"
DATASET_TIPE = "Total reduced"
# Model parameters
CHECKPOINT = BIOBERT_ITALIAN_ENCODER
DROPOUT_RATE = 0.2
ADD_COMMON_LAYER = True
# Training parameters
N_EPOCHS = 40
BATCH_SIZE = 8
BATCH_SIZE_VALIDATION = 4
LEARNING_RATE_HEADS = 1e-5
LEARNING_RATE_ENCODER = 1e-9
ONLY_HEADS = True
EXCLUDE_LONG_REPORTS = False  # if False, truncates long reports
EMBEDDING_TYPE = "mean_pooling"  # "cls", "pooler", "mean_pooling"


#######
# Utils
#######
base_dir = Path(__file__).parent.parent
# Set plotting style
plt.style.use('ggplot')
# Set device
device = get_device()
print(f'{device = }')
# Set the API key for HuggingFace
load_dotenv()  # Load environment variables from .env file
hf_api_key = os.getenv("HF_TOKEN")
login(token=hf_api_key)


###########
# Load data
###########
file_names = {
    'train': TRAIN_FILE_NAME,
    'validation': VALIDATION_FILE_NAME,
}
paths = {split: base_dir / "data" / file_name
         for split, file_name in file_names.items()}
data = {split: pd.read_csv(path).fillna(NAN_VALUE) for split, path in paths.items()}
train_data, validation_data = data['train'], data['validation']
# Log
print(f"{len(train_data) = }")
print(f"{len(validation_data) = }")
# Create nan-flag columns
for split, df in data.items():
    add_nan_flag_to_df(df)
# Create lists of Annotated reports
annotated_reports =  {split: create_list_of_annotated_reports(data[split]) for split in file_names}


##########################
# Load model and tokenizer
##########################
model = ReportExtractor(checkpoint=CHECKPOINT,
                        dropout_rate=DROPOUT_RATE,
                        add_common_layer=ADD_COMMON_LAYER,
                        embedding_type=EMBEDDING_TYPE).to(device)
tokenizer = AutoTokenizer.from_pretrained(model.checkpoint)
# Check the maximum number of tokens for each report
if EXCLUDE_LONG_REPORTS:
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
    return tokenizer(examples['text'],
                     truncation=True,
                     padding="max_length",
                     max_length=model.encoder.config.max_position_embeddings)
dataset = dataset.map(tokenize_function, batched=True)
cols_to_remove = [col for col in ["token_type_ids", "text"] if col in dataset.column_names]
dataset = dataset.remove_columns(cols_to_remove)
dataset.set_format('torch')
# Log before adding columns
print(dataset)
# Add annotation fields to the dataset
#label_to_id_map = create_label_to_id_map(model.annotations_model)
normalization_stats = get_normalization_stats(annotated_reports['train'], model.regression_fields)
model.normalization_stats = normalization_stats
for split, reports in annotated_reports.items():
    dataset[split] = add_target_columns_to_dataset(dataset=dataset[split],
                                                   annotated_reports=reports,
                                                   label_to_id_map=model.label_to_id_map,
                                                   classification_columns=model.classification_fields,
                                                   binary_classification_columns=model.binary_classification_fields,
                                                   multiple_choice_columns=model.multiple_choice_fields,
                                                   regression_columns=model.regression_fields,
                                                   normalization_stats=normalization_stats)
# Log after adding columns
print(dataset)


##########
# Training
##########
# Set trainable parameters
if ONLY_HEADS:
    for param in model.encoder.parameters():
        param.requires_grad = False
else:
    for name, param in model.encoder.named_parameters():
        if any(f"layer.{i}." in name for i in [8,9,10,11]):
            param.requires_grad = True
        else:
            param.requires_grad = False     
    
    
            
# Visualize trainable parameters
model_parameters_info(model)

# Set logging dict for WandB
wandb_dict = {
    'entity': "luca-tramonti-PRIN",
    'project': "PRIN",
    'config':{
        "learning_rate_heads": LEARNING_RATE_HEADS,
        "learning_rate_encoder": LEARNING_RATE_ENCODER,
        "architecture": f'{CHECKPOINT} + extraction heads',
        "dataset": DATASET_TIPE,
        "epochs": N_EPOCHS,
        "dropout": DROPOUT_RATE,
        "train_batch_size": BATCH_SIZE,
        "validation_batch_size": BATCH_SIZE_VALIDATION,
        "add_common_layer": ADD_COMMON_LAYER,
        "train_only_heads": ONLY_HEADS,
        "exclude_long_reports": EXCLUDE_LONG_REPORTS,
        "embedding_type": EMBEDDING_TYPE
    }
}
# Start training
tracking = loop.train(
    model,
    dataset_train=dataset['train'],
    dataset_validation=dataset['validation'],
    epochs=N_EPOCHS,
    batch_size=BATCH_SIZE,
    lr_heads=LEARNING_RATE_HEADS,
    lr_encoder=LEARNING_RATE_ENCODER,
    verbose=1,
    batch_size_val=BATCH_SIZE_VALIDATION,
    #wandb_dict=wandb_dict
    wandb_dict=None
)


###########
# Plot loss
###########
df_1 = pd.DataFrame.from_records(tracking['train'])
df_2 = pd.DataFrame.from_records(tracking['validation'])
df_1['split'] = 'train'
df_2['split'] = 'validation'
df = pd.concat([df_1, df_2])
sns.lineplot(data=df, x=df.index, y='epoch', hue='split')
plt.xlabel('Epoch')
plt.ylabel('Loss')
if True:
    plt.show()


############
# Save model
############
#model.save_pretrained(base_dir / "saved_models" / "report_extractor")
#tokenizer.save_pretrained(base_dir / "saved_models" / "report_extractor")


########################################################################
# -------------------- Inference and save results -------------------- #
########################################################################
from torch.utils.data import DataLoader
import json

#############
# Preliminari
#############
base_dir = Path.cwd()
device = get_device()
print(f'{device = }')


############
# Load model
############
#model = ReportExtractor.from_pretrained(base_dir / "saved_models" / "report_extractor", device=device)
#model.to(device)
model.eval()
#tokenizer = AutoTokenizer.from_pretrained(base_dir / "saved_models" / "report_extractor")


###########
# Load data
###########
file_names = {
    'validation': VALIDATION_FILE_NAME,
    'test': TEST_FILE_NAME
}
paths = {split: base_dir / "data" / file_name
         for split, file_name in file_names.items()}
data = {split: pd.read_csv(path).fillna(NAN_VALUE) for split, path in paths.items()}
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
if EXCLUDE_LONG_REPORTS:
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
    return tokenizer(examples['text'],
                     truncation=True,
                     padding="max_length",
                     max_length=model.encoder.config.max_position_embeddings)
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
save_path = base_dir / "ENCODER_CAMPI_RIDOTTI" / "results.json"
with open(save_path, "w") as f:
    json.dump(results, f, indent=4)
print(f"Results saved to {save_path}")
