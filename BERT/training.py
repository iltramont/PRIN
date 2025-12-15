import os
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

from ast import literal_eval
from dotenv import load_dotenv
from pathlib import Path
from huggingface_hub import login
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict

import loop
from constants import AnnotatedReport, Annotations
from classifiers import ReportExtractor
from BERT_utils import create_label_to_id_map, labels_to_bits, NAN_VALUE, get_multiple_choice_fields,  get_optional_regression_fields
from BERT_utils import SEED

# Ripetibilità
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Parameters
TRAIN_FILE_NAME = "train_split.csv"
VALIDATION_FILE_NAME = "validation_split.csv"
# Model parameters
CHECKPOINT = "bert-base-multilingual-cased"
DROPOUT_RATE = 0.2
# Training parameters
N_EPOCHS = 50
BATCH_SIZE = 4
BATCH_SIZE_VALIDATION = 4
LEARNING_RATE = 1e-5
ADD_COMMON_LAYER = False
ONLY_HEADS = True


# Set base directory
base_dir = Path(__file__).parent.parent

# Set plotting style
plt.style.use('ggplot')

# Set device
print(f'{torch.cuda.is_available() = }')  # True se la GPU è disponibile
print(f'{torch.cuda.device_count() = }')  # Numero di GPU disponibili
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'{device = }')

# Set the API key for HuggingFace
load_dotenv()  # Load environment variables from .env file
hf_api_key = os.getenv("HF_TOKEN")
login(token=hf_api_key)



# Functions
def create_hugging_face_dataset(annotated_reports: list[AnnotatedReport]) -> Dataset:
    text = [report.report_text for report in annotated_reports]
    return Dataset.from_dict({'text': text})


# Load data
file_names = {
    'train': TRAIN_FILE_NAME,
    'validation': VALIDATION_FILE_NAME,
}
paths = {split: base_dir / "data" / file_name
         for split, file_name in file_names.items()}

data = dict()
for split, path in paths.items():
    with open(path, "r", encoding="utf-8") as f:
        data[split] = pd.read_csv(f)

train_data, validation_data = data['train'], data['validation']

print(f"{len(train_data) = }")
print(f"{len(validation_data) = }")

# Creiamo colonne per flag quando i campi numerici sono mancanti
for split, df in data.items():
    for col in get_optional_regression_fields(Annotations):
        new_name = f'{col}_is_nan'
        df[new_name] = df[col].isna()

annotated_reports: dict[str, list[AnnotatedReport]] = {split: [] for split in file_names.keys()}
mc_fields = get_multiple_choice_fields(Annotations)
for split in data:
    df = data[split].fillna(NAN_VALUE)
    for _, row in df.iterrows():
        annotations_dict = dict()
        for field in Annotations.model_fields.keys():
            v = row[field]
            if v == NAN_VALUE:
                v = None
            if field in mc_fields:
                v = literal_eval(v)
            annotations_dict[field] = v
        annotated_reports[split].append(AnnotatedReport(report_text=row['report_text'], report_data=annotations_dict))


# Load model and tokenizer
model = ReportExtractor(dropout_rate=DROPOUT_RATE, add_common_layer=ADD_COMMON_LAYER).to(device)
tokenizer = AutoTokenizer.from_pretrained(model.checkpoint)

# Check the maximum number of tokens for each report
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

# Create Hugging Face Datasets
dataset = DatasetDict({
    split: create_hugging_face_dataset(annotated_reports[split])
    for split in annotated_reports
})

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="longest", max_length=model.encoder.config.max_position_embeddings)

dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.remove_columns(["token_type_ids", "text"])
dataset.set_format('torch')
print(dataset)

# Add annotation fields to the dataset
label_to_id_map = create_label_to_id_map(model.annotations_model)

# Classification fields
for f in model.classification_fields:
    for split in annotated_reports:
        target: list[int] = []
        for r in annotated_reports[split]:
            label = getattr(r.report_data, f)
            if label is None:
                label = NAN_VALUE
            id = label_to_id_map[f]['label_to_id'][label]
            target.append(id)
        dataset[split] = dataset[split].add_column(f, target)
# Binary classification fields
for f in model.binary_classification_fields:
    for split in ('train', 'validation'):
        target: list[int] = []
        for r in annotated_reports[split]:
            label = getattr(r.report_data, f)
            id = label_to_id_map[f]['label_to_id'][label]
            target.append(id)
        dataset[split] = dataset[split].add_column(f, target)
# Regression fields
# Mean and std for normalization
regression_stats = {}
for f in model.regression_fields:
    values = []
    for r in annotated_reports['train']:
        v = getattr(r.report_data, f)
        if v is not None:
            values.append(float(v))
    mu = np.mean(values)
    std = np.std(values)
    regression_stats[f] = (mu, std)
for f in model.regression_fields:
    for split in ('train', 'validation'):
        target: list[float] = []
        for r in annotated_reports[split]:
            value = getattr(r.report_data, f)
            if value is None:
                value = 0
            mu = regression_stats[f][0]
            std = regression_stats[f][1]
            value = (float(value) - mu) / std
            target.append(value)
        dataset[split] = dataset[split].add_column(f, target)
# Multiple choice fields
for f in model.multiple_choice_fields:
    for split in ('train', 'validation'):
        target: list[list[int]] = []
        for r in annotated_reports[split]:
            values = getattr(r.report_data, f)
            bits = labels_to_bits(values, label_to_id_map[f]['label_to_id'])
            target.append(bits)
        dataset[split] = dataset[split].add_column(f, target)       
        
print(dataset)

# Training
# Set trainable parameters
if ONLY_HEADS:
    for param in model.encoder.parameters():
        param.requires_grad = False
# Total parameters
total_params = sum(p.numel() for p in model.parameters())

# Parametri allenabili (quelli con gradiente)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Print parameters info
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
out_feat = 0
for head in model.heads.values():
    out_feat += head.out_features
print(f'Extraction heads parameters: {model.encoder.config.hidden_size * out_feat + out_feat}')

wandb_dict = {
    'entity': "luca-tramonti-PRIN",
    'project': "PRIN",
    'config':{
        "learning_rate": LEARNING_RATE,
        "architecture": f'{CHECKPOINT} + extraction heads',
        "dataset": "Only Guido reports",
        "epochs": N_EPOCHS,
        "dropout": DROPOUT_RATE,
        "train_batch_size": BATCH_SIZE,
        "validation_batch_size": BATCH_SIZE_VALIDATION,
        "add_common_layer": ADD_COMMON_LAYER,
        "train_only_heads": ONLY_HEADS,
    }
}

# Start training
tracking = loop.train(
    model,
    dataset_train=dataset['train'],
    dataset_validation=dataset['validation'],
    epochs=N_EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    verbose=1,
    batch_size_val=BATCH_SIZE_VALIDATION,
    wandb_dict=wandb_dict
    #wandb_dict=None
)

df_1 = pd.DataFrame.from_records(tracking['train'])
df_2 = pd.DataFrame.from_records(tracking['validation'])
df_1['split'] = 'train'
df_2['split'] = 'validation'
df = pd.concat([df_1, df_2])
sns.lineplot(data=df, x=df.index, y='epoch', hue='split')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.show()