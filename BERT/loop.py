import sys
import os
from dotenv import load_dotenv
from pathlib import Path

# if notebook is in PRIN/notebooks, parent() is PRIN
#project_root = Path.cwd().resolve().parent
#sys.path.insert(0, str(project_root))
#print("Added to sys.path:", project_root)

import json
from constants import Annotations, AnnotatedReport
import time
from IPython.display import clear_output

from huggingface_hub import login

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam


from transformers import AutoTokenizer, DefaultDataCollator
from datasets import load_dataset, Dataset, DatasetDict


from classifiers import SimpleExtractor
from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, f1_score, precision_score, recall_score
import numpy as np



def get_loss(field, prediction, target, regression_fields, classification_fields, multiple_choice_fields):
    """Seleziona la loss corretta in base al tipo di campo."""
    if field in regression_fields:
        return torch.nn.functional.mse_loss(prediction.squeeze(-1), target.float())
    elif field in classification_fields:
        return torch.nn.functional.cross_entropy(prediction, target.long())
    elif field in multiple_choice_fields:
        return torch.nn.functional.binary_cross_entropy_with_logits(prediction, target.float())
    return None


def evaluate(model, dataset, batch_size=16):
    """Evaluation loop: calcola la loss media e accuracy per classificazione."""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = model.device
    model.eval()

    total_loss = 0.0
    # Contatori per metriche
    classification_preds, classification_targets = ({f: [] for f in model.regression_fields},
                                                    {f: [] for f in model.regression_fields})
    regression_preds, regression_targets = ({f: [] for f in model.regression_fields},
                                            {f: [] for f in model.regression_fields})
    multilabel_preds, multilabel_targets = ({f: [] for f in model.multiple_choice_fields},
                                            {f: [] for f in model.multiple_choice_fields})

    with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask, labels_dict = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                for field, prediction in outputs.items():
                    if field not in labels_dict:
                        continue
                    target = labels_dict[field].to(device)

                    # Loss
                    field_loss = get_loss(
                        field, prediction, target,
                        model.regression_fields,
                        model.classification_fields,
                        model.multiple_choice_fields
                    )
                    total_loss += field_loss.item()

                    # Regressione
                    if field in model.regression_fields:
                        regression_preds[field].extend(prediction.squeeze(-1).cpu().numpy())
                        regression_targets[field].extend(target.cpu().numpy())

                    # Classificazione
                    elif field in model.classification_fields:
                        preds = prediction.argmax(dim=-1)
                        classification_preds[field].extend(preds.cpu().numpy())
                        classification_targets[field].extend(target.cpu().numpy())

                    # Multilabel / multiple choice
                    elif field in model.multiple_choice_fields:
                        preds = (prediction > 0).int()
                        multilabel_preds[field].extend(preds.cpu().numpy())
                        multilabel_targets[field].extend(target.cpu().numpy())

    # Regressione
    for field in regression_preds:
        if regression_preds[field]:
            mae = mean_absolute_error(regression_targets[field], regression_preds[field])
            r2 = r2_score(regression_targets[field], regression_preds[field])
            print(f"Regression {field}: MAE={mae:.4f}, R²={r2:.4f}")

    # Classificazione
    for field in classification_preds:
        if classification_preds[field] > 0:
            acc = accuracy_score(classification_targets[field], classification_preds[field])
            print(f"Accuracy {field}: {acc:.2%}")

    # Multilabel / multiple choice
    for field in multilabel_preds:
        if multilabel_preds[field]:
            f1 = f1_score(multilabel_targets[field], multilabel_preds[field], average="micro")
            precision = precision_score(multilabel_targets[field], multilabel_preds[field], average="micro")
            recall = recall_score(multilabel_targets[field], multilabel_preds[field], average="micro")
            print(f"Multilabel {field}: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")


def train(model, dataset, epochs=3, batch_size=16, lr=2e-5):
    """
    Training loop generico per ReportExtractor.
    - model: istanza del modello
    - dataset: oggetto Dataset che restituisce (input_ids, attention_mask, labels_dict)
    - epochs: numero di epoche
    - batch_size: dimensione batch
    - lr: learning rate
    """
    device = model.device
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(dataloader):
            input_ids, attention_mask, labels_dict = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Forward
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Calcola loss per ogni campo
            loss = 0.0
            for field, prediction in outputs.items():
                if field not in labels_dict:
                    continue
                target = labels_dict[field].to(device)
                field_loss = get_loss(
                    field, prediction, target,
                    model.regression_fields,
                    model.classification_fields,
                    model.multiple_choice_fields
                )
                loss += field_loss

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Training loss: {total_loss/len(dataloader):.4f}")
