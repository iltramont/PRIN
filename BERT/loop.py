import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW

from datasets import Dataset

from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, f1_score, precision_score, recall_score


def get_loss(field, prediction, target, regression_fields, classification_fields, multiple_choice_fields, optional_regression_fields):
    """Seleziona la loss corretta in base al tipo di campo."""
    if field in regression_fields:
        return torch.nn.functional.mse_loss(prediction.squeeze(-1), target.float())
    elif field in optional_regression_fields:
        return torch.nn.functional.binary_cross_entropy_with_logits(prediction, target.float())        
    elif field in classification_fields:
        return torch.nn.functional.cross_entropy(prediction, target.long())
    elif field in multiple_choice_fields:
        return torch.nn.functional.binary_cross_entropy_with_logits(prediction, target.float())
    return None


def evaluate(model, dataset: Dataset, batch_size: int, verbose: int = 1):
    """Evaluation loop: calcola la loss media e altre metriche"""
    dataloader_eval = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = next(model.parameters()).device
    model.eval()

    total_loss = 0.0
    # Contatori per metriche
    classification_preds, classification_targets = ({f: [] for f in model.classification_fields},
                                                    {f: [] for f in model.classification_fields})
    regression_preds, regression_targets = ({f: [] for f in model.regression_fields},
                                            {f: [] for f in model.regression_fields})
    optional_regression_preds, optional_regression_targets = ({f: [] for f in model.optional_regression_fields},
                                                              {f: [] for f in model.optional_regression_fields})
    multilabel_preds, multilabel_targets = ({f: [] for f in model.multiple_choice_fields},
                                            {f: [] for f in model.multiple_choice_fields})

    with torch.no_grad():
        for batch in dataloader_eval:
            input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            losses = []
            for field, prediction in outputs.items():
                target = batch[field].to(device)
                # Loss
                field_loss = get_loss(
                    field, prediction, target,
                    model.regression_fields,
                    model.classification_fields,
                    model.multiple_choice_fields,
                    model.optional_regression_fields
                )
                if field_loss is not None:
                    losses.append(field_loss)

                # Regressione
                if field in model.regression_fields:
                    regression_preds[field].extend(prediction.squeeze(-1).cpu().numpy())
                    regression_targets[field].extend(target.cpu().numpy())
                    
                # Regressione opzionale
                elif field in model.optional_regression_fields:
                    preds = (torch.sigmoid(prediction) > 0.5).int()
                    optional_regression_preds[field].extend(preds.cpu().numpy())
                    optional_regression_targets[field].extend(target.cpu().numpy())

                # Classificazione
                elif field in model.classification_fields:
                    preds = prediction.argmax(dim=-1)
                    classification_preds[field].extend(preds.cpu().numpy())
                    classification_targets[field].extend(target.cpu().numpy())

                # Multilabel / multiple choice
                elif field in model.multiple_choice_fields:
                    preds = (torch.sigmoid(prediction) > 0.5).int()
                    multilabel_preds[field].extend(preds.cpu().numpy())
                    multilabel_targets[field].extend(target.cpu().numpy())      
            
            loss = sum(losses) / len(losses)
            total_loss += loss.item()
                
        # Loss
        print(f"Validation loss: {total_loss / len(dataloader_eval):.4f}")

        # Regressione
        for field in regression_preds:
            target = regression_targets[field]
            preds = regression_preds[field]
            mae = mean_absolute_error(target, preds)
            r2 = r2_score(target, preds)
            if verbose > 1:
                print(f"Regression {field}: MAE={mae:.4f}, R²={r2:.4f}")
        
        # Regressione opzionale
        for field in optional_regression_preds:
            acc = accuracy_score(optional_regression_targets[field], optional_regression_preds[field])
            if verbose > 1:
                print(f"Optional Regression {field}: Accuracy={acc:.2%}")

        # Classificazione
        for field in classification_preds:
            acc = accuracy_score(classification_targets[field], classification_preds[field])
            if verbose > 1:
                print(f"Accuracy {field}: {acc:.2%}")

        # Multilabel / multiple choice
        for field in multilabel_preds:
            f1 = f1_score(multilabel_targets[field], multilabel_preds[field], average="micro", zero_division=0)
            precision = precision_score(multilabel_targets[field], multilabel_preds[field], average="micro", zero_division=0)
            recall = recall_score(multilabel_targets[field], multilabel_preds[field], average="micro", zero_division=0)
            if verbose > 1:
                print(f"Multilabel {field}: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    return total_loss / len(dataloader_eval)


def train(model, dataset_train: Dataset, epochs: int, batch_size: int, lr: float = 2e-5,
          dataset_validation: Dataset| None =None, batch_size_val: int = 2, verbose: int = 1):
    """
    Training loop generico per ReportExtractor.
    - model: istanza del modello
    - dataset: oggetto Dataset che restituisce (input_ids, attention_mask)
    - epochs: numero di epoche
    - batch_size: dimensione batch
    - lr: learning rate
    """
    device = next(model.parameters()).device
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    #optimizer = Adam(model.parameters(), lr=lr)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    loss_lists = {'train': [] , 'validation': []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(dataloader_train):
            input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Forward
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Calcola loss per ogni campo
            losses = []
            for field, prediction in outputs.items():
                target = batch[field].to(device)
                field_loss = get_loss(
                    field, prediction, target,
                    model.regression_fields,
                    model.classification_fields,
                    model.multiple_choice_fields,
                    model.optional_regression_fields
                )
                if field_loss is not None:
                    losses.append(field_loss)
            loss = sum(losses) / len(losses)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
        
        loss_lists['train'].append(total_loss/len(dataloader_train))
        print(f"Epoch {epoch+1}/{epochs} - Training loss: {total_loss/len(dataloader_train):.4f}")
        if dataset_validation is not None:
            eval_loss = evaluate(model, dataset_validation, batch_size_val, verbose)
            loss_lists['validation'].append(eval_loss)
            
    return loss_lists
