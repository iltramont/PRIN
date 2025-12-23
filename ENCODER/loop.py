import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
import torch.nn.functional as F
import wandb


from datasets import Dataset

from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, f1_score, precision_score, recall_score

from constants import SEED


def get_loss(
    field: str,
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    regression_fields: list[str],
    classification_fields: list[str],
    multiple_choice_fields: list[str],
    binary_classification_fields: list[str],
    loss_dict: dict[str, list[float]],
    alpha_nan: float = 1.0,
) -> None | torch.Tensor:
    """
    Loss coerente con il concetto:
    - {field}_is_nan == True → informazione NON estraibile dal testo
    - la loss del valore/classe è calcolata SOLO se estraibile
    """
    pred = outputs[field]
    device = pred.device
    # ===========
    # REGRESSIONE
    # ===========
    if field in regression_fields:
        target = batch[field].float().to(device)
        pred_value = pred.squeeze(-1)
        nan_field = f"{field}_is_nan"
        if nan_field in outputs:
            t_nan = batch[nan_field].float().to(device)
            pred_nan = outputs[nan_field].squeeze(-1)
            # Loss estraibilità
            loss_nan = alpha_nan * F.binary_cross_entropy_with_logits(pred_nan, t_nan)
            loss_dict[nan_field].append(loss_nan.item())
            # Loss valore SOLO se estraibile
            mask = (t_nan == 0)
            if mask.any():
                # calcola MSE loss solo se presente almeno un valore non mascherato
                loss_value = F.mse_loss(pred_value[mask], target[mask])
                loss_dict[field].append(loss_value.item())
                return loss_nan + loss_value
            else:
                return loss_nan
        else:
            # Se il campo non può essere NAN, calcola direttamente MSE loss
            return F.mse_loss(pred_value, target)
    # ===========================
    # CLASSIFICAZIONE MULTICLASSE
    # ===========================
    elif field in classification_fields:
        target = batch[field].long().to(device)
        loss = F.cross_entropy(pred, target)
        loss_dict[field].append(loss.item())
        return loss
    # =======================
    # CLASSIFICAZIONE BINARIA
    # =======================
    elif field in binary_classification_fields:
        target = batch[field].float().to(device)
        loss = F.binary_cross_entropy_with_logits(pred.squeeze(-1), target)
        loss_dict[field].append(loss.item())
        return loss
    # =============================
    # MULTI-LABEL / MULTIPLE CHOICE
    # =============================
    elif field in multiple_choice_fields:
        target = batch[field].float().to(device)
        loss = F.binary_cross_entropy_with_logits(pred, target)
        loss_dict[field].append(loss.item())
        return loss
    # ==================
    # Campo non presente
    # ==================
    else:
        print('LOSS NULLA!!!!')
        return None
    

def evaluate(model, dataset: Dataset, batch_size: int, verbose: int = 1):
    """Evaluation loop: calcola la loss media e altre metriche"""
    dataloader_eval = DataLoader(dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator().manual_seed(SEED))
    device = next(model.parameters()).device
    model.eval()

    epoch_loss = 0.0
    with torch.no_grad():
        count_batch = 0
        batch_loss_dict = {field: [] for field in model.heads.keys()}
        for batch in dataloader_eval:
            count_batch += 1
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            # Inference
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Batch loss
            # Calcola loss per ogni campo
            batch_losses: list[torch.Tensor] = []

            for field in outputs:
                # Ignora eventuale chiave "_preds" se presente
                if field.startswith("_"):
                    continue
                if field.endswith('_is_nan'):
                    continue
                field_loss = get_loss(
                    field=field,
                    outputs=outputs,
                    batch=batch,
                    regression_fields=model.regression_fields,
                    classification_fields=model.classification_fields,
                    multiple_choice_fields=model.multiple_choice_fields,
                    binary_classification_fields=model.binary_classification_fields,
                    loss_dict=batch_loss_dict,
                    alpha_nan=1.0
                )
                if field_loss is not None:
                    batch_losses.append(field_loss)

            batch_loss = sum(batch_losses) / len(batch_losses)  # media delle loss per ogni campo del batch
            # Aumenta epoch loss con batch loss
            epoch_loss += batch_loss.item()

        epoch_loss_dict = dict()
        for f, l in batch_loss_dict.items():
            if len(l) > 0:
                epoch_loss_dict[f] = sum(l) / len(l)
            else:
                epoch_loss_dict[f] = 0.0
        epoch_loss_dict['epoch'] = epoch_loss / count_batch  # media delle loss dei batch
        print(f"Validation loss: {epoch_loss_dict['epoch']:.4f}")
    return epoch_loss_dict


def train(model,
          dataset_train: Dataset,
          epochs: int,
          batch_size: int,
          lr: float = 2e-5,
          dataset_validation: Dataset | None = None,
          batch_size_val: int = 2,
          verbose: int = 1,
          wandb_dict: dict | None = None,
          ) -> dict[str, list[dict[str, float]]]:
    """
    Training loop generico per ReportExtractor.
    - model: istanza del modello
    - dataset: oggetto Dataset che restituisce (input_ids, attention_mask)
    - epochs: numero di epoche
    - batch_size: dimensione batch
    - lr: learning rate
    """

    if wandb_dict is not None:
        wandb.login()
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity=wandb_dict['entity'],
            # Set the wandb project where this run will be logged.
            project=wandb_dict['project'],
            # Track hyperparameters and run metadata.
            config=wandb_dict['config']
        )
    else:
        run = None

    device = next(model.parameters()).device
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    #optimizer = Adam(model.parameters(), lr=lr)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    loss_lists = {'train': [] , 'validation': []}
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        count_batch = 0
        batch_loss_dict = {field: [] for field in model.heads.keys()}
        for batch in tqdm(dataloader_train):
            count_batch += 1
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Inference
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Batch loss
            # Calcola loss per ogni campo
            batch_losses: list[torch.Tensor] = []
            for field in outputs:
                # Ignora eventuale chiave "_preds" se presente
                if field.startswith("_"):
                    continue
                if field.endswith('_is_nan'):
                    continue
                field_loss = get_loss(
                    field=field,
                    outputs=outputs,
                    batch=batch,
                    regression_fields=model.regression_fields,
                    classification_fields=model.classification_fields,
                    multiple_choice_fields=model.multiple_choice_fields,
                    binary_classification_fields=model.binary_classification_fields,
                    loss_dict=batch_loss_dict,
                    alpha_nan=1.0
                )
                if field_loss is not None:
                    batch_losses.append(field_loss)
            batch_loss = sum(batch_losses) / len(batch_losses)

            # Backpropagation
            optimizer.zero_grad()
            batch_loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Aumenta epoch loss con batch loss
            epoch_loss += batch_loss.item()

        epoch_loss_dict = dict()
        for f, l in batch_loss_dict.items():
            if len(l) > 0:
                epoch_loss_dict[f] = sum(l) / len(l)
            else:
                epoch_loss_dict[f] = 0.0
        epoch_loss_dict['epoch'] = epoch_loss / count_batch

        if wandb_dict is not None:
            log_dict = {f"train_loss_{f}": l for f, l in epoch_loss_dict.items() if f != 'epoch'}
            log_dict['train_loss'] = epoch_loss_dict['epoch']
        loss_lists['train'].append(epoch_loss_dict)
        print(f"Epoch {epoch+1}/{epochs} - Training loss: {epoch_loss_dict['epoch']:.4f}")
        if dataset_validation is not None:
            eval_loss_dict = evaluate(model, dataset_validation, batch_size_val, verbose)
            if wandb_dict is not None:
                for f, l in eval_loss_dict.items():
                    if f != 'epoch':
                        log_dict[f"eval_loss_{f}"] = l
                log_dict['eval_loss'] = eval_loss_dict['epoch']
            loss_lists['validation'].append(eval_loss_dict)
        if wandb_dict is not None:
            run.log(log_dict)
    if wandb_dict is not None:
        run.finish()
    return loss_lists
