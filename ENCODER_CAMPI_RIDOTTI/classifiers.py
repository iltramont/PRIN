import os

import numpy
import torch
import torch.nn as nn
from transformers import AutoModel
from model_utils import get_number_of_classes
from model_utils import (get_regression_fields,
                        get_multiple_choice_fields,
                        get_classification_fields,
                        get_optional_regression_fields,
                        get_binary_classification_fields,
                        create_label_to_id_map
                    )
from pydantic import BaseModel
from constants import Annotations, BERT_ENCODER_CHECKPOINT, XLM_ROBERTA_ENCODER_CHECKPOINT


class ReportExtractor(nn.Module):
    def __init__(self, checkpoint=BERT_ENCODER_CHECKPOINT,
                 annotations_model: type[BaseModel] = Annotations,
                 embedding_type: str = "mean_pooling",  # "cls", "pooler", "mean_pooling"
                 add_common_layer: bool = True,
                 dropout_rate: float = 0.2,
                 normalization_stats: None | dict[str, tuple[float]] = None):
        assert embedding_type in {"cls", "pooler", "mean_pooling"}
        super().__init__()
        # Model attributes
        self.checkpoint = checkpoint
        self.annotations_model = annotations_model
        self.embedding_type = embedding_type
        self.add_common_layer = add_common_layer
        self.dropout_rate = dropout_rate
        self.regression_fields = get_regression_fields(annotations_model)
        self.normalization_stats = normalization_stats
        self.label_to_id_map = create_label_to_id_map(annotations_model)
        self.multiple_choice_fields = get_multiple_choice_fields(annotations_model)
        self.binary_classification_fields = get_binary_classification_fields(annotations_model)
        self.optional_regression_fields = get_optional_regression_fields(annotations_model)
        self.classification_fields = get_classification_fields(annotations_model)
        self.num_classes = get_number_of_classes(annotations_model)
        # Layers
        self.encoder = AutoModel.from_pretrained(checkpoint)  # Encoder base model (like ENCODER)
        # Get embedding length
        hidden = self.encoder.config.hidden_size
        if add_common_layer:
            self.common_layer = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.SiLU()
            )
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)  
        # Task adapters
        self.task_adapters = nn.ModuleDict({
            "regression": nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(0.2)),
            "classification": nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(0.2)),
            "binary": nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(0.2)),
            "multilabel": nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(0.2))
        })
        # Heads
        self.heads = nn.ModuleDict()
        # Regression
        for field in self.regression_fields:
            self.heads[field] = nn.Linear(hidden, 1)
        # Binary classification
        for field in self.binary_classification_fields:
            self.heads[field] = nn.Linear(hidden, 1)
        # Multi-class classification
        for field in self.classification_fields:
            n_classes = self.num_classes[field]
            self.heads[field] = nn.Linear(hidden, n_classes)
        # Multi-label classification (multiple choice)
        for field in self.multiple_choice_fields:
            n_classes = self.num_classes[field]
            self.heads[field] = nn.Linear(hidden, n_classes) 

    def forward(self, input_ids, attention_mask) -> dict[str, torch.Tensor]:
        """
        embedding -> Common layer -> dropout ->task adapters -> Heads
        """
        # Get embeddings from encoder
        embeddings = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use pooler output or [CLS] token
        if self.embedding_type == "pooler" and hasattr(embeddings, "pooler_output"):
            embeddings = embeddings.pooler_output
        elif self.embedding_type == "cls":
            embeddings = embeddings.last_hidden_state[:, 0, :]
        elif self.embedding_type == "mean_pooling":
            last_hidden = embeddings.last_hidden_state
            mask = attention_mask.unsqueeze(-1)
            den = mask.sum(1).clamp(min=1e-6)
            embeddings = (last_hidden * mask).sum(1) / den  
        # Common layer
        if self.add_common_layer:
            embeddings = self.common_layer(embeddings)
        # Apply dropout
        embeddings = self.dropout(embeddings)
        # Task adapters
        adapter_regression = self.task_adapters["regression"](embeddings)
        adapter_classification = self.task_adapters["classification"](embeddings)
        adapter_binary = self.task_adapters["binary"](embeddings)
        adapter_multilabel = self.task_adapters["multilabel"](embeddings)
        # Apply each head to get predictions for each field
        outputs = dict()
        for field, head in self.heads.items():
            if field in self.regression_fields:
                outputs[field] = head(adapter_regression)
            elif field in self.binary_classification_fields:
                outputs[field] = head(adapter_binary)
            elif field in self.multiple_choice_fields:
                outputs[field] = head(adapter_multilabel)
            elif field in self.classification_fields:
                outputs[field] = head(adapter_classification)
        return outputs


    def all_fields(self) -> list[str]:
        return list(self.heads.keys())


    def rework_output(self, model_output: dict[str, torch.Tensor]) -> dict[str, numpy.ndarray]:
        with torch.no_grad():
            result = dict()
            for field in self.regression_fields:
                tensor = model_output[field]
                if self.normalization_stats is not None:
                    mu, std = self.normalization_stats[field]
                    tensor = (tensor * std) + mu
                tensor = torch.nn.functional.relu(tensor)
                result[field] = tensor.reshape(-1).cpu().numpy()
            for field in self.binary_classification_fields:
                tensor = torch.nn.functional.sigmoid(model_output[field])
                result[field] = tensor.reshape(-1).cpu().numpy()
            for field in self.multiple_choice_fields:
                tensor = torch.nn.functional.sigmoid(model_output[field])
                result[field] = tensor.cpu().numpy()
            for field in self.classification_fields:
                tensor = torch.nn.functional.softmax(model_output[field], dim=-1)
                result[field] = tensor.cpu().numpy()
        return result


    def save_pretrained(self, save_directory: str):
        """Salva encoder HuggingFace + heads custom"""
        os.makedirs(save_directory, exist_ok=True)
        # Salva encoder HuggingFace
        self.encoder.save_pretrained(save_directory)
        # Salva configurazione custom
        torch.save({
            "checkpoint": self.checkpoint,
            "annotations_model": self.annotations_model.__name__,
            "embedding_type": self.embedding_type,
            "add_common_layer": self.add_common_layer,
            "dropout_rate": self.dropout_rate,
            "normalization_stats": self.normalization_stats,
            "label_to_id_map": self.label_to_id_map,
            "state_dict": self.state_dict()
        }, os.path.join(save_directory, "report_extractor_trained.pt"))

    @classmethod
    def from_pretrained(cls, load_directory: str, annotations_model: type[BaseModel] = Annotations, device="cpu"):
        """Ricarica encoder HuggingFace + heads custom"""
        # Carica encoder HuggingFace
        encoder = AutoModel.from_pretrained(load_directory)
        # Carica configurazione custom
        checkpoint = torch.load(os.path.join(load_directory, "report_extractor_trained.pt"),
                                map_location=device,
                                weights_only=False)
        model = cls(checkpoint=checkpoint["checkpoint"],
                    annotations_model=annotations_model,
                    embedding_type=checkpoint["embedding_type"],
                    add_common_layer=checkpoint["add_common_layer"],
                    dropout_rate=checkpoint["dropout_rate"],
                    normalization_stats=checkpoint["normalization_stats"])
        # Sostituisci encoder con quello caricato
        model.encoder = encoder
        # Carica pesi
        model.load_state_dict(checkpoint["state_dict"])
        return model