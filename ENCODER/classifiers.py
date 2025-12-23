import os
import torch
import torch.nn as nn
from transformers import AutoModel
from model_utils import get_number_of_classes
from model_utils import (get_regression_fields,
                        get_multiple_choice_fields,
                        get_classification_fields,
                        get_optional_regression_fields,
                        get_binary_classification_fields
                    )
from pydantic import BaseModel
from constants import Annotations, BERT_ENCODER_CHECKPOINT, XLM_ROBERTA_ENCODER_CHECKPOINT


class ReportExtractor(nn.Module):
    def __init__(self, checkpoint=BERT_ENCODER_CHECKPOINT,
                 annotations_model: type[BaseModel] = Annotations,
                 use_pooler_output: bool = False,
                 add_common_layer: bool = True,
                 dropout_rate: float = 0.2):
        super().__init__()
        # Model attributes
        self.checkpoint = checkpoint
        self.annotations_model = annotations_model
        self.use_pooler_output = use_pooler_output
        self.add_common_layer = add_common_layer
        self.dropout_rate = dropout_rate
        self.regression_fields = get_regression_fields(annotations_model)
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
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer to reduce overfitting

        # Heads
        self.heads = nn.ModuleDict()
        # Regression (apply ReLU to ensure non-negative outputs)
        for field in self.regression_fields:
            self.heads[field] = nn.Linear(hidden, 1)
        # Binary classification (apply sigmoid for inference)
        for field in self.binary_classification_fields:
            self.heads[field] = nn.Linear(hidden, 1)
        # Multi-class classification (apply softmax for inference)
        for field in self.classification_fields:
            n_classes = self.num_classes[field]
            self.heads[field] = nn.Linear(hidden, n_classes)
        # (apply sigmoid for inference)
        for field in self.multiple_choice_fields:
            n_classes = self.num_classes[field]
            self.heads[field] = nn.Linear(hidden, n_classes) 

    def forward(self, input_ids, attention_mask) -> dict[str, torch.Tensor]:
        """
        embedding -> Dropout -> Heads
        """
        # Get embeddings from encoder
        embeddings = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use pooler output or [CLS] token
        if self.use_pooler_output and hasattr(embeddings, "pooler_output"):
            embeddings = embeddings.pooler_output
        else:
            embeddings = embeddings.last_hidden_state[:, 0, :]
        # Common layer
        if self.add_common_layer:
            embeddings = self.common_layer(embeddings)
        # Apply dropout
        embeddings = self.dropout(embeddings)
        # Apply each head to get predictions for each field
        outputs = dict()
        for field, head in self.heads.items():
            out = head(embeddings)
            #if field in self.regression_fields:
            #    out = nn.functional.relu(out)
            outputs[field] = out
        # outputs["_preds"] = outputs
        return outputs

    def save_pretrained(self, save_directory: str):
        """Salva encoder HuggingFace + heads custom"""
        os.makedirs(save_directory, exist_ok=True)
        # Salva encoder HuggingFace
        self.encoder.save_pretrained(save_directory)
        # Salva configurazione custom
        torch.save({
            "checkpoint": self.checkpoint,
            "annotations_model": self.annotations_model.__name__,
            "use_pooler_output": self.use_pooler_output,
            "add_common_layer": self.add_common_layer,
            "dropout_rate": self.dropout_rate,
            "state_dict": self.state_dict()
        }, os.path.join(save_directory, "report_extractor_trained.pt"))

    @classmethod
    def from_pretrained(cls, load_directory: str, annotations_model: type[BaseModel] = Annotations, device="cpu"):
        """Ricarica encoder HuggingFace + heads custom"""
        # Carica encoder HuggingFace
        encoder = AutoModel.from_pretrained(load_directory)
        # Carica configurazione custom
        checkpoint = torch.load(os.path.join(load_directory, "report_extractor_trained.pt"), map_location=device)
        model = cls(checkpoint=checkpoint["checkpoint"],
                    annotations_model=annotations_model,
                    use_pooler_output=checkpoint["use_pooler_output"],
                    add_common_layer=checkpoint["add_common_layer"],
                    dropout_rate=checkpoint["dropout_rate"])
        # Sostituisci encoder con quello caricato
        model.encoder = encoder
        # Carica pesi
        model.load_state_dict(checkpoint["state_dict"])
        return model