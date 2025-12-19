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
        self.encoder = AutoModel.from_pretrained(checkpoint)  # Encoder base model (like BERT)
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