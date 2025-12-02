import torch
import torch.nn as nn
import sys
from pathlib import Path
from transformers import AutoModel
from BERT_utils import get_number_of_classes
from BERT_utils import get_regression_fields, get_multiple_choice_fields, get_classification_fields
from pydantic import BaseModel
from constants import Annotations


class SimpleExtractor(nn.Module):
    def __init__(self, bert_checkpoint="bert-base-multilingual-cased", num_classes_morfologia=4, num_classes_tessuto_adiposo=5):
        super().__init__()
        self.tokenizer_checkpoint = bert_checkpoint
        self.bert = AutoModel.from_pretrained(bert_checkpoint)
        hidden = self.bert.config.hidden_size  # Embedding length
        self.dropout = nn.Dropout(0.2)

        # Classification heads
        self.class_morfologia = nn.Linear(hidden, num_classes_morfologia)
        self.class_tessuto_adiposo = nn.Linear(hidden, num_classes_tessuto_adiposo)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Better than pooler_output
        cls = out.last_hidden_state[:, 0, :]
        # Alternative:
        # cls = out.pooler_output
        cls = self.dropout(cls)

        return {
            "morfologia": self.class_morfologia(cls),
            "tessuto_adiposo": self.class_tessuto_adiposo(cls),
        }


class ReportExtractor(nn.Module):
    def __init__(self, checkpoint="bert-base-multilingual-cased", annotations_model: type[BaseModel]=Annotations, use_pooler_output: bool=False):
        super().__init__()
        self.checkpoint = checkpoint
        self.use_pooler_output = use_pooler_output
        self.regression_fields = get_regression_fields(annotations_model)
        self.multiple_choice_fields = get_multiple_choice_fields(annotations_model)
        self.classification_fields = get_classification_fields(annotations_model)
        
        self.encoder = AutoModel.from_pretrained(checkpoint)  # Encoder base model (like BERT)
        self.dropout = nn.Dropout(0.2)  # Dropout layer to reduce overfitting
        
        # Get number of classes for each field and embedding length
        num_classes = get_number_of_classes(annotations_model)
        hidden = self.encoder.config.hidden_size

        # Classification heads
        self.heads = nn.ModuleDict()
        for field in annotations_model.model_fields.keys():
            if field in self.regression_fields:
                n_classes = 1
            else:
                n_classes = num_classes[field]
            head = nn.Linear(hidden, n_classes)
            self.heads[field] = head
            #setattr(self, field, head)

    def forward(self, input_ids, attention_mask) -> dict[str, torch.Tensor]:
        """
        embedding -> Dropout -> Heads
        """
        # Get embeddings from encoder
        embeddings = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use pooler output or [CLS] token
        if self.use_pooler_output:
            embeddings = embeddings.pooler_output
        else:
            embeddings = embeddings.last_hidden_state[:, 0, :]
        # Apply dropout
        embeddings = self.dropout(embeddings)
        # Apply each head to get predictions for each field
        return {field: head(embeddings) for field, head in self.heads.items()}
        