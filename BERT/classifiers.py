import torch
import torch.nn as nn
from transformers import AutoModel
from BERT_utils import get_number_of_classes
from BERT_utils import get_regression_fields, get_multiple_choice_fields, get_classification_fields, get_optional_regression_fields
from pydantic import BaseModel
from constants import Annotations


class ReportExtractor(nn.Module):
    def __init__(self, checkpoint="bert-base-multilingual-cased",
                 annotations_model: type[BaseModel] = Annotations,
                 use_pooler_output: bool = False,
                 dropout_rate: float = 0.2):
        super().__init__()
        # Model attributes
        self.checkpoint = checkpoint
        self.annotations_model = annotations_model
        self.use_pooler_output = use_pooler_output
        self.dropout_rate = dropout_rate
        self.regression_fields = get_regression_fields(annotations_model)
        self.multiple_choice_fields = get_multiple_choice_fields(annotations_model)
        self.classification_fields = get_classification_fields(annotations_model)
        self.num_classes = get_number_of_classes(annotations_model)
        # Layers
        self.encoder = AutoModel.from_pretrained(checkpoint)  # Encoder base model (like BERT)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer to reduce overfitting
        
        # Get embedding length
        hidden = self.encoder.config.hidden_size

        # Classification heads
        self.heads = nn.ModuleDict()
        for field in (self.regression_fields + self.classification_fields + self.multiple_choice_fields):
            if field in self.regression_fields:
                n_classes = 1
            elif field in self.multiple_choice_fields:
                n_classes = self.num_classes[field]
            else:
                n_classes = self.num_classes[field]
                if n_classes < 3:
                    n_classes = 1
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
        # Apply dropout
        embeddings = self.dropout(embeddings)
        # Apply each head to get predictions for each field
        outputs = dict()
        for field, head in self.heads.items():
            out = head(embeddings)
            if field in self.regression_fields:
                out = nn.functional.relu(out)
            outputs[field] = out
        return outputs