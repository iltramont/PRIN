import torch
import torch.nn as nn
import sys
from pathlib import Path
from transformers import AutoModel

project_root = Path.cwd().resolve().parent
sys.path.insert(0, str(project_root))

from utils.schema_json import *


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
    def __init__(self, checkpoint="bert-base-multilingual-cased", num_classes_morfologia=4, num_classes_tessuto_adiposo=5):
        super().__init__()
        self.checkpoint = checkpoint
        self.encoder = AutoModel.from_pretrained(checkpoint)
        hidden = self.encoder.config.hidden_size  # Embedding length
        self.dropout = nn.Dropout(0.2)

        # Classification heads
        self.class_morfologia = nn.Linear(hidden, num_classes_morfologia)
        self.class_tessuto_adiposo = nn.Linear(hidden, num_classes_tessuto_adiposo)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Better than pooler_output
        cls = out.last_hidden_state[:, 0, :]
        # Alternative:
        # cls = out.pooler_output
        cls = self.dropout(cls)

        return {
            "morfologia": self.class_morfologia(cls),
            "tessuto_adiposo": self.class_tessuto_adiposo(cls),
        }

