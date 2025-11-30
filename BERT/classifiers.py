import sys
import os
from dotenv import load_dotenv
from pathlib import Path

import json
from utils.schema_json import ReportData, AnnotatedReport
import time
from IPython.display import clear_output

from huggingface_hub import login

import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, Dataset, DatasetDict



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

