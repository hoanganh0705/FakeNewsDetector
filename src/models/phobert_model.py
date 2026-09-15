from __future__ import annotations

import os
from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
)
from transformers.modeling_outputs import SequenceClassifierOutput


class PhoBertClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        num_classes: int = 2,
        dropout: float = 0.1,
        freeze_bert: bool = False,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.label_smoothing = label_smoothing

        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_classes,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
        )

        if freeze_bert:
            for name, param in self.transformer.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        return self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            **kwargs,
        )

    def save_pretrained(self, save_directory: str) -> None:
        os.makedirs(save_directory, exist_ok=True)
        self.transformer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ) -> "PhoBertClassifier":
        instance = cls(
            model_name=pretrained_model_name_or_path,
            **kwargs,
        )
        return instance
