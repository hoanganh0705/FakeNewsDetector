from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn


class StudentBiLSTM(nn.Module):
    DEFAULT_EMBEDDING_DIM: int = 300
    DEFAULT_HIDDEN_DIM: int = 64
    DEFAULT_NUM_LAYERS: int = 1

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dropout: float = 0.3,
        padding_idx: int = 0,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout_layer = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

        self._init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embedded = self.embedding(input_ids)                # (B, L, D)
        _, (hidden, _) = self.lstm(embedded)                # hidden: (2*L, B, H)
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (B, 2*H)
        dropped = self.dropout_layer(last_hidden)
        return self.classifier(dropped)                     # (B, C)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_mb(self) -> float:
        bytes_per_param = 4  # float32
        return (self.count_parameters() * bytes_per_param) / (1024 ** 2)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "StudentBiLSTM":
        # Security: weights_only=True prevents arbitrary code execution from tampered checkpoints.
        state = torch.load(path, map_location="cpu", weights_only=True)
        instance = cls(**kwargs)
        instance.load_state_dict(state)
        return instance
        
    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight_ih" in name:      # LSTM input-to-hidden
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:    # LSTM hidden-to-hidden
                nn.init.orthogonal_(param.data)
            elif "weight" in name and param.dim() == 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:         # LSTM bias
                nn.init.zeros_(param)
