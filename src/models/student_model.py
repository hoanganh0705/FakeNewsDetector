"""
Student BiLSTM for Knowledge Distillation.

A smaller BiLSTM trained to mimic the soft predictions of a teacher model
(PhoBERT or BiLSTM teacher).  Used in ``src/training/train_student.py``.

The class interface mirrors :class:`BiLSTMClassifier` so that
evaluation and inference code written for the teacher works unchanged.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn


class StudentBiLSTM(nn.Module):
    """
    Smaller bidirectional LSTM for knowledge-distillation (KD) student.

    This is a deliberately compact model (small hidden size) so that the
    resulting checkpoint is much smaller than the teacher PhoBERT while
    retaining a large fraction of its accuracy.

    Class-level defaults (``DEFAULT_*``) are read by
    ``StudentBiLSTMTrainer`` to initialise the trainer when no explicit
    values are provided.
    """

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
        """
        Parameters
        ----------
        vocab_size : int
            Vocabulary size (number of embedding rows).
        embedding_dim : int, default 300
            Token embedding dimension.
        hidden_dim : int, default 64
            LSTM hidden size (per direction).  Intentionally smaller than
            the teacher BiLSTM to keep the model compact.
        num_layers : int, default 1
            Number of LSTM layers.
        dropout : float, default 0.3
            Dropout probability between LSTM layers and before the head.
        padding_idx : int, default 0
            Index of the padding token in the vocabulary.
        num_classes : int, default 2
            Output dimension (binary classification = 2).
        """
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

    # ------------------------------------------------------------------
    # Forward — identical signature to BiLSTMClassifier.forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run a forward pass.

        Parameters
        ----------
        input_ids : ``torch.Tensor`` — shape ``(B, seq_len)``
            Token indices.
        attention_mask : ``torch.Tensor``, optional
            Ignored; accepted only for API compatibility with the teacher.

        Returns
        -------
        ``torch.Tensor`` — shape ``(B, num_classes)``
            Raw logits.
        """
        embedded = self.embedding(input_ids)                # (B, L, D)
        _, (hidden, _) = self.lstm(embedded)                # hidden: (2*L, B, H)
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (B, 2*H)
        dropped = self.dropout_layer(last_hidden)
        return self.classifier(dropped)                     # (B, C)

    # ------------------------------------------------------------------
    # Model-size helpers used by the KD trainer
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """
        Return the total number of **trainable** parameters.

        Returns
        -------
        int
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_mb(self) -> float:
        """
        Approximate model size in megabytes (float32).

        Returns
        -------
        float
        """
        bytes_per_param = 4  # float32
        return (self.count_parameters() * bytes_per_param) / (1024 ** 2)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the model state dict to ``path``."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "StudentBiLSTM":
        """Load a saved model checkpoint."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        instance = cls(**kwargs)
        instance.load_state_dict(state)
        return instance

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "weight" in name and param.dim() == 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
