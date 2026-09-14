"""
BiLSTM model for Vietnamese Fake News Detection.

A bidirectional LSTM classifier with optional pretrained FastText embeddings.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM for binary text classification.

    Architecture
    -------------
    * **Embedding** — ``(V, D)`` learnable lookup table (initialised from
      FastText when ``load_pretrained_embeddings`` is called).
    * **BiLSTM** — ``num_layers`` bidirectional LSTM layers; output is the
      last hidden state of both directions concatenated.
    * **Dropout** — applied after the LSTM output.
    * **Classifier head** — linear projection to ``num_classes`` logits.

    Forward signature matches the interface expected by
    ``src/training/train_bilstm.py`` (and indirectly by
    ``src/training/reproduce_predictions.py``).
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.3,
        padding_idx: int = 0,
        num_classes: int = 2,
    ) -> None:
        """
        Parameters
        ----------
        vocab_size : int
            Size of the vocabulary (number of rows in the embedding matrix).
        embedding_dim : int
            Dimensionality of each token embedding.
        hidden_dim : int
            LSTM hidden-state dimensionality (per direction).
        num_layers : int, default 1
            Number of stacked LSTM layers.
        dropout : float, default 0.3
            Dropout probability applied after the LSTM.
        padding_idx : int, default 0
            Index of the padding token in the vocabulary (its embedding is
            fixed at zero and excluded from gradient computation).
        num_classes : int, default 2
            Number of output classes (binary = 2).
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
    # Public API used by trainers
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
            Token indices into the vocabulary.
        attention_mask : ``torch.Tensor``, optional — shape ``(B, seq_len)``
            Not used in this implementation but accepted for compatibility.

        Returns
        -------
        ``torch.Tensor`` — shape ``(B, num_classes)``
            Raw logits (no softmax applied).
        """
        # input_ids: (B, L)
        embedded = self.embedding(input_ids)           # (B, L, D)
        lstm_out, (hidden, _) = self.lstm(embedded)   # lstm_out: (B, L, 2*H)
        # Use the last hidden state from both directions
        # hidden: (num_layers * 2, B, H) → take the top layer
        last_hidden = torch.cat(
            [hidden[-2], hidden[-1]], dim=1
        )                                              # (B, 2*H)
        dropped = self.dropout_layer(last_hidden)
        logits = self.classifier(dropped)              # (B, num_classes)
        return logits

    def load_pretrained_embeddings(self, matrix: torch.Tensor) -> None:
        """
        Replace the learnable embedding matrix with pretrained weights.

        Parameters
        ----------
        matrix : ``torch.Tensor`` — shape ``(vocab_size, embedding_dim)``
            Pretrained embedding matrix (e.g. FastText).
            Must match the model's ``vocab_size`` and ``embedding_dim``.
        """
        if matrix.shape != (self.vocab_size, self.embedding_dim):
            raise ValueError(
                f"Pretrained matrix shape {matrix.shape} does not match "
                f"(vocab_size={self.vocab_size}, embedding_dim={self.embedding_dim})"
            )
        self.embedding.weight.data.copy_(matrix)
        self.embedding.weight.requires_grad = True  # fine-tune by default

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the full model state plus metadata to a ``.pt`` file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "BiLSTMClassifier":
        """Load a saved model, overriding constructor args with saved weights."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        # Reconstruct from the state dict keys to get correct dimensions
        # Assumes the checkpoint was saved with the same vocab_size, etc.
        instance = cls(**kwargs)
        instance.load_state_dict(state)
        return instance

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight_ih" in name:      # LSTM input-to-hidden
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:    # LSTM hidden-to-hidden
                nn.init.orthogonal_(param.data)
            elif "weight" in name and param.dim() == 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
