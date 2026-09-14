"""
PhoBERT model wrapper for Vietnamese Fake News Detection.

Fine-tunes ``vinai/phobert-base`` (or any HuggingFace model) for binary
sequence classification.  Follows the HuggingFace ``AutoModel`` convention
so that ``save_pretrained`` / ``from_pretrained`` work as expected.
"""

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
    """
    Lightweight wrapper around a HuggingFace transformer for sequence
    classification.

    This class follows the standard HuggingFace pattern (AutoModel +
    classification head) so that downstream code can call
    ``save_pretrained`` / ``from_pretrained`` without any custom logic.

    Forward signature is compatible with ``train_phobert.py``:
        ``forward(input_ids, attention_mask=None, token_type_ids=None,
                 labels=None, **kwargs) -> SequenceClassifierOutput``
    """

    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        num_classes: int = 2,
        dropout: float = 0.1,
        freeze_bert: bool = False,
        label_smoothing: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        model_name : str, default ``"vinai/phobert-base"``
            HuggingFace model identifier (local path or model hub name).
        num_classes : int, default 2
            Number of output labels (binary classification = 2).
        dropout : float, default 0.1
            Dropout probability applied before the classifier head.
        freeze_bert : bool, default False
            If True, freeze all BERT parameters so only the head is trained.
        label_smoothing : float, default 0.0
            Label smoothing factor (passed to CrossEntropyLoss in the trainer,
            not used directly here).
        """
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.label_smoothing = label_smoothing

        # Use AutoModelForSequenceClassification for a pre-built head
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

    # ------------------------------------------------------------------
    # Forward — mirrors the HuggingFace interface used in train_phobert.py
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        """
        Forward pass through the transformer.

        Parameters
        ----------
        input_ids : ``torch.Tensor`` — shape ``(B, seq_len)``
            Token IDs.
        attention_mask : ``torch.Tensor``, optional — shape ``(B, seq_len)``
            Mask to avoid attending to padding tokens.
        token_type_ids : ``torch.Tensor``, optional
            Not used by PhoBERT but accepted for compatibility.
        labels : ``torch.Tensor``, optional — shape ``(B,)``
            Integer class labels. If provided, ``loss`` is computed
            automatically by HuggingFace.

        Returns
        -------
        ``SequenceClassifierOutput``
            A dataclass with ``loss`` (if ``labels`` provided),
            ``logits`` (shape ``(B, num_classes)``), and other HuggingFace
            standard fields.
        """
        return self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Persistence (HuggingFace standard)
    # ------------------------------------------------------------------

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the model and config to ``save_directory`` using the HuggingFace
        convention (model.safetensors / config.json).

        Parameters
        ----------
        save_directory : str
            Target directory. Created if it does not exist.
        """
        os.makedirs(save_directory, exist_ok=True)
        self.transformer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ) -> "PhoBertClassifier":
        """
        Load a model from a HuggingFace checkpoint directory or hub name.

        Parameters
        ----------
        pretrained_model_name_or_path : str
            Local directory or HuggingFace model ID.
        **kwargs : dict
            Passed to the constructor (e.g. ``dropout``).

        Returns
        -------
        ``PhoBertClassifier``
        """
        instance = cls(
            model_name=pretrained_model_name_or_path,
            **kwargs,
        )
        return instance
