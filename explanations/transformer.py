"""Base utilities for Transformer attribution extraction.

Provides tokenization, padding, and architecture-aware helpers that
the attention and integrated-gradients extractors build on.  All public
helpers in this module produce ``(n_samples, max_seq_len)`` arrays with
padding positions masked to zero so downstream drift metrics receive a
consistent fixed-width attribution matrix.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def tokenize_texts(
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    max_length: int = 128,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Tokenize a batch of texts with truncation and padding.

    Returns a dict of tensors (input_ids, attention_mask, etc.) moved to
    *device*, ready to feed into a HuggingFace model.
    """
    encoded = tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in encoded.items()}


def mask_padding(
    attributions: np.ndarray,
    attention_mask: np.ndarray,
) -> np.ndarray:
    """Zero-out attribution values at padding positions.

    Args:
        attributions: ``(n_samples, seq_len)`` attribution scores.
        attention_mask: ``(n_samples, seq_len)`` binary mask (1 = real token).

    Returns:
        Attribution array with padding positions set to 0.
    """
    return attributions * attention_mask


def get_classification_token_index(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
) -> int | np.ndarray:
    """Return the token position used for classification.

    * **Encoder models** (BERT, RoBERTa, …) → index ``0`` (the [CLS] token).
    * **Decoder-only models** (Phi-2, Llama, Mistral, …) → index of the last
      non-pad token in each sequence (since causal LMs classify from the final
      position).

    Returns either a scalar ``int`` (encoder) or an ``(n_samples,)`` array of
    per-sample indices (decoder).
    """
    config = model.config

    # Encoder models: always position 0
    if getattr(config, "model_type", "") in (
        "bert", "roberta", "distilbert", "electra", "albert",
    ):
        return 0

    # Decoder-only: last real token per sample
    # input_ids shape: (batch, seq_len)
    if isinstance(input_ids, torch.Tensor):
        input_ids_np = input_ids.detach().cpu().numpy()
    else:
        input_ids_np = np.asarray(input_ids)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    # For each row, find the last non-pad position
    mask = input_ids_np != pad_id
    # sum of True values minus 1 gives the last non-pad index
    last_indices = mask.sum(axis=1) - 1
    return last_indices
