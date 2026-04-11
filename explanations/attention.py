"""Attention-weight attribution extractor for Transformer models.

Extracts attention weights from a HuggingFace Transformer and aggregates
them into a ``(n_samples, max_seq_len)`` attribution matrix compatible
with the expl_drift drift-detection pipeline.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .transformer import mask_padding, tokenize_texts


AggregationStrategy = Literal["last_layer_mean", "all_layer_mean", "last_layer_max"]


def explain_attention(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    *,
    max_length: int = 128,
    strategy: AggregationStrategy = "last_layer_mean",
    batch_size: int = 32,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Extract attention-based attributions for a batch of texts.

    The model is run in eval / no-grad mode.  Attention weights are
    aggregated according to *strategy*, then the row corresponding to
    the classification token (CLS for encoders, last real token for
    decoders) is taken as the per-token attribution vector.

    Args:
        model: HuggingFace model (must support ``output_attentions``).
        tokenizer: Matching tokenizer.
        texts: Input texts to explain.
        max_length: Tokenizer max sequence length.
        strategy: How to aggregate across heads/layers.
        batch_size: Forward-pass batch size.
        device: Device override; defaults to ``model.device``.

    Returns:
        ``(n_samples, max_length)`` array with padding masked to 0.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_attributions: list[np.ndarray] = []

    # Some models use SDPA by default which doesn't support
    # output_attentions — temporarily switch to eager attention.  We
    # track whether the attribute existed before so that we can fully
    # restore the config (including deleting it if it was absent)
    # rather than leaving a stray ``_attn_implementation = "eager"``
    # behind after the call returns.
    _MISSING = object()
    orig_impl = getattr(model.config, "_attn_implementation", _MISSING)
    model.config._attn_implementation = "eager"
    try:
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            inputs = tokenize_texts(
                tokenizer, batch_texts, max_length=max_length, device=device
            )

            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)

            # attentions: tuple of (batch, n_heads, seq_len, seq_len) per layer
            attentions = outputs.attentions
            batch_attrs = _aggregate_attentions(
                attentions, inputs, strategy, model, tokenizer
            )
            all_attributions.append(batch_attrs)
    finally:
        if orig_impl is _MISSING:
            try:
                delattr(model.config, "_attn_implementation")
            except AttributeError:
                pass
        else:
            model.config._attn_implementation = orig_impl

    return np.concatenate(all_attributions, axis=0)


def _aggregate_attentions(
    attentions: tuple[torch.Tensor, ...],
    inputs: dict[str, torch.Tensor],
    strategy: AggregationStrategy,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
) -> np.ndarray:
    """Aggregate raw attention into (batch, seq_len) attributions."""
    if strategy == "last_layer_mean":
        # Last layer, mean over heads → (batch, seq_len, seq_len)
        attn = attentions[-1].mean(dim=1)
    elif strategy == "all_layer_mean":
        # Stack all layers, mean over layers and heads
        stacked = torch.stack(attentions, dim=0)  # (n_layers, batch, n_heads, seq, seq)
        attn = stacked.mean(dim=(0, 2))  # (batch, seq, seq)
    elif strategy == "last_layer_max":
        # Last layer, max over heads
        attn = attentions[-1].max(dim=1).values  # (batch, seq, seq)
    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy}")

    # Extract the classification token's attention row
    cls_attrs = _extract_cls_row(attn, model, inputs, tokenizer)

    # Mask padding
    attention_mask = inputs["attention_mask"].detach().cpu().numpy()
    return mask_padding(cls_attrs, attention_mask)


def _extract_cls_row(
    attn: torch.Tensor,
    model: PreTrainedModel,
    inputs: dict[str, torch.Tensor],
    tokenizer: PreTrainedTokenizerBase,
) -> np.ndarray:
    """Get the attention row for the classification token.

    For encoder models: row 0 ([CLS]).
    For decoder-only models: row at the last non-pad position per sample.
    """
    from .transformer import get_classification_token_index

    cls_idx = get_classification_token_index(model, inputs["input_ids"], tokenizer)

    attn_np = attn.detach().cpu().numpy()  # (batch, seq, seq)

    if isinstance(cls_idx, (int, np.integer)):
        # Encoder: same index for all samples
        return attn_np[:, cls_idx, :]
    else:
        # Decoder: per-sample index
        batch_size = attn_np.shape[0]
        return np.array([attn_np[i, cls_idx[i], :] for i in range(batch_size)])
