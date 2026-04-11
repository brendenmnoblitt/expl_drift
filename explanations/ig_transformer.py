"""Integrated Gradients attribution extractor for Transformer models.

Uses Captum's ``IntegratedGradients`` on pre-computed token embeddings
to produce token-level attributions compatible with the expl_drift drift
pipeline.  Output shape: ``(n_samples, max_seq_len)`` with padding
masked to zero.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from captum.attr import IntegratedGradients
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .transformer import mask_padding, tokenize_texts


def _get_embedding_layer(model: PreTrainedModel) -> torch.nn.Module:
    """Resolve the token embedding layer for a HuggingFace model."""
    model_type = getattr(model.config, "model_type", "")

    # Try common attribute paths
    if model_type in ("bert", "distilbert", "albert"):
        base = getattr(model, model_type, None)
        if base is not None:
            return base.embeddings.word_embeddings
    if model_type == "roberta":
        return model.roberta.embeddings.word_embeddings
    if model_type == "electra":
        return model.electra.embeddings.word_embeddings

    # Decoder-only models (Phi, Llama, Mistral, GPT-2, etc.)
    for attr in ("model", "transformer"):
        base = getattr(model, attr, None)
        if base is not None:
            embed = getattr(base, "embed_tokens", None) or getattr(base, "wte", None)
            if embed is not None:
                return embed

    raise ValueError(
        f"Cannot auto-detect embedding layer for model_type={model_type!r}. "
        "Pass the embedding layer explicitly."
    )


def explain_ig_transformer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    *,
    max_length: int = 128,
    n_steps: int = 50,
    batch_size: int = 16,
    target_class: int | None = None,
    device: torch.device | str | None = None,
    embedding_layer: torch.nn.Module | None = None,
) -> np.ndarray:
    """Compute Integrated Gradients attributions for a Transformer.

    Computes IG on the token embedding vectors using Captum's
    ``IntegratedGradients``.  Per-token attributions are summed across
    the embedding dimension to produce a single scalar per token.

    Args:
        model: HuggingFace model with a classification head.
        tokenizer: Matching tokenizer.
        texts: Input texts to explain.
        max_length: Tokenizer max sequence length.
        n_steps: Number of interpolation steps for IG.
        batch_size: Forward-pass batch size (smaller than attention since
            IG requires gradients and is more memory-intensive).
        target_class: Class index to attribute towards.  If ``None``,
            attributes towards each sample's predicted class.
        device: Device override; defaults to ``model.device``.
        embedding_layer: Override auto-detection of the embedding layer.

    Returns:
        ``(n_samples, max_length)`` array with padding masked to 0.
    """
    if device is None:
        device = next(model.parameters()).device

    if embedding_layer is None:
        embedding_layer = _get_embedding_layer(model)

    model.eval()

    def forward_fn(input_embeds, attention_mask):
        outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask)
        logits = outputs.logits  # (batch, n_classes)
        if target_class is not None:
            return logits[:, target_class]
        # Default: use predicted class per sample
        preds = logits.argmax(dim=-1)
        return logits.gather(1, preds.unsqueeze(1)).squeeze(1)

    ig = IntegratedGradients(forward_fn)

    all_attributions: list[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        inputs = tokenize_texts(tokenizer, batch_texts, max_length=max_length, device=device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Baseline: pad token embedding everywhere
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        baseline_ids = torch.full_like(input_ids, pad_id)

        # Get embeddings for input and baseline
        with torch.no_grad():
            input_embeds = embedding_layer(input_ids)
            baseline_embeds = embedding_layer(baseline_ids)

        # input_embeds needs gradients for IG
        input_embeds = input_embeds.detach().requires_grad_(True)

        # Compute IG attributions directly on embeddings
        # Returns (batch, seq_len, embed_dim)
        attrs = ig.attribute(
            input_embeds,
            baselines=baseline_embeds,
            additional_forward_args=(attention_mask,),
            n_steps=n_steps,
        )

        # Sum across embedding dim → (batch, seq_len)
        token_attrs = attrs.sum(dim=-1).detach().cpu().numpy()

        # Mask padding
        mask = attention_mask.detach().cpu().numpy()
        token_attrs = mask_padding(token_attrs, mask)

        all_attributions.append(token_attrs)

    return np.concatenate(all_attributions, axis=0)
