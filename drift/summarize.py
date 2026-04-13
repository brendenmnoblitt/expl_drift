"""Summarize per-token attribution vectors into semantically stable features.

When attributions are indexed by token position (as in Transformer models),
each position corresponds to a different word in every sample, making
per-position drift metrics unreliable.  This module reduces a raw
``(n_samples, seq_len)`` attribution matrix into a compact
``(n_samples, k)`` matrix of summary statistics that are position-invariant
and can be fed directly into :class:`~expl_drift.drift.detector.DriftDetector`.

Typical usage::

    from expl_drift.drift.summarize import summarize_attributions

    raw_attrs = explain_attention(model, tokenizer, texts)  # (n, seq_len)
    summary = summarize_attributions(raw_attrs)              # (n, k)
    detector = DriftDetector(summary)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import entropy as _entropy


def _gini(values: np.ndarray) -> float:
    """Gini coefficient of a 1-D array (0 = perfectly uniform, 1 = maximally concentrated)."""
    v = np.sort(np.abs(values))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * v) - (n + 1) * np.sum(v)) / (n * np.sum(v)))


def summarize_attributions(
    attributions: np.ndarray,
    *,
    top_k: int = 5,
) -> np.ndarray:
    """Reduce per-token attributions to position-invariant summary features.

    For each sample, computes the following over non-padding (non-zero)
    token positions:

    - **entropy**: Shannon entropy of the normalized |attribution| distribution.
      High entropy = attention spread evenly; low = concentrated on few tokens.
    - **gini**: Gini coefficient of |attributions|.  Complementary to entropy;
      more sensitive to concentration in the tail.
    - **top_k_concentration**: Fraction of total |attribution| mass in the
      top-*k* positions.  Captures how peaked the distribution is.
    - **max_attribution**: Maximum |attribution| value.
    - **std_attribution**: Standard deviation of |attributions| across positions.
    - **n_active_tokens**: Number of non-zero positions (sequence length proxy).

    Args:
        attributions: ``(n_samples, seq_len)`` array.  Padding positions
            should already be masked to 0.
        top_k: Number of top positions for concentration feature.

    Returns:
        ``(n_samples, 6)`` array of summary features.
    """
    attributions = np.asarray(attributions, dtype=float)
    n_samples = attributions.shape[0]

    results = np.zeros((n_samples, 6), dtype=float)

    for i in range(n_samples):
        row = np.abs(attributions[i])
        mask = row > 0
        active = row[mask]

        if len(active) == 0:
            continue

        total = active.sum()
        normed = active / total if total > 0 else active

        # Entropy of normalized attribution distribution
        results[i, 0] = float(_entropy(normed))

        # Gini coefficient
        results[i, 1] = _gini(active)

        # Top-k concentration
        k = min(top_k, len(active))
        top_k_sum = np.sort(active)[-k:].sum()
        results[i, 2] = top_k_sum / total if total > 0 else 0.0

        # Max attribution
        results[i, 3] = active.max()

        # Std of attributions
        results[i, 4] = active.std()

        # Number of active tokens
        results[i, 5] = len(active)

    return results


SUMMARY_FEATURE_NAMES = [
    "entropy",
    "gini",
    "top_k_concentration",
    "max_attribution",
    "std_attribution",
    "n_active_tokens",
]
