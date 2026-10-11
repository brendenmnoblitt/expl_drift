"""Fixed, signed, position-invariant document attribution summaries.

Retain signed raw token arrays separately for secondary analyses. Token count and
truncation are input controls, not coordinates of this explanation representation.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import entropy as _entropy

SUMMARY_FEATURE_NAMES = (
    "normalized_entropy",
    "gini",
    "top_k_concentration",
    "mean_absolute",
    "max_absolute",
    "std_absolute",
    "signed_mean",
    "signed_mass_balance",
)


def _gini(values: np.ndarray) -> float:
    """Gini of absolute document-token values, including true zeros."""
    v = np.sort(np.abs(values))
    n = len(v)
    total = v.sum()
    if n == 0 or total == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * v) - (n + 1) * total) / (n * total))


def summarize_attributions(
    attributions: np.ndarray,
    *,
    document_mask: np.ndarray | None = None,
    top_k: int = 5,
) -> np.ndarray:
    """Return the fixed eight-feature document representation.

    Columns in ``SUMMARY_FEATURE_NAMES`` separate concentration (normalized
    Shannon entropy, Gini, top-k mass), magnitude (mean/max/std absolute value),
    and direction (signed mean, signed sum / absolute sum). Entropy is divided
    by log(document-token count), and is zero for zero mass or <=1 token.
    All features are zero for an empty document or zero attribution mass.

    ``document_mask`` is a boolean or 0/1 matrix with the same shape as the
    finite, two-dimensional input. It excludes padding, instructions, candidate
    tokens, and other non-document segments, while retaining true zero-valued
    document tokens. Without a mask every position is a document token: zeros
    are never used to guess token identity. ``top_k`` must be fixed in the study
    representation manifest; targets and columns never follow the prediction.
    The input signed arrays are not modified.
    """
    attributions = np.asarray(attributions, dtype=float)
    if attributions.ndim != 2 or not np.isfinite(attributions).all():
        raise ValueError("attributions must be a finite two-dimensional array")
    if isinstance(top_k, bool | np.bool_) or not isinstance(top_k, int | np.integer) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if document_mask is None:
        mask = np.ones(attributions.shape, dtype=bool)
    else:
        mask = np.asarray(document_mask)
        if mask.shape != attributions.shape or not np.isin(mask, (0, 1)).all():
            raise ValueError("document_mask must match attributions and contain only 0/1")
        mask = mask.astype(bool)

    results = np.zeros((len(attributions), len(SUMMARY_FEATURE_NAMES)), dtype=float)
    for i, row in enumerate(attributions):
        signed = row[mask[i]]
        absolute = np.abs(signed)
        total = absolute.sum()
        if len(signed) == 0 or total == 0:
            continue
        probabilities = absolute / total
        results[i, 0] = (
            float(_entropy(probabilities) / np.log(len(signed))) if len(signed) > 1 else 0.0
        )
        results[i, 1] = _gini(absolute)
        k = min(top_k, len(signed))
        results[i, 2] = np.sort(absolute)[-k:].sum() / total
        results[i, 3] = absolute.mean()
        results[i, 4] = absolute.max()
        results[i, 5] = absolute.std()
        results[i, 6] = signed.mean()
        results[i, 7] = signed.sum() / total
    if not np.isfinite(results).all():
        raise ValueError("attribution summary overflowed; rescale the finite input")
    return results
