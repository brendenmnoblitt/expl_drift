"""Integrated Gradients explanation helper."""

import numpy as np
import torch
from captum.attr import IntegratedGradients


def explain_ig(
    model: torch.nn.Module, X: np.ndarray, baseline: np.ndarray | None = None
) -> np.ndarray:
    """Compute Integrated Gradients attributions for a PyTorch model.

    If baseline is None, uses zero vector.

    Args:
        model: PyTorch model to explain.
        X: (n_samples, n_features) array of input data to explain.
        baseline: Optional (n_samples, n_features) array of baseline inputs. If None,
            uses zero vector.
    Returns:
        np.ndarray: Attributions of shape (n_samples, n_features).
    """
    X_arr = X.values if hasattr(X, "values") else np.asarray(X)
    X_tensor = torch.tensor(X_arr, dtype=torch.float32)
    X_tensor.requires_grad_(True)

    if baseline is None:
        baseline_tensor = torch.zeros_like(X_tensor)
    else:
        bl_arr = baseline.values if hasattr(baseline, "values") else np.asarray(baseline)
        baseline_tensor = torch.tensor(bl_arr, dtype=torch.float32)

    ig = IntegratedGradients(model)
    attributions = ig.attribute(X_tensor, baselines=baseline_tensor)

    return attributions.detach().numpy()
