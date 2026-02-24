"""SHAP explanation helpers for XGBoost and PyTorch models."""

import numpy as np
import shap
import torch
import xgboost


def _xgboost_shap(model: object, X_arr: np.ndarray) -> np.ndarray:
    """Native XGBoost SHAP via pred_contribs (C++ path, no Python SHAP overhead).
    
    Args:
        model: XGBoost model to explain.
        X_arr: (n_samples, n_features) array of input data to explain.
    Returns:
        np.ndarray: SHAP values of shape (n_samples, n_features).
    """
    booster = model.get_booster()
    # preserve feature names so the booster doesn't reject unnamed arrays
    feature_names = booster.feature_names
    dmatrix = xgboost.DMatrix(X_arr, feature_names=feature_names)
    # pred_contribs returns (n_samples, n_features + 1); last column is the bias
    contribs = booster.predict(dmatrix, pred_contribs=True)
    return contribs[:, :-1]


def _nn_shap(
    model: object, X_arr: np.ndarray, bg_arr: np.ndarray,
) -> np.ndarray:
    """GradientExplainer for PyTorch models (backprop-based, much faster than KernelExplainer).
    
    Args:
        model: PyTorch model to explain.
        X_arr: (n_samples, n_features) array of input data to explain.
        bg_arr: (n_bg_samples, n_features) array of background data for SHAP
    Returns:
        np.ndarray: SHAP values of shape (n_samples, n_features).
    """
    bg_sample = bg_arr[:min(100, len(bg_arr))]
    bg_tensor = torch.tensor(bg_sample, dtype=torch.float32)

    # SHAP gradient/deep explainers require 2D output (batch, classes).
    # wrap the model if its forward() squeezes to 1D.
    class _Ensure2D(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, x):
            out = self.wrapped(x)
            if out.dim() == 1:
                out = out.unsqueeze(-1)
            return out

    model.eval()
    wrapper = _Ensure2D(model)
    explainer = shap.GradientExplainer(wrapper, bg_tensor)
    shap_values = explainer.shap_values(torch.tensor(X_arr, dtype=torch.float32))

    # GradientExplainer returns list of arrays (one per output class) for 2D output
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    result = np.asarray(shap_values)
    # squeeze trailing dimension from the 2D wrapper: (n, features, 1) -> (n, features)
    if result.ndim == 3 and result.shape[-1] == 1:
        result = result.squeeze(-1)
    return result


def explain_shap(
    model: object,
    X: np.ndarray,
    background_data: np.ndarray,
    model_type: str = "xgboost",
) -> np.ndarray:
    """Compute SHAP values for X.

    For XGBoost: uses native pred_contribs (fast C++ path).
    For NN: uses DeepExplainer (backprop-based).

    Args:
        model: Model to explain (XGBoost or PyTorch).
        X: (n_samples, n_features) array of input data to explain.
        background_data: (n_bg_samples, n_features) array of background data for SHAP
        model_type: Type of model, either "xgboost" or "nn" (PyTorch neural network).
    Returns:
        np.ndarray: SHAP values of shape (n_samples, n_features).
    """
    X_arr = X.values if hasattr(X, "values") else np.asarray(X)
    bg_arr = background_data.values if hasattr(background_data, "values") \
        else np.asarray(background_data)

    if model_type == "xgboost":
        shap_values = _xgboost_shap(model, X_arr)
    elif model_type == "nn":
        shap_values = _nn_shap(model, X_arr, bg_arr)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return np.asarray(shap_values)
