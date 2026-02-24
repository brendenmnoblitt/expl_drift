"""LIME explanation helper."""

import numpy as np
import torch
from lime.lime_tabular import LimeTabularExplainer


def explain_lime(
    model: object,
    X: np.ndarray,
    training_data: np.ndarray,
    feature_names: list[str],
    model_type: str = "xgboost",
    max_samples: int = 500,
    seed: int | None = None,
) -> np.ndarray:
    """Compute LIME attributions for up to max_samples instances.

    Args:
        model: Model to explain.
        X: (n_samples, n_features) array of input data to explain.
        training_data: (n_train_samples, n_features) array of data to use for LIME
            background distribution.
        feature_names: List of feature names corresponding to columns in X and training_data.
        model_type: Type of model, either "xgboost" or "nn" (PyTorch neural network).
            Determines how to call predict_fn.
        max_samples: Maximum number of instances to explain (for speed). If X has fewer than
            max_samples, explains all instances.
        seed: Random seed for reproducibility when sampling instances to explain.
    Returns:
        np.ndarray: Attributions of shape (n_explain, n_features) where n_explain
            is min(max_samples, n_samples).
    """
    X_arr = X.values if hasattr(X, "values") else np.asarray(X)
    train_arr = training_data.values if hasattr(training_data, "values") \
        else np.asarray(training_data)

    explainer = LimeTabularExplainer(
        train_arr,
        feature_names=feature_names,
        mode="classification",
        random_state=seed,
    )

    if model_type == "xgboost":
        predict_fn = model.predict_proba
    elif model_type == "nn":
        def predict_fn(x: np.ndarray) -> np.ndarray:
            model.eval()
            with torch.no_grad():
                probs = model(torch.tensor(x, dtype=torch.float32)).numpy()
            return np.column_stack([1 - probs, probs])
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    rng = np.random.RandomState(seed)
    n_explain = min(max_samples, len(X_arr))
    indices = rng.choice(len(X_arr), size=n_explain, replace=False)

    n_features = X_arr.shape[1]
    attributions = np.zeros((n_explain, n_features))

    for i, idx in enumerate(indices):
        exp = explainer.explain_instance(X_arr[idx], predict_fn, num_features=n_features)
        label_map = exp.as_map()
        # for binary classification, use label 1
        label_key = 1 if 1 in label_map else list(label_map.keys())[0]
        for feat_idx, weight in label_map[label_key]:
            attributions[i, feat_idx] = weight

    return attributions
