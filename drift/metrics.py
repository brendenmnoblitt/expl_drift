"""Metric functions for quantifying attribution drift."""

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist, cosine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

MONITORED_METRICS: tuple[str, ...] = ("cosine_drift", "max_jsd", "max_wasserstein")

def _histogram_per_feature(
    attrs: np.ndarray, n_bins: int, range_vals: list[tuple[float, float]]
) -> list[np.ndarray]:
    """Compute normalized histograms per feature. Adds small epsilon to avoid zero bins.
    
    Args:
        attrs: (n_samples, n_features) array of attributions.
        n_bins: Number of histogram bins.
        range_vals: List of (min, max) tuples for each feature to define histogram range.
    Returns:
        list[np.ndarray]: List of length n_features, each element is a (n_bins,) array
            of normalized histogram values.
    """
    epsilon = 1e-10
    n_features = attrs.shape[1]
    hists: list[np.ndarray] = []
    for f in range(n_features):
        hist, _ = np.histogram(attrs[:, f], bins=n_bins, range=range_vals[f])
        hist = hist.astype(float) + epsilon
        hist /= hist.sum()
        hists.append(hist)
    return hists

def _per_feature_jsd(
    baseline_attrs: np.ndarray, current_attrs: np.ndarray, n_bins: int = 20
) -> np.ndarray:
    """Compute JSD per feature. Returns array of shape (n_features,).
    
    Args:
        baseline_attrs: (n_samples, n_features) array of baseline attributions.
        current_attrs: (n_samples, n_features) array of current attributions.
        n_bins: Number of bins to use for histogram estimation.
        
    Returns:
        np.ndarray: Array of shape (n_features,) containing JSD values for each feature.
    """
    baseline_attrs = np.asarray(baseline_attrs)
    current_attrs = np.asarray(current_attrs)
    n_features = baseline_attrs.shape[1]

    range_vals: list[tuple[float, float]] = []
    for f in range(n_features):
        lo = min(baseline_attrs[:, f].min(), current_attrs[:, f].min())
        hi = max(baseline_attrs[:, f].max(), current_attrs[:, f].max())
        if lo == hi:
            hi = lo + 1.0
        range_vals.append((lo, hi))

    base_hists = _histogram_per_feature(baseline_attrs, n_bins, range_vals)
    curr_hists = _histogram_per_feature(current_attrs, n_bins, range_vals)

    jsd_values: list[float] = []
    for b, c in zip(base_hists, curr_hists):
        m = 0.5 * (b + c)
        jsd = 0.5 * stats.entropy(b, m) + 0.5 * stats.entropy(c, m)
        jsd_values.append(jsd)

    return np.array(jsd_values)


def compute_jsd(
    baseline_attrs: np.ndarray, current_attrs: np.ndarray, n_bins: int = 20
) -> float:
    """Per-feature Jensen-Shannon divergence, averaged across features.

    Binned, non-parametric measure of distributional shift. More stable than
    KS statistic for small samples, but less sensitive to subtle shape changes
    than Wasserstein distance.

    Args:
        baseline_attrs: (n_samples, n_features) array of baseline attributions.
        current_attrs: (n_samples, n_features) array of current attributions.
        n_bins: Number of bins to use for histogram estimation.
    Returns:
        float: Average JSD across features.
    """
    return float(np.mean(_per_feature_jsd(baseline_attrs, current_attrs, n_bins)))


def compute_max_jsd(
    baseline_attrs: np.ndarray, current_attrs: np.ndarray, n_bins: int = 20
) -> float:
    """Maximum per-feature JSD — captures drift in any single feature.
    
    Args:
        baseline_attrs: (n_samples, n_features) array of baseline attributions.
        current_attrs: (n_samples, n_features) array of current attributions.
        n_bins: Number of bins to use for histogram estimation.
    Returns:
        float: Maximum JSD across features.
    """
    return float(np.max(_per_feature_jsd(baseline_attrs, current_attrs, n_bins)))


def compute_cosine_attribution_drift(
    baseline_attrs: np.ndarray, current_attrs: np.ndarray
) -> float:
    """Cosine distance between mean |attribution| vectors.

    Compares the overall feature importance profile rather than per-sample
    distributions. Very stable and sensitive to shifts in which features
    the model relies on.

    Args:
        baseline_attrs: (n_samples, n_features) array of baseline attributions.
        current_attrs: (n_samples, n_features) array of current attributions.
    Returns:
        float: Cosine distance between mean |attribution| vectors.
    """
    baseline_attrs = np.asarray(baseline_attrs)
    current_attrs = np.asarray(current_attrs)

    base_profile = np.mean(np.abs(baseline_attrs), axis=0)
    curr_profile = np.mean(np.abs(current_attrs), axis=0)

    # Handle edge case of zero vectors
    if np.allclose(base_profile, 0) or np.allclose(curr_profile, 0):
        return 0.0

    return float(cosine(base_profile, curr_profile))


def _per_feature_wasserstein(
    baseline_attrs: np.ndarray, current_attrs: np.ndarray
) -> np.ndarray:
    """Compute Wasserstein distance per feature. Returns array of shape (n_features,).
    
    Non-parametric, no binning needed. More sensitive than JSD for detecting shifts in
    distribution shape.

    Args:
        baseline_attrs: (n_samples, n_features) array of baseline attributions.
        current_attrs: (n_samples, n_features) array of current attributions.
    Returns:
        np.ndarray: Array of shape (n_features,) containing Wasserstein distances for each feature.
    """
    baseline_attrs = np.asarray(baseline_attrs)
    current_attrs = np.asarray(current_attrs)
    n_features = baseline_attrs.shape[1]

    distances: list[float] = []
    for f in range(n_features):
        d = stats.wasserstein_distance(baseline_attrs[:, f], current_attrs[:, f])
        distances.append(d)

    return np.array(distances)


def compute_wasserstein_drift(
    baseline_attrs: np.ndarray, current_attrs: np.ndarray
) -> float:
    """Mean per-feature Wasserstein (Earth Mover's) distance.

    Non-parametric, no binning needed. More sensitive than JSD for
    detecting shifts in distribution shape.

    Args:
        baseline_attrs: (n_samples, n_features) array of baseline attributions.
        current_attrs: (n_samples, n_features) array of current attributions.
    Returns:
        float: Average Wasserstein distance across features.
    """
    return float(np.mean(_per_feature_wasserstein(baseline_attrs, current_attrs)))


def compute_max_wasserstein_drift(
    baseline_attrs: np.ndarray, current_attrs: np.ndarray
) -> float:
    """Maximum per-feature Wasserstein distance.
    
    Args:
        baseline_attrs: (n_samples, n_features) array of baseline attributions.
        current_attrs: (n_samples, n_features) array of current attributions.
    Returns:
        float: Maximum Wasserstein distance across features.
    """
    return float(np.max(_per_feature_wasserstein(baseline_attrs, current_attrs)))


def compute_energy_distance(
    baseline_attrs: np.ndarray, current_attrs: np.ndarray
) -> float:
    """Multivariate energy distance between two attribution distributions.
    
    Captures overall distributional shift across all features, including changes in feature
    interactions. More sensitive than per-feature metrics for detecting shifts in joint
    distribution, but less interpretable and more computationally expensive.
    
    Args:
        baseline_attrs: (n_samples, n_features) array of baseline attributions.
        current_attrs: (n_samples, n_features) array of current attributions.
    Returns:
        float: Energy distance between two attribution distributions.
    """
    baseline_attrs = np.asarray(baseline_attrs, dtype=float)
    current_attrs = np.asarray(current_attrs, dtype=float)

    # Subsample for efficiency if large
    max_n = 500
    if len(baseline_attrs) > max_n:
        rng = np.random.RandomState(0)
        idx = rng.choice(len(baseline_attrs), max_n, replace=False)
        baseline_attrs = baseline_attrs[idx]
    if len(current_attrs) > max_n:
        rng = np.random.RandomState(1)
        idx = rng.choice(len(current_attrs), max_n, replace=False)
        current_attrs = current_attrs[idx]

    d_xy = cdist(baseline_attrs, current_attrs, metric="euclidean").mean()
    d_xx = cdist(baseline_attrs, baseline_attrs, metric="euclidean").mean()
    d_yy = cdist(current_attrs, current_attrs, metric="euclidean").mean()

    return float(2 * d_xy - d_xx - d_yy)


def compute_ks_statistic(
    baseline_attrs: np.ndarray, current_attrs: np.ndarray
) -> tuple[float, float]:
    """Per-feature KS test. Returns (max_statistic, fraction_significant).
    
    Non-parametric, no binning needed. More sensitive to subtle shape changes than
    JSD, but less stable for small samples.
    
    Args:
        baseline_attrs: (n_samples, n_features) array of baseline attributions.
        current_attrs: (n_samples, n_features) array of current attributions.
    Returns:
        tuple: (max_statistic, fraction_significant) where max_statistic is the maximum
            KS statistic across features, and fraction_significant is the fraction of features
            with p-value < 0.05.
    """
    baseline_attrs = np.asarray(baseline_attrs)
    current_attrs = np.asarray(current_attrs)
    n_features = baseline_attrs.shape[1]

    ks_stats: list[float] = []
    p_values: list[float] = []
    for f in range(n_features):
        stat, pval = stats.ks_2samp(baseline_attrs[:, f], current_attrs[:, f])
        ks_stats.append(stat)
        p_values.append(pval)

    max_stat = float(np.max(ks_stats))
    frac_sig = float(np.mean(np.array(p_values) < 0.05))
    return max_stat, frac_sig


def compute_classifier_drift(
    baseline_attrs: np.ndarray, current_attrs: np.ndarray, seed: int | None = None
) -> float:
    """Train a classifier to distinguish baseline vs current attributions. Returns
    Area Under the ROC Curve (AUC).
    
    Args:
        baseline_attrs: (n_samples, n_features) array of baseline attributions.
        current_attrs: (n_samples, n_features) array of current attributions.
        seed: Random seed for reproducibility.
    Returns:
        float: AUC of classifier.
    """
    baseline_attrs = np.asarray(baseline_attrs)
    current_attrs = np.asarray(current_attrs)

    X = np.vstack([baseline_attrs, current_attrs])
    y = np.concatenate([np.zeros(len(baseline_attrs)), np.ones(len(current_attrs))])

    clf = LogisticRegression(random_state=seed, max_iter=1000)
    scores = cross_val_score(clf, X, y, cv=min(5, min(len(baseline_attrs), len(current_attrs))),
                             scoring="roc_auc")
    return float(np.mean(scores))

def compute_all_metrics(
    baseline: np.ndarray,
    current: np.ndarray,
    seed: int | None = None,
    *,
    metrics: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Compute drift metrics. Returns dict of metric_name -> score.

    Args:
        baseline: (n_samples, n_features) array of baseline attributions.
        current: (n_samples, n_features) array of current attributions.
        seed: Random seed for reproducibility (used in classifier metric).
        metrics: Optional tuple of metric names to compute. If None, computes all metrics.
    Returns:
        dict[str, float]: Dictionary mapping metric names to their computed scores.
    """
    all_metrics: tuple[str, ...] = (
        "cosine_drift",
        "jsd",
        "max_jsd",
        "wasserstein",
        "max_wasserstein",
        "energy_distance",
        "ks_max_statistic",
        "ks_fraction_significant",
        "classifier_auc",
    )
    requested = set(all_metrics if metrics is None else metrics)
    result: dict[str, float] = {}

    if "cosine_drift" in requested:
        result["cosine_drift"] = float(
            compute_cosine_attribution_drift(baseline, current)
        )

    if "energy_distance" in requested:
        result["energy_distance"] = float(compute_energy_distance(baseline, current))

    if "classifier_auc" in requested:
        result["classifier_auc"] = float(
            compute_classifier_drift(baseline, current, seed)
        )

    if "jsd" in requested or "max_jsd" in requested:
        jsd_values = _per_feature_jsd(baseline, current)
        if "jsd" in requested:
            result["jsd"] = float(np.mean(jsd_values))
        if "max_jsd" in requested:
            result["max_jsd"] = float(np.max(jsd_values))

    if "wasserstein" in requested or "max_wasserstein" in requested:
        distances = _per_feature_wasserstein(baseline, current)
        if "wasserstein" in requested:
            result["wasserstein"] = float(np.mean(distances))
        if "max_wasserstein" in requested:
            result["max_wasserstein"] = float(np.max(distances))

    if "ks_max_statistic" in requested or "ks_fraction_significant" in requested:
        ks_max, ks_frac = compute_ks_statistic(baseline, current)
        if "ks_max_statistic" in requested:
            result["ks_max_statistic"] = float(ks_max)
        if "ks_fraction_significant" in requested:
            result["ks_fraction_significant"] = float(ks_frac)

    return result
