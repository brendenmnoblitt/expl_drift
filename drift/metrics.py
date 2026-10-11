"""Metric functions for quantifying attribution drift."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist, cosine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

MONITORED_METRICS: tuple[str, ...] = ("cosine_drift", "max_jsd", "max_wasserstein")

def _feature_array(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or 0 in values.shape or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a nonempty finite two-dimensional array")
    return values


def _reference_identity(reference: np.ndarray) -> str:
    canonical = np.ascontiguousarray(reference, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(json.dumps(list(canonical.shape)).encode())
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _histogram(values: np.ndarray, edges: tuple[float, ...]) -> np.ndarray:
    """Keep underflow/overflow explicitly; include both endpoints in support."""
    lo, hi = edges[0], edges[-1]
    if len(edges) == 1:
        counts = np.array([np.sum(values < lo), np.sum(values == lo), np.sum(values > lo)])
    else:
        interior, _ = np.histogram(values[(values >= lo) & (values <= hi)], bins=edges)
        counts = np.concatenate(([np.sum(values < lo)], interior, [np.sum(values > hi)]))
    return counts.astype(float) / len(values)


@dataclass(frozen=True)
class FittedMetricState:
    """Immutable reference-only scaling, histogram support, and provenance.

    Varying features use reference population standard deviation (no centering,
    so signed means retain direction). Constant nonzero features use absolute
    reference value. A constant-zero feature has no identifiable magnitude
    unit: transform its departures to -1/0/+1, detecting support/direction but
    deliberately not departure intensity. All saved values are JSON finite.
    """

    reference_id: str
    feature_names: tuple[str, ...]
    scales: tuple[float, ...]
    constant_zero: tuple[bool, ...]
    bin_edges: tuple[tuple[float, ...], ...]
    reference_histograms: tuple[tuple[float, ...], ...]
    n_bins: int

    @classmethod
    def fit(
        cls, reference: np.ndarray, *, n_bins: int = 20,
        feature_names: tuple[str, ...] | list[str] | None = None,
    ) -> "FittedMetricState":
        reference = _feature_array(reference, "reference")
        if (
            isinstance(n_bins, bool | np.bool_)
            or not isinstance(n_bins, int | np.integer) or n_bins < 1
        ):
            raise ValueError("n_bins must be a positive integer")
        names = tuple(feature_names) if feature_names is not None else tuple(
            f"feature_{i}" for i in range(reference.shape[1])
        )
        if len(names) != reference.shape[1] or len(set(names)) != len(names) or not all(
            isinstance(name, str) and name for name in names
        ):
            raise ValueError(
                "feature_names must be unique nonempty names matching the feature count"
            )
        constant = np.all(reference == reference[0], axis=0)
        zero = constant & (reference[0] == 0)
        scales = np.where(
            constant, np.where(zero, 1.0, np.abs(reference[0])), reference.std(axis=0)
        )
        if not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError("reference scaling overflowed or underflowed; rescale the input")
        normalized = reference / scales
        if not np.isfinite(normalized).all():
            raise ValueError("reference transform overflowed; rescale the input")
        edges = tuple(
            (float(column[0]),) if constant[i] else tuple(
                float(value) for value in np.linspace(column.min(), column.max(), n_bins + 1)
            )
            for i, column in enumerate(normalized.T)
        )
        if any(len(edge) > 1 and np.any(np.diff(edge) <= 0) for edge in edges):
            raise ValueError("reference histogram edges are not numerically distinct")
        histograms = tuple(
            tuple(_histogram(column, edge)) for column, edge in zip(normalized.T, edges)
        )
        return cls(
            _reference_identity(reference), names, tuple(float(s) for s in scales),
            tuple(bool(z) for z in zero), edges, histograms, int(n_bins),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = _feature_array(values, "features")
        if values.shape[1] != len(self.scales):
            raise ValueError("feature count does not match fitted state")
        normalized = values / self.scales
        for i, zero in enumerate(self.constant_zero):
            if zero:
                normalized[:, i] = np.sign(values[:, i])
        if not np.isfinite(normalized).all():
            raise ValueError("feature transform overflowed; rescale the input")
        return normalized

    def validate_reference(self, reference: np.ndarray) -> None:
        reference = _feature_array(reference, "reference")
        if _reference_identity(reference) != self.reference_id:
            raise ValueError("reference array identity does not match fitted state")

    def _payload(self) -> dict[str, Any]:
        return {
            "version": 1, "reference_id": self.reference_id,
            "feature_names": list(self.feature_names), "scales": list(self.scales),
            "constant_zero": list(self.constant_zero),
            "bin_edges": [list(edge) for edge in self.bin_edges],
            "reference_histograms": [list(hist) for hist in self.reference_histograms],
            "n_bins": self.n_bins,
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self._payload(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(encoded.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, saved: dict[str, Any]) -> "FittedMetricState":
        if saved.get("version") != 1:
            raise ValueError("unsupported fitted metric state version")
        try:
            state = cls(
                saved["reference_id"], tuple(saved["feature_names"]), tuple(saved["scales"]),
                tuple(saved["constant_zero"]), tuple(tuple(e) for e in saved["bin_edges"]),
                tuple(tuple(h) for h in saved["reference_histograms"]), saved["n_bins"],
            )
            if state.fingerprint != saved["fingerprint"]:
                raise ValueError("fitted metric state fingerprint mismatch")
            size = len(state.feature_names)
            if (
                size == 0 or len(set(state.feature_names)) != size
                or not all(isinstance(name, str) and name for name in state.feature_names)
                or not isinstance(state.n_bins, int) or isinstance(state.n_bins, bool)
                or state.n_bins < 1
                or len(state.scales) != size or len(state.constant_zero) != size
                or len(state.bin_edges) != size or len(state.reference_histograms) != size
                or not np.isfinite(state.scales).all() or np.any(np.asarray(state.scales) <= 0)
                or not all(type(zero) is bool for zero in state.constant_zero)
                or not isinstance(state.reference_id, str) or len(state.reference_id) != 64
            ):
                raise ValueError("invalid fitted metric state dimensions or scaling")
            for edge, histogram in zip(state.bin_edges, state.reference_histograms):
                if (
                    len(edge) not in (1, state.n_bins + 1) or not np.isfinite(edge).all()
                    or np.any(np.diff(edge) <= 0)
                    or len(histogram) != (3 if len(edge) == 1 else state.n_bins + 2)
                    or not np.isfinite(histogram).all() or np.any(np.asarray(histogram) < 0)
                    or not np.isclose(sum(histogram), 1)
                ):
                    raise ValueError("invalid fitted histogram support")
            return state
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError("invalid fitted metric state") from error


def _per_feature_jsd(
    baseline_attrs: np.ndarray, current_attrs: np.ndarray, n_bins: int = 20,
    *, metric_state: FittedMetricState | None = None,
) -> np.ndarray:
    """Reference-fitted histogram JSD in nats, with explicit outlier bins."""
    state = metric_state or FittedMetricState.fit(baseline_attrs, n_bins=n_bins)
    state.validate_reference(baseline_attrs)
    current = state.transform(current_attrs)
    values = []
    for baseline, column, edge in zip(state.reference_histograms, current.T, state.bin_edges):
        histogram = _histogram(column, edge)
        mixture = 0.5 * (np.asarray(baseline) + histogram)
        values.append(
            0.5 * stats.entropy(baseline, mixture) + 0.5 * stats.entropy(histogram, mixture)
        )
    return np.clip(values, 0.0, np.log(2))


def compute_jsd(
    baseline_attrs: np.ndarray, current_attrs: np.ndarray, n_bins: int = 20,
    *, metric_state: FittedMetricState | None = None,
) -> float:
    """Per-feature Jensen-Shannon divergence, averaged across features.

    Histogram edges are fitted on baseline alone and never updated from the
    monitored window. Explicit below/above-support bins retain outlier mass.
    The natural-log divergence lies in [0, log(2)] without pseudo-counts.

    Args:
        baseline_attrs: (n_samples, n_features) array of baseline attributions.
        current_attrs: (n_samples, n_features) array of current attributions.
        n_bins: Number of bins to use for histogram estimation.
    Returns:
        float: Average JSD across features.
    """
    return float(np.mean(_per_feature_jsd(
        baseline_attrs, current_attrs, n_bins, metric_state=metric_state,
    )))


def compute_max_jsd(
    baseline_attrs: np.ndarray, current_attrs: np.ndarray, n_bins: int = 20,
    *, metric_state: FittedMetricState | None = None,
) -> float:
    """Maximum per-feature JSD — captures drift in any single feature.
    
    Args:
        baseline_attrs: (n_samples, n_features) array of baseline attributions.
        current_attrs: (n_samples, n_features) array of current attributions.
        n_bins: Number of bins to use for histogram estimation.
    Returns:
        float: Maximum JSD across features.
    """
    return float(np.max(_per_feature_jsd(
        baseline_attrs, current_attrs, n_bins, metric_state=metric_state,
    )))


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
    baseline_attrs = _feature_array(baseline_attrs, "baseline")
    current_attrs = _feature_array(current_attrs, "current")
    if baseline_attrs.shape[1] != current_attrs.shape[1]:
        raise ValueError("baseline and current feature counts must match")
    return _cosine_profiles(
        np.mean(np.abs(baseline_attrs), axis=0),
        np.mean(np.abs(current_attrs), axis=0),
    )


def _cosine_profiles(baseline: np.ndarray, current: np.ndarray) -> float:
    base_zero = np.all(baseline == 0)
    current_zero = np.all(current == 0)
    if base_zero or current_zero:
        return 0.0 if base_zero and current_zero else 1.0
    # Rescale profiles to avoid norm overflow and preserve very small directions.
    baseline = baseline / np.max(np.abs(baseline))
    current = current / np.max(np.abs(current))
    return float(np.clip(cosine(baseline, current), 0.0, 2.0))


def compute_cosine_feature_drift(
    baseline: np.ndarray, current: np.ndarray, *,
    metric_state: FittedMetricState | None = None,
) -> float:
    """Signed mean-profile cosine using reference-fitted feature scales.

    Unlike the raw magnitude diagnostic ``compute_cosine_attribution_drift``,
    this default detector score does not erase sign flips. Both zero means
    give zero distance; exactly one zero mean gives distance one.
    """
    state = metric_state or FittedMetricState.fit(baseline)
    state.validate_reference(baseline)
    return _cosine_profiles(
        state.transform(baseline).mean(axis=0), state.transform(current).mean(axis=0)
    )


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
    baseline_attrs: np.ndarray, current_attrs: np.ndarray,
    *, metric_state: FittedMetricState | None = None,
) -> float:
    """Mean Wasserstein distance in reference-fitted feature units.

    Constant-zero departures use the documented sign-support transform.

    Args:
        baseline_attrs: (n_samples, n_features) array of baseline attributions.
        current_attrs: (n_samples, n_features) array of current attributions.
    Returns:
        float: Average Wasserstein distance across features.
    """
    state = metric_state or FittedMetricState.fit(baseline_attrs)
    state.validate_reference(baseline_attrs)
    return float(np.mean(_per_feature_wasserstein(
        state.transform(baseline_attrs), state.transform(current_attrs),
    )))


def compute_max_wasserstein_drift(
    baseline_attrs: np.ndarray, current_attrs: np.ndarray,
    *, metric_state: FittedMetricState | None = None,
) -> float:
    """Maximum per-feature Wasserstein in reference-fitted units.
    
    Args:
        baseline_attrs: (n_samples, n_features) array of baseline attributions.
        current_attrs: (n_samples, n_features) array of current attributions.
    Returns:
        float: Maximum Wasserstein distance across features.
    """
    state = metric_state or FittedMetricState.fit(baseline_attrs)
    state.validate_reference(baseline_attrs)
    return float(np.max(_per_feature_wasserstein(
        state.transform(baseline_attrs), state.transform(current_attrs),
    )))


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
    metric_state: FittedMetricState | None = None,
) -> dict[str, float]:
    """Compute signed, reference-scaled drift scores with fixed reference JSD edges.

    Args:
        baseline: (n_samples, n_features) array of baseline attributions.
        current: (n_samples, n_features) array of current attributions.
        seed: Random seed for reproducibility (used in classifier metric).
        metrics: Optional tuple of metric names to compute. If None, computes all metrics.
        metric_state: Reusable reference-only fitted transform and histogram state.
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
    if requested - set(all_metrics):
        raise ValueError(f"unknown metrics: {sorted(requested - set(all_metrics))}")
    state = metric_state or FittedMetricState.fit(baseline)
    state.validate_reference(baseline)
    scaled_baseline = state.transform(baseline)
    scaled_current = state.transform(current)
    result: dict[str, float] = {}

    if "cosine_drift" in requested:
        result["cosine_drift"] = float(
            _cosine_profiles(scaled_baseline.mean(axis=0), scaled_current.mean(axis=0))
        )

    if "energy_distance" in requested:
        result["energy_distance"] = float(compute_energy_distance(scaled_baseline, scaled_current))

    if "classifier_auc" in requested:
        result["classifier_auc"] = float(
            compute_classifier_drift(scaled_baseline, scaled_current, seed)
        )

    if "jsd" in requested or "max_jsd" in requested:
        jsd_values = _per_feature_jsd(baseline, current, metric_state=state)
        if "jsd" in requested:
            result["jsd"] = float(np.mean(jsd_values))
        if "max_jsd" in requested:
            result["max_jsd"] = float(np.max(jsd_values))

    if "wasserstein" in requested or "max_wasserstein" in requested:
        distances = _per_feature_wasserstein(scaled_baseline, scaled_current)
        if "wasserstein" in requested:
            result["wasserstein"] = float(np.mean(distances))
        if "max_wasserstein" in requested:
            result["max_wasserstein"] = float(np.max(distances))

    if "ks_max_statistic" in requested or "ks_fraction_significant" in requested:
        ks_max, ks_frac = compute_ks_statistic(scaled_baseline, scaled_current)
        if "ks_max_statistic" in requested:
            result["ks_max_statistic"] = float(ks_max)
        if "ks_fraction_significant" in requested:
            result["ks_fraction_significant"] = float(ks_frac)

    return result
