"""Drift detector wrapper around attribution drift metrics."""

import numpy as np
import pandas as pd

from .metrics import (
    FittedMetricState,
    _feature_array,
    _per_feature_jsd,
    _per_feature_wasserstein,
    compute_all_metrics,
)


class DriftDetector:
    """Detects explanation drift by comparing window attributions against a baseline.
    
    Attributes:
        baseline: (n_samples, n_features) array of baseline attributions.
    """

    def __init__(
        self, baseline_attributions: np.ndarray, *,
        metric_state: FittedMetricState | None = None,
        feature_names: tuple[str, ...] | list[str] | None = None,
        n_bins: int = 20,
    ) -> None:
        """Fit once on reference, or restore state for the identical reference.

        Pass summary feature names explicitly when using document summaries.
        The detector owns a read-only reference copy so caller mutation cannot
        silently change fitted metric semantics.
        """
        self.baseline = _feature_array(baseline_attributions, "baseline").copy()
        self.baseline.setflags(write=False)
        self.metric_state = metric_state or FittedMetricState.fit(
            self.baseline, feature_names=feature_names, n_bins=n_bins,
        )
        self.metric_state.validate_reference(self.baseline)
        if feature_names is not None and tuple(feature_names) != self.metric_state.feature_names:
            raise ValueError("feature names do not match fitted state")

    def evaluate_window(
        self,
        window_attributions: np.ndarray,
        seed: int | None = None,
        *,
        metrics: tuple[str, ...] | None = None,
    ) -> dict[str, float]:
        """Compute drift metrics for a single window vs baseline.

        Args:
            window_attributions: (n_samples, n_features) array of attributions for the current
                window.
            seed: Random seed for reproducibility (used in classifier metric).
            metrics: Optional tuple of metric names to compute. If None, computes all metrics.
        Returns:
            dict[str, float]: Dictionary mapping metric names to their computed scores.
        """
        return compute_all_metrics(
            self.baseline, window_attributions, seed, metrics=metrics,
            metric_state=self.metric_state,
        )

    def feature_drivers(self, window_attributions: np.ndarray) -> dict[str, dict[str, float]]:
        """Report each feature's JSD, normalized Wasserstein, and signed mean shift.

        Drivers are diagnostics, not extra monitor metrics or independently
        calibrated alerts. Their maxima reproduce the corresponding scores.
        """
        baseline = self.metric_state.transform(self.baseline)
        current = self.metric_state.transform(window_attributions)
        jsd = _per_feature_jsd(
            self.baseline, window_attributions, metric_state=self.metric_state,
        )
        wasserstein = _per_feature_wasserstein(baseline, current)
        mean_shift = current.mean(axis=0) - baseline.mean(axis=0)
        return {
            name: {
                "jsd": float(jsd[i]),
                "wasserstein": float(wasserstein[i]),
                "signed_mean_shift": float(mean_shift[i]),
            }
            for i, name in enumerate(self.metric_state.feature_names)
        }

    def evaluate_all_windows(
        self,
        list_of_window_attributions: list[np.ndarray],
        seed: int | None = None,
        *,
        metrics: tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        """Compute drift metrics for all windows. Returns DataFrame with rows as windows and columns
            as metrics.

        Args:
            list_of_window_attributions: List of (n_samples, n_features) arrays of attributions for
                each window.
            seed: Random seed for reproducibility (used in classifier metric).
            metrics: Optional tuple of metric names to compute. If None, computes all metrics.
        Returns:
            pd.DataFrame: DataFrame where each row corresponds to a window and columns are
                metric scores.
        """
        records: list[dict[str, float]] = []
        for i, attrs in enumerate(list_of_window_attributions):
            m = self.evaluate_window(attrs, seed, metrics=metrics)
            m["window"] = i
            records.append(m)
        df = pd.DataFrame(records)
        df = df.set_index("window")
        return df
