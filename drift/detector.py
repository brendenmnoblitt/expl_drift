"""Drift detector wrapper around attribution drift metrics."""

import numpy as np
import pandas as pd

from .metrics import compute_all_metrics


class DriftDetector:
    """Detects explanation drift by comparing window attributions against a baseline.
    
    Attributes:
        baseline: (n_samples, n_features) array of baseline attributions.
    """

    def __init__(self, baseline_attributions: np.ndarray) -> None:
        """Initialize the drift detector with baseline attributions.
        
        Args:
            baseline_attributions: (n_samples, n_features) array of baseline attributions.
        """
        self.baseline = baseline_attributions

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
        )

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
