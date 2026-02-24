"""Class-conditional drift monitoring.

Used to catch drift that is localized to specific predicted classes, which may be missed by a
pooled monitor.  For example, if a model's performance degrades only on one class, the overall
distribution of attributions may not shift enough to trigger an alert from a pooled monitor.
By maintaining separate monitors for each predicted class, we can detect drift that affects
only a subset of classes.
"""

from typing import Any

import numpy as np

from ..drift.detector import DriftDetector
from .alert_level import ALERT_SEVERITY
from .drift_monitor import DriftMonitor


def _class_cal_windows(
    cls: int,
    baseline_windows: list[tuple[np.ndarray, np.ndarray]],
    min_class_samples: int,
) -> list[np.ndarray]:
    """Return class-specific calibration windows with enough samples.
    
    Args:
        cls: Class label to filter by.
        baseline_windows: List of (attributions, predictions) tuples for known-good windows.
        min_class_samples: Minimum number of samples for the class in a window to be included.
    Returns:
        list[np.ndarray]: List of attribution arrays for the class, one per calibration window.
    """
    windows: list[np.ndarray] = []
    for cal_attrs, cal_preds in baseline_windows:
        mask = cal_preds == cls
        if mask.sum() >= min_class_samples:
            windows.append(cal_attrs[mask])
    return windows


class ClassConditionalMonitor:
    """Monitors drift per predicted class and aggregates to an overall alert.

    Maintains a separate DriftMonitor for each predicted class, calibrated on windows where that
    class has enough samples. Also maintains a pooled monitor on all samples together. When
    evaluating a new window, it checks each class-specific monitor (if calibrated) and
    the pooled monitor, and returns the worst alert level among them. This allows detection
    of drift that may only affect a subset of classes, which could be missed by a
    pooled monitor alone.
    """

    def __init__(
        self,
        baseline_attributions: np.ndarray,
        baseline_predictions: np.ndarray,
        baseline_windows: list[tuple[np.ndarray, np.ndarray]],
        *,
        min_class_samples: int = 30,
        **monitor_kwargs: Any,
    ) -> None:
        """Initialize class-conditional monitors from baseline windows.
        
        Args:
            baseline_attributions: SHAP attribution array for the baseline period.
            baseline_predictions: Predicted class labels for the baseline period.
            baseline_windows: List of (attributions, predictions) tuples for known-good windows.
            min_class_samples: Minimum number of samples for a class in a window
                to be included in that class's monitor calibration. Default is 30.
            **monitor_kwargs: Additional keyword arguments to pass to each DriftMonitor.
        """
        self.min_class_samples = min_class_samples
        self._monitor_kwargs = monitor_kwargs

        # discover classes from baseline
        self._classes = sorted(set(int(c) for c in np.unique(baseline_predictions)))

        # partition baseline attributions by class
        baseline_by_class = self._partition_by_class(
            baseline_attributions, baseline_predictions
        )

        # create per-class detectors and monitors
        self._monitors: dict[int, DriftMonitor] = {}
        for cls in self._classes:
            cls_cal_windows = _class_cal_windows(
                cls, baseline_windows, min_class_samples
            )
            if len(cls_cal_windows) < 2:
                continue
            self._monitors[cls] = DriftMonitor(
                DriftDetector(baseline_by_class[cls]), cls_cal_windows, **monitor_kwargs
            )

        # pooled monitor (all classes together)
        pooled_detector = DriftDetector(baseline_attributions)
        pooled_cal_windows = [attrs for attrs, _ in baseline_windows]
        self._pooled_monitor = DriftMonitor(
            pooled_detector, pooled_cal_windows, **monitor_kwargs
        )

    @staticmethod
    def _partition_by_class(
        attributions: np.ndarray, predictions: np.ndarray
    ) -> dict[int, np.ndarray]:
        """Split attributions array by predicted class label.
        
        Args:
            attributions: SHAP attribution array.
            predictions: Predicted class labels.
        Returns:
            dict[int, np.ndarray]: Mapping from class label to attributions for samples
                predicted as that class.
        """
        classes = np.unique(predictions)
        return {int(c): attributions[predictions == c] for c in classes}

    def evaluate(
        self,
        window_attributions: np.ndarray,
        window_predictions: np.ndarray,
    ) -> dict:
        """Evaluate a single window with per-class and pooled drift monitoring.

        Args:
            window_attributions: SHAP attribution array for the window to evaluate.
            window_predictions: Predicted class labels for the window.
        Returns:
            dict: Dictionary containing overall alert level, pooled monitor result,
                per-class results, and list of skipped classes.
        """
        by_class = self._partition_by_class(window_attributions, window_predictions)
        pooled_result = self._pooled_monitor.evaluate(window_attributions)
        worst = pooled_result["alert_level"]

        per_class_results: dict[int, dict] = {}
        skipped: list[int] = []

        for cls in self._classes:
            if cls not in self._monitors:
                skipped.append(cls)
                continue
            cls_attrs = by_class.get(cls)
            if cls_attrs is None or len(cls_attrs) < self.min_class_samples:
                skipped.append(cls)
                continue
            cls_result = self._monitors[cls].evaluate(cls_attrs)
            per_class_results[cls] = cls_result
            if ALERT_SEVERITY[cls_result["alert_level"]] > ALERT_SEVERITY[worst]:
                worst = cls_result["alert_level"]

        # also flag classes in the window that weren't in the baseline
        for cls in by_class:
            if cls not in self._classes and cls not in skipped:
                skipped.append(cls)

        return {
            "alert_level": worst,
            "pooled": pooled_result,
            "per_class": per_class_results,
            "skipped_classes": sorted(skipped),
        }

    def reset(self) -> None:
        """Reset all internal monitors (e.g., after model retrain)."""
        self._pooled_monitor.reset()
        for monitor in self._monitors.values():
            monitor.reset()
