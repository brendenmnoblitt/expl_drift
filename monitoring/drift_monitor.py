"""Stateful drift monitor with tiered alerting.

Wraps DriftDetector with threshold calibration and stateful alert tracking.
Designed for production use: calibrate once on baseline windows, then call
evaluate() on each incoming batch to get an alert level.

Usage
-----
    detector = DriftDetector(baseline_shap)
    monitor = DriftMonitor(detector, baseline_window_shaps)
    result = monitor.evaluate(new_window_shap)
    if result['alert_level'] == AlertLevel.CRITICAL:
        send_page()
"""

import numpy as np

from ..drift.detector import DriftDetector
from ..drift.metrics import MONITORED_METRICS
from .alert_level import AlertLevel


class DriftMonitor:
    """Stateful drift monitor with tiered alerting.

    Attributes:
        detector: DriftDetector instance to compute drift metrics.
        warning_std: Number of std deviations above baseline to trigger WARNING.
        critical_std: Number of std deviations above baseline to trigger CRITICAL.
        prewarning_std: Number of std deviations above baseline to trigger pre-warning trend signal.
        min_consecutive: Minimum consecutive windows of WARNING before escalating to CRITICAL.
        prewarning_trend_consecutive: Number of consecutive windows with increasing metric values
            to consider for pre-warning trend signal.
        prewarning_min_delta_std: Minimum increase in metric value (in std units) over the trend
            window to consider for pre-warning signal.
        warning_clear_consecutive: Number of consecutive no-signal windows required to
            clear a WARNING state.
    """

    def __init__(
        self,
        detector: DriftDetector,
        baseline_windows: list[np.ndarray],
        *,
        warning_std: float = 2.0,
        critical_std: float = 4.0,
        prewarning_std: float = 1.5,
        min_consecutive: int = 2,
        prewarning_trend_consecutive: int = 2,
        prewarning_min_delta_std: float = 0.25,
        warning_clear_consecutive: int = 3,
    ) -> None:
        """Initialize the DriftMonitor with a DriftDetector and calibration parameters.
        
        Args:
            detector: DriftDetector instance to compute drift metrics.
            baseline_windows: List of attribution arrays for known-good windows.
            warning_std: Number of std deviations above baseline to trigger WARNING.
            critical_std: Number of std deviations above baseline to trigger CRITICAL.
            prewarning_std: Number of std deviations above baseline to trigger pre-warning
                trend signal.
            min_consecutive: Minimum consecutive windows of WARNING before escalating to CRITICAL.
            prewarning_trend_consecutive: Number of consecutive windows with increasing metric
                values to consider for pre-warning trend signal.
            prewarning_min_delta_std: Minimum increase in metric value (in std units) over the trend
                window to consider for pre-warning signal.
            warning_clear_consecutive: Number of consecutive no-signal windows required to
                clear a WARNING state.
        Raises:
            ValueError: If any of the threshold parameters are invalid
                (e.g., critical_std <= warning_std).
        """
        self._validate_params(
            warning_std=warning_std,
            critical_std=critical_std,
            prewarning_std=prewarning_std,
            prewarning_trend_consecutive=prewarning_trend_consecutive,
            prewarning_min_delta_std=prewarning_min_delta_std,
            warning_clear_consecutive=warning_clear_consecutive,
        )
        self.detector = detector
        self.warning_std = warning_std
        self.critical_std = critical_std
        self.prewarning_std = prewarning_std
        self.min_consecutive = min_consecutive
        self.prewarning_trend_consecutive = prewarning_trend_consecutive
        self.prewarning_min_delta_std = prewarning_min_delta_std
        self.warning_clear_consecutive = warning_clear_consecutive

        # state tracking
        self._consecutive_warnings: int = 0
        self._consecutive_criticals: int = 0
        self._consecutive_no_signal: int = 0
        self._last_alert_level: AlertLevel = AlertLevel.OK
        self._metric_warning_streaks: dict[str, int] = {
            metric: 0 for metric in MONITORED_METRICS
        }
        self._metric_value_history: dict[str, list[float]] = {
            metric: [] for metric in MONITORED_METRICS
        }

        # calibrate thresholds from baseline
        self.thresholds: dict[str, dict[str, float]] = {}
        self._calibrate(baseline_windows)

    @staticmethod
    def _validate_params(
        *,
        warning_std: float,
        critical_std: float,
        prewarning_std: float,
        prewarning_trend_consecutive: int,
        prewarning_min_delta_std: float,
        warning_clear_consecutive: int,
    ) -> None:
        """Validate threshold and state parameters.
        
        Args:
            warning_std: Number of std deviations above baseline to trigger WARNING.
            critical_std: Number of std deviations above baseline to trigger CRITICAL.
            prewarning_std: Number of std deviations above baseline to trigger pre-warning
                trend signal.
            prewarning_trend_consecutive: Number of consecutive windows with increasing metric
                values to consider for pre-warning trend signal.
            prewarning_min_delta_std: Minimum increase in metric value (in std units) over the
                trend window to consider for pre-warning signal.
            warning_clear_consecutive: Number of consecutive no-signal windows required to
                clear a WARNING state.
        Raises:
            ValueError: If any of the threshold parameters are invalid
                (e.g., critical_std <= warning_std).
        """
        if critical_std <= warning_std:
            raise ValueError("critical_std must be greater than warning_std")
        if prewarning_std <= 0:
            raise ValueError("prewarning_std must be > 0")
        if prewarning_std >= warning_std:
            raise ValueError("prewarning_std must be less than warning_std")
        if prewarning_trend_consecutive < 1:
            raise ValueError("prewarning_trend_consecutive must be >= 1")
        if prewarning_min_delta_std <= 0:
            raise ValueError("prewarning_min_delta_std must be > 0")
        if warning_clear_consecutive < 1:
            raise ValueError("warning_clear_consecutive must be >= 1")

    def _calibrate(self, baseline_windows: list[np.ndarray]) -> None:
        """Compute per-metric mean and std from baseline windows.
        
        Args:
            baseline_windows: List of numpy arrays representing baseline windows.
        """
        baseline_metrics = [
            self.detector.evaluate_window(w, metrics=MONITORED_METRICS)
            for w in baseline_windows
        ]

        for metric in MONITORED_METRICS:
            values = np.array([m[metric] for m in baseline_metrics])
            mean = float(np.mean(values))
            std = float(np.std(values)) + 1e-10  # epsilon for zero-variance
            self.thresholds[metric] = {
                "mean": mean,
                "std": std,
                "prewarning": mean + self.prewarning_std * std,
                "warning": mean + self.warning_std * std,
                "critical": mean + self.critical_std * std,
            }

    def evaluate(self, window_attributions: np.ndarray) -> dict:
        """Evaluate a single window and return tiered alert result.

        Args:
            window_attributions: SHAP attribution array for the window to evaluate.
        Returns:
            dict: Dictionary containing overall alert level, per-metric results,
                and consecutive counts.
        """
        metrics = self.detector.evaluate_window(
            window_attributions, metrics=MONITORED_METRICS,
        )

        # per-metric alert levels
        per_metric: dict[str, AlertLevel] = {}
        for metric in MONITORED_METRICS:
            value = metrics[metric]
            thresh = self.thresholds[metric]
            if value >= thresh["critical"]:
                per_metric[metric] = AlertLevel.CRITICAL
            elif value >= thresh["warning"]:
                per_metric[metric] = AlertLevel.WARNING
            elif self._is_trend_prewarning(metric=metric, value=value, threshold=thresh):
                per_metric[metric] = AlertLevel.WARNING
            else:
                per_metric[metric] = AlertLevel.OK

        # snapshot previous per-metric warning streaks before updating them.
        prior_metric_warning_streaks = dict(self._metric_warning_streaks)
        had_signal = any(
            level in (AlertLevel.WARNING, AlertLevel.CRITICAL)
            for level in per_metric.values()
        )
        if had_signal:
            self._consecutive_no_signal = 0
        else:
            self._consecutive_no_signal += 1

        # determine overall alert level
        alert_level = self._resolve_overall(
            per_metric,
            prior_metric_warning_streaks,
            consecutive_no_signal=self._consecutive_no_signal,
        )

        # update consecutive counters
        if alert_level == AlertLevel.CRITICAL:
            self._consecutive_criticals += 1
            self._consecutive_warnings += 1
        elif alert_level == AlertLevel.WARNING:
            self._consecutive_criticals = 0
            self._consecutive_warnings += 1
        else:
            self._consecutive_criticals = 0
            self._consecutive_warnings = 0

        # update per-metric warning streaks after resolving the current window.
        for metric, level in per_metric.items():
            self._metric_value_history[metric].append(float(metrics[metric]))
            if len(self._metric_value_history[metric]) > self.prewarning_trend_consecutive:
                self._metric_value_history[metric].pop(0)
            if level in (AlertLevel.WARNING, AlertLevel.CRITICAL):
                self._metric_warning_streaks[metric] += 1
            else:
                self._metric_warning_streaks[metric] = 0

        self._last_alert_level = alert_level

        return {
            "alert_level": alert_level,
            "metrics": metrics,
            "alerts": per_metric,
            "consecutive_warnings": self._consecutive_warnings,
            "consecutive_criticals": self._consecutive_criticals,
        }

    def _resolve_overall(
        self,
        per_metric: dict[str, AlertLevel],
        prior_metric_warning_streaks: dict[str, int],
        *,
        consecutive_no_signal: int,
    ) -> AlertLevel:
        """Determine overall alert level from per-metric alerts.

        Rules
        ------
        CRITICAL if any metric is CRITICAL and has met the min_consecutive threshold.
        WARNING if any metric is WARNING or CRITICAL.
        WARNING if pre-warning trend signal is present.
        WARNING if we are in a WARNING state and haven't had enough consecutive no-signal
            windows to clear it.
        OK otherwise.

        Args:
            per_metric: Dictionary mapping metric name to its AlertLevel for the current window.
            prior_metric_warning_streaks: Dictionary mapping metric name to the count of
                consecutive WARNING windows for that metric prior to the current window.
            consecutive_no_signal: Count of consecutive windows with no WARNING or CRITICAL signal.
        Returns:
            AlertLevel: Overall alert level for the current window.
        """
        for metric, level in per_metric.items():
            if (
                level == AlertLevel.CRITICAL
                and prior_metric_warning_streaks.get(metric, 0) >= self.min_consecutive
            ):
                return AlertLevel.CRITICAL

        if any(level in (AlertLevel.WARNING, AlertLevel.CRITICAL) for level in per_metric.values()):
            return AlertLevel.WARNING

        if (
            self._last_alert_level in (AlertLevel.WARNING, AlertLevel.CRITICAL)
            and consecutive_no_signal < self.warning_clear_consecutive
        ):
            return AlertLevel.WARNING

        return AlertLevel.OK

    def _is_trend_prewarning(
        self,
        *,
        metric: str,
        value: float,
        threshold: dict[str, float],
    ) -> bool:
        """Return True if a soft-threshold + trend signal warrants WARNING.
        
        Args:
            metric: Name of the metric being evaluated.
            value: Current value of the metric for the window being evaluated.
            threshold: Dictionary containing 'mean', 'std', 'prewarning', 'warning', and
                'critical' thresholds for the metric.
        Returns:
            bool: True if the metric value is above the pre-warning threshold and has shown a
                consistent increasing trend over the last few windows, indicating a potential
                upcoming drift signal.
        """
        if value < threshold["prewarning"]:
            return False

        history = self._metric_value_history[metric]
        if len(history) < self.prewarning_trend_consecutive:
            return False

        seq = history[-self.prewarning_trend_consecutive:] + [float(value)]
        if not all(curr > prev for prev, curr in zip(seq, seq[1:])):
            return False

        min_rise = self.prewarning_min_delta_std * threshold["std"]
        return (seq[-1] - seq[0]) >= min_rise

    def reset(self) -> None:
        """Reset consecutive counters (e.g., after model retrain)."""
        self._consecutive_warnings = 0
        self._consecutive_criticals = 0
        self._consecutive_no_signal = 0
        self._last_alert_level = AlertLevel.OK
        for metric in self._metric_warning_streaks:
            self._metric_warning_streaks[metric] = 0
            self._metric_value_history[metric] = []
