"""Primary lead-time algorithm for drift detection.

This module intentionally contains only the production/default lead-time
computation based on threshold crossing relative to accuracy degradation.
"""

import numpy as np
import pandas as pd


def smooth(arr: np.ndarray, window: int = 1) -> np.ndarray:
    """Centered rolling mean. window=1 is a no-op.
    
    Args:
        arr: 1D array to smooth.
        window: rolling mean window size. 1 = no smoothing.
    Returns:
        np.ndarray: Smoothed array of same length as input.
    """
    if window <= 1:
        return np.asarray(arr, dtype=float)
    return pd.Series(arr).rolling(window, min_periods=1, center=True).mean().values

def compute_detection_lead_time(
    drift_series: np.ndarray,
    accuracy_series: np.ndarray,
    threshold_std: float = 2.0,
    smooth_acc_window: int = 1,
    drift_baseline: tuple[float, float] | None = None,
    accuracy_baseline: tuple[float, float] | None = None,
) -> int | None:
    """Compute windows between drift flag and accuracy drop.

    Drift flag = first window where drift > mean + threshold_std * std (of pre-drift windows).
    Accuracy drop = first window where smoothed accuracy < mean - threshold_std * std.
    Lead time = accuracy drop window - drift flag window.

    Args:
        drift_series: 1D array of drift metric values per window.
        accuracy_series: 1D array of accuracy values per window.
        threshold_std: Number of standard deviations from baseline to set threshold.
        smooth_acc_window: Window size for smoothing accuracy. 1 = no smoothing.
        drift_baseline: Optional ``(mean, std)`` for the drift metric, computed
            from a known-clean reference (e.g. a monitor's calibration windows).
            If omitted, the first ``max(3, len(drift_series)//4)`` windows are
            used — which contaminates the baseline when drift starts at window 0.
        accuracy_baseline: Optional ``(mean, std)`` for accuracy, same idea.
    Returns:
        int | None: Lead time in windows, or None if no drift or accuracy drop detected
    """
    drift_arr = np.asarray(drift_series)
    raw_acc = np.asarray(accuracy_series, dtype=float)
    smooth_acc = smooth(raw_acc, smooth_acc_window)

    # use first few windows as "normal" baseline
    n_baseline = max(3, len(drift_arr) // 4)

    if drift_baseline is not None:
        drift_mean, drift_std = drift_baseline
        drift_std = drift_std + 1e-10
    else:
        drift_mean = np.mean(drift_arr[:n_baseline])
        drift_std = np.std(drift_arr[:n_baseline]) + 1e-10
    drift_threshold = drift_mean + threshold_std * drift_std

    # threshold calibrated from RAW accuracy (preserves natural window-to-window variance).
    # detection applied to SMOOTHED accuracy (removes transient single-window dips).
    if accuracy_baseline is not None:
        acc_mean, acc_std = accuracy_baseline
        acc_std = acc_std + 1e-10
    else:
        acc_mean = np.mean(raw_acc[:n_baseline])
        acc_std = np.std(raw_acc[:n_baseline]) + 1e-10
    acc_threshold = acc_mean - threshold_std * acc_std

    # when drift exceeds threshold, record window index and break
    drift_flag: int | None = None
    for i, v in enumerate(drift_arr):
        if v > drift_threshold:
            drift_flag = i
            break

    # when smoothed accuracy drops below threshold, record window index and break
    acc_flag: int | None = None
    for i, v in enumerate(smooth_acc):
        if v < acc_threshold:
            acc_flag = i
            break

    if drift_flag is None or acc_flag is None:
        return None

    return acc_flag - drift_flag


def evaluate_stateful_events(
    alert_levels: list,
    accuracy_series: np.ndarray,
    *,
    reference_accuracy: float,
    evaluation_start: int,
    stable_start: int = 0,
    quality_drop: float = 0.05,
    quality_persistence: int = 2,
) -> dict:
    """Classify actual warning onsets against a sustained observed quality drop.

    Quality is an observed-window estimand, not evidence of underlying
    degradation. The event timestamp is the confirming window. Quality
    persistence starts afresh at evaluation entry; monitor state does not.
    An already active stable-period warning is excluded until it clears and
    a new warning starts. Inputs must include stable-period alert history;
    slicing it away would falsely credit a carried warning as shift detection.
    Null qualification and group-aware reference uncertainty are caller duties.
    """
    from .alert_level import AlertLevel

    accuracy = np.asarray(accuracy_series, dtype=float)
    if accuracy.ndim != 1 or len(accuracy) != len(alert_levels) or not len(accuracy):
        raise ValueError("alert states and window accuracies must be nonempty and aligned")
    if not np.isfinite(accuracy).all() or np.any((accuracy < 0) | (accuracy > 1)):
        raise ValueError("window accuracies must be finite probabilities")
    if not np.isfinite(reference_accuracy) or not 0 <= reference_accuracy <= 1:
        raise ValueError("reference accuracy must be a finite probability")
    if not np.isfinite(quality_drop) or not 0 < quality_drop <= 1:
        raise ValueError("quality drop must be in (0, 1]")
    if (
        isinstance(quality_persistence, bool)
        or not isinstance(quality_persistence, int) or quality_persistence < 1
    ):
        raise ValueError("quality persistence must be a positive integer")
    if (
        type(stable_start) is not int or type(evaluation_start) is not int
        or not 0 <= stable_start <= evaluation_start < len(accuracy)
    ):
        raise ValueError("stable and evaluation boundaries must be ordered within the horizon")
    states = [
        level if isinstance(level, AlertLevel) else AlertLevel(level) for level in alert_levels
    ]
    warning = [level in (AlertLevel.WARNING, AlertLevel.CRITICAL) for level in states]
    stable_onsets = []
    detection = None
    for index in range(stable_start, len(states)):
        prior_active = warning[index - 1] if index else False
        if warning[index] and not prior_active:
            if index < evaluation_start:
                stable_onsets.append(index)
            elif detection is None:
                detection = index
    quality_event = None
    streak = 0
    for index in range(evaluation_start, len(accuracy)):
        # Comparing drops, rather than a subtracted threshold, includes exactly
        # five percentage points despite ordinary binary floating-point rounding.
        drop = reference_accuracy - accuracy[index]
        qualifies = drop >= quality_drop or np.isclose(drop, quality_drop, rtol=0, atol=1e-12)
        streak = streak + 1 if qualifies else 0
        if streak >= quality_persistence:
            quality_event = index
            break
    lead_time = None if detection is None or quality_event is None else quality_event - detection
    if quality_event is None:
        outcome = (
            "warning_without_quality_event" if detection is not None
            else "no_warning_no_quality_event"
        )
    elif detection is None:
        outcome = "miss"
    elif lead_time > 0:
        outcome = "positive_lead"
    elif lead_time == 0:
        outcome = "simultaneous"
    else:
        outcome = "late_detection"
    return {
        "estimand": "sustained_observed_window_accuracy_drop",
        "detection_timestamp": detection,
        "quality_event_timestamp": quality_event,
        "lead_time": lead_time,
        "outcome": outcome,
        "stable_warning_onsets": stable_onsets,
        "warning_active_at_entry": bool(evaluation_start and warning[evaluation_start - 1]),
        "quality_event_absent": quality_event is None,
        "detection_absent": detection is None,
        "evaluation_start": evaluation_start,
        "reference_accuracy": float(reference_accuracy),
        "quality_drop": float(quality_drop),
        "quality_persistence": quality_persistence,
    }
