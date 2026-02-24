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
    Returns:
        int | None: Lead time in windows, or None if no drift or accuracy drop detected
    """
    drift_arr = np.asarray(drift_series)
    raw_acc = np.asarray(accuracy_series, dtype=float)
    smooth_acc = smooth(raw_acc, smooth_acc_window)

    # use first few windows as "normal" baseline
    n_baseline = max(3, len(drift_arr) // 4)

    drift_mean = np.mean(drift_arr[:n_baseline])
    drift_std = np.std(drift_arr[:n_baseline]) + 1e-10
    drift_threshold = drift_mean + threshold_std * drift_std

    # threshold calibrated from RAW accuracy (preserves natural window-to-window variance).
    # detection applied to SMOOTHED accuracy (removes transient single-window dips).
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
