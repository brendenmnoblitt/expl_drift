"""Tests for drift alert state transitions and validation rules."""

import numpy as np
import pytest

from expl_drift.drift.detector import DriftDetector
from expl_drift.monitoring import AlertLevel, ClassConditionalMonitor, DriftMonitor


@pytest.fixture
def baseline_and_detector():
    rng = np.random.RandomState(42)
    base = rng.randn(200, 5)
    detector = DriftDetector(base)
    baseline_windows = [base + rng.randn(200, 5) * 0.01 for _ in range(5)]
    return detector, baseline_windows, base


def test_invalid_threshold_configuration_raises(baseline_and_detector):
    detector, baseline_windows, _ = baseline_and_detector
    with pytest.raises(ValueError, match="critical_std must be greater than warning_std"):
        DriftMonitor(detector, baseline_windows, warning_std=2.0, critical_std=2.0)
    with pytest.raises(ValueError, match="prewarning_std must be less than warning_std"):
        DriftMonitor(detector, baseline_windows, prewarning_std=2.0, warning_std=2.0)


def test_thresholds_are_monotonic(baseline_and_detector):
    detector, baseline_windows, _ = baseline_and_detector
    monitor = DriftMonitor(detector, baseline_windows)
    for metric in ("cosine_drift", "max_jsd", "max_wasserstein"):
        t = monitor.thresholds[metric]
        assert t["critical"] > t["warning"] > t["prewarning"] > t["mean"]


def test_baseline_like_window_is_ok(baseline_and_detector):
    detector, baseline_windows, base = baseline_and_detector
    monitor = DriftMonitor(detector, baseline_windows)
    rng = np.random.RandomState(99)
    normal_window = base + rng.randn(200, 5) * 0.01
    result = monitor.evaluate(normal_window)
    assert result["alert_level"] == AlertLevel.OK
    assert result["consecutive_warnings"] == 0


def test_heavy_drift_triggers_alert(baseline_and_detector):
    detector, baseline_windows, base = baseline_and_detector
    monitor = DriftMonitor(detector, baseline_windows, min_consecutive=1)
    result = monitor.evaluate(base + 10.0)
    assert result["alert_level"] in (AlertLevel.WARNING, AlertLevel.CRITICAL)


def test_warning_latch_then_clear(baseline_and_detector):
    detector, baseline_windows, base = baseline_and_detector
    monitor = DriftMonitor(detector, baseline_windows, warning_clear_consecutive=2)
    monitor.thresholds = {
        metric: {
            "mean": 0.0,
            "std": 1.0,
            "prewarning": 1.0,
            "warning": 2.0,
            "critical": 4.0,
        }
        for metric in ("cosine_drift", "max_jsd", "max_wasserstein")
    }

    values = iter([3.0, 0.0, 0.0])  # warning signal, then two no-signal windows

    def fake_eval(_, **kwargs):
        v = float(next(values))
        return {
            "cosine_drift": v,
            "max_jsd": 0.0,
            "max_wasserstein": 0.0,
        }

    monitor.detector.evaluate_window = fake_eval

    r1 = monitor.evaluate(base)
    r2 = monitor.evaluate(base)
    r3 = monitor.evaluate(base)

    assert r1["alert_level"] == AlertLevel.WARNING
    assert r2["alert_level"] == AlertLevel.WARNING
    assert r3["alert_level"] == AlertLevel.OK


def test_prewarning_trend_promotes_to_warning(baseline_and_detector):
    detector, baseline_windows, base = baseline_and_detector
    monitor = DriftMonitor(
        detector,
        baseline_windows,
        prewarning_std=1.0,
        warning_std=2.0,
        critical_std=4.0,
        prewarning_trend_consecutive=2,
        prewarning_min_delta_std=0.2,
        warning_clear_consecutive=1,
    )

    # Deterministic synthetic thresholds/history
    monitor.thresholds = {
        metric: {
            "mean": 0.0,
            "std": 1.0,
            "prewarning": 1.0,
            "warning": 2.0,
            "critical": 4.0,
        }
        for metric in ("cosine_drift", "max_jsd", "max_wasserstein")
    }

    values = iter([0.2, 1.1, 1.4])

    def fake_eval(_, **kwargs):
        return {
            "cosine_drift": float(next(values)),
            "max_jsd": 0.0,
            "max_wasserstein": 0.0,
        }

    monitor.detector.evaluate_window = fake_eval

    assert monitor.evaluate(base)["alert_level"] == AlertLevel.OK
    assert monitor.evaluate(base)["alert_level"] == AlertLevel.OK
    r3 = monitor.evaluate(base)
    assert r3["alerts"]["cosine_drift"] == AlertLevel.WARNING
    assert r3["alert_level"] == AlertLevel.WARNING


def test_reset_clears_state(baseline_and_detector):
    detector, baseline_windows, base = baseline_and_detector
    monitor = DriftMonitor(detector, baseline_windows, min_consecutive=1)
    monitor.evaluate(base + 10.0)
    assert monitor._consecutive_warnings > 0
    monitor.reset()
    assert monitor._consecutive_warnings == 0
    assert monitor._consecutive_criticals == 0


@pytest.fixture
def class_conditional_setup():
    rng = np.random.RandomState(42)
    n_per_class = 100
    n_features = 5

    base_0 = rng.randn(n_per_class, n_features) - 1.0
    base_1 = rng.randn(n_per_class, n_features) + 1.0
    baseline_attrs = np.vstack([base_0, base_1])
    baseline_preds = np.array([0] * n_per_class + [1] * n_per_class)

    cal_windows = []
    for _ in range(5):
        w_0 = rng.randn(n_per_class, n_features) * 0.01 - 1.0
        w_1 = rng.randn(n_per_class, n_features) * 0.01 + 1.0
        w_attrs = np.vstack([w_0, w_1])
        w_preds = np.array([0] * n_per_class + [1] * n_per_class)
        cal_windows.append((w_attrs, w_preds))

    return baseline_attrs, baseline_preds, cal_windows


def test_class_conditional_ok_when_no_drift(class_conditional_setup):
    baseline_attrs, baseline_preds, cal_windows = class_conditional_setup
    monitor = ClassConditionalMonitor(baseline_attrs, baseline_preds, cal_windows)

    rng = np.random.RandomState(99)
    w_0 = rng.randn(100, 5) * 0.01 - 1.0
    w_1 = rng.randn(100, 5) * 0.01 + 1.0
    window_attrs = np.vstack([w_0, w_1])
    window_preds = np.array([0] * 100 + [1] * 100)

    result = monitor.evaluate(window_attrs, window_preds)
    assert result["alert_level"] == AlertLevel.OK


def test_class_conditional_detects_single_class_drift(class_conditional_setup):
    baseline_attrs, baseline_preds, cal_windows = class_conditional_setup
    monitor = ClassConditionalMonitor(baseline_attrs, baseline_preds, cal_windows)

    rng = np.random.RandomState(99)
    w_0 = rng.randn(100, 5) * 0.01 - 1.0
    w_1 = rng.randn(100, 5) + 10.0
    window_attrs = np.vstack([w_0, w_1])
    window_preds = np.array([0] * 100 + [1] * 100)

    result = monitor.evaluate(window_attrs, window_preds)
    assert result["alert_level"] in (AlertLevel.WARNING, AlertLevel.CRITICAL)
    assert result["per_class"][1]["alert_level"] in (AlertLevel.WARNING, AlertLevel.CRITICAL)
