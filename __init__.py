"""expl_drift: Explanation drift monitoring library.

Detects model degradation early by monitoring how attribution
distributions shift over time, before accuracy drops are observable.

Typical usage
-------------
>>> from expl_drift import DriftDetector, DriftMonitor, AlertLevel, explain_shap
>>> baseline = explain_shap(model, X_baseline, X_baseline)
>>> detector = DriftDetector(baseline)
>>> monitor = DriftMonitor(detector, calibration_windows)
>>> result = monitor.evaluate(new_window_shap_values)

For experiment-specific code (data loading, model training,
plotting), see the ``expl_drift_experiments`` package.
"""

# Drift detection (core)
from .drift.detector import DriftDetector
from .drift.metrics import MONITORED_METRICS, compute_all_metrics

# Monitoring (production alerting + threshold lead-time algorithm)
from .monitoring import (
    AlertLevel,
    ClassConditionalMonitor,
    DriftMonitor,
    compute_detection_lead_time,
)

# Explanations (XAI algorithms)
from .explanations.shap_explainer import explain_shap
from .explanations.lime_explainer import explain_lime
from .explanations.ig_explainer import explain_ig

__version__ = "0.1.0"

__all__ = [
    # drift
    "DriftDetector",
    "MONITORED_METRICS",
    "compute_all_metrics",
    # monitoring
    "AlertLevel",
    "ClassConditionalMonitor",
    "DriftMonitor",
    "compute_detection_lead_time",
    # explanations
    "explain_shap",
    "explain_lime",
    "explain_ig",
]
