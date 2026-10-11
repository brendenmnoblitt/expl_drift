"""Core explanation-drift detection primitives."""

from .detector import DriftDetector
from .metrics import FittedMetricState, compute_all_metrics, compute_cosine_feature_drift
from .summarize import SUMMARY_FEATURE_NAMES, summarize_attributions

__all__ = [
    "DriftDetector",
    "FittedMetricState",
    "compute_cosine_feature_drift",
    "compute_all_metrics",
    "summarize_attributions",
    "SUMMARY_FEATURE_NAMES",
]
