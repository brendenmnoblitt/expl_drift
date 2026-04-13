"""Core explanation-drift detection primitives."""

from .detector import DriftDetector
from .metrics import compute_all_metrics
from .summarize import SUMMARY_FEATURE_NAMES, summarize_attributions

__all__ = [
    "DriftDetector",
    "compute_all_metrics",
    "summarize_attributions",
    "SUMMARY_FEATURE_NAMES",
]
