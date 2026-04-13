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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .explanations.attention import explain_attention
    from .explanations.ig_transformer import explain_ig_transformer

# Drift detection (core)
from .drift.detector import DriftDetector
from .drift.metrics import MONITORED_METRICS, compute_all_metrics
from .drift.summarize import SUMMARY_FEATURE_NAMES, summarize_attributions

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

# Transformer explanations are loaded lazily so that the base package
# does not require ``transformers`` to be installed.  Install the
# optional extra with ``pip install expl_drift[transformer]`` to enable.
_LAZY_TRANSFORMER_EXPORTS = {
    "explain_attention": ".explanations.attention",
    "explain_ig_transformer": ".explanations.ig_transformer",
}


def __getattr__(name):
    if name in _LAZY_TRANSFORMER_EXPORTS:
        import importlib
        import importlib.util

        # Only rewrite the error when ``transformers`` is actually
        # missing.  If it's installed, let any ImportError from the
        # target module (typo, missing transitive dep, …) surface
        # unchanged so real bugs are not masked.
        if importlib.util.find_spec("transformers") is None:
            raise ImportError(
                f"{name} requires the 'transformer' extra. "
                "Install with: pip install expl_drift[transformer]"
            )
        module = importlib.import_module(
            _LAZY_TRANSFORMER_EXPORTS[name], package=__name__
        )
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.1.0"

__all__ = [
    # drift
    "DriftDetector",
    "MONITORED_METRICS",
    "compute_all_metrics",
    "summarize_attributions",
    "SUMMARY_FEATURE_NAMES",
    # monitoring
    "AlertLevel",
    "ClassConditionalMonitor",
    "DriftMonitor",
    "compute_detection_lead_time",
    # explanations
    "explain_shap",
    "explain_lime",
    "explain_ig",
    # transformer explanations (lazy-loaded via __getattr__)
    "explain_attention",  # pyright: ignore[reportUnsupportedDunderAll]
    "explain_ig_transformer",  # pyright: ignore[reportUnsupportedDunderAll]
]
