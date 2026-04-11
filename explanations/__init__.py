"""Model explanation backends used by the drift pipeline."""

from typing import TYPE_CHECKING

from .ig_explainer import explain_ig
from .lime_explainer import explain_lime
from .shap_explainer import explain_shap

if TYPE_CHECKING:
    from .attention import explain_attention
    from .ig_transformer import explain_ig_transformer

# Transformer backends are lazy — they require the optional
# ``transformer`` extra (HuggingFace ``transformers``).  Importing this
# package should not fail when that extra is not installed.
_LAZY_TRANSFORMER_EXPORTS = {
    "explain_attention": ".attention",
    "explain_ig_transformer": ".ig_transformer",
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


__all__ = [
    "explain_ig",
    "explain_lime",
    "explain_shap",
    # transformer backends (lazy-loaded via __getattr__)
    "explain_attention",  # pyright: ignore[reportUnsupportedDunderAll]
    "explain_ig_transformer",  # pyright: ignore[reportUnsupportedDunderAll]
]
