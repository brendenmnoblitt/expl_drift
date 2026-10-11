"""Behavior regressions for signed summaries and frozen reference measurement."""

import copy
import json

import numpy as np
import pytest

from expl_drift.drift import (
    SUMMARY_FEATURE_NAMES,
    DriftDetector,
    FittedMetricState,
    summarize_attributions,
)
from expl_drift.drift.metrics import (
    MONITORED_METRICS,
    compute_all_metrics,
    compute_cosine_attribution_drift,
    compute_cosine_feature_drift,
    compute_jsd,
    compute_max_jsd,
    compute_max_wasserstein_drift,
)
from expl_drift.monitoring import DriftMonitor


def test_summary_separates_sign_magnitude_and_concentration():
    raw = np.array([[3.0, -1.0, 0.0, 20.0], [1.0, 2.0, 0.0, -20.0]])
    mask = np.array([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=bool)
    original = raw.copy()
    summary = summarize_attributions(raw, document_mask=mask, top_k=1)
    flipped = summarize_attributions(-raw, document_mask=mask, top_k=1)
    assert summary.shape == (2, 8)
    assert "n_active_tokens" not in SUMMARY_FEATURE_NAMES
    np.testing.assert_allclose(flipped[:, :6], summary[:, :6])
    np.testing.assert_allclose(flipped[:, 6:], -summary[:, 6:])
    np.testing.assert_array_equal(raw, original)
    assert compute_cosine_feature_drift(summary, flipped) > 0
    assert compute_cosine_attribution_drift(raw, -raw) == pytest.approx(0)
    assert compute_cosine_feature_drift(
        np.array([[1.0, -2.0]]), np.array([[-1.0, 2.0]])
    ) == pytest.approx(2)


def test_summary_permutation_and_padding_invariance_with_true_zero_tokens():
    raw = np.array([[2.0, 0.0, -1.0, 90.0]])
    mask = np.array([[1, 1, 1, 0]])
    summary = summarize_attributions(raw, document_mask=mask, top_k=1)
    permutation = [3, 1, 0, 2]
    np.testing.assert_allclose(
        summarize_attributions(raw[:, permutation], document_mask=mask[:, permutation], top_k=1),
        summary,
    )
    np.testing.assert_allclose(
        summarize_attributions(
            np.array([[2.0, 0.0, -1.0, 0.0, 0.0]]),
            document_mask=np.array([[1, 1, 1, 0, 0]]),
            top_k=1,
        ),
        summary,
    )
    without_zero = summarize_attributions(raw, document_mask=np.array([[1, 0, 1, 0]]), top_k=1)
    assert summary[0, 3] == pytest.approx(1)
    assert without_zero[0, 3] == pytest.approx(1.5)
    assert summary[0, 1] > without_zero[0, 1]
    np.testing.assert_allclose(summarize_attributions(raw[:, :3], top_k=1), summary)
    np.testing.assert_array_equal(summarize_attributions(np.zeros((2, 3))), np.zeros((2, 8)))
    np.testing.assert_array_equal(
        summarize_attributions(raw, document_mask=np.zeros_like(mask)), np.zeros((1, 8))
    )


@pytest.mark.parametrize(
    "raw,mask",
    [
        (np.array([1.0, 2.0]), None),
        (np.array([[np.nan]]), None),
        (np.array([[np.inf]]), np.array([[0]])),
        (np.ones((1, 2)), np.ones((1, 1))),
        (np.ones((1, 2)), np.array([[1, 2]])),
    ],
)
def test_summary_rejects_invalid_inputs(raw, mask):
    with pytest.raises(ValueError):
        summarize_attributions(raw, document_mask=mask)


def test_default_scores_are_invariant_to_feature_units_including_constants():
    baseline = np.array([[1.0, 10.0, 3.0, 0.0], [2.0, 30.0, 3.0, 0.0], [4.0, 50.0, 3.0, 0.0]])
    current = baseline + np.array([1.0, 20.0, -6.0, 7.0])
    units = np.array([1000.0, 0.001, 50.0, 0.01])
    original = compute_all_metrics(baseline, current, metrics=MONITORED_METRICS)
    rescaled = compute_all_metrics(baseline * units, current * units, metrics=MONITORED_METRICS)
    assert rescaled == pytest.approx(original)
    assert compute_max_wasserstein_drift(baseline * units, current * units) == pytest.approx(
        original["max_wasserstein"]
    )
    state = FittedMetricState.fit(baseline)
    assert state.constant_zero == (False, False, False, True)
    assert state.transform(current)[0, 3] == 1
    assert state.transform(current * np.array([1.0, 1.0, 1.0, 1e6]))[0, 3] == 1


def test_reference_edges_retain_outliers_and_do_not_refit_from_window():
    baseline = np.linspace(0, 1, 101).reshape(-1, 1)
    state = FittedMetricState.fit(baseline, n_bins=5)
    saved = state.to_dict()
    current = baseline.copy()
    current[-1, 0] = 1e9
    value = compute_jsd(baseline, current, n_bins=5)
    assert 0 < value < 0.02
    assert value == pytest.approx(compute_jsd(baseline, current, metric_state=state))
    assert state.to_dict() == saved
    for outside in (-1e9, 1e9):
        outliers = np.full_like(baseline, outside)
        assert compute_jsd(baseline, outliers, metric_state=state) == pytest.approx(np.log(2))
        assert compute_max_jsd(baseline, outliers, n_bins=5) <= np.log(2)
    assert state.reference_histograms[0][0] == 0
    assert state.reference_histograms[0][-1] == 0
    assert compute_jsd(baseline, baseline, metric_state=state) == pytest.approx(0)


def test_constant_support_distinguishes_both_departure_directions():
    baseline = np.array([[0.0, 3.0], [0.0, 3.0]])
    state = FittedMetricState.fit(baseline)
    assert state.reference_histograms == ((0.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    for current in (baseline + 1, baseline - 1):
        assert compute_jsd(baseline, current, metric_state=state) == pytest.approx(np.log(2))
    assert compute_cosine_feature_drift(np.zeros((2, 2)), np.ones((2, 2))) == pytest.approx(1)
    assert compute_cosine_feature_drift(np.zeros((2, 2)), np.zeros((2, 2))) == 0


def test_fitted_state_persistence_identity_and_tampering():
    reference = np.array([[1.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    window = np.array([[2.0, -10.0], [8.0, 0.0]])
    detector = DriftDetector(reference, feature_names=("signed", "constant_zero"), n_bins=4)
    manifest = json.loads(json.dumps(detector.metric_state.to_dict(), allow_nan=False))
    state = FittedMetricState.from_dict(manifest)
    restored = DriftDetector(reference.copy(), metric_state=state)
    assert restored.evaluate_window(window, metrics=MONITORED_METRICS) == detector.evaluate_window(
        window, metrics=MONITORED_METRICS
    )
    assert restored.feature_drivers(window) == detector.feature_drivers(window)
    assert state.fingerprint == detector.metric_state.fingerprint
    altered = copy.deepcopy(manifest)
    altered["scales"][0] *= 2
    with pytest.raises(ValueError, match="fingerprint"):
        FittedMetricState.from_dict(altered)
    with pytest.raises(ValueError, match="identity"):
        DriftDetector(reference + 1, metric_state=state)
    reference[:] = 100
    assert restored.evaluate_window(window, metrics=MONITORED_METRICS) == detector.evaluate_window(
        window, metrics=MONITORED_METRICS
    )
    with pytest.raises(ValueError):
        detector.baseline[0, 0] = 10


def test_monitor_reports_drivers_without_changing_metric_names():
    baseline = np.array([[1.0, -1.0], [2.0, -3.0], [4.0, -2.0]])
    detector = DriftDetector(baseline, feature_names=("positive", "negative"))
    monitor = DriftMonitor(detector, [baseline, baseline + 0.01, baseline - 0.01])
    result = monitor.evaluate(baseline + np.array([0.0, 5.0]))
    assert set(result["metrics"]) == set(MONITORED_METRICS)
    assert set(result["feature_drivers"]) == {"positive", "negative"}
    assert max(d["jsd"] for d in result["feature_drivers"].values()) == pytest.approx(
        result["metrics"]["max_jsd"]
    )
    assert max(d["wasserstein"] for d in result["feature_drivers"].values()) == pytest.approx(
        result["metrics"]["max_wasserstein"]
    )
    assert result["metric_state_fingerprint"] == detector.metric_state.fingerprint


@pytest.mark.parametrize("values", [np.ones(3), np.empty((0, 2)), np.array([[np.nan]])])
def test_fitted_metrics_reject_invalid_reference(values):
    with pytest.raises(ValueError):
        DriftDetector(values)
