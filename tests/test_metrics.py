"""Tests for explanation-drift metric behavior and edge cases."""

import numpy as np
import pytest

from expl_drift.drift.metrics import (
    compute_all_metrics,
    compute_classifier_drift,
    compute_cosine_attribution_drift,
    compute_energy_distance,
    compute_jsd,
    compute_ks_statistic,
    compute_max_jsd,
    compute_max_wasserstein_drift,
    compute_wasserstein_drift,
)


@pytest.fixture
def identical_distributions():
    rng = np.random.RandomState(42)
    data = rng.randn(200, 5)
    return data, data.copy()


@pytest.fixture
def different_distributions():
    rng = np.random.RandomState(42)
    baseline = rng.randn(200, 5)
    current = rng.randn(200, 5) + 5
    return baseline, current


@pytest.fixture
def single_feature_shift():
    rng = np.random.RandomState(42)
    baseline = rng.randn(200, 5)
    current = baseline.copy()
    current[:, 0] += 5
    return baseline, current


def test_jsd_near_zero_for_identical(identical_distributions):
    baseline, current = identical_distributions
    jsd = compute_jsd(baseline, current)
    assert jsd < 0.01
    assert 0 <= jsd <= np.log(2) + 0.01


def test_jsd_high_for_shifted(different_distributions):
    baseline, current = different_distributions
    assert compute_jsd(baseline, current) > 0.1


def test_max_jsd_captures_single_feature_shift(single_feature_shift):
    baseline, current = single_feature_shift
    max_jsd = compute_max_jsd(baseline, current)
    avg_jsd = compute_jsd(baseline, current)
    assert max_jsd > avg_jsd * 2


def test_cosine_detects_distribution_change(different_distributions):
    baseline, current = different_distributions
    assert compute_cosine_attribution_drift(baseline, current) > 0.0


def test_wasserstein_and_max_wasserstein_behave(single_feature_shift, different_distributions):
    b1, c1 = single_feature_shift
    b2, c2 = different_distributions
    assert compute_max_wasserstein_drift(b1, c1) > compute_wasserstein_drift(b1, c1)
    assert compute_wasserstein_drift(b2, c2) > 1.0


def test_energy_distance_increases_under_shift(different_distributions):
    baseline, current = different_distributions
    assert compute_energy_distance(baseline, current) > 0.1


def test_ks_significance_detects_large_shift(different_distributions):
    baseline, current = different_distributions
    ks_max, frac_sig = compute_ks_statistic(baseline, current)
    assert ks_max > 0.5
    assert frac_sig > 0.5


def test_classifier_drift_auc_high_for_shift(different_distributions):
    baseline, current = different_distributions
    assert compute_classifier_drift(baseline, current) > 0.9


def test_compute_all_metrics_keys_and_finite(identical_distributions):
    baseline, current = identical_distributions
    result = compute_all_metrics(baseline, current)
    expected_keys = {
        "jsd",
        "max_jsd",
        "cosine_drift",
        "wasserstein",
        "max_wasserstein",
        "energy_distance",
        "ks_max_statistic",
        "ks_fraction_significant",
        "classifier_auc",
    }
    assert set(result.keys()) == expected_keys
    assert all(np.isfinite(v) for v in result.values())


@pytest.fixture
def gradual_drift_scenario():
    rng = np.random.RandomState(42)
    drift = np.concatenate([
        rng.normal(0.1, 0.02, 5),
        0.1 + 0.08 * np.arange(1, 16),
    ])
    accuracy = np.concatenate([
        rng.normal(0.85, 0.005, 8),
        0.85 - 0.03 * np.arange(1, 13),
    ])
    return drift, accuracy


@pytest.fixture
def no_drift_scenario():
    rng = np.random.RandomState(42)
    drift = rng.normal(0.1, 0.02, 20)
    accuracy = rng.normal(0.85, 0.01, 20)
    return drift, accuracy


def test_threshold_detection_algorithm_detects_gradual_drift(gradual_drift_scenario):
    from expl_drift.monitoring.algorithms import compute_detection_lead_time

    drift, accuracy = gradual_drift_scenario
    assert compute_detection_lead_time(drift, accuracy) is not None


def test_threshold_detection_algorithm_avoids_false_alarm(no_drift_scenario):
    from expl_drift.monitoring.algorithms import compute_detection_lead_time

    drift, accuracy = no_drift_scenario
    assert compute_detection_lead_time(drift, accuracy) is None
