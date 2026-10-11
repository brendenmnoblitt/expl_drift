"""Consumer-visible event timing and censoring, not metric threshold proxies."""

import numpy as np
import pytest

from expl_drift.monitoring.algorithms import evaluate_stateful_events


def test_quality_event_uses_confirming_window_and_actual_state_onset():
    result = evaluate_stateful_events(
        ["ok", "ok", "warning", "warning", "critical"],
        np.array([0.9, 0.9, 0.9, 0.85, 0.85]),
        reference_accuracy=0.9,
        evaluation_start=2,
    )
    assert result["detection_timestamp"] == 2
    assert result["quality_event_timestamp"] == 4
    assert result["lead_time"] == 2
    assert result["outcome"] == "positive_lead"


def test_carried_stable_warning_is_not_shift_detection():
    result = evaluate_stateful_events(
        ["ok", "warning", "warning", "critical", "ok", "warning"],
        np.array([0.9, 0.9, 0.84, 0.84, 0.84, 0.84]),
        reference_accuracy=0.9,
        evaluation_start=2,
    )
    assert result["stable_warning_onsets"] == [1]
    assert result["warning_active_at_entry"] is True
    assert result["detection_timestamp"] == 5
    assert result["quality_event_timestamp"] == 3
    assert result["lead_time"] == -2
    assert result["outcome"] == "late_detection"


@pytest.mark.parametrize(
    "states,accuracy,outcome,quality,detection",
    [
        (["ok", "ok", "ok"], [0.9, 0.84, 0.84], "miss", 2, None),
        (["ok", "warning", "ok"], [0.9, 0.9, 0.9], "warning_without_quality_event", None, 1),
        (["ok", "ok", "ok"], [0.9, 0.84, 0.9], "no_warning_no_quality_event", None, None),
        (["ok", "ok", "warning"], [0.9, 0.84, 0.84], "simultaneous", 2, 2),
    ],
)
def test_censored_and_simultaneous_outcomes_are_retained(
    states, accuracy, outcome, quality, detection
):
    result = evaluate_stateful_events(
        states, np.array(accuracy), reference_accuracy=0.9, evaluation_start=1
    )
    assert result["outcome"] == outcome
    assert result["quality_event_timestamp"] == quality
    assert result["detection_timestamp"] == detection
    if quality is None or detection is None:
        assert result["lead_time"] is None


def test_quality_persistence_does_not_borrow_stable_window():
    result = evaluate_stateful_events(
        ["ok", "ok", "ok"], np.array([0.9, 0.84, 0.84]), reference_accuracy=0.9, evaluation_start=2
    )
    assert result["quality_event_timestamp"] is None


@pytest.mark.parametrize("accuracy", [np.array([0.9, np.nan]), np.array([0.9, 1.1])])
def test_invalid_quality_cannot_be_reported_as_absent_degradation(accuracy):
    with pytest.raises(ValueError):
        evaluate_stateful_events(
            ["ok", "warning"], accuracy, reference_accuracy=0.9, evaluation_start=1
        )
