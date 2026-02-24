# Alert State Machine Diagram

Source: `diagram.mmd`

## Purpose

This diagram explains how active production metrics feed into `DriftMonitor.evaluate`, and how evaluation results map to `OK`, `WARNING`, and `CRITICAL` states.

## How to read it

1. Inputs (`A_b`, `A_t`) are transformed into production metrics via `compute_all_metrics`.
2. Active metrics (`cosine_drift`, `max_jsd`, `max_wasserstein`) and calibrated thresholds are evaluated together.
3. The state machine applies streak/cooldown logic to transition between severities.
4. Transitions use both immediate threshold breaches and persistence/clear conditions.

## Notes

- Trend-based prewarning can promote `OK -> WARNING`.
- Consecutive warning windows can promote `WARNING -> CRITICAL`.
- Cooldown and clear criteria can de-escalate alerts.

## Code links

- `monitoring/drift_monitor.py`
- `monitoring/constants.py`
