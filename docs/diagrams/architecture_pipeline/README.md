# Architecture Pipeline Diagram

Source: `diagram.mmd`

## Purpose

This diagram shows the end-to-end production flow for explanation drift monitoring, from baseline/window data through SHAP attribution generation, drift scoring, and tiered alerting. It also shows where offline lead-time analysis fits.

## How to read it

1. `Data and Model`: Baseline (`X_b`) and current window (`X_t`) are fed into the trained model.
2. `Attribution Generation`: SHAP attributions are computed for baseline (`A_b`) and current window (`A_t`).
3. `Drift Scoring`: `DriftDetector` compares `A_t` against `A_b` and emits monitored metrics.
4. `Tiered Alerting`: `DriftMonitor` applies calibrated thresholds and stateful logic to produce `AlertLevel`.
5. `Lead-Time Analysis (offline)`: Alert/metric time series are aligned with accuracy time series to compute detection lead time.

## Code links

- `drift/detector.py`
- `monitoring/drift_monitor.py`
- `monitoring/algorithms.py`
