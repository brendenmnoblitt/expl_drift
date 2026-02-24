# Class-Conditional Flow Diagram

Source: `diagram.mmd`

## Purpose

This diagram documents how class-conditional monitoring combines pooled monitoring with per-class monitors and returns a single worst-case alert level.

## How to read it

1. Current-window attributions and predictions are partitioned by predicted class.
2. The pooled monitor is always evaluated on all samples.
3. For each class, evaluation only runs if a calibrated class monitor exists and sample count meets `min_class_samples`.
4. Skipped classes are recorded.
5. Final severity is aggregated using the worst severity across pooled and eligible per-class results.

## Code links

- `monitoring/class_conditional.py`
- `monitoring/drift_monitor.py`
