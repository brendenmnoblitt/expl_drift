# expl_drift

Core library for explanation-drift detection and tiered alerting.

## Scope

- `expl_drift/`: production-facing drift metrics, detector, and alert monitor.
- `expl_drift_experiments/` (sibling repo): experiment runners, notebooks, and results artifacts used for paper analysis.

## Method Summary

1. Compute SHAP attributions per window.
2. Compare current window attributions to baseline.
3. Compute drift metrics.
4. Alert with stateful thresholds and hysteresis.
5. Estimate offline lead time with threshold-based `compute_detection_lead_time`.

Production alerting metrics:

- `cosine_drift`
- `max_jsd`
- `max_wasserstein`

Archived CUSUM/EWMA experiments are retained as non-runtime reference code only.

See [ALGORITHMS.md](/home/brendenadm/projects/expl_drift/docs/ALGORITHMS.md) for formal definitions.

## Environment

- Install pinned dependencies with `pip install -r requirements.txt`.
- Use Python `>=3.12,<3.14` for parity with the current supported environment.
- Run lint checks with `make lint`.

## Paper Readiness Checklist

- [x] Algorithm description matches implementation (`ALGORITHMS.md`).
- [x] Tiered alert logic documented with state machine diagram (`docs/diagrams/alert_state_machine/diagram.mmd`).
- [x] Detection metrics and lead-time logic implemented in code.
- [ ] Repositories committed and tagged for a citable snapshot.
- [ ] Final manuscript tables/figures frozen to a named run ID.

## Key Code References

- Drift metrics: [metrics.py](/home/brendenadm/projects/explanation-drift-mlops-demo/expl_drift/drift/metrics.py)
- Drift detector: [detector.py](/home/brendenadm/projects/explanation-drift-mlops-demo/expl_drift/drift/detector.py)
- Alert monitor: [drift_monitor.py](/home/brendenadm/projects/explanation-drift-mlops-demo/expl_drift/monitoring/drift_monitor.py)
- Lead-time methods: [algorithms.py](/home/brendenadm/projects/explanation-drift-mlops-demo/expl_drift/monitoring/algorithms.py)
- Class-conditional monitoring: [class_conditional.py](/home/brendenadm/projects/explanation-drift-mlops-demo/expl_drift/monitoring/class_conditional.py)

## Citation

- License: [LICENSE](/home/brendenadm/projects/explanation-drift-mlops-demo/expl_drift/LICENSE)
- Citation metadata: [CITATION.cff](/home/brendenadm/projects/explanation-drift-mlops-demo/expl_drift/CITATION.cff)
