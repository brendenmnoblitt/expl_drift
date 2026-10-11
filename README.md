# expl_drift

Core library for explanation-drift detection and tiered alerting. Background, initial experimentation, and results can be found in the paper: _Monitoring Explanation Drift: A Multi-Metric Framework for Early Detection of ML System Degradation_ (review in-progress)

## Scope

- `expl_drift/`: production-facing drift metrics, detector, and alert monitor.
- `expl_drift_experiments/` (sibling repo): experiment runners, notebooks, and results artifacts used for paper analysis.

## Method Summary

1. Compute signed attributions for a fixed model and fixed decision target.
2. Fit measurement state on the designated reference array, then compare each window without refitting.
3. Compute drift metrics.
4. Alert with stateful thresholds and hysteresis.
5. Estimate offline lead time with threshold-based `compute_detection_lead_time`.

Production alerting metrics:

- `cosine_drift`
- `max_jsd`
- `max_wasserstein`

Archived CUSUM/EWMA experiments are retained as non-runtime reference code only.

### Research-extension measurement contract

`summarize_attributions(raw, document_mask=mask, top_k=5)` returns a fixed
eight-column, position-invariant representation. The ordered
`SUMMARY_FEATURE_NAMES` distinguish concentration (`normalized_entropy`, `gini`,
`top_k_concentration`), magnitude (`mean_absolute`, `max_absolute`,
`std_absolute`), and direction (`signed_mean`, `signed_mass_balance`).
Normalized entropy divides Shannon entropy by log(document-token count).
Zero attribution mass and empty documents yield zeros. Retained token count and
truncation remain separate input controls, not primary explanation features.
Keep signed raw arrays for secondary analysis, and keep target identity, column
order, mask semantics, and `top_k` fixed rather than changing with predictions.

A document mask excludes padding and non-document segments while retaining true
zero-valued document tokens. Without a mask **every position is a document token**;
zero attribution is not evidence of padding. Inputs, including masked positions,
must be finite. This is a changed representation, not a relabeling of historical
six-feature summaries.

`DriftDetector(reference, feature_names=SUMMARY_FEATURE_NAMES)` fits one
`FittedMetricState` by default. Varying features are divided by reference
population standard deviation, without centering. Constant nonzero features use
their absolute reference value as scale. For constant-zero features, magnitude
units cannot be learned: departures become -1/0/+1, detecting support/direction
but not departure intensity. This convention avoids raw maxima across
heterogeneous units and is invariant to positive feature-unit rescaling.

JSD edges and reference histograms are fitted only on reference data. Each
feature has explicit below/above-reference-support bins (endpoints remain inside
support); constant features have below/equal/above bins. Direct `compute_jsd`
and `compute_max_jsd` use the same baseline-only fitting. JSD uses natural logs,
no pseudo-counts, and is bounded by log(2). Monitored windows never change edges.

The detector's `cosine_drift` is cosine distance between **signed, scaled feature
means**; a sign reversal is not erased. Both zero profiles yield 0, exactly one
zero profile yields 1. The separate `compute_cosine_attribution_drift` function
is an unscaled **raw absolute-attribution magnitude-profile diagnostic** that
intentionally cannot detect a pure sign flip. Direct Wasserstein summary scores
and detector maxima use fitted reference units; optional detector energy and
classifier scores use the same transformed features.

Save and restore reference measurement state without refitting:

```python
import json
from expl_drift.drift import DriftDetector, FittedMetricState, SUMMARY_FEATURE_NAMES

detector = DriftDetector(reference_summary, feature_names=SUMMARY_FEATURE_NAMES)
saved = json.dumps(detector.metric_state.to_dict(), allow_nan=False)
state = FittedMetricState.from_dict(json.loads(saved))
restored = DriftDetector(reference_summary, metric_state=state)
scores = restored.evaluate_window(window_summary)
drivers = restored.feature_drivers(window_summary)
```

State records a canonical float64 reference-array SHA-256 identity and a
fingerprint covering version, feature names, scales, constant-zero treatment,
bin edges, and reference histograms. Restoration rejects fingerprint or
reference-identity mismatches. The detector owns a read-only reference copy.
Persist role/sample identities and the full representation/target manifest
alongside this numeric state; array identity alone does not establish scientific
role separation.

`DriftMonitor.evaluate` retains the three metric return names and existing
stateful alert logic, and adds per-feature `feature_drivers` (JSD, normalized
Wasserstein, signed mean shift) plus `metric_state_fingerprint`. Drivers are
diagnostics, not independently calibrated warnings.

These changes implement local measurement prerequisites only. Existing threshold
mean/std calibration does **not** qualify a horizon-level false-alarm target,
nor do CPU fixtures qualify real model signals. Freeze and qualify the complete
detector on designated tuning/null streams before held-out evaluation. Preserve
paper snapshots and raw pilots unchanged; compare historical formulas against
corrected measurements on saved arrays in a separately labeled sensitivity
report, never silently rewrite historical tables or run the old formulas as a
second permanent engine.

## Environment

- Install pinned dependencies with `pip install -r requirements.txt`.
- Use Python `>=3.12,<3.14` for parity with the current supported environment.
- Run lint checks with `make lint`.

## Paper Readiness Checklist

- [x] Algorithm description matches implementation (`ALGORITHMS.md`).
- [x] Tiered alert logic documented with state machine diagram (`docs/diagrams/alert_state_machine/diagram.mmd`).
- [x] Detection metrics and lead-time logic implemented in code.
- [x] Repositories committed and tagged for a citable snapshot.
- [x] Final manuscript tables/figures frozen to a named run ID.

## Key Code References

- Drift metrics: [metrics.py](/home/brendenadm/projects/explanation-drift-mlops-demo/expl_drift/drift/metrics.py)
- Drift detector: [detector.py](/home/brendenadm/projects/explanation-drift-mlops-demo/expl_drift/drift/detector.py)
- Alert monitor: [drift_monitor.py](/home/brendenadm/projects/explanation-drift-mlops-demo/expl_drift/monitoring/drift_monitor.py)
- Lead-time methods: [algorithms.py](/home/brendenadm/projects/explanation-drift-mlops-demo/expl_drift/monitoring/algorithms.py)
- Class-conditional monitoring: [class_conditional.py](/home/brendenadm/projects/explanation-drift-mlops-demo/expl_drift/monitoring/class_conditional.py)

## Citation

- License: [LICENSE](/home/brendenadm/projects/explanation-drift-mlops-demo/expl_drift/LICENSE)
- Citation metadata: [CITATION.cff](/home/brendenadm/projects/explanation-drift-mlops-demo/expl_drift/CITATION.cff)
