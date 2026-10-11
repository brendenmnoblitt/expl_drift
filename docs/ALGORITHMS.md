# Algorithm Design
This document describes the implemented explanation-drift workflow used in this repository.
Archived experimental variants (for example CUSUM/EWMA) are intentionally excluded from the production runtime path.

## 1) Drift Metric Computation

The detector compares finite numeric reference features (`A_base`) against current-window features (`A_t`). Raw signed attributions and fixed-target document summaries are distinct representations; preserve the representation manifest with the fitted numeric state.

Production alerting uses three metrics:
- `cosine_drift`
- `max_jsd`
- `max_wasserstein`

Optional analysis metrics (not used by production alerting) include:
- `jsd`, `wasserstein`
- `energy_distance`
- `ks_max_statistic`, `ks_fraction_significant`
- `classifier_auc`

For the research extension, document summaries have eight fixed columns:
normalized entropy, Gini, top-k absolute mass concentration, mean/max/std
absolute attribution, signed mean, and signed mass balance. Explicit document
masks retain true zeros and exclude non-document segments. Without a mask,
all positions are retained. Count/truncation are separate input controls.

`FittedMetricState` fits scales and finite histogram edges only on reference
data. Varying features use population standard deviation without centering;
constant nonzero features use absolute reference value. Constant-zero features
map departures to -1/0/+1: support/direction are measurable, departure intensity
has no learnable reference unit and is deliberately discarded. State is
JSON-serializable, immutable, reference-array-identified, and fingerprinted.
Restoration checks the fingerprint and the exact canonical float64 reference
array. Window evaluation never fits a transform or changes the reference edges.

```text
Algorithm: Compute Drift Metrics

Input:
  A_base in R^(n_base x d)
  A_t in R^(n_t x d)
  metric set M

Output:
  metric dictionary D_t

Steps:
1. Apply the saved reference-only feature transform to A_base and A_t.
2. For each requested metric in M, compute score(s):
   - cosine_drift: cosine distance between signed scaled feature means; both
     zero profiles give 0, exactly one zero profile gives 1.
   - JSD metrics: reference-fitted per-feature histograms with explicit underflow/
     overflow bins (reference endpoints stay inside), then mean or max.
     Constant support uses below/equal/above bins. No pseudo-counts; divergence
     in natural logs lies in [0, log(2)].
   - Wasserstein metrics: per-feature distance in fitted reference units, then
     mean or max; never a raw maximum across heterogeneous feature units.
   - energy_distance: multivariate energy distance of transformed features.
   - KS metrics: per-feature KS test on transformed features, returning max
     statistic and fraction significant.
   - classifier_auc: logistic classifier AUC using transformed features.
3. Return D_t.
```

The separate `compute_cosine_attribution_drift` API remains an unscaled
absolute-attribution magnitude-profile diagnostic; it intentionally erases pure
sign reversals and is not the default signed-summary cosine. Direct JSD and
Wasserstein score APIs use the same reference fitting as the detector and accept
reusable `metric_state`. Detector feature drivers expose per-feature JSD,
normalized Wasserstein, and signed mean shifts without adding alert metrics.
The monitor includes these diagnostics and the fitted-state fingerprint while
retaining its three metric names and its existing state machine.

These corrected definitions change historical scores. Preserve paper snapshots
and raw pilots, and produce a separately labeled saved-array sensitivity bridge
to the old formulas rather than relabeling historical tables or maintaining a
second engine. Local behavioral fixtures do not establish real-signal
qualification or a horizon-level false-alarm guarantee. The threshold calibration
below still requires complete-detector null tuning/qualification before a frozen
study may claim comparable false-alarm rates.

## 2) Stateful Tiered Alerting

`DriftMonitor` calibrates thresholds on baseline windows, then evaluates each new window with stateful escalation/de-escalation logic.

```text
Algorithm: Tiered Alerting

Inputs:
  monitored metrics M = {cosine_drift, max_jsd, max_wasserstein}
  baseline windows B
  parameters:
    prewarning_std, warning_std, critical_std
    min_consecutive
    prewarning_trend_consecutive, prewarning_min_delta_std
    warning_clear_consecutive

Calibration:
  For each metric m in M:
    mu[m], sigma[m] = mean/std over baseline-window metric values
    T_pre[m]  = mu[m] + prewarning_std * sigma[m]
    T_warn[m] = mu[m] + warning_std * sigma[m]
    T_crit[m] = mu[m] + critical_std * sigma[m]

Per window t:
1. Compute current metric values x[m].
2. Assign per-metric level:
   - CRITICAL if x[m] >= T_crit[m]
   - WARNING  if x[m] >= T_warn[m]
   - WARNING  if trend-prewarning rule is satisfied
   - else OK
3. Resolve overall level:
   - CRITICAL if any metric is CRITICAL and its prior warning streak >= min_consecutive
   - else WARNING if any metric is WARNING/CRITICAL
   - else WARNING if previous overall state was WARNING/CRITICAL and
     consecutive_no_signal < warning_clear_consecutive
   - else OK
4. Update histories and streak counters.
5. Emit overall alert level, per-metric levels, and metric values.
```

Trend-prewarning rule:
- Current value is at least `T_pre[m]`.
- Recent `prewarning_trend_consecutive` values plus current value are strictly increasing.
- Total rise over that sequence is at least
  `prewarning_min_delta_std * sigma[m]`.

## 3) Stateful Event Evaluation (Offline)

`evaluate_stateful_events` consumes actual sequential `DriftMonitor` alert
states and window accuracies, with explicit stable/evaluation boundaries and
an independently estimated reference accuracy. It never thresholds a metric
series to stand in for the monitor.

- Detection is the first new WARNING/CRITICAL onset in the evaluation horizon.
  A stable-period warning carried into evaluation is not credited; the state
  must clear before another warning can count. Stable warning onsets and the
  entry state are reported separately.
- The provisional observed quality event is a drop of at least 0.05 from
  reference accuracy for two consecutive evaluation windows. Its timestamp is
  the confirming second window; persistence starts at evaluation entry.
  Both parameters are explicit caller inputs to be qualified and frozen before G2.
- Lead time is quality-event timestamp minus detection timestamp. Positive,
  simultaneous, late, missed, warning-without-event, and neither-event outcomes
  are all retained. Missing timestamps produce undefined lead time, not zero.
- The function validates finite accuracies and boundaries. The observed-drop
  estimand is not evidence of underlying degradation. Group dependence,
  reference-estimation uncertainty, and horizon-level null-event qualification
  remain required study-level evidence.

`compute_detection_lead_time` remains the historical threshold-series analysis
used by preserved experiments. It is not the research extension's event rule;
do not use its centered smoothing or retrospective thresholds for new claims.
