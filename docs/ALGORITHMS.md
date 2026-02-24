# Algorithm Design
This document describes the implemented explanation-drift workflow used in this repository.
Archived experimental variants (for example CUSUM/EWMA) are intentionally excluded from the production runtime path.

## 1) Drift Metric Computation

The detector compares baseline SHAP attributions (`A_base`) against current-window attributions (`A_t`).

Production alerting uses three metrics:
- `cosine_drift`
- `max_jsd`
- `max_wasserstein`

Optional analysis metrics (not used by production alerting) include:
- `jsd`, `wasserstein`
- `energy_distance`
- `ks_max_statistic`, `ks_fraction_significant`
- `classifier_auc`

```text
Algorithm: Compute Drift Metrics

Input:
  A_base in R^(n_base x d)
  A_t in R^(n_t x d)
  metric set M

Output:
  metric dictionary D_t

Steps:
1. For each requested metric in M, compute score(s) from A_base vs A_t:
   - cosine_drift: cosine distance between mean absolute attribution profiles.
   - JSD metrics: per-feature histogram JSD, then mean (jsd) or max (max_jsd).
   - Wasserstein metrics: per-feature Wasserstein distance, then mean (wasserstein)
     or max (max_wasserstein).
   - energy_distance: multivariate energy distance.
   - KS metrics: per-feature KS test, return max statistic and fraction significant.
   - classifier_auc: logistic classifier AUC for distinguishing baseline vs current.
2. Return D_t.
```

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

## 3) Lead-Time Evaluation (Offline)

Lead time is computed by `compute_detection_lead_time` as a single value for one drift series and one accuracy series.

```text
Algorithm: Detection Lead Time

Inputs:
  drift_series d_1...d_T
  accuracy_series a_1...a_T
  threshold_std (default 2.0)
  smooth_acc_window (default 1)

Steps:
1. Smooth accuracy with centered rolling mean (window = smooth_acc_window).
2. Use first n_baseline = max(3, floor(T/4)) windows as baseline.
3. Drift threshold:
     tau_d = mean(d_baseline) + threshold_std * std(d_baseline)
4. Accuracy threshold:
     tau_a = mean(a_baseline_raw) - threshold_std * std(a_baseline_raw)
5. Find first indices:
     t_d   = first i with d_i > tau_d
     t_acc = first i with smoothed_accuracy_i < tau_a
6. If either index does not exist, return NA; else return t_acc - t_d.
```

Interpretation:
- Positive lead time: drift signal appears before accuracy degradation.
- Zero lead time: both occur in the same window.
- Negative lead time: accuracy drops before the drift signal.
