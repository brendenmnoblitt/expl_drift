# Modern-classifier explanation-drift research plan

Status: implementation started on the execution branches; G1–G5 remain pending.
The [first-session evidence](research-extension-session-2026-10-11.json) records
source identities, corrected data, verified development behavior and gate blockers.
The [checkpoint-preparation continuation](research-extension-checkpoints-2026-10-11.json)
adds pinned AG News, current-schema role/schedule artifacts, four new task
checkpoints and proper quality-measure implementation. G1–G5 are not closed.
This is not completed qualification or a promise of positive results.

Target delivery: November 1, 2026. Execution window: October 12–November 1.

## Branch strategy and carryover

Execute this extension on a new `codex/research-extension` branch in each repository, based on the existing development head rather than `main`. Preserve the infrastructure/pilot branches and paper tags; do not merge to `main` or move tags as part of this plan.

| Repository | Starting development branch | Reviewed HEAD | New execution branch |
| --- | --- | --- | --- |
| `expl_drift` | `codex/hf-infra-planning` | `4c84674` | `codex/research-extension` |
| `expl_drift_experiments` | `laya/remote-experiment-next` | `7ed291a` | `codex/research-extension` |

These HEADs record the branch-strategy review, not immutable bases after preparatory commits. Before branching, commit the reviewed research plan and its `TODO.md` reference on the core planning branch. Review the uncommitted experiments root/endpoint README changes separately before including them; preserve `results/laya-youtube-spam-calibrated/` as a historical pilot without blindly adding raw results to Git. Leave unrelated unreviewed changes out of those commits. Branch from the resulting development heads and record their full source revisions.

Carry forward:

- The amended plan, infrastructure checklist, baseline/qualification documentation and historical provenance manifests.
- Endpoint schemas/service, dependency lock, container and publication workflows.
- Remote extraction/client machinery, resumable subrequests, numerical integration, token alignment and saved-artifact handling.
- The existing core metrics, calibration, monitoring and event-evaluation implementation as the single starting engine for corrections.
- Historical experiment code and manifests for preserved reproduction and sensitivity comparisons.

Carryover means available for reuse and correction, not scientifically qualified for the new study. Candidate-logit IG still requires the selected margin-target qualification; pooled representations and all other required signals must pass G1. Cache acceptance must enforce the new exact fingerprints.

Do not carry Latin-1-transformed inputs, duplicate-unaware schedules, mixed-image or incomplete pilot caches into primary evidence. Do not treat pilot thresholds/targets/representations as the frozen protocol, relabel AG News heads as YouTube checkpoints, or present historical results as outputs of the corrected implementation. Preserve these historical artifacts separately; do not delete them or silently resume them.

Implement on the new execution branches until the later clean-repository cutover after stable contracts. This section specifies the branch procedure; it does not claim that preparatory commits or branch creation have occurred.


## Outcome and scope

Deliver a scientifically qualified extension of the tabular and BERT/GPT-2 studies, completed AG News and corrected YouTube Spam experiments, and a clean paper-facing repository with installable packages and offline reproduction of its tables and figures.

A successful experiment answers the question reliably. Ties, negative results, undetectable label-only drift, and streams without degradation remain reportable outcomes; superiority and positive lead time are not release requirements.

Primary question: for a fixed modern classifier and decision schema, does attribution monitoring provide useful advance warning or additional detection coverage beyond input, embedding, and prediction signals at a comparable false-alarm rate?

Required model matrix: Laya as the new classifier, with BERT and GPT-2 as reference families on matched tasks. This is one newly studied classifier family, not evidence across multiple newly added families. The existing CLM infrastructure proposal remains separate; qualifying an additional 8B model is not on this three-week critical path. Adding it requires an explicit scope and compute decision.

Do not compare archived AG News performance directly with new YouTube performance. Do not claim architecture-level accuracy superiority from zero-shot Laya versus task-fine-tuned reference models. Compare monitoring methods within each fixed model; identify training/deployment regimes when comparing across models.

## Evidence and constraints

- [Existing infrastructure checklist](../TODO.md): current implementation boundaries and preservation requirements.
- [Preserved baseline](hf-infra-baseline-2026-10-03.md): paper tags, source snapshots, and historical provenance limits. Saved legacy tabular manifests do not establish exact generating commits.
- Existing study code: `../expl_drift_experiments/experiments/laya_endpoint/`, `experiments/transformer_drift/`, and `endpoint/`.
- Existing core: `drift/summarize.py`, `drift/metrics.py`, `monitoring/drift_monitor.py`, and `monitoring/algorithms.py`.
- The reviewed Laya pilot has 24 examples. The calibrated cache has 832 of 1,000 planned examples, mixed endpoint images, duplicate comments, and Latin-1-transformed inputs. Preserve both as historical pilots; neither supplies the new primary evidence.
- BERT/GPT-2 weights were not found in the configured `experiments/transformer_drift/models/` directory or the supplied course-project directory. Recover identifiable artifacts or produce and pin new task checkpoints; do not assume the published checkpoints are available.
- Current package imports/runners rely on sibling checkouts and path injection. The new release must install and run without either.
- GPU time, spending ceiling, endpoint access, and extraction throughput are not yet established for the proposed matrix. Measure and approve them before full extraction; planning does not authorize publication, deployment, or paid jobs.

## Schedule and gates

| Dates | Work | Required exit evidence |
| --- | --- | --- |
| Oct 12–14 | Task 1 data repair and artifact recovery; Task 2 measurement design | Corrected, group-aware dataset inventories; checkpoint decision; written target/event definitions; reusable signal-bank design |
| Oct 15–18 | Complete Tasks 1–2; qualify all required model signals, extract reference/tuning banks and estimate compute | G1 extraction qualification and G2 frozen protocol, including null qualification, complete role allocations and fitted/tuned state; resource estimate and approved run ceiling |
| Oct 19–23 | Task 3 controlled AG News study | Complete required extraction bank, all scheduled replays, paired detector comparisons, and explicit null/negative controls |
| Oct 24–27 | Task 4 corrected YouTube replication | G4 complete second-corpus results; no threshold or representation retuning after viewing evaluation outcomes |
| Oct 28–30 | New-repository cutover, paper tables/figures, methods and limitations | Installed-package replay equivalent to frozen pre-cutover artifacts; complete claim-to-evidence mapping |
| Oct 31–Nov 1 | Independent clean-environment reproduction and corrections | G5 reproducible release candidate and manuscript-ready evidence bundle |

Prepare the YouTube split, binary reference checkpoints, and packaging layout while AG News extraction runs. Once extraction contracts are frozen, independent corpus/model extraction jobs may overlap if their measured cost is approved. Scientific dependencies still apply: G1 precedes reference/threshold-tuning extraction under the qualified contract; G1 and G2 precede held-out primary evaluation extraction. The final repository cutover follows stable contracts.

Use three explicit extraction stages:

1. **Qualification/development:** use designated development examples to check targets, representations, numerical alignment, transformation label preservation and extraction cost. These examples do not enter held-out evaluation.
2. **Reference and threshold-tuning:** after G1, extract designated reference/calibration and threshold-tuning banks. Fit scales/bin edges, calibrate complete detectors, and qualify the proposed quality-event rule against development/tuning null streams before G2. These results may inform the protocol; their role and every fitted/tuned artifact must be recorded.
3. **Held-out evaluation:** after G2, extract and replay the frozen held-out stable and shifted schedules. Keep their model outputs, detector results and quality outcomes sealed before protocol freeze. Source/group inventories may be inspected for allocation feasibility, but held-out performance cannot guide any definition, transformation, combination rule or threshold.

Any pre-G2 access to held-out model outputs, detector results or quality outcomes invalidates the affected held-out allocation for confirmatory use; replace it with untouched groups and record the amendment. Exact-fingerprint reuse of qualified signals within authorized roles remains permitted; changing a role does not erase prior access.

If the Oct 18 gate fails, publish the exact failed prerequisite and revised completion forecast before committing to the full run. Do not silently remove models, detectors, seeds, controls, or a corpus to preserve the date. An incomplete matrix remains incomplete, not a renamed successful study.

## Task 1 — Correct data and qualify one immutable runtime per model

### Work

- [x] Decode the YouTube CSVs as UTF-8, retaining source text other than documented normalization. Pin the archive checksum and record source-to-input transformations.
- [x] Resolve duplicate source IDs, identify exact-text/template groups, inspect conflicting labels, and preserve the source-video grouping. Save a machine-readable exclusion/group manifest.
- [ ] Build explicit training, reference/feature-fitting, calibration, threshold-tuning, held-out stable, and shifted evaluation roles. Keep duplicate/template groups out of incompatible roles across all schedules; prevent duplicate source groups across windows within a schedule except deliberately recorded paired controls. Record any permitted reference/calibration overlap and keep both separate from threshold-tuning and held-out evaluation.
- [ ] Hash the dataset, text, labels, sample/group IDs, window membership, task schema, and transformations. Save immutable sampling schedules before extraction.
- [x] Recover and identify the earlier AG News reference checkpoints or train new ones using only the designated training split. Record tokenizer, weights, training settings, and checkpoint hashes. Newly trained checkpoints are new baselines, not reproductions of the old weights.
- [ ] Qualify Laya on the actual AG News four-candidate task and YouTube binary task, rather than extrapolating from sentiment examples. Freeze instructions, candidate IDs/order/descriptions, token budget, precision, pooling, SDK and tokenizer revisions.
- [ ] Save raw logits/decision quantities, probabilities, attribution target, input and baseline target values, signed token values, token segments, truncation, completeness residual, actual gradient evaluations, latency, and runtime identity.
- [ ] Compare the reported score path with the attributed function on identical inputs. Use normal, ambiguous, long/truncated, short, and both-class/multiclass examples. Candidate-logit completeness alone does not qualify probability or margin explanation.
- [ ] Measure score/prediction/attribution reproducibility across batch sizes 1, 4, and 8, repeated requests, and singleton attribution versus batched scoring. Set numerical tolerances before the comparison; if interchangeability fails, fix the path or bind cache identity to the exact batch context and narrow the stated semantics.
- [ ] Apply equivalent target/alignment/numerical checks to BERT and GPT-2 extraction. Completeness is necessary but not a proof of explanation faithfulness; include controlled signed and token-permutation checks appropriate to the declared representation.
- [ ] Freeze a single endpoint image/runtime for each model's primary study. No mixed-image continuation. A changed implementation starts a new qualified revision and invalidates affected cached signals.
- Current endpoint fix (`expl_drift_experiments` commit `b9d69a1`): transport batch size no longer changes model execution context; scoring, representation, and fixed-margin IG use singleton collation per sample. Endpoint tests pass (50). Immutable image `sha256:4c47ca3bb84b75f84c4a4a3bfbe8db79923205f58602ce34dbd6abe415077725` was built by workflow [38389756154](https://github.com/brendenmnoblitt/expl_drift_experiments/actions/runs/38389756154) and configured on the existing L4 x1 endpoint without changing scaling.
  YouTube batch-stability run: 22/22 requests, 25,344 actual gradient evaluations (786,432 cap), no retries, and 17.9 minutes of runner wall time. Across eight development examples, batch sizes 1/4/8, and two repeats, prediction parity passed and all maximum absolute signal discrepancies were 0 at `atol=rtol=1e-5`. Report: `../expl_drift_experiments/results/research-extension-data/remote-laya-qualification-v3/youtube-2026-10-13-singleton-b9d69a1/qualification.json`.
  AG News batch-stability run: 66/66 requests, 47,232 actual gradient evaluations (2,359,296 cap), no retries, and 34.7 minutes of runner wall time. The same eight-example, three-target comparison had prediction parity and zero maximum absolute signal discrepancies at the same tolerances. Report: `../expl_drift_experiments/results/research-extension-data/remote-laya-qualification-v3/ag-news-2026-10-13-singleton-b9d69a1/qualification.json`.
  These runs report different checkpoint-artifact fingerprints (`35b599d63dfa3816f29a8c1280499560e4950b5a5ebf6b75d1437a2769e1abd1` for YouTube and `f81510fe53d225e1553f302b08fc724752b1c535cc6c0325320dd33dcb7f3c1a` for AG News) despite the same configured image and pinned model revision; keep them as separate runtime evidence, do not combine them, and investigate fingerprint stability before a same-runtime cross-task claim. Neither run establishes G1. No held-out evaluation ran; the endpoint is paused.
- [ ] Measure GPU-hour estimates from representative samples, including slow/high-evaluation and truncated cases. Record the spending ceiling, concurrency, timeouts, failure accounting, and shutdown procedure before primary runs.

### G1 acceptance

A saved qualification report identifies the exact function explained, checkpoint/runtime/environment, task and token semantics, tolerances and observed discrepancies. Completeness is independently recalculable from saved forward quantities. Every required signal passes its declared gates. Corrected data, group manifests, checkpoints and schedules are reproducible; old caches are not resumed under corrected text.

## Task 2 — Freeze the measurement and evaluation contract

### Work

- [ ] Choose a common decision target before qualification is finalized. Preferred primary target: fixed candidate margins against a designated reference candidate (spam minus ham for YouTube; fixed reference-class margins for AG News). Keep class ordering/reference identity constant. Do not switch the target with the current prediction.
- [ ] Implement and qualify the selected target in every adapter. The existing single-candidate-logit IG path is a pilot path until then; never relabel its cached outputs as margin attributions. If a different target is needed, decide and document it before primary results and amend the hypothesis accordingly.
- [ ] Freeze the primary attribution representation: per-fixed-target position-invariant summary features, with concentration/magnitude and directional statistics explicitly distinguished. Keep retained token count/truncation as separate input controls, not part of the primary explanation vector. Save signed raw attribution arrays for secondary analyses.
- [ ] Fit feature scales only on reference/calibration data. Normalize magnitude-sensitive summaries, or calibrate per-feature scores before aggregation; never take raw maxima across heterogeneous units. Define treatment of constant features. Freeze the transform and fit-data identities.
- [ ] Fit and save JSD histogram edges from reference/calibration data, with explicit underflow/overflow handling. Never rebuild edges from the monitored window. Add behavior checks for outliers, support shifts, and the JSD bound.
- [ ] Preserve one core implementation of metric scoring, calibration, stateful alerting, and event evaluation. Compare the old and corrected definitions on saved arrays as a sensitivity report; do not rewrite historical tables or move paper tags.
- [ ] Lock reference/calibration, threshold-tuning, held-out stable evaluation, and shifted evaluation periods. An illustrative 20-window schedule is window 0 reference, 1–4 calibration, 5–9 held-out stable, and 10–19 evaluation under increasing shift exposure; the no-shift control remains clean throughout. Do not treat five stable windows as sufficient false-alarm evidence by themselves.
- [ ] Calibrate the complete stateful detector, including metric combination, trend rules and hysteresis, against separate clean threshold-tuning streams. Target a 5% probability of any warning over the chosen null evaluation horizon; record the estimate and uncertainty. Freeze tuned settings before held-out evaluation. Report residual false-alarm differences rather than claiming perfect matching.
- [ ] Give false-alarm calibration its own evidence budget: freeze separate counts of threshold-tuning and held-out null horizons, horizon length, uncertainty method, and acceptable calibration error before tuning. Specify initialization, reference/calibration warm-up, retained history, reset boundaries and treatment of warnings already active at evaluation entry; use the same state convention for tuning and evaluation. Size the budget for the declared precision, not merely the ten primary schedules. Reuse signal banks for additional null replays without repeating IG. Report conditional finite-corpus schedule uncertainty separately from corpus-level uncertainty; resampled or shared-group horizons are not independent binomial trials. If comparability is not established, report detection/false-alarm tradeoffs rather than superiority at matched rates.
- [ ] Define the quality-event estimand before outcomes: an observed window-accuracy drop or evidence of underlying degradation. The proposed observed event is at least a 5-percentage-point accuracy drop from reference quality sustained for two windows; it is provisional until null qualification. Specify reference-quality estimation, grouping, uncertainty and the confirming timestamp at the second window, not retrospectively at the first. An observed-drop event alone does not establish underlying degradation; any underlying-degradation claim needs a prespecified uncertainty-aware decision rule.
- [ ] Qualify the quality-event rule on stable development/tuning null streams at the planned window sizes, including reference-estimation uncertainty and source-group dependence. Freeze its acceptable horizon-level null-event frequency before qualification and save observed frequencies and uncertainty. Two-window persistence alone is not qualification. Resolve an unacceptable rule before G2 without inspecting held-out outcomes; evaluate and report its null-event frequency again on held-out no-shift streams after freeze. Label lead times, misses and warnings without degradation according to the declared estimand; keep any alternative threshold/rule analysis separately labeled and never select it for favorable lead times.
- [ ] Report overall accuracy, balanced/standardized-prior accuracy, per-class recall, and proper scoring measures. Under class-prior shift, calculate the composition-only quality expectation from pre-shift class-conditional performance.
- [ ] Define detection as the first actual stateful WARNING/CRITICAL during the evaluation horizon. Lead time is the quality-event timestamp minus that detection timestamp. Record prior stable-period false alerts separately; do not credit a carried-over false warning as a new shift detection.
- [ ] Report positive, simultaneous and late detection, misses, false warnings without degradation, and absent quality events separately. Do not drop censored/undefined lead times and report only favorable detected cases.
- [ ] Lock primary comparisons and statistical analysis: paired methods on identical schedules; finite-corpus schedule variability distinguished from independent datasets or training seeds; group-aware uncertainty; multiplicity treatment for several superiority claims. No one-sample test against zero substitutes for a paired baseline comparison.
- [ ] Freeze the incremental-value comparison within each fixed model: **B** is the prespecified input + embedding + prediction baseline system; **B + A** retains those baseline components and adds the qualified attribution representation. Fix feature transforms, component/combination rules and tuning procedures before held-out evaluation; calibrate each complete system separately on the same tuning schedules to the common null target. Use paired held-out differences to assess added early warning or coverage with false-alarm uncertainty retained. Individual-signal comparisons are diagnostics, not substitutes for B versus B + A. A tie limits the claim to this qualified representation and baseline system, not attribution universally.

### G2 acceptance

A versioned protocol/configuration records all targets, representations, fitted transforms/bin edges, complete corpus role allocations, sample sizes, schedules, thresholds, event estimands, controls, comparisons, compute commitments and claim boundaries. It includes quality-event null qualification and reference uncertainty, separate tuning/held-out false-alarm evidence budgets and state conventions, and the paired B versus B + A analysis. Both corpora's designated reference/tuning artifacts and fitted state exist before freeze; held-out results do not. Smoke scenarios prove sign changes, unit changes, outliers, label-only controls, state transitions and censored outcomes have the intended meaning. No primary result is inspected before protocol freeze; later amendments are dated and require separate confirmatory evidence.

## Task 3 — Controlled AG News bridge study

### Work

- [x] Pin the AG News revision, labels and text construction. Use only the training split for training references; reserve grouped portions of the held-out split for tuning and final evaluation.
- [ ] Run the three fixed classifiers on the same text/label/window schedules, with matched document visibility where practical. Account for Laya's instruction/candidate overhead: equal total token limits do not imply equal retained document lengths. Report residual visibility differences and truncation-stratified results.
- [ ] Required conditions: held-out no shift; class-prior shift without label corruption; a within-class, length-controlled text shift; and a label-only negative control on identical inputs. Prefer graded exposure to a frozen, label-preserving transformation rather than changing its definition after viewing outcomes.
- [ ] Select and audit the length-controlled transformation on development data. Save exact edits, token-length changes and label-preservation evidence. Harmful performance changes are measured, not assumed; a shift without degradation is a valid specificity case.
- [ ] Required signals per model: qualified IG representation, document-length/input statistics, full probability vector and derived confidence/margin, and a qualified pooled representation. Define and pin Laya pooling; its existing endpoint does not yet qualify representation extraction.
- [ ] Evaluate individual signals, the prespecified baseline-only system B, and B + A under the same null false-alarm target and paired schedules. Report the incremental-value comparison required by Task 2, including achieved held-out false-alarm rates and uncertainty. Attention may be a labeled supplementary replication signal, not a substitute for missing primary IG or embeddings.
- [ ] Initial planning target: 10 predetermined schedules, 20 windows, 100 examples/window. Schedules are resamplings of a finite corpus, not ten independent datasets. Lock final sizes at G2 from cluster feasibility and measured extraction cost; no post-outcome downsizing.
- [ ] Build reusable per-example signal banks for unchanged texts and the frozen transformed variant. After batch invariance is qualified, reuse those signals across detectors, prior-shift/no-shift schedules and label-only controls. Do not repeat IG simply because the detector or schedule changed. Cache keys include text, model, schema, target, precision, tokenization and runtime.
- [ ] Produce full per-window/raw results, actual stateful event histories, quality outcomes, group-aware intervals and paired detector differences. Include failures and all planned cells.

### G3 acceptance

All frozen AG News cells are complete, including null controls and qualified embedding baselines. Every detector sees the same underlying examples and task outcomes. Figures/tables regenerate offline. Report held-out quality-event null frequencies, achieved detector false-alarm rates and the paired B versus B + A results. Claims distinguish transfer, early warning, additional coverage and superiority; superiority is stated only when paired evidence at comparable false-alarm rates supports it.

## Task 4 — Corrected YouTube second-corpus study

### Work

- [ ] Use the corrected/group-aware corpus from Task 1. Save the exclusions and the resulting effective counts by class and source video.
- [x] Produce separate binary-task BERT/GPT-2 reference checkpoints with training groups excluded from evaluation. The old four-class AG News heads cannot simply be relabeled ham/spam. Laya's zero-shot task schema is fixed and its different training regime is disclosed.
- [ ] Provisional allocation: 300 unique training examples, 200 separate threshold-tuning examples, and a 1,000-source-example bank partitioned by the role table below. The bank includes reference/calibration and held-out roles, not 1,000 shifted evaluation examples. Confirm every role's class/group feasibility before G2; do not promise 20 disjoint 100-example windows from this small deduplicated corpus. Training quality is measured, not guaranteed.
- [ ] Save the final allocation table with actual unique source-example and duplicate/template-group counts by class and video for every role, exclusions and all permitted reuse. Freeze role membership across schedules; resampling may change window placement within a role but cannot move groups between incompatible roles. Distinguish source texts from transformed variants and pair variants with their source role/group.
- [ ] Decide explicitly whether source video is a role-disjoint holdout unit or a retained grouping/uncertainty variable. Duplicate/template-disjoint roles do not imply video-disjoint roles. Record videos per role and bound the generalization claim accordingly; any video-held-out design must pass allocation feasibility with only five videos.
- [ ] Use 10 predetermined schedules and the same detector/event definitions as the bridge study. Refit reference-dependent transforms/calibration on this corpus as specified, but do not optimize definitions or thresholds against its shifted evaluation results.
- [ ] Run the same four condition types and required signals as Task 3. Label-only controls reuse saved extraction. Audit label preservation for within-class text edits on this corpus too.
- [ ] Analyze per-class quality, composition-only expected accuracy, source-video/template dependence, retained token length and truncation. Intervals must reflect shared texts/groups and the five-video corpus; do not claim production representativeness or independent environments.
- [ ] Report whether the AG News conclusions replicate, weaken, tie, or fail. Record training-overlap uncertainty for Laya rather than certifying an undisclosed training mix.

Starting allocation for the illustrative 20-window, 50-example schedule; these are planning targets, not verified effective counts:

| Role | Provisional unique source examples | Use and reuse boundary |
| --- | ---: | --- |
| Training | 300 | Train binary reference checkpoints; excluded from all other roles |
| Threshold-tuning | 200 | Tune full detectors and qualify null behavior before G2; excluded from reference/calibration and held-out roles |
| Reference/feature-fitting | 50 | Window 0; fit reference-dependent state; excluded from threshold-tuning and held-out roles |
| Calibration | 200 | Windows 1–4; designated calibration state; separate from reference in this starting allocation |
| Held-out stable | 250 | Windows 5–9; untouched until G2; measure pre-shift false warnings |
| Held-out evaluation | 500 | Windows 10–19; unchanged for no-shift controls or exposed to frozen shifts |
| Total | 1,500 | Includes the 1,000-example reference/calibration/held-out bank; transformed variants do not add unique source examples |

Repeated null replays from the tuning bank and paired reuse across frozen conditions/schedules must be declared with their dependence; they do not create new independent examples. Do not reuse a duplicate/template group across windows within a schedule. This allocation does not establish that 200 tuning examples support the required false-alarm precision or that every shift has feasible class counts. Resolve those constraints and approve any changed sampling design before G2, never after viewing evaluation outcomes.


### G4 acceptance

A complete second-corpus result bundle covers all required models, methods, conditions and schedules under the frozen protocol and verified role/group allocation. It reports held-out quality-event null frequencies, achieved detector false-alarm rates and paired B versus B + A replication results. No old transformed-text or mixed-image cache enters the primary results. Cross-corpus claims are supported by both bundles, with failures and sampling limits retained.

## Clean repository and interfaces

Decision: one new repository, retaining separately installable `expl_drift` and `expl_drift_experiments` packages plus the endpoint service. Do not create a second drift engine, general plugin framework, or compatibility-shim layer. Preserve the old repositories and paper snapshots as historical provenance.

Proposed layout and responsibilities:

| Path | Responsibility and dependency rule |
| --- | --- |
| `packages/expl_drift/` | Core feature transforms, fitted metric state, calibration, alerts and event evaluation; no dataset, endpoint, training or plotting dependency on import |
| `packages/expl_drift_experiments/` | Dataset/group manifests, task/model adapters, extraction orchestration, immutable study configs, offline replay and paper reports; depends on installed core |
| `services/endpoint/` | Existing versioned extraction contract extended for qualified decision quantities/representations; independent runtime and lock; returns model signals, not drift verdicts |
| `configs/`, `docs/`, `tests/` | Frozen protocols, scientific methods, release instructions and behavioral/integration checks |
| External artifact store/release assets | Checkpoint/raw-signal/result bundles with checksums, licenses and download manifests; no credentials or large weights silently embedded in Git |

The minimal runtime view has a host experiment CLI consuming dataset/signal artifacts, an authenticated GPU extraction service, and local offline core evaluation. Labels stay in host evaluation; the extraction service never chooses quality events or drift thresholds.

Freeze three interfaces during Week 1:

1. **Signal extraction:** extend the existing request/response schema, preserving explicit model/schema/target identities, ordered sample outputs, numerical semantics and runtime attestation limits. Cache/resume accepts exact fingerprints only.
2. **Monitor replay:** consume numeric feature arrays plus a saved representation/fit manifest; fit on designated reference data; sequential evaluation returns metrics, feature drivers and actual stateful events without model calls.
3. **Study/report:** consume signal banks, grouped schedules, labels and a frozen protocol; emit complete raw results, event outcomes, uncertainty, tables and figures. Fail loudly on absent primary signals/cells.

Proposed user commands are extraction, offline replay and paper report generation through one experiments CLI. They are future release requirements, not commands that exist today. Qualify package installation and command entry points without `sys.path` insertion, hard-coded home paths, sibling checkouts or implicit environment overlays. Keep model/training extras separate from core replay dependencies; endpoint and host locks may differ only where their artifact contract makes that difference explicit.

Start layout/packaging preparation after G2. Perform the final cutover on Oct 28–30: migrate active callers/configs/CI/docs once, remove obsolete active imports/path hacks, compare replay artifacts before and after relocation, and pin extraction-source and analysis-source revisions separately when they differ. Publishing a new hosted repository, creating a release, or merging preserved branches requires a separate explicit instruction.

### G5 release acceptance

- [ ] Both packages install in a clean environment; importing core requires no model SDK or heavy attribution stack.
- [ ] Endpoint deployment uses its locked environment and immutable image; a real qualified request and saved replay exercise the release path.
- [ ] A clean checkout obtains checksummed public/released artifacts and regenerates every primary table/figure without endpoint credentials or new inference.
- [ ] A bounded extract → save → replay → report smoke run exercises the installed CLI and real service; failure/resume and incompatibility cases are observable.
- [ ] Replay reproduces saved metrics, stateful events and event outcomes at declared tolerances; report artifacts identify complete run IDs and checksums.
- [ ] Source revisions, installed package versions, dependency locks, model/tokenizer/checkpoint hashes, hardware, requests, actual IG cost, timing, splits and protocol version are recorded. Dirty/uncommitted source is either prohibited for primary runs or content-hashed explicitly.
- [ ] Legacy snapshots remain preserved and documented. Historical provenance gaps are disclosed, not filled with inferred commits.
- [ ] Manuscript-ready methods, results, limitations and a claim-to-table/figure checklist exist. Valid negative results remain included; unknown training contamination and five-video limits remain explicit.
- [ ] Credentials/private identifiers are excluded from publication; artifact licenses and anonymization needs are reviewed; GPU services are paused/stopped and temporary endpoint settings restored.

## Risks and decisions before full execution

| Risk | Required decision or mitigation |
| --- | --- |
| Adaptive IG is much slower than ordinary inference, especially for multiclass margins | Measure every required target/model, deduplicate extraction through signal banks, approve compute/concurrency by Oct 18; revise scope only explicitly and before outcomes |
| Reference weights are unavailable or not identifiable | Recover verified weights early or train new pinned checkpoints; never claim exact old-model replication |
| Laya pooled embeddings or margin IG cannot be qualified | Fix the actual extraction prerequisite before primary runs; do not substitute unsupported signals or omit the baseline silently |
| Small YouTube corpus cannot meet group-disjoint allocation/class counts | Inventory before protocol freeze; obtain explicit approval for a revised sampling design if needed |
| Detector edits change historical results | Keep frozen paper tags and artifacts intact; publish a sensitivity bridge and new study revision |
| Packaging consumes scientific time | Prepare only concrete boundaries first; final migration after frozen signals, no framework redesign or extra model family |
| Unexpected ties or no degradation under a chosen shift | Report the observed result and limit the claim; do not retune shifts or discard schedules for positive leads |

## First execution session

1. Preserve run/source identities and inventory recoverable reference checkpoints.
2. Correct decoding and generate duplicate/group/class inventories and a complete role allocation, including reference/calibration and explicit source-video handling.
3. Extend saved forward quantities and implement the selected target/pooling qualification fixtures.
4. Profile representative real extraction and count unique signal-bank work for the complete proposed matrix.
5. After G1, extract the designated reference/tuning banks, fit and save measurement state, qualify the quality-event null behavior, and establish the separate false-alarm evidence budget. Freeze B versus B + A, the written protocol and resource approval at G2 only after those prerequisites pass; keep held-out evaluation sealed until then.

At plan drafting on 2026-10-11, this document authorized planning only; no new
deployment, paid extraction, repository publication, or legacy-result replacement
had been performed then. Later resource decisions and results are recorded below.

## Execution update — 2026-10-13

After explicit resource approval, the existing HF Laya endpoint was updated to
an immutable schema-2 image and its pinned runtime readiness was verified. The
bounded YouTube development attempt persisted 20 completed requests; request 21
timed out at its 900-second limit, and request 22 was not sent. Available batch
sizes 1 and 4 were exactly repeatable within each size, but attribution values
differed across batch sizes by up to 0.046; all 32 returned completeness checks
passed. No complete comparison report or G1 pass resulted. The endpoint was
paused, and G1–G5 remain open. Full run evidence:
`../../expl_drift_experiments/results/research-extension-data/remote-laya-qualification-v2/youtube-2026-10-13/`.
Any further extraction requires a fresh resource decision and output directory.
No held-out model outcomes were evaluated, and no null-budget or sampling
amendment was applied.
