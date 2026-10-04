# HF endpoint infrastructure TODO

Set up remote classifier signal extraction for studies in
`expl_drift_experiments`, with `expl_drift` as the authoritative implementation
of drift metrics, calibration, alerting, and lead-time calculations.

This checklist is saved in `expl_drift` for planning. Implementation belongs in
`expl_drift_experiments` unless an item explicitly names the core library.
No image publication or endpoint deployment has been performed as part of
this plan.

## Preserve the documented baseline

Keep this work on dedicated `codex/` branches in both repositories. Do not merge
to `main` during development; merging requires a later explicit instruction
from the user. Published or manuscript-backed code, configurations, figures,
tables, and result artifacts must remain reproducible.

The starting core-library commit for this planning work is
`e2f528c710c2b971464c16996a9e8c4bcb3d526b`. This records the current checkout,
not a claim that it is the snapshot used by every existing manuscript.

- [x] Create the core planning branch: `codex/hf-infra-planning`.
- [x] Create the experiments development branch: `codex/hf-endpoint-infra`.
- [x] Record current commits, paper-tag revisions, qualification dependencies,
  tracked-file hashes, and existing test/lint results in the
  [infrastructure baseline](docs/hf-infra-baseline-2026-10-03.md).
- [ ] Identify and record the tags/commits, dependency environment, and run
  manifests supporting the existing writeups. Preserve those snapshots.
- [ ] Add the new study in dedicated files/directories and save outputs under
  new run IDs. Leave legacy experiment code, configurations, and manuscript
  artifacts intact.
- [ ] Keep proposed core behavior corrections separate from infrastructure
  changes. Record their effect on earlier calculations before incorporating
  them into a new study.
- [ ] Make image publication and HF deployment support the selected development
  commit without merging to `main`, creating a release, or moving paper tags.
- [ ] Before any eventual merge, review the diff against the preserved baseline
  and verify the existing documented behavior remains reproducible.

## Repository and dependency setup

- [x] Add and locally qualify a standalone endpoint input-contract and CPU
  container scaffold in the experiments branch. See the
  [scaffold qualification](docs/hf-endpoint-scaffold-2026-10-03.md).
- [ ] Add the endpoint service, Dockerfile, remote client, and build/deploy
  workflow to `expl_drift_experiments`.
- [ ] Pin the `expl_drift` commit installed by the experiment environment and
  container. Verify imports resolve to that installation rather than an
  incidental sibling checkout or stale package.
- [ ] Lock a compatible Python, PyTorch, CUDA, Transformers, and model SDK
  environment. Use the same dependency versions for local extraction checks
  and the remote container where practical.
- [ ] Adapt Scry's container, image-publication workflow, and request/provenance
  checks selectively. Keep the experiment runtime independent of Scry's
  package and QA engine.

## Extraction contract

- [ ] Define one versioned request/response contract covering sample IDs,
  document text, instructions, candidate IDs/descriptions, extraction method,
  attribution target, token limits, and the expected model revisions.
- [ ] Return decision scores/probabilities, requested attribution and
  representation arrays, and token/segment mappings. Specify array shapes,
  feature meanings, dtype, and candidate ordering.
- [ ] Validate identities, dimensions, finite values, and checksums when
  loading results. Reject incompatible representations before computing drift.
- [ ] Record the attribution target and representation settings so changing
  predictions or candidate order cannot silently change the quantity monitored.
- [ ] Keep drift calculations in the experiment runner through `expl_drift`;
  the endpoint returns model signals.

## Model loading and extraction

- [ ] Choose and pin the first model and tokenizer revisions. Start with one
  classifier; qualify CLM after the Laya path works.
- [ ] Implement Laya loading and extraction through its actual decision head.
  Verify score parity with its native SDK on a few fixed examples.
- [ ] Verify attribution extraction supports the selected model and target.
  Preserve gradient access for integrated gradients; advertise only methods
  that have passed qualification.
- [ ] Record document/instruction/candidate token boundaries and truncation.
  Fix serialization, token budgets, and candidate ordering explicitly.
- [ ] For CLM, pin the compatible Qwen backbone, tokenizer, projection-head
  checkpoint, pooling, and scoring settings. Define how all artifacts are
  obtained inside the HF container.
- [ ] Calculate shared confidence/margin baselines from the score distributions
  rather than assuming each model's native confidence field means the same thing.

## Container and endpoint service

- [ ] Build a Linux GPU image containing the experiment extraction service,
  pinned core library, and model dependencies.
- [ ] Configure model artifact loading from HF's `/repository` mount and any
  separately required pinned assets.
- [ ] Add readiness and extraction routes. Readiness follows successful model
  loading and reports configured revisions and qualified capabilities.
- [ ] Bound batch size, input length, request duration, and concurrent model
  access. Report truncation, unsupported requests, and extraction failures
  explicitly.
- [ ] Add CPU checks for the service contract and a small real-model extraction
  check before image publication. Reserve GPU qualification for the remote
  preflight.

## CI and image publication

- [x] Add an experiments workflow that checks the extraction service and
  builds the image on pull requests and development-branch pushes without
  publishing or deploying it. Locally qualified and committed on the development
  branch; hosted execution awaits a push. See the
  [CI qualification](docs/hf-endpoint-ci-2026-10-04.md).
- [ ] Configure registry credentials and permissions for an explicitly
  triggered image-publication workflow.
- [ ] Publish an image tagged by source commit and capture its immutable digest.
  Pin the base image and CI actions used for the build.
- [ ] Include the experiments commit, core-library commit, and dependency
  environment in image metadata and workflow output.
- [ ] Keep core-library CI in `expl_drift` focused on its calculations and
  regression checks. Add targeted checks there if adapter work exposes a core
  defect, and consume the corrected pinned revision in experiments.

## HF deployment and lifecycle

- [ ] Configure the HF deployment credential as a CI secret with access to the
  intended endpoint and model repositories; keep credentials out of artifacts
  and logs.
- [ ] Choose the research endpoint name, region, GPU, access mode, scaling,
  idle behavior, and bounded preflight budget. Verify registry pull access.
- [ ] Add an explicitly triggered deployment step that creates or updates a
  dedicated research endpoint using the published image digest and pinned model
  revision. Keep existing Scry endpoints separate.
- [ ] Wait for readiness with a bounded timeout and run a small extraction
  request. Fail deployment qualification if capabilities or outputs differ
  from the requested configuration.
- [ ] Save endpoint configuration and deployment identity. Compare observed
  configuration with requested pins; distinguish configured provenance from
  what the runtime can independently attest.
- [ ] Document how to restore the previous image/model configuration and how
  to pause or remove the research endpoint after a study.

## Remote execution and saved artifacts

- [ ] Add a small endpoint client to the experiments runner with authentication,
  explicit timeouts, and recorded request/response identities.
- [ ] Define bounded retry behavior. Treat timeouts as potentially completed
  requests; record attempts and avoid silently duplicating extraction work.
- [ ] Save completed extraction outputs atomically and resume from validated
  artifacts. Support analysis replay without new model calls.
- [ ] Select persistent artifact storage and verify upload/download integrity.
  Keep labels local for evaluation unless an extraction method requires them.
- [ ] Save a run manifest with code and model revisions, image digest, dataset
  fingerprint, sample/window IDs, extraction settings, array hashes, library
  versions, hardware, timings, and remote attempt counts.

## Consistency and completion checks

- [ ] Use fixed examples to compare native model scores with adapter outputs
  and local extraction with remote extraction at declared numeric tolerances.
- [ ] Replay identical saved arrays through the pinned `expl_drift` library
  locally and in the container. Verify metrics, calibrated thresholds, alerts,
  and lead-time results agree.
- [ ] Verify calibration uses only designated baseline windows and every
  detector receives the same sample IDs, windows, and evaluation protocol.
- [ ] Exercise wrong revisions, malformed arrays, interrupted requests, and
  resume behavior with small targeted checks.
- [ ] Complete a bounded GPU preflight on a few examples and record memory,
  extraction latency, attribution cost, and deployment configuration.
- [ ] Document setup, build, deploy, extract, replay, and shutdown commands in
  the experiments README. Link the completed preflight artifacts.

The infrastructure milestone is complete when one pinned classifier produces
qualified remote signals, saved results replay through `expl_drift`, and the
endpoint can be reproduced and shut down using the documented workflow.
Larger studies and additional classifier families follow that milestone.

## References

- Existing experiments: `../expl_drift_experiments/experiments/transformer_drift/`
- Scry container: `../scry/endpoint/Dockerfile`
- Scry publication workflow: `../scry/.github/workflows/release-endpoint.yml`
  (currently publishes an image; HF endpoint deployment needs a separate step)
- Scry extraction service: `../scry/src/scry/remote/hf_endpoint.py`
- [HF custom container documentation](https://huggingface.co/docs/inference-endpoints/en/guides/custom_container)
- [Laya model card](https://huggingface.co/convaiinnovations/laya)
- [CLM model card](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B)
