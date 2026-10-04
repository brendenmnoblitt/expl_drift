# Endpoint CI qualification — October 4 2026

The experiments branch `codex/hf-endpoint-infra` now contains
`.github/workflows/endpoint-ci.yml`. It runs for endpoint/workflow changes on
`codex/` branch pushes and pull requests. The single job installs the locked
standalone endpoint environment, checks Ruff and formatting, runs the contract
tests, builds Linux amd64, and starts the image's default Uvicorn command for an
HTTP smoke check. The job has a 15-minute limit and read-only repository access.
Checkout and Python setup actions are pinned to verified full commit SHAs.

The reusable smoke check lives in `endpoint/scripts/smoke_container.py`. It uses
only the Python standard library and synthetic model revisions. Startup retries
are bounded and handle connection resets while Uvicorn starts. The workflow
prints container logs and removes its test container on exit. The script is
excluded from the runtime image through `endpoint/.dockerignore`.

Local validation passed:

- Fresh isolated Python 3.12.13 environment installed with `uv sync --locked`.
- Endpoint contract suite: 13 passed; the existing upstream Starlette/AnyIO
  deprecation warning remains.
- Ruff lint and formatting: passed for all five source, test, and script files.
- Workflow syntax: passed actionlint 1.7.7; external ShellCheck and Pyflakes
  checks were disabled because those tools are not installed locally.
- Linux amd64 Docker build: passed, reusing cached unchanged runtime layers.
  Local image ID: `sha256:b16f5d24500c4e2fbc787c1d149dfacb22359d56036050ad7adfa0408c40fc29`.
- Real container smoke: liveness 200, readiness and valid extraction 503,
  mutable model revision 422. The test container was removed afterward.
- All 105 recorded legacy file hashes, both repositories' HEAD/main revisions,
  development branches, and paper-tag objects/commits match the baseline.

The workflow and scaffold were committed locally on `codex/hf-endpoint-infra` as
`bdc55b728983fe0c6f0313da2f59397c7103951d`. No hosted GitHub run has been
performed; that requires pushing the development branch.
No image was published, HF endpoint deployed, release created, or main branch
merged. The image still contains only the API scaffold; classifier loading and
numerical extraction qualification are subsequent work. Drift calculations
remain assigned to the experiment runner through a pinned `expl_drift` revision.
