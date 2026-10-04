# HF endpoint scaffold qualification

October 3, 2026. The second infrastructure increment adds a standalone service
under `expl_drift_experiments/endpoint/`, on `codex/hf-endpoint-infra`. The core
planning branch remains `codex/hf-infra-planning`. The implementation adds ten
files and changes no previously tracked files in either repository.

## Implemented behavior

The new package has its own project definition and committed dependency lock.
It avoids importing the legacy experiments package, model training code, and
plotting dependencies. Its CPU Docker base is pinned by digest, and the image
installs only the locked runtime dependencies.

The schema version 1 input contract requires full model/tokenizer commit SHAs,
ordered candidate descriptions, unique sample/candidate IDs, a corpus
fingerprint, and an explicit attribution target when requested. Unknown fields,
implicit scalar conversions, unsupported schema versions, mutable revisions,
and invalid targets are rejected. Input text is preserved exactly.

`/live` reports process liveness. `/health` returns HTTP 503 with
`extraction_available: false`, and valid `/extract` requests return HTTP 503.
No real or synthetic signal outputs are produced by the service. Model adapters
and successful signal-response schemas remain subsequent work.

## Validation

| Check | Result |
| --- | --- |
| New endpoint contract suite in installed source | 13 passed |
| New source Ruff lint and formatting | Passed |
| Existing core suite after installation | 34 passed |
| Existing experiments suite after installation | 13 passed |
| Linux amd64 image build | Passed |
| Actual Uvicorn startup and HTTP smoke in container | Passed |
| Original tracked-file hash verification | All 105 preserved |
| Local main references and paper tags | Preserved |

The live container smoke checked `/live` 200, `/health` 503, valid extraction 503,
and rejection of a mutable model revision with 422. The temporary container
removed itself after the test and exposed no host port. The new test client
emits one upstream Starlette/AnyIO deprecation warning; all tests pass.

Service tests used an isolated Python 3.12.13 environment. The container uses
Python 3.12.15. Existing suites used the qualification environment described in
the [baseline report](hf-infra-baseline-2026-10-03.md). Installed source paths
were checked explicitly so the endpoint tests did not accidentally exercise
only the temporary staging copy.

Local image tag: `expl-drift-endpoint:scaffold`.

Local image ID:
`sha256:b16f5d24500c4e2fbc787c1d149dfacb22359d56036050ad7adfa0408c40fc29`.
This is a local image identity, not a published registry digest.

Pinned base-image digest:
`sha256:54c85f3c47607a77f32adec749d3c81d1348bf25833671f512b26a9b6d778cb3`.

## Next increment

Add branch CI that runs the isolated contract checks and builds the container
without publishing or deploying it. Then qualify a pinned Laya adapter against
its native SDK on a few fixed examples. Define actual decision scores,
representation features, attribution extraction, and successful response
validation around that qualification. Keep readiness unavailable until the
advertised capabilities work. Publication and HF deployment follow in separate
increments.

No models were downloaded, images published, endpoints deployed, or changes
merged to main in this increment. The scaffold files were local and uncommitted
at qualification. On October 4, the scaffold and subsequent build-only CI were
committed locally on `codex/hf-endpoint-infra` as
`bdc55b728983fe0c6f0313da2f59397c7103951d`.
