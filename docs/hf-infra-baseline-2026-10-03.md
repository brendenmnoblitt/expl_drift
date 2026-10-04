# HF infrastructure baseline

October 3, 2026. The first infrastructure increment establishes a comparison
point before adding remote extraction. Existing tracked code and manuscript
artifacts remain unchanged. Both repositories are on development branches;
their local `main` references and paper tags are preserved.

The [baseline manifest](hf-infra-baseline-2026-10-03.json) records SHA-256 hashes
for all 35 tracked core files and 70 tracked experiments files, including the
tracked figures, tables, notebooks, and run manifest. It also records paper-tag
objects and resolved commits, current revisions, and qualification dependencies.
Untracked and ignored files, including local model weights, are outside this
inventory.

## Revisions and branches

| Repository | Development branch | Starting commit |
| --- | --- | --- |
| `expl_drift` | `codex/hf-infra-planning` | `e2f528c710c2b971464c16996a9e8c4bcb3d526b` |
| `expl_drift_experiments` | `codex/hf-endpoint-infra` | `255f22ceeaeac71e602cf5334fd2bee2c303b077` |

| Paper tag | Core commit | Experiments commit |
| --- | --- | --- |
| `paper-tabular-v1` | `909d95a1ebef663964829cf2322eb099f2000171` | `6262c238c1071c9b1a2f35cafe286d90c00df7e6` |
| `paper-transformer-v1` | `e2f528c710c2b971464c16996a9e8c4bcb3d526b` | `255f22ceeaeac71e602cf5334fd2bee2c303b077` |

## Existing checks

| Check | Result |
| --- | --- |
| Core pytest suite | 34 passed, 16.97 seconds |
| Experiments pytest suite | 13 passed, 11.39 seconds |
| Core Ruff 0.12.9 | 6 existing findings, exit code 1 |
| Experiments Ruff 0.12.9 | 66 existing findings, exit code 1 |

The lint findings precede infrastructure implementation. Core findings concern
imports and typing style. Experiments findings also include notebook imports,
unused variables/imports, and line lengths. They were recorded without modifying
legacy files. New infrastructure should pass its own lint checks; a future CI
workflow must account for this existing baseline without hiding new findings.

The core virtual environment uses Python 3.12.13, PyTorch 2.10.0, Transformers
5.10.1, and Captum 0.7.0. The experiments suite initially could not collect
because `requests` and then `seaborn` were absent. Qualification used the existing
core environment with a temporary import overlay for those dependencies and
Ruff; it did not modify the installed environment. The exact versions, including
the overlay's transitive dependencies, are in the manifest. This qualification
environment is not a proposed container dependency lock.

Both pytest suites were run through the core `.venv/bin/python`, with the common
parent directory on `PYTHONPATH`. The experiments run additionally used
`/private/tmp/expl-drift-baseline-extras` on `PYTHONPATH`. Bytecode and pytest
cache writes were disabled, and test/plotting caches used temporary directories.
The existing tests require no new model downloads or HF compute.

## Historical evidence limits

`results/latest_run.txt` identifies `stats_20260223_152707`. Its reproducibility
manifest contains null Git commit fields and reports an unborn HEAD when it
was created. The named paper tags therefore identify preserved code snapshots,
but do not independently prove which exact commits generated those saved
tabular results. The original manifest remains intact.

Passing unit tests and preserving file hashes do not reproduce the manuscript
tables or qualify Laya, CLM, Docker, CUDA, or HF Endpoints. Those checks belong
to subsequent increments.

## Next increment

Add a minimal extraction service and container scaffold in the experiments
development branch. First verify its request contract and dependency boundary
locally. Then qualify real-model extraction, image publication, and remote
deployment in separate increments. Keep all drift calculations in the pinned
core library, and keep merges to `main` deferred until explicitly requested.
