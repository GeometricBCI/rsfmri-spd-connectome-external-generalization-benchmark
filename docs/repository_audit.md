# Repository audit

Audit date: 2026-07-26

Scope: release-preparation branch, code and documentation only

Data-safety statement: no real `.pkl`, NIfTI, DICOM, BIDS participant folder,
participant metadata, or benchmark result payload was opened or inspected.

OpenAI Codex assisted with the repository inventory, refactor, test design,
release tooling, and editorial review documented here. The project authors
remain responsible for the scientific design, interpretation, dataset
permissions, privacy decisions, authorship, and publication approval. Codex is
not an author, creator, rights holder, designated human reviewer, approver, or
source of dataset permission.

This statement describes the initial code-only audit. On 2026-07-27, after
that audit, the project owner separately directed a controlled local conversion
of the ABIDE, COBRE, and CamCAN prepared tables. The trusted exporter read those
local inputs outside Git and wrote only allowlisted derived fields to an
ignored release workspace. That later conversion does not change the scope of
the historical audit above, and its output remains a draft until the release
report says otherwise.

## Structure at audit start

The repository contained five top-level entry points:

- `prepare_fmri_datasets.py`: optional source-data preparation;
- `run_single_dataset_benchmark.py`: within-dataset grouped K-fold;
- `run_pooled_benchmark.py`: pooled grouped K-fold and LODO;
- `run_spdnet_ablation.py`: pooled SPDNet architecture ablation;
- `make_results.py`: tables and figures.

Reusable code lived under `spd_connectome_benchmark/`:

- `config.py` for constants and default paths;
- `connectomes/` for OAS/correlation construction and CorrVec;
- `models/` for SPDNet;
- `benchmark_tools/runtime.py` for loading, splitting, metrics, and output;
- `benchmark_tools/harmonization.py` for tangent-space/ComBat operations;
- `analysis_outputs/` for aggregate tables and figures.

One smoke-test file and a pinned `requirements.txt` existed. There was no YAML
configuration loader, stable unified CLI, `pyproject.toml`, CI workflow,
data-schema document, data-availability policy, reproducibility guide, or
citation metadata.

## Existing entry points and input assumptions

Prepared input names were constructed as:

```text
<data-root>/atlas_<atlas>/<dataset>_X_y.pkl
```

The loader expected a pandas table with `SubjectID`, `TimeSeries`, and `Age`.
Each row represented a scan. Optional preparation/description fields included
session, sex, diagnosis/group, and site. Loading used pickle, so inputs must be
trusted local artifacts.

The six paper datasets were COBRE, ADNI-DOD, Cam-CAN, ABIDE, OASIS-3, and ADNI.
No operational user-specific absolute path, legacy parent-relative dataset
root, or 1000BRAINS path was present at audit time. Before this refactor,
prepared-data defaults were repository-local, with data-root environment
overrides. The release default is now the checkout sibling
`../rsfmri_spd_data/`, and the preparation script rejects data roots inside
Git. Output roots were CLI-configurable but named inconsistently across scripts.

## Existing outputs

Benchmark writers produced metric-by-fold CSVs:

- within-dataset `R1..RK`, `Avg`, `Time(sec)`;
- pooled grouped K-fold `R1..RK`;
- pooled LODO `TEST_<dataset>`;
- neighboring runtime metadata JSON;
- Torch checkpoints;
- harmonized feature/SPD `.npz` caches.

`make_results.py` produced tables and figures. Its dataset-description path also
produced scan-level metadata and an ABIDE excluded-participant table containing
identifiers and sensitive row-level fields. Those public-workflow exports
violated the release data-safety requirements.

## Hard-coded and inconsistent paths

No user-specific absolute path remained. Problems were instead:

- import-time default roots with no YAML layer;
- different output option names in single, pooled, ablation, and plotting CLIs;
- raw-source subdirectory conventions embedded in preparation code;
- a nonexistent generic “project data link” described in the old README;
- metadata sidecars recording full working directories, command paths, and
  environment data paths.

## Duplicated or obsolete code

- Three similar SPDNet training loops remain. They deliberately preserve
  historical dtype and model differences; consolidating them now would create
  unnecessary numerical risk.
- The ablation script duplicated harmonized-SPD cache logic and accepted weak
  legacy caches. It now delegates fresh/cache computation to the common signed
  cache helper.
- The preparation script contains an inherited COBRE fetcher and uses Nilearn
  private utilities. Replacing source acquisition code requires provenance and
  license review and was not guessed here.
- The `msdl_39` CLI option previously called a masker branch named only `msdl`;
  the registered alias is now consistent.

## Reproducibility risks found

1. pooled and ablation scripts parsed `--seed` but never applied the global
   Python/NumPy/Torch seed;
2. harmonization caches did not bind strongly to input contents or policy;
3. old SPD caches could omit the `apply_harm_to_test` field and still load;
4. metadata sidecars disclosed local paths;
5. result writers did not reject metric/fold length mismatches;
6. all SPDNet loaders use `drop_last=True`, so a training subset smaller than
   its batch size produced zero optimizer updates;
7. no CI or unified dry-run existed;
8. a public pytest could open real prepared data when an environment variable
   was set.

## Data-leakage and scientific-method risks

- Fresh LODO behavior is dataset-disjoint and source-only: the held-out target
  dataset is neither used to fit nor transformed by ComBat.
- Historical grouped-K-fold harmonization fits on outer train but uses true
  test ages when applying the transform to test. This is target-informed and
  not deployment-style evaluation. It is preserved and made explicit.
- Tangent reference/ComBat fitting occurs on full outer train before inner
  Ridge tuning or SPDNet validation selection. This is not a fully nested
  preprocessing design, but changing it would define a new method.
- Harmonization uses logical dataset label as `SITE`, not necessarily physical
  acquisition site.
- OAS failure falls back to `np.cov`; harmonization replaces non-finite
  transformed values with zero. These legacy behaviors remain and should be
  quantified in a future methods review.
- Checkpoint names and some historical result-discovery logic can mix or
  overwrite configurations. A canonical long-form result manifest remains
  desirable.

## Dependency and environment problems

Direct dependencies were pinned but runtime, preparation, plotting, and test
packages shared one file. Test tooling is now separated into
`requirements-dev.txt`. Some scientific dependencies remain platform-sensitive
and heavy. A minimal `pyproject.toml` defines Python 3.11+, the package, and the
unified console script while retaining pinned requirements as the authoritative
dependency files. A future lock strategy should cover supported CPU/GPU
platforms separately.

## Missing documentation found

- no explicit dataset redistribution policy;
- no internal scan/connectome/split/output schema;
- no end-to-end reproducibility guide;
- no safe dry-run or configuration precedence;
- no explanation of GitHub versus future Zenodo roles;
- no valid `CITATION.cff`, paper DOI, complete author list, changelog, release
  tag, or frozen result manifest.

The missing scientific citation fields cannot be invented and remain a human
release decision.

## Changes made on this branch

- added `run_benchmark.py` and a lightweight stable CLI;
- added CLI > YAML > environment > default resolution;
- added dataset/atlas registry and filename-only dry-run;
- added typed time-series, SPD-array, label, and split validation;
- moved split construction into a focused module with compatibility re-exports;
- made harmonization test policies explicit;
- strengthened cache schema/signatures and removed cached covariate duplication;
- made pooled/ablation run-level seeding effective without fold reseeding;
- added a clear failure for zero-batch `drop_last=True` runs;
- redacted runtime path metadata and validated fold counts;
- added deterministic, path-free logical run-configuration manifests;
- made dry-run reports use logical filenames while redacting local roots;
- removed scan-level and excluded-participant public outputs;
- removed the real-data-enabled pytest path;
- fixed the MSDL alias mismatch;
- expanded `.gitignore` for participant data, outputs, caches, and credentials;
- rejected in-project stable input roots and added a tracked-filename safety
  test for participant/raw-neuroimaging patterns;
- added pinned YAML support, packaging metadata, synthetic CI, configuration
  examples, synthetic tests, and release documentation.

## Deliberately preserved behaviors

- age regression only;
- six-dataset ordering;
- subject grouping and LODO dataset isolation;
- matrix construction and dtype differences between legacy runners;
- full-outer-train tangent reference before inner tuning;
- grouped-K-fold target-informed test transform;
- LODO source-only transform;
- SPDNet MSE objective, early-stopping semantics, and `drop_last=True`;
- Ridge parameter grid/scoring and legacy result layouts.

## Potential behavior changes

- pooled and ablation model randomness is now controlled by `--seed`, so new
  results can differ from historical unseeded runs;
- Windows-safe timestamps replace colon-containing timestamps;
- incompatible or legacy harmonization caches are recomputed;
- malformed inputs/results fail earlier with explicit errors;
- undersized training folds now fail rather than silently training zero batches;
- public result generation no longer emits scan-level or excluded-person files.

## Unresolved human decisions

1. confirm manuscript authors, DOI/status, software citation, release version,
   and changelog;
2. document exact upstream dataset versions, applications, DUAs, preprocessing
   snapshots, and redistribution decisions;
3. retain the confirmed public connectome scope for CamCAN, ABIDE, and COBRE,
   while separately reviewing every source-bound artifact, any additional
   metadata, and any fold-level result bundle before release;
4. decide whether grouped-K-fold target-informed harmonization should remain a
   primary reported protocol or be supplemented by a label-free protocol;
5. decide whether preprocessing should become fully nested;
6. decide whether to change `drop_last`, OAS fallback, or non-finite
   harmonization replacement in a new scientific version;
7. define a canonical long-form result schema and immutable input/result
   manifests;
8. review inherited data-fetch code and third-party attribution;
9. determine platform-specific dependency locks and supported accelerators;
10. supply the frozen, de-identified material needed by the separate Zenodo
    packaging workflow;
11. review the contents and PDF metadata of the eight aggregate artifacts
    already tracked under `results/`; they predate this hardening work and were
    intentionally not opened during the data-safe audit.
