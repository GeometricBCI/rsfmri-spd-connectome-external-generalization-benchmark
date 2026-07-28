# Reproducibility guide

## Scope

This release supports chronological-age regression with SPDNet,
tangent-space Ridge, CorrVec Ridge, and a dummy baseline. It supports
subject-grouped K-fold and leave-one-dataset-out evaluation. Unsupported
classification, RF, XGBoost, and Schaefer-400 variants are rejected rather than
silently approximated.

## Reproducible setup

1. Check out an exact commit on a dedicated branch or tag.
2. Use Python 3.11 or newer in a clean virtual environment.
3. Install the pinned direct dependencies from `requirements.txt`.
4. Record the OS, Python version, accelerator, and complete resolved dependency
   set for the final published run.
5. Obtain source datasets independently and reconstruct inputs outside Git.
6. Retain a private manifest with logical dataset/atlas names and checksums;
   never publish participant filenames or paths.
7. Resolve and inspect the complete run with `--dry-run`.
8. Run the benchmark and retain aggregate CSVs plus redacted metadata sidecars.

Example:

```bash
export PREPARED_DATA_ROOT="<prepared-data-directory-outside-this-checkout>"
export BENCHMARK_RESULTS_ROOT="<benchmark-results-directory>"

python run_benchmark.py \
  --config configs/examples/synthetic_dry_run.yaml \
  --input-root "$PREPARED_DATA_ROOT" \
  --output-dir "$BENCHMARK_RESULTS_ROOT" \
  --dry-run

python run_benchmark.py \
  --config configs/examples/synthetic_dry_run.yaml \
  --input-root "$PREPARED_DATA_ROOT" \
  --output-dir "$BENCHMARK_RESULTS_ROOT"
```

## Configuration resolution

Precedence is CLI, YAML, environment, default. YAML-relative paths resolve from
the configuration file; CLI/environment-relative paths resolve from the current
working directory. The local defaults are the checkout sibling
`../rsfmri_spd_data/` and the ignored `results/benchmark_csv/` tree.
`RSFMRI_SPD_DATA_ROOT` overrides the data root without modifying source code.
`RSFMRI_SPD_OUTPUT_ROOT` names the shared results root, whose
`benchmark_csv/` child is used for benchmark outputs.
`RSFMRI_SPD_BENCHMARK_OUTPUT_ROOT` is an exact benchmark-only override. A
non-editable installation anchors these defaults to the invocation directory
rather than writing into `site-packages`.

Data preparation rejects output and raw-data roots inside the checkout. The
stable CLI permits an in-checkout output only beneath `results/`; arbitrary
external output roots remain supported. Input and output roots may not overlap.

## Randomness

`seed` controls Python, NumPy, Torch, model initialization, dropout, validation
splits, and the ambient run-level DataLoader shuffle stream.
`data_shuffle_seed` separately controls prepared-row ordering. The refactor
does not add fold-specific DataLoader reseeding, preserving the established run
sequence. Exact GPU determinism can still depend on platform and Torch kernels.

Earlier pooled and ablation entry points parsed `--seed` without applying it
globally. The release now applies that seed. This is an intentional
reproducibility correction and can change numerical results relative to an
unseeded historical run.

## Preserved scientific boundaries

- scans are grouped by subject in every outer K-fold and validation split;
- pooled subject keys include the logical dataset name;
- LODO holds out a full dataset;
- tangent references and harmonization models fit on outer train only;
- Ridge hyperparameter selection remains group-aware;
- tangent features are still computed on full outer train before inner Ridge
  tuning, matching the historical code rather than introducing a new fully
  nested protocol;
- SPDNet harmonization still precedes its inner train/validation subdivision;
- LODO source-only harmonization never transforms the unseen target dataset;
- grouped-K-fold harmonization retains the historical known-test-age transform;
- SPD construction, matrix dtypes, metrics, early stopping, `drop_last`, and
  legacy CSV names remain otherwise unchanged.

The grouped-K-fold harmonization mode uses true test age as a smooth biological
covariate when applying a model fitted on outer train. It is target-informed and
must be labelled as such. Whether to replace it with a deployment-style
protocol is a human scientific decision and would constitute a new experiment.

## Cache and output integrity

Harmonization cache files include a schema version and SHA-256 signature over
the relevant in-memory arrays, split indices, metric, and test policy. Old or
incompatible caches are rejected. Covariate arrays are no longer duplicated
inside feature-cache files. Cache files remain non-public derived artifacts.

Result writers verify that each metric contains exactly one value per fold.
Metadata sidecars remove the working directory and redact path-valued command
arguments; environment fields record only whether a data-root variable was set.
The stable CLI writes a deterministic, path-free run manifest whose filename
contains a fingerprint over experiment selection, all resolved non-path legacy
runner parameters, and available Git revision state. This includes connectome,
Ridge, and SPDNet defaults that are not individually exposed by the stable CLI.

## Synthetic validation

The public CI:

- installs pinned dependencies on Python 3.11;
- compiles Python modules;
- runs only synthetic/mock pytest fixtures;
- checks CLI help;
- exercises dry run against empty filename placeholders that are never opened.

It never downloads data, reads secrets, contacts Zenodo, or performs a full
benchmark experiment.

## Result reconstruction limits

Figures and tables that depend on real benchmark CSVs or aggregate summaries
cannot be reconstructed from a clean Git clone until the reviewed, de-identified
result bundle is supplied by the release integrator. Do not fabricate missing
results. Track the exact configuration and commit for any supplied bundle.

The eight aggregate CSV/PDF artifacts already tracked under `results/` predate
this hardening work and were not opened during the data-safe audit. An
authorized release reviewer must inspect their contents and PDF metadata or
replace them with a reviewed release bundle before publication.
