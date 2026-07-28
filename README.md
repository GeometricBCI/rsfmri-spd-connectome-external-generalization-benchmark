# rs-fMRI SPD Connectome External-Generalization Benchmark

This repository contains the research code for *Benchmarking External
Generalization of SPD Matrix Learning for Resting-State fMRI Connectome
Prediction*. The project asks a practical question: how well do models trained
on one collection of resting-state fMRI connectomes generalize to data acquired
in a different cohort?

The current benchmark predicts chronological age from symmetric
positive-definite (SPD) functional-connectivity matrices. It supports:

- SPDNet;
- tangent-space Ridge regression;
- vectorized-correlation Ridge regression (`corr_ridge`);
- mean and median dummy baselines;
- subject-grouped K-fold evaluation;
- leave-one-dataset-out (LODO) evaluation.

Classification, random forest, XGBoost, and Schaefer-400 experiments are not
implemented in the stable command-line interface.

## Quick start

Python 3.11 or newer is required. Create an isolated environment and install
the project:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Inspect a complete run before loading any data:

```bash
python run_benchmark.py \
  --config configs/examples/synthetic_dry_run.yaml \
  --dry-run
```

The dry run resolves the configuration, checks the required logical filenames,
and prints the model × cross-validation × harmonization plan. It does not
deserialize pickle files, import the training stack, create output
directories, or fit a model. In a fresh clone it will report the six expected
input files as missing and exit with status 2; that confirms the configuration
without implying that participant data were bundled with the repository.

A typical LODO Ridge run is:

```bash
export PREPARED_DATA_ROOT="<prepared-data-directory-outside-this-checkout>"
export BENCHMARK_RESULTS_ROOT="<benchmark-results-directory>"

python run_benchmark.py \
  --task regression \
  --target Age \
  --model ridge \
  --cv lodo \
  --harm none \
  --atlas schaefer_100 \
  --datasets cobre adnidod camcan abide oasis3 adni \
  --seed 1 \
  --input-root "$PREPARED_DATA_ROOT" \
  --output-dir "$BENCHMARK_RESULTS_ROOT"
```

Run `python run_benchmark.py --help` for the stable interface. The historical
specialist entry points remain available for reproducing established
experiments:

```bash
python run_single_dataset_benchmark.py --help
python run_pooled_benchmark.py --help
python run_spdnet_ablation.py --help
python prepare_fmri_datasets.py --help
python make_results.py --help
```

## Data

Participant data are deliberately kept out of Git. The tracked GitHub
repository does not contain raw MRI, ROI time series, participant identifiers,
phenotypic tables, or prepared benchmark pickle tables.

Prepared inputs are expected outside the checkout:

```text
<input-root>/
  atlas_schaefer_100/
    abide_X_y.pkl
    adni_X_y.pkl
    oasis3_X_y.pkl
    camcan_X_y.pkl
    cobre_X_y.pkl
    adnidod_X_y.pkl
```

Each pickle contains a trusted local pandas table with one scan or session per
row. Pickle can execute code while loading, so use only files you created
yourself or obtained from a source whose checksum and provenance you have
verified.

Dataset access and redistribution terms are handled separately:

- ADNI, ADNI-DOD, and OASIS-3 participant-level materials remain restricted;
- ABIDE, COBRE, and CamCAN derived connectomes may enter the separate Zenodo
  release only through the dataset-specific policy, metadata allowlist,
  source-binding, and privacy checks;
- 1000BRAINS and any unregistered dataset are non-public by default.

See [data availability](docs/data_availability.md) for the release boundary and
[data schema](docs/data_schema.md) for the internal table and connectome
contracts.

## Configuration

Configuration values are resolved in this order:

1. command-line argument;
2. YAML configuration;
3. `RSFMRI_SPD_*` environment variable;
4. documented local default.

Paths written in YAML are relative to the YAML file. Relative command-line and
environment paths are interpreted from the current working directory.

By default, prepared data live in the checkout sibling
`../rsfmri_spd_data/`, and benchmark outputs are written beneath the ignored
`results/benchmark_csv/` directory. External input and output locations can be
set explicitly. Inputs may never be placed inside the Git checkout; outputs
inside the checkout must remain under `results/`.

The supported YAML schema and environment variables are documented in
[configs/README.md](configs/README.md) and
[the reproducibility guide](docs/reproducibility.md).

## Scientific protocol

The release keeps the established scientific behavior explicit:

- rows represent scans or sessions, while split groups represent subjects;
- pooled subject keys are prefixed by dataset to prevent cross-cohort
  collisions;
- outer K-fold, validation, and Ridge tuning are subject-grouped;
- LODO holds out one complete dataset;
- tangent references and ComBat models are fitted on outer-training data;
- LODO harmonization is source-only and leaves the unseen dataset
  untransformed;
- connectomes use per-scan OAS covariance, symmetrization, regularization,
  correlation normalization, and final SPD jitter;
- legacy metric names and fold labels are retained for result compatibility.

The historical grouped-K-fold harmonization protocol uses known test age as a
biological covariate when applying the learned transform. It is therefore
target-informed and should not be described as deployment-style, label-free
evaluation. Changing that behavior would define a new experiment rather than a
software-only refactor.

More detail is available in the
[reproducibility guide](docs/reproducibility.md) and
[repository audit](docs/repository_audit.md).

## Outputs

Benchmark CSVs are written to the selected output directory. Each result has a
path-redacted metadata sidecar, and the stable CLI writes a deterministic
`run_config_<fingerprint>.json` describing the logical experiment and available
Git revision.

Harmonization caches and model checkpoints are working artifacts. They can
contain scan-level derived information and are not part of the public release.
Aggregate tables and figures can be generated with `make_results.py` after the
underlying benchmark outputs have been reviewed.

## GitHub and Zenodo

GitHub and Zenodo serve different purposes:

- GitHub contains evolving source code, tests, configuration examples, and
  documentation;
- Zenodo receives a frozen dataset archive built from explicitly approved
  derived data and reviewed metadata.

The local `zenodo_upload/` work area is ignored by Git. The release tools
validate source bindings, metadata allowlists, privacy rules, matrix structure,
manifests, and checksums before creating an upload archive. They do not publish
a Zenodo record, reserve a DOI, or create a GitHub release.

See [the Zenodo release guide](docs/zenodo_release.md) for the controlled export
and packaging workflow.

## Testing

The public test suite uses synthetic or mock inputs only:

```bash
python -m pip install -r requirements-dev.txt
python -m compileall -q spd_connectome_benchmark tools *.py
python -m pytest
python run_benchmark.py --help
```

CI never downloads participant data, opens a real prepared pickle, contacts
Zenodo, or runs a full scientific benchmark.

## License and citation

Repository source code is released under the
[BSD 3-Clause License](LICENSE). Dataset-specific terms are documented
separately and are not replaced by the software license.

The final paper citation and Zenodo DOI will be added only after the relevant
records exist; they are not inferred in advance.

## Development assistance

The codebase, test structure, release tooling, and documentation were organized
and edited with assistance from OpenAI Codex. Scientific decisions, data-use
permissions, interpretation of results, and final review remain the
responsibility of the project authors. Codex is not an author, creator, rights
holder, designated human reviewer, approver, or source of dataset permission.
