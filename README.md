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
rsfmri-spd-benchmark \
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

rsfmri-spd-benchmark \
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

Use the package CLI for the supported interface:

```bash
rsfmri-spd-benchmark --help
python prepare_fmri_datasets.py --help
python make_results.py --help
```

## Project layout

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   └── examples/
│       └── synthetic_dry_run.yaml
├── docs/
│   └── reproducibility.md
├── spd_connectome_benchmark/
├── tests/
├── results/
├── tools/
└── release_templates/
```

This is intentionally kept small: the package code lives under
[spd_connectome_benchmark](spd_connectome_benchmark), the example run config is in
[configs/examples/synthetic_dry_run.yaml](configs/examples/synthetic_dry_run.yaml),
and the main write-up is in [docs/reproducibility.md](docs/reproducibility.md).

## Data and outputs

Participant data are not checked into Git. The benchmark expects prepared
connectome tables outside the repository checkout. Those files should be kept in
an external data directory and referenced through the config or CLI arguments.

The project writes output artifacts such as benchmark CSVs and model metadata to
an output directory, typically under `results/`.

## Configuration

The benchmark reads a single YAML config file passed through `--config`.
The example config is minimal and intentionally readable:

```bash
rsfmri-spd-benchmark --config configs/examples/synthetic_dry_run.yaml --dry-run
```

Configuration precedence is:

1. command-line option;
2. YAML value;
3. environment variable;
4. built-in local default.

## Usage

```bash
python -m pip install -r requirements-dev.txt
python -m compileall -q spd_connectome_benchmark tools *.py
rsfmri-spd-benchmark --help
```

## License

Repository source code is released under the [BSD 3-Clause License](LICENSE).
Dataset-specific terms are handled separately and are not replaced by the
software license.
