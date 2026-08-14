# rs-fMRI SPD Connectome External-Generalization Benchmark

This project benchmarks how well SPD-based connectome models trained on one
resting-state fMRI cohort generalize to another cohort.

The supported workflow is a small, package-first benchmark CLI for regression on
functional-connectivity matrices. It supports the core evaluated modes used in
this repository: SPDNet, Ridge baselines, and dataset-level generalization
settings such as K-fold and leave-one-dataset-out evaluation.

## Quick start

Python 3.11+ is required.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Check the configuration before running any data-intensive job:

```bash
rsfmri-spd-benchmark --config configs/examples/mini_run.yaml --dry-run
```

This dry run validates the selected config and prints the resolved benchmark
plan without opening participant data or fitting models.

## Typical usage

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

For the supported CLI surface:

```bash
rsfmri-spd-benchmark --help
python prepare_fmri_datasets.py --help
python make_results.py --help
```

## Repository layout

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   └── examples/
│       └── mini_run.yaml
├── docs/
│   └── reproducibility.md
├── spd_connectome_benchmark/
├── tests/
├── results/
└── LICENSE
```

The package implementation lives under [spd_connectome_benchmark](spd_connectome_benchmark).
The example configuration is in [configs/examples/mini_run.yaml](configs/examples/mini_run.yaml).
The longer write-up is in [docs/reproducibility.md](docs/reproducibility.md).

## Data expectations

Participant-level data are not committed to Git. The benchmark expects
prepared connectome inputs in an external directory and reads them through the
config or CLI arguments.

Outputs such as benchmark CSVs and metadata are written to the configured output
root, typically under `results/`.

## License

Source code in this repository is released under the [BSD 3-Clause License](LICENSE).
Dataset-specific terms remain separate from the software license.
