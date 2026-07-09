# rs-fMRI SPD Matrix Learning Benchmark

This repository contains the code used to benchmark age-regression models on
resting-state fMRI SPD connectomes. The pooled experiments use six datasets:

`cobre`, `adnidod`, `camcan`, `abide`, `oasis3`, `adni`.

The implementation keeps experiment defaults explicit and documents known
paper/code alignment details directly in the relevant code paths and CLI help.

## Repository Layout

The runnable experiment code lives in the top-level scripts so the benchmark
protocols are visible without chasing thin wrappers. The
`spd_connectome_benchmark` package contains only reusable building blocks:
configuration, connectome construction, SPD models, shared training utilities,
and figure/table helpers.

```text
prepare_fmri_datasets.py                    Optional raw-data extraction and QC
run_single_dataset_benchmark.py             Within-dataset GroupKFold benchmark
run_pooled_benchmark.py                     Pooled GroupKFold and LODO benchmark
run_spdnet_ablation.py                      Figure 5 SPDNet ablation benchmark
make_results.py                             Result tables and figures
spd_connectome_benchmark/
  config.py                    Dataset list, default paths, and benchmark constants
  connectomes/vec_connectomes.py
                                OAS covariance, correlation, and CorrVec features
  models/spd.py                 SPDNet model definitions backed by spd_learn
  benchmark_tools/cli.py        Shared CLI argument groups
  benchmark_tools/harmonization.py
                                Split-local ComBat/tangent-space harmonization
  benchmark_tools/runtime.py    Loading, splits, training helpers, metrics, CSV writers
  analysis_outputs/dataset_description.py
  analysis_outputs/dataset_shift.py
  analysis_outputs/lodo_tables.py
  analysis_outputs/result_io.py
  analysis_outputs/result_grid_plots.py
  analysis_outputs/plot_style.py
  analysis_outputs/result_plotting.py
  analysis_outputs/result_figures.py
results/
  benchmark_csv/                 Default benchmark CSV output directory
  tables/
    table1_dataset_summary.csv
    table3_lodo_per_dataset_metrics.csv
  figures/
    figure2_dataset_overview.pdf
    figure3_within_dataset_negmae.pdf
    figure3_within_dataset_r2.pdf
    figure4_pooled_benchmark.pdf
    figure5_spdnet_ablation.pdf
```

## Data Setup

The benchmark scripts expect prepared pickle files with this layout:

```text
data/
  atlas_schaefer_100/
    cobre_X_y.pkl
    adnidod_X_y.pkl
    camcan_X_y.pkl
    abide_X_y.pkl
    oasis3_X_y.pkl
    adni_X_y.pkl
```

There are two supported ways to create that directory:

1. Download the processed benchmark archive from the project data link and
   extract it under `data/`. The archive should contain the `atlas_schaefer_100`
   directory shown above.
2. Rebuild the files locally from raw/source datasets:

```bash
python prepare_fmri_datasets.py --dataset all --atlas schaefer_100
```

Use `RSFMRI_SPD_DATA_ROOT=/path/to/data` or `--data_root /path/to/data` when the
prepared files live outside the repository.

Raw/source directories are configurable without editing code:

```bash
python prepare_fmri_datasets.py \
  --data_root /path/to/data \
  --raw_data_dir /path/to/raw_data \
  --adni_adnidod_raw_dir /path/to/adni_adnidod_raw \
  --oasis3_raw_dir /path/to/oasis3_raw
```

Equivalent environment variables are `RSFMRI_SPD_RAW_DATA_DIR`,
`RSFMRI_SPD_ADNI_ADNIDOD_RAW_DIR`, and `RSFMRI_SPD_OASIS3_RAW_DIR`.

## Main Commands

```bash
python prepare_fmri_datasets.py --help
python run_single_dataset_benchmark.py --help
python run_pooled_benchmark.py --help
python run_spdnet_ablation.py --help
python make_results.py --help
```

## Paper Protocol Map

- Connectomes: `spd_connectome_benchmark.connectomes.estimate_connectome_matrices`
  implements per-scan OAS, `eps=1e-5`, correlation normalization with
  `sqrt(eps)` floor, and final SPD jitter.
- Tangent-Space Ridge:
  `run_single_dataset_benchmark.run_tangent_space_ridge_age_regression`
  and `run_pooled_benchmark._run_tangent_space_ridge` fit the tangent
  reference on the outer training split only.
- CorrVec:
  `spd_connectome_benchmark.connectomes.vectorize_correlation_matrices(..., include_diagonal=False)`
  uses off-diagonal upper-triangle entries without tangent-space weighting.
- SPDNet: `spd_connectome_benchmark.models.spd.SPDNetRegressor` implements
  BiMap/ReEig/LogEig plus the shared MLP head and the Figure 5 dimensions.
- SPD layers: `SPDNetRegressor` uses `spd_learn.modules` directly.
- Harmonization: `run_pooled_benchmark.py` fits the tangent reference and
  ComBat model inside each split; GroupKFold applies correction to the test fold
  with true ages, while LODO leaves the held-out dataset unchanged.

## Requirements

Use Python 3.11 or newer. The recommended setup uses the public `spd-learn`
package for SPDNet layers:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` pins the direct dependencies used by the Python 3.11
validation environment:
`spd-learn`, `torch`, `pyriemann`, `scikit-learn`, `nilearn`, `nibabel`,
`neuroHarmonize`, `statsmodels`, `neuroCombat`, `pandas`, `numpy`,
`matplotlib`, `seaborn`, `tabulate`, `joblib`, and `pytest`.

The repository was smoke-tested in an isolated Python 3.11 environment with
the `spd_learn` backend. Full benchmark runs require the prepared
`atlas_<atlas_name>/*_X_y.pkl` files described above.

## Quick Checks

```bash
python -m compileall -q .
python -m pytest
```

`pytest` is optional in the runtime environment but recommended for development.
The smoke tests are intentionally lightweight: they check the paper dataset
list, SPD connectome construction, grouped validation splits, CorrVec
vectorization, and an SPDNet forward pass.

To verify that the external SPD layers are available:

```bash
python -c "from spd_learn.modules import BiMap, ReEig, LogEig; print('spd_learn ok')"
```
The command should print `spd_learn ok`.

## Example Runs

Paper-like pooled benchmark:

```bash
python run_pooled_benchmark.py \
  --DATASETS cobre adnidod camcan abide oasis3 adni \
  --algorithms spdnet ridge corr_ridge dummy \
  --protocol both \
  --harm_mode both
```

Paper Figure 5 ablation:

```bash
python run_spdnet_ablation.py \
  --DATASETS cobre adnidod camcan abide oasis3 adni \
  --spdnet_variants quarterdim halfdim one two \
  --protocol both \
  --harm_mode both
```

Paper-like within-dataset run:

```bash
python run_single_dataset_benchmark.py \
  --datasets cobre,adnidod,camcan,abide,oasis3,adni \
  --train_batch_size 1024 \
  --test_batch_size 1024 \
  --run_ridge \
  --run_dummy \
  --run_vec_corr_ridge
```

Result tables and figures:

```bash
python make_results.py support
python make_results.py dataset-shift
python make_results.py lodo-tables
python make_results.py result-figures
```

By default, benchmark CSV files are written under `results/benchmark_csv/`,
checkpoints under `results/model_weights/`, tables under `results/tables/`,
and figures under `results/figures/`.
Each generated benchmark CSV also gets a neighboring `.metadata.json` sidecar
with the command, Python runtime, selected environment variables, and key
dependency versions.
