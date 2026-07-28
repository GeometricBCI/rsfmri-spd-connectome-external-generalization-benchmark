# Internal data schema

This document describes the structures inferred from the code. It contains no
real participant identifiers, filenames, or values.

## Unit of observation

One prepared table row represents one scan/session, not necessarily one unique
subject. A subject may therefore occur in multiple rows. `SubjectID` is used
only as a local grouping key; pooled loading prefixes that key with the logical
dataset name to prevent collisions across datasets. Identifiers must never be
written to public result artifacts.

## Prepared table contract

Required columns:

| Column | Meaning | Expected representation |
|---|---|---|
| `SubjectID` | Local subject grouping key | non-empty scalar/string-like |
| `TimeSeries` | Atlas time series for one scan | finite floating array `(T_i, P)` |
| `Age` | Chronological-age target | finite numeric scalar |

Optional columns used by preparation or aggregate descriptions include
`Session`, `Sex`, `Diagnosis`, `Group`, and a site label such as `Site`, `SITE`,
or `SiteID`. Other source metadata may exist locally but is not part of the
public benchmark contract.

All scans in one experiment must have the same `P`. The preparation code
applies scan-level quality criteria before serialization, including a minimum
time-series length of 100, no fully null atlas regions, and covariance
condition-number bounds. These are established preprocessing semantics, not
generic validation defaults.

## Atlas representation

The registered atlases are:

| Name | Regions | Representation |
|---|---:|---|
| `schaefer_100` | 100 | label image |
| `msdl_39` | 39 | probabilistic maps |

Schaefer-400 is not implemented in this release. Prepared filenames live under
`atlas_<name>/`. Atlas images and raw neuroimaging files are not stored in the
repository.

## Connectome contract

`connectomes` is a floating array with shape `(N, P, P)`:

- `N` scans;
- square matrices with a consistent `P`;
- finite values;
- symmetric within numerical tolerance;
- strictly positive-definite.

The established construction is per-scan OAS covariance, symmetrization,
`epsilon * I`, correlation normalization with standard-deviation floor
`sqrt(epsilon)`, symmetrization, then final `epsilon * I`. The default epsilon
is `1e-5`.

Synthetic tests generate SPD matrices as:

```python
A @ A.T + epsilon * I
```

## Target and label arrays

- regression targets: floating 1D array `(N,)`, currently chronological age;
- dataset labels: non-empty string-like 1D array `(N,)`;
- subject groups: non-missing hashable scalar 1D array `(N,)`; numeric and
  string identifiers are both accepted;
- site labels: optional; current pooled harmonization uses the logical dataset
  label as `SITE`, not an acquisition-site field.

Classification targets are not implemented.

## Split representation

A split is `(train_idx, test_idx)`, two non-empty integer arrays indexing scan
rows. Indices must be unique within each side, disjoint across sides, and in
bounds. Subject groups must be disjoint between outer train and test.

- grouped K-fold labels are `R1` through `RK`;
- LODO test labels are `TEST_<dataset>`;
- each LODO test fold contains one complete dataset;
- training/validation division is group-aware and occurs within outer train.

LODO harmonization fits and transforms source training data but leaves the
unseen target dataset unchanged.

## Result tables

Legacy CSV compatibility is retained:

- metrics are rows;
- grouped K-fold columns are `R1..RK`;
- LODO columns are `TEST_<dataset>`;
- final columns are `Avg` and `Time(sec)`.

Grouped K-fold writes `MAE`, `R2`, and `Pearson_r`. LODO additionally writes
`Spearman_rho`, `corr(BAG,age)`, and `age_bias_slope`. Sidecars contain runtime
and dependency information but redact local paths. A
`run_config_<fingerprint>.json` manifest records experiment selection, all
resolved non-path legacy execution parameters, harmonization policy, and
available Git revision state without local paths. Aggregate description outputs
may contain counts and summary statistics; scan-level metadata and identifiers
are prohibited.

## Validation helpers

`spd_connectome_benchmark.data_contract` provides typed helpers for time-series,
SPD arrays, aligned target/label vectors, and split disjointness. These helpers
are exercised only with synthetic inputs in the public test suite.
