# Restricted-data reconstruction: {{DATASET}}

Release version: `{{VERSION}}`

## Redistribution boundary

Participant-level connectomes, metadata, original or hashed identifiers, exact
split memberships, source filenames, and source paths for {{DATASET}} are not
included in this release.

Researchers must obtain access independently from the original custodian under
the current governing terms. This document provides no credentials, private
links, copied metadata, or participant lists.

## Required local environment

Use Python 3.11 or newer and the dependency versions recorded in the release.
Keep both raw/source material and prepared outputs outside the Git checkout.
Define controlled local paths only in the trusted environment:

```bash
export AUTHORIZED_SOURCE_ROOT="<authorized-source-directory>"
export PREPARED_OUTPUT_ROOT="<prepared-output-directory>"
export BENCHMARK_RESULTS_ROOT="<benchmark-results-directory>"
export DATASET_KEY="<logical-key-from-selection-config>"
```

Do not commit those values or include them in a public manifest.

## Local reconstruction command

After independently confirming authorization:

```bash
python prepare_fmri_datasets.py \
  --dataset "$DATASET_KEY" \
  --atlas {{ATLAS_NAME}} \
  --data_root "$PREPARED_OUTPUT_ROOT" \
  {{DATASET_SOURCE_ARGUMENT}} "$AUTHORIZED_SOURCE_ROOT"
```

Use the logical key in the adjacent `selection_config.yaml`; do not derive it
from a private filename.

## Expected preprocessing and connectome construction

The repository's established pipeline:

1. loads the independently obtained, locally authorized source representation;
2. applies the dataset-specific preprocessing path documented in the source
   code and provenance snapshot;
3. extracts atlas time series;
4. applies established scan-level quality checks;
5. computes per-scan OAS covariance;
6. symmetrizes and regularizes the matrix;
7. normalizes it to a correlation representation;
8. applies final positive-definite jitter;
9. writes a trusted local prepared table outside Git.

Do not replace missing source-specific selection rules with assumptions.

## Selection rules

The adjacent `selection_config.yaml` records the configured, provisional
logical selection and split-reconstruction settings. Before publication, a
reviewer must confirm its inclusion/exclusion logic, longitudinal handling,
required fields, target definition, and expected aggregate counts against an
authoritative protocol or the reviewed repository implementation. Missing
details are not inferred. Neither this document nor the selection configuration
may expose source identifiers or exact memberships.

## Internal prepared schema

The benchmark expects one row per scan/session with:

- a non-public subject grouping key;
- a finite `(timepoints, atlas_regions)` time-series array;
- finite chronological age for the current regression task;
- optional local sex, diagnosis/group, and site fields.

The expected local filename pattern is documented by the benchmark CLI but must
not be copied into the public package as participant data.

## Deterministic validation and split reconstruction

First validate only the required prepared filename set:

```bash
python run_benchmark.py \
  --dry-run \
  --task regression \
  --target Age \
  --atlas {{ATLAS_NAME}} \
  --datasets "$DATASET_KEY" \
  --cv kfold \
  --seed 1 \
  --data-shuffle-seed 42 \
  --input-root "$PREPARED_OUTPUT_ROOT" \
  --output-dir "$BENCHMARK_RESULTS_ROOT"
```

The command above validates the single-dataset, subject-grouped K-fold setup.
To reconstruct leave-one-dataset-out evaluation, select at least two
independently authorized prepared datasets and use the dataset order, grouping
rules, and seeds recorded in this release. Restricted reconstruction materials
report aggregate fold counts only; they do not contain original, hashed, or
release-safe mappings to the restricted participants.

## Local validation checklist

- input and output roots are outside the Git checkout and do not overlap;
- the atlas and matrix dimensions match the release configuration;
- time series and targets are finite and aligned;
- grouping keys are non-missing and split-disjoint;
- connectomes are finite, square, symmetric, and positive definite;
- the dry run reports the intended dataset, atlas, model, CV, harmonization,
  and seed matrix;
- private checksums and provenance remain in the controlled environment.

Public release validation cannot certify access compliance; the authorized
researcher remains responsible for the source terms.
