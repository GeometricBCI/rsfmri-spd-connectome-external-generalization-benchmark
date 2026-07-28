# {{PROJECT_TITLE}}

Version `{{VERSION}}`

Release status: `{{VALIDATION_STATUS}}`

## About this release

This archive accompanies the rs-fMRI SPD connectome external-generalization
benchmark. It brings the policy-approved derived-connectome scope,
release-safe metadata, data dictionaries, provenance, and validation records
into one versioned resource.

The archive is generated from a source-bound safe export. Every
participant-level artifact must pass the dataset policy, metadata allowlist,
matrix schema, privacy scan, manifest, and checksum checks before it can appear
here.

## Data included in this archive

{{DATASET_INVENTORY}}

The table in `metadata/dataset_inventory.tsv` is the authoritative inventory.
The participant-level scope is intentionally narrow:

- ABIDE includes derived Schaefer-100 connectivity matrices and generated
  `sample_uid` values;
- COBRE includes derived Schaefer-100 connectivity matrices and generated
  `sample_uid` values;
- CamCAN includes derived Schaefer-100 connectivity matrices with generated
  `sample_uid`, age, and sex.

ABIDE and COBRE do not include an age target. Their public matrices can be
inspected and reused under the applicable source terms, but the full
age-prediction experiments cannot be rerun from this archive alone. Required
phenotypes must be obtained independently and used under the governing
dataset conditions.

The NPZ and TSV files are release-safe exchange formats; they are not direct
inputs to `run_benchmark.py`. The benchmark CLI expects authorized local
ROI-time-series tables and constructs connectomes during execution. This
archive does not include an adapter that converts its public matrices back into
that internal input format.

## Data not included

The archive does not contain:

- source pickle files or ROI time series;
- raw MRI, NIfTI, DICOM, BIDS, or T1-weighted images;
- original, encoded, or hashed participant, scan, visit, image, or session
  identifiers;
- source filenames, local paths, credentials, or data-use agreements;
- diagnosis, site, questionnaire, Home Interview, free-text, or other
  unapproved participant variables;
- exact participant-level benchmark split membership;
- participant-level ADNI, ADNI-DOD, OASIS-3, 1000BRAINS, or unknown-dataset
  material.

## File layout

- `data/` contains the connectomes and metadata admitted to this build;
- `metadata/` contains the inventory, data dictionary, provenance, Zenodo
  metadata worksheet, and frozen policy snapshots;
- `configs/` records the logical dataset and preprocessing configuration;
- `restricted_reconstruction/` explains how authorized users can reconstruct
  restricted datasets independently;
- `benchmark_results/` contains reviewed aggregate outputs when present;
- `reproducibility/` records the source revision, environment, and validation
  commands;
- `manifests/` contains the file inventory, SHA-256 values, and validation
  report.

No exact split tables are included in this version.

## Integrity and validation

First verify the checksums from the extracted release root:

```bash
shasum -a 256 --check manifests/SHA256SUMS.txt
```

On platforms that provide GNU coreutils, `sha256sum --check` is equivalent.

The Python validator is part of the GitHub repository, not duplicated inside
this dataset archive. Check out the commit recorded in
`reproducibility/git_commit.txt`, install its release dependencies, and run:

```bash
export RELEASE_DIR="<extracted-release-directory>"
export RELEASE_CONFIG="<external-release-config-file>"

python -m tools.zenodo.validate_release \
  --release-dir "$RELEASE_DIR" \
  --config "$RELEASE_CONFIG"
```

The external configuration and every policy document it references must match
the frozen snapshots exactly. Publication packaging requires that binding.
Snapshot-only checks are reserved for preserved-archive verification or
synthetic testing; they do not establish publication readiness, and any
package created with `--archive-verification` must not be uploaded.

Connectome archives are read with `numpy.load(..., allow_pickle=False)`.
Validation checks numeric dtype, finite values, shape, symmetry, correlation
diagonal, positive definiteness, metadata alignment, privacy rules, manifests,
and checksums.

## Rights and citation

Read `LICENSES.md` before reuse. CamCAN, ABIDE, and COBRE retain separate
dataset-specific conditions; neither the repository's BSD software license nor
a single Zenodo record-level selection replaces those terms.

A valid `CITATION.cff` is generated only after the creator records are
complete. Use it and the Zenodo citation only after publication-ready
validation has passed.

## Publication review

The validation report is the source of truth for release readiness. A
structural pass confirms that the files are internally consistent; it is not
the same as permission to publish.

The detailed review queue is recorded in
`manifests/validation_report.md`. Before publication, a designated human
reviewer must complete:

- the ordered creator list, affiliations, identifiers, and funding record;
- the record-level mixed-rights presentation and documentation license;
- the exact ABIDE and COBRE license/citation wording;
- authoritative evidence for each included source-bound artifact;
- the restricted-dataset exclusion and reconstruction review;
- the version, privacy, manifest, checksum, archive, and final Zenodo-form
  approvals.

## Source and support

Repository: {{REPOSITORY_URL}}

Source revision: `{{GIT_COMMIT}}`

GitHub hosts the evolving software. This archive is the corresponding frozen
dataset resource. Issues concerning code or documentation should be reported
through the repository; access questions remain with the original dataset
custodians.

## Release-preparation assistance

OpenAI Codex was used as a software assistant to help organize and review the
release configuration, documentation, tests, and validation code. The named
human authors and designated reviewers retain responsibility for the
scientific content, dataset permissions, licensing decisions, privacy review,
authorship, and publication approval. Codex is not an author, creator, rights
holder, designated human reviewer, approver, or source of dataset permission.
