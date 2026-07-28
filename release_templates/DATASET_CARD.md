# Dataset card: {{PROJECT_TITLE}}

Release version: `{{VERSION}}`

Archive: `spd_connectome_benchmark_v{{VERSION}}`

Release status is determined by `manifests/validation_report.md`.

## Purpose

This dataset card documents the resting-state fMRI connectivity matrices
prepared for the external-generalization benchmark. The benchmark compares
SPDNet, tangent-space Ridge, vectorized-correlation Ridge, and a dummy baseline
under subject-grouped K-fold and leave-one-dataset-out evaluation.

The archive does not contain an age target for ABIDE or COBRE. Those matrices
support inspection, method development, and reuse under their source terms,
but a complete reconstruction of the age-regression experiments requires
phenotypes obtained independently from the original custodians. CamCAN is the
only participant-level dataset in the current policy for which age and sex may
be included.

## Dataset policy inventory

Included participant-level dataset inventory:

{{DATASET_INVENTORY}}

The generated list above describes files actually admitted from the safe
export. Policy decisions remain authoritative in
`metadata/release_policy_snapshot.yaml`. Empty or omitted public-data
directories do not imply approval.

The release policy is:

- ADNI, ADNI-DOD, and OASIS-3 participant-level release is forbidden;
- CamCAN, ABIDE, and COBRE permissions for participant-level derived
  functional-connectivity matrices are confirmed under separate,
  dataset-specific terms;
- a real export of any confirmed dataset still requires the exact
  dataset-specific metadata allowlist, reviewed artifact identity, and
  source-binding SHA-256;
- 1000BRAINS is forbidden unless both the policy and approval record are
  deliberately changed;
- unknown datasets are forbidden.

Reviewed aggregate, non-identifying outputs may be included only through the
explicit content allowlist.

### CamCAN dataset card

Dataset name: Cambridge Centre for Ageing and Neuroscience (CamCAN).

Source citation: Shafto et al. (2014),
*The Cambridge Centre for Ageing and Neuroscience (Cam-CAN) study protocol: a
cross-sectional, lifespan, multidisciplinary examination of healthy cognitive
ageing*, BMC Neurology 14:204,
<https://doi.org/10.1186/s12883-014-0204-1>.

Derived data description: participant-level Schaefer-100 ROI-to-ROI
resting-state functional-connectivity matrices.

Permitted public scope:

- participant-level ROI-to-ROI functional-connectivity matrices;
- release-safe `sample_uid`;
- age;
- sex.

Excluded:

- raw MRI images;
- T1-weighted images;
- identifiable images;
- identifiable or other unapproved behavioural variables;
- Home Interview variables;
- `CCID`, original or hashed identifiers, source paths, and exact split
  membership.

Metadata fields: `sample_uid`, `age`, and `sex`.

License: `CC-BY-4.0`.

Citation requirement: cite the CamCAN cohort paper above and preserve
attribution.

`sample_uid` is release-generated as `s` followed by 32 lowercase hexadecimal
UUIDv5 digits; a CCID-shaped value is invalid. `age` must be finite and between
0 and 130, and `sex` must be normalized to `F` or `M`. The generated release
must retain the full citation above in `data/camcan/LICENSE.txt`.

When a completed artifact binding admits CamCAN, the archive contains
exactly:

```text
data/camcan/connectomes/camcan_schaefer100_fc.npz
data/camcan/metadata/participants.tsv
data/camcan/data_dictionary.tsv
data/camcan/LICENSE.txt
```

No duplicate or legacy Cam-CAN copy may appear under
`data/public_connectomes/` or `data/public_metadata/`. Confirmed permission
does not replace the required artifact-review identity, date, non-sensitive
evidence reference, Schaefer-100 scope, and trusted-export source-binding
SHA-256. Its `license_identifier` must also exactly match `CC-BY-4.0`.

### ABIDE dataset card

Dataset name: Autism Brain Imaging Data Exchange (ABIDE).

Source terms: the
[official ABIDE I project page](https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html)
states that the data are available for non-commercial research under a
Creative Commons Attribution-NonCommercial-ShareAlike license and requires
dataset-specific acknowledgements. The page does not identify a license
version. The complete ABIDE/INDI bibliography, funding acknowledgements, and
applicable license URI must therefore be confirmed during final metadata
review.

Derived data description: participant-level ROI-to-ROI resting-state
functional-connectivity matrices derived from ABIDE source data.

Permitted public scope:

- approved participant-level functional-connectivity matrices;
- release-safe `sample_uid`.

Excluded:

- raw MRI, NIfTI, DICOM, and T1-weighted images;
- original identifiers, source filenames, and source paths;
- unrestricted clinical or free text;
- questionnaire responses and arbitrary behavioural variables;
- exact split membership.

Metadata fields: `sample_uid` only. No age, sex, diagnosis, site, questionnaire,
or other phenotype is admitted without a future explicit confirmation and
allowlist change.

Citation and reuse requirements: preserve ABIDE/INDI attribution,
acknowledgements, non-commercial-use restrictions, and applicable share-alike
conditions.

When a completed artifact binding admits ABIDE, its participant-level files
are:

```text
data/public_connectomes/abide/connectomes.npz
data/public_metadata/abide/metadata.tsv
```

### COBRE dataset card

Dataset name: Center for Biomedical Research Excellence (COBRE).

Source terms: the
[official COBRE project page](https://fcon_1000.projects.nitrc.org/indi/retro/cobre.html)
states that the data are available under a Creative Commons
Attribution-NonCommercial license. The page does not identify a license
version. The complete COBRE/FCP bibliography, attribution wording, and
applicable license URI must therefore be confirmed during final metadata
review.

Derived data description: participant-level ROI-to-ROI resting-state
functional-connectivity matrices derived from COBRE source data.

Permitted public scope:

- approved participant-level functional-connectivity matrices;
- release-safe `sample_uid`.

Excluded:

- raw MRI, NIfTI, DICOM, and T1-weighted images;
- original identifiers, source filenames, and source paths;
- unrestricted clinical or free text;
- questionnaire responses and arbitrary behavioural variables;
- exact split membership.

Metadata fields: `sample_uid` only until additional fields are explicitly
confirmed and added to the COBRE allowlist.

Citation requirements: preserve COBRE and FCP/INDI citation, attribution, and
non-commercial-use requirements.

When a completed artifact binding admits COBRE, its participant-level files
are:

```text
data/public_connectomes/cobre/connectomes.npz
data/public_metadata/cobre/metadata.tsv
```

## Unit of observation and internal reconstruction contract

One internal prepared row represents one scan/session, and a subject may
contribute more than one row. Subject grouping is used for split construction
but internal grouping keys are never public release fields.

Authorized local reconstruction expects:

- a finite time-series array with shape `(timepoints, atlas_regions)` per scan;
- chronological age as the current regression target;
- a non-public subject grouping key;
- optional local sex, diagnosis/group, and site metadata.

These internal fields do not become public merely because they exist locally.

## Public connectome schema

For a dataset admitted by the release policy and its completed artifact review,
a public NPZ may contain:

- `connectomes`: `float32` or `float64`, shape
  `(n_samples, n_regions, n_regions)`;
- no additional NPZ keys in this release schema. Adding numeric targets in a
  future version requires deliberate schema, policy, validator, and approval
  changes.

Object dtype is forbidden. Files must load with
`numpy.load(path, allow_pickle=False)`. Matrices must be finite, square,
symmetric within the configured tolerance, approximately unit-diagonal for the
correlation representation, and positive definite.

No original or hashed identifiers, source paths, or original filenames are
permitted. A release-safe `sample_uid` may appear only in approved metadata and
split tables, must not encode a source identifier, and must be unique.

## Atlases and connectome construction

The current benchmark registry supports Schaefer-100 and MSDL-39. The
established connectome construction uses per-scan OAS covariance,
symmetrization, regularization, correlation normalization, and a final
positive-definite jitter. The concrete atlas and numerical parameters for this
release are recorded in the configuration and provenance tables.

## Splits

- outer K-fold and inner validation are grouped by subject;
- leave-one-dataset-out holds out one complete logical dataset;
- training and test memberships are disjoint;
- LODO harmonization fits source data and does not transform the unseen target
  dataset;
- the historical grouped-K-fold harmonization transform uses known test age
  and is therefore target-informed.

No exact participant-level split membership is included in this release.
Current policy explicitly forbids exact split files for ABIDE, COBRE, and
CamCAN. Restricted-dataset reconstruction material may record seeds, grouping
rules, selection rules, and reviewed aggregate fold counts, but never exact
memberships or hashed identifiers.

## Included outputs

See `metadata/dataset_inventory.tsv` and `manifests/manifest.tsv` for the
complete generated output inventory.

Every included output is described in `metadata/dataset_inventory.tsv`,
`metadata/data_dictionary.tsv`, and `metadata/provenance.tsv`. The manifest
records its dataset, content category, access category, source category, and
policy decision.

## Limitations

- This release does not make restricted source datasets publicly available.
- The public NPZ/TSV files are an exchange format, not direct inputs to
  `run_benchmark.py`; the benchmark CLI reconstructs connectomes from
  authorized ROI-time-series tables.
- Redistribution approval is independent of technical deidentification.
- Exact numerical reproducibility can depend on hardware and numerical
  libraries. The archive records direct environment constraints; a final
  publication build should additionally preserve the complete resolved
  environment for the published run.
- The benchmark currently implements chronological-age regression, not
  classification.

## Review and validation

Publication requires complete creator, license, dataset-scope, privacy,
provenance, manifest, checksum, and final-form approvals. See
`metadata/manual_approvals_snapshot.yaml` and
`manifests/validation_report.md`.

The current validation report contains the detailed review queue. Publication
remains blocked until a designated human completes the creator, licensing,
artifact-evidence, restricted-scope, privacy, provenance, checksum, version,
and final Zenodo-form approvals.
