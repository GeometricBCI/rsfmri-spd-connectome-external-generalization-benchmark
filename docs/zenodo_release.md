# Zenodo release workflow

## Scope and safety boundary

This workflow builds a frozen, versioned **dataset** resource for:

> Benchmarking External Generalization of SPD Matrix Learning for Resting-State
> fMRI Connectome Prediction

GitHub remains the evolving source-code repository. The Zenodo package contains
only materials admitted by an explicit release policy and safe-export manifest.
This workflow does not call the Zenodo API, read or store a Zenodo token,
reserve a DOI, create a Git tag, publish a GitHub release, or upload anything.

The committed release configuration is intentionally fail-closed. The
participant-level derived-data scope is recorded for ABIDE, COBRE, and CamCAN,
but the repository still cannot produce a publication-ready package until
human reviewers complete the artifact bindings, creator metadata, license
presentation, and final publication gate.

Never use the public build against raw/internal data. Only the separately gated
trusted exporter may read an internal pickle, and only a human may run that
command in a controlled environment.

## Components

The versioned source inputs are:

- `configs/release/zenodo_release.yaml`: project metadata, numerical validation
  tolerances, content allowlist, and publication gates;
- `configs/release/dataset_release_policy.yaml`: dataset-by-content decisions;
- `configs/release/metadata_allowlist.yaml`: exact public columns and NPZ keys;
- `configs/release/forbidden_patterns.yaml`: redacting privacy-scan rules;
- `configs/release/manual_approvals.yaml`: deliberate human decisions;
- `release_templates/`: release-facing documentation;
- `tools/zenodo/`: trusted export, public build, validation, checksum, privacy,
  metadata, and deterministic packaging code.

Generated material belongs under `zenodo_upload/`, which is ignored by Git.
Controlled inputs and the safe-export working directory must also remain
outside Git.

## Default release policy

Unknown datasets, unknown fields, and unknown content categories are denied.

| Dataset | Participant-level policy | Artifact requirement |
|---|---|---|
| ADNI | Forbidden | Reviewed aggregates and reconstruction documentation only |
| ADNI-DOD | Forbidden | Reviewed aggregates and reconstruction documentation only |
| OASIS-3 | Forbidden | Reviewed aggregates and reconstruction documentation only |
| ABIDE | Confirmed derived-connectome scope; public metadata is limited to generated `sample_uid` | Exact safe-export binding, evidence record, and final review |
| COBRE | Confirmed derived-connectome scope; public metadata is limited to generated `sample_uid` | Exact safe-export binding, evidence record, and final review |
| CamCAN | Written confirmation covers derived Schaefer-100 connectomes plus age and sex under CC BY 4.0 | Exact safe-export binding, evidence record, and final review |
| 1000BRAINS | Forbidden unless the policy is deliberately changed | Non-public |
| Unknown dataset | Forbidden | Excluded |

For every known dataset, aggregate metrics, statistical summaries, and
non-identifying figure source data still require an explicit column allowlist
and artifact review. A directory name never implies approval.

The policy distinguishes:

- participant-level connectomes;
- participant-level metadata;
- exact split membership;
- aggregate metrics;
- statistical summaries;
- figure source data;
- configuration files;
- processing scripts;
- reconstruction instructions.

An approval cannot override `forbidden`. For
`forbidden_unless_explicitly_approved`, both the policy and the approval record
must be deliberately changed.

## Manual approval contract

The only recognized approving status is `approved`. It is valid only when the
record contains non-empty:

- `approved_by`;
- `approved_on`;
- `scope`;
- `evidence`.

`approval_protocol.designated_reviewer` records who is responsible for the
human review process. That designation does not approve any dataset, license,
artifact, metadata record, or publication gate. Each approval still requires
its own completed record after the named reviewer has inspected the relevant
evidence.

The project configuration additionally requires the deliberate confirmation
sentence in `manual_approvals.yaml`. Empty, missing, unknown, or conflicting
fields are treated as `manual_required`.

Entries in `manual_action_items` are unconditional publication blockers. Remove
one only after its underlying configuration and evidence are genuinely
complete. The semantic version requires
`release.version_confirmed: true`, and the separate `publication_gate` record
must be complete with `publish_ready: true` only after all prior reviews pass.

For participant-level ABIDE, COBRE, or Cam-CAN export, `scope` must use only
the exact tool scopes that are actually authorized:

- `participant_connectomes`;
- `participant_metadata`;
- `exact_splits`, only for a dataset whose policy separately authorizes exact
  membership. Cam-CAN exact split membership is forbidden.

The same record must bind the approval to:

- canonical `dataset`;
- `atlas`;
- `n_regions`;
- the `source_binding_sha256` emitted by trusted-export dry run.

Stage 2 requires those values to match the safe-export attestation exactly.
Changing the export namespace, source checksum, declared schema, atlas, region
count, or source identity attestation changes the binding and invalidates the
old approval. The namespace is bound because it deterministically controls the
release-safe `sample_uid` values.

Approval is not sufficient by itself. The reviewer must also make a deliberate
dataset-specific edit to `metadata_allowlist.yaml`, adding only `sample_uid`
and the exact approved public metadata columns. Original or hashed identifiers
remain forbidden even if someone attempts to add them to an allowlist.

### Cam-CAN written-confirmation review

The Cam-CAN policy records written confirmation that derived Schaefer-100
participant-level connectomes and the metadata fields `age` and `sex` are
allowed for release under CC-BY-4.0. The public Cam-CAN portal terms separately
state both that source data must not be further disclosed and that derived data
and processing scripts used to produce them should be made available in an
open-access repository:

https://opendata.mrc-cbu.cam.ac.uk/projects/camcan/

This policy authorization is a ceiling, not a completed operational approval.
Before an actual Cam-CAN participant-level export can pass, the controlled
approval record must still identify the authorized reviewer, ISO review date,
non-sensitive evidence reference, exact CC-BY-4.0 scope, Schaefer-100 schema,
and the source-binding SHA-256 emitted by trusted-export dry run. Do not invent
an approver, date, evidence reference, or binding to make validation pass.

The exact metadata schema is `sample_uid`, `age`, and `sex`. `sample_uid` is
release-generated as `s` followed by 32 lowercase hexadecimal UUIDv5 digits;
it is not a transformed source identifier, and a CCID-shaped value cannot be
substituted. `age` must be a finite numeric value from 0 through 130, and `sex`
must be normalized to `F` or `M` before release. `CCID`, home-interview
variables, and other original or hashed identifiers are forbidden; raw T1
images, identifiable images, and exact split membership are also forbidden.
The release must retain the configured
`Shafto_et_al_CamCAN_cohort_paper` citation, resolved in the generated license
notice to Shafto et al. (2014),
<https://doi.org/10.1186/s12883-014-0204-1>.

When Cam-CAN is admitted by a completed source-bound approval, its sole
canonical archive layout is:

```text
data/camcan/
├── connectomes/
│   └── camcan_schaefer100_fc.npz
├── metadata/
│   └── participants.tsv
├── data_dictionary.tsv
└── LICENSE.txt
```

Do not also place Cam-CAN files under the legacy
`data/public_connectomes/camcan/` or `data/public_metadata/camcan/` paths. The
strict validator permits exactly one Cam-CAN representation. The
`participants.tsv` exception applies only at the path above and must contain
exactly `sample_uid`, `age`, and `sex`, never `CCID`.

The completed approval record's `license_identifier` must exactly equal the
policy and generated notice value `CC-BY-4.0`; a different license keeps both
Cam-CAN participant scopes closed.

## Two-stage workflow

### Stage 1: trusted internal export

Stage 1 is a human-run conversion in a controlled environment. It may
deserialize a trusted internal pandas DataFrame pickle. The command:

- always prints a pickle-execution warning;
- refuses to proceed without `--trusted-internal-input`;
- accepts only an attested, lower-case SHA-256-bound manifest;
- rejects unknown manifest or schema fields;
- requires exactly one declared connectome or time-series column;
- applies dataset policy and metadata allowlists;
- never derives a public ID by hashing an original ID;
- writes only release-safe NPZ/TSV/JSON into a designated, non-overlapping
  safe-export directory.

#### Controlled internal manifest

The internal manifest is private and must not be committed. Its top-level
schema is:

```yaml
schema_version: 1
export_namespace: "<deliberately-generated-uuid>"
trust_attestation:
  attested_by: "<authorized-reviewer>"
  attested_on: "<review-date>"
  evidence: "<non-sensitive-controlled-review-reference>"
  source_controlled: true
  checksums_verified: true
datasets:
  - dataset: abide
    input_path: "<controlled-absolute-input-path>"
    sha256: "<lowercase-sha256-of-controlled-input>"
    input_format: pandas_dataframe
    atlas: schaefer_100
    expected_regions: 100
    source_identity_attestation:
      dataset: abide
      approved_by: "<authorized-reviewer>"
      evidence: "<non-sensitive-source-identity-reference>"
    schema:
      timeseries_column: TimeSeries
      metadata_columns: {}
      fold_column: null
      partition_column: null
      matrix_type: correlation
      ignored_internal_columns:
        - SubjectID
```

Each dataset entry must contain exactly:

- `dataset`;
- absolute controlled `input_path`;
- verified lowercase `sha256`;
- `input_format: pandas_dataframe`;
- supported `atlas`;
- matching `expected_regions`;
- `source_identity_attestation` with dataset, reviewer, and evidence;
- `schema`.

Within `schema`, declare exactly one of `connectome_column` or
`timeseries_column`. `metadata_columns` maps approved public names to internal
source columns; an empty mapping releases only the generated `sample_uid`.
`fold_column` and `partition_column` are optional and must be declared together
for exact split export. A declared connectome column may use `correlation`,
`covariance`, or `spd` when the supplied matrices satisfy that contract. A
time-series column currently supports only `correlation`, because the trusted
exporter applies the benchmark's OAS-to-correlation construction.
`ignored_internal_columns` is an explicit non-release declaration, not an
alternate allowlist.

Do not place credentials, participant lists, original filenames, or private
agreement text in the manifest.

#### Approval-binding dry run

Set controlled locations only in the trusted shell:

```bash
export INTERNAL_MANIFEST="<controlled-internal-manifest>"
export SAFE_EXPORT_DIR="<controlled-safe-export-directory>"
```

Run:

```bash
python -m tools.zenodo.export_internal \
  --config configs/release/zenodo_release.yaml \
  --trusted-internal-input \
  --input-manifest "$INTERNAL_MANIFEST" \
  --output-dir "$SAFE_EXPORT_DIR" \
  --dry-run
```

Dry run validates the manifest and policy, opens the named controlled input only
as bytes to verify its regular-file status and pinned SHA-256, and does not
deserialize the pickle or write output. Its plan includes
`source_binding_sha256`. The committed real-release configuration remains
fail-closed until the emitted bindings, reviewer identity, evidence references,
and final metadata approvals are complete. The public metadata allowlists
already limit ABIDE and COBRE to `sample_uid`, and CamCAN to `sample_uid`,
`age`, and `sex`.

An authorized reviewer must then:

1. verify the planned dataset, scope, source identity, checksum, schema, atlas,
   and region count;
2. copy the emitted binding into the matching dataset approval record;
3. fill dataset, atlas, `n_regions`, approval identity/date/scope/evidence, and
   confirmation;
4. add only the approved public columns to the dataset-specific metadata
   allowlist;
5. rerun dry run and confirm every requested scope is explicitly allowed.

ADNI, ADNI-DOD, and OASIS-3 participant-level export remains forbidden; do not
change their records merely to make the command pass.

#### Actual trusted export

Only after the dry-run review succeeds may the human run:

```bash
python -m tools.zenodo.export_internal \
  --config configs/release/zenodo_release.yaml \
  --trusted-internal-input \
  --input-manifest "$INTERNAL_MANIFEST" \
  --output-dir "$SAFE_EXPORT_DIR"
```

The output directory must be absent and must not overlap an input.
The safe export contains only deterministic numeric NPZ, allowlisted TSV, a
marker, and a manifest with checksums and policy bindings. NPZ writing forbids
pickle and object dtype. Export is assembled in a private sibling temporary
directory, privacy-scanned, and atomically renamed only after every check passes.

### Stage 2: public release build

Stage 2 must receive only the reviewed safe-export directory. It does not
import or call a pickle loader.

Set:

```bash
export SAFE_EXPORT_DIR="<reviewed-safe-export-directory>"
export ZENODO_OUTPUT_DIR="zenodo_upload"
```

Plan without writing:

```bash
python -m tools.zenodo.build_release \
  --config configs/release/zenodo_release.yaml \
  --safe-export-dir "$SAFE_EXPORT_DIR" \
  --output-dir "$ZENODO_OUTPUT_DIR" \
  --dry-run
```

Build a fresh staging tree:

```bash
python -m tools.zenodo.build_release \
  --config configs/release/zenodo_release.yaml \
  --safe-export-dir "$SAFE_EXPORT_DIR" \
  --output-dir "$ZENODO_OUTPUT_DIR"
```

The builder rejects unmanifested files, symlinks, unsafe extensions, unknown
datasets/categories/columns, incomplete participant-level approvals, source
binding mismatches, and input/output overlap. It refuses to overwrite an
existing staging directory. Dry run performs the same NPZ, TSV, split, checksum,
and privacy validation as a real build, but does not create output.

Every production aggregate artifact also requires an exact
`reviewed_artifacts` entry under the aggregate review record. The entry binds
canonical dataset, content category, safe-export relative path, and SHA-256;
changing or adding a file therefore requires a new deliberate review.

The configured publication basename is
`spd_connectome_benchmark_v0.1.0`, derived from the current `pyproject.toml`
version. The workflow does not create a `v1.0.0` tag or silently promote the
version.

## Safe public formats

- Connectomes: NPZ containing numeric `connectomes` with shape
  `(n_samples, n_regions, n_regions)`.
- Metadata and split tables: TSV or CSV with exact allowlisted headers.
- Aggregate metrics and statistical tables: TSV, CSV, or JSON with exact
  category allowlists.
- Configuration: YAML or JSON.
- Documentation: Markdown or plain text.

Public NPZ validation always uses:

```python
numpy.load(path, allow_pickle=False)
```

Arrays must be float32 or float64, finite, square, symmetric, and positive
definite. The default release binds correlation matrices to Schaefer-100
(`100 × 100`), checks an approximately unit diagonal, and uses the tolerances
recorded in `zenodo_release.yaml`.

Object arrays, pickle, NPY, NIfTI, DICOM, raw BIDS, time series, checkpoints,
individual predictions, embeddings, residuals, source paths, original
filenames, and original/hashed identifiers are blocked.

## Split handling

For an explicitly approved public dataset:

- memberships reference only release-safe `sample_uid` values;
- UIDs must exist in the approved metadata table and be unique;
- train, validation, and test roles must be recognized;
- memberships must be unique and disjoint within a fold;
- duplicates or unknown UIDs fail validation.

For restricted datasets, the package contains no original, hashed, or public
mapping to participants and no exact memberships. Reconstruction folders record
selection logic, atlas, matrix construction, seeds, grouping rules, and
aggregate fold information only.

## Expected staging tree

Unapproved public dataset directories are omitted. The builder creates a tree
of this form:

```text
zenodo_upload/
  staging/
    spd_connectome_benchmark_v0.1.0/
      README.md
      DATASET_CARD.md
      LICENSES.md
      VERSION
      CITATION.cff                    # generated only after creators are complete
      metadata/
        zenodo_record_metadata.json
        zenodo_form_values.md
        dataset_inventory.tsv
        data_dictionary.tsv
        provenance.tsv
        release_policy_snapshot.yaml
        manual_approvals_snapshot.yaml
        metadata_allowlist_snapshot.yaml
        forbidden_patterns_snapshot.yaml
        release_config_snapshot.yaml
      data/
        README.md
        public_connectomes/
        public_metadata/
        camcan/                         # only after source-bound approval
          connectomes/
            camcan_schaefer100_fc.npz
          metadata/
            participants.tsv
          data_dictionary.tsv
          LICENSE.txt
      splits/
      restricted_reconstruction/
        adni/
          README.md
          selection_config.yaml
          reconstruction_commands.md
        adnidod/
          README.md
          selection_config.yaml
          reconstruction_commands.md
        oasis3/
          README.md
          selection_config.yaml
          reconstruction_commands.md
      configs/
        datasets/
        experiments/
        preprocessing/
      benchmark_results/
        aggregate_metrics/
        statistical_tests/
        figure_source_data/
      reproducibility/
        commands.md
        environment.yml
        requirements-lock.txt
        git_commit.txt
        software_versions.json
        repository_snapshot.json
      manifests/
        manifest.tsv
        SHA256SUMS.txt
        validation_report.json
        validation_report.md
```

## Validation

Dry-run validation checks that the staging directory exists without writing
reports:

```bash
python -m tools.zenodo.validate_release \
  --release-dir zenodo_upload/staging/spd_connectome_benchmark_v0.1.0 \
  --config configs/release/zenodo_release.yaml \
  --dry-run
```

Structural validation writes JSON and Markdown reports:

```bash
python -m tools.zenodo.validate_release \
  --release-dir zenodo_upload/staging/spd_connectome_benchmark_v0.1.0 \
  --config configs/release/zenodo_release.yaml
```

Publication-gate validation is stricter:

```bash
python -m tools.zenodo.validate_release \
  --release-dir zenodo_upload/staging/spd_connectome_benchmark_v0.1.0 \
  --config configs/release/zenodo_release.yaml \
  --publication-ready
```

It must remain incomplete until all manual metadata, licenses, approvals, and
artifact reviews are genuinely complete.

Validation checks:

1. required release files and policy snapshots exist;
2. no pickle, NIfTI, DICOM, raw BIDS, time-series, checkpoint, or forbidden
   extension exists;
3. symlinks do not escape the release and unsafe filesystem objects are absent;
4. no absolute local path is present;
5. every dataset and content category is known;
6. likely participant identifiers and original filenames are absent;
7. metadata columns are explicitly approved;
8. NPZ files load with `allow_pickle=False` and contain no object array;
9. arrays contain only finite values;
10. connectomes are three-dimensional and square;
11. connectomes are symmetric within tolerance;
12. correlation diagonals are approximately one;
13. SPD matrices are positive definite;
14. atlas names and region counts match;
15. connectome sample counts match metadata rows;
16. release-safe sample UIDs are valid and unique;
17. split roles, membership, uniqueness, and disjointness are valid;
18. restricted folders contain documentation/configuration only;
19. manifest paths, sizes, categories, policy decisions, and hashes match;
20. SHA-256 coverage and values match;
21. required Zenodo metadata contains no unresolved fake identifiers or
    incomplete manual decisions;
22. a packaged archive extracts safely and passes revalidation;
23. repeated archive builds from identical inputs are byte-deterministic.

Privacy findings identify the file and line when possible while redacting the
middle of a suspected value. Approved creator email addresses are the only
possible email exception, and only when explicitly present in creator metadata.

Validation does not replace legal, ethical, license, or governance review.

## Deterministic packaging

Plan:

```bash
python -m tools.zenodo.package_release \
  --release-dir zenodo_upload/staging/spd_connectome_benchmark_v0.1.0 \
  --upload-dir zenodo_upload/upload_files \
  --config configs/release/zenodo_release.yaml \
  --dry-run
```

After publication-ready validation succeeds, build the upload folder:

```bash
python -m tools.zenodo.package_release \
  --release-dir zenodo_upload/staging/spd_connectome_benchmark_v0.1.0 \
  --upload-dir zenodo_upload/upload_files \
  --config configs/release/zenodo_release.yaml
```

Publication packaging always requires `--config`. Every validation pass
compares that complete external configuration bundle exactly with the five
frozen snapshots in `metadata/`; any difference fails closed.

The package command:

1. requires and binds the external configuration bundle;
2. refuses incomplete metadata or approvals;
3. validates the staged release in publication-ready mode;
4. regenerates manifest/checksum catalogs after reports are final;
5. creates the ZIP twice with normalized metadata and compares hashes;
6. extracts the archive into a temporary directory;
7. rejects unsafe archive members;
8. revalidates the extracted tree against the external configuration;
9. writes the upload companions and their checksums.

`--allow-synthetic-test-package` exists only for a configuration with
`release.test_only: true`; such an artifact must never be uploaded.

For a read-only audit of a preserved archive when its external configuration
bundle is unavailable, snapshot-only packaging must be requested explicitly:

```bash
python -m tools.zenodo.package_release \
  --release-dir "<extracted-release-directory>" \
  --upload-dir "<new-verification-directory>" \
  --archive-verification \
  --dry-run
```

Archive-verification output is marked `DO NOT UPLOAD OR PUBLISH`; embedded
snapshots alone do not establish publication readiness.

The upload directory is:

```text
zenodo_upload/upload_files/
  spd_connectome_benchmark_v0.1.0.zip
  README.md
  LICENSES.md
  SHA256SUMS.txt
  zenodo_record_metadata.json
  ZENODO_UPLOAD_CHECKLIST.md
```

The final ZIP excludes internal exports, synthetic fixtures, caches, and
validation scratch files.

## Human upload procedure

There is deliberately no API upload command.

1. Confirm the upload directory checksum file locally.
2. Open a new Zenodo dataset deposit through the authorized human account.
3. Upload only the files listed in `upload_files/SHA256SUMS.txt`.
4. Select resource type **Dataset**.
5. Compare every web-form value with
   `zenodo_record_metadata.json`, `ZENODO_UPLOAD_CHECKLIST.md`, and
   `metadata/zenodo_form_values.md`.
6. Manually verify creators/order, affiliations, ORCIDs, description, licenses,
   funding, communities, related identifiers, and manuscript DOI against
   authoritative records.
7. Confirm the displayed upload filenames and sizes.
8. Leave the deposit unpublished if any approval, field, checksum, or preview
   differs.
9. Only an authorized human may select Zenodo's publish action.
10. Record the Zenodo-created DOI after publication; do not invent or reserve
    one in advance.

The metadata JSON is a companion and possible API-shaped input. Uploading it as
a file does not automatically populate every Zenodo web-form field.

## Required manual fields

The current real-release configuration intentionally blocks publication until
reviewers confirm:

- complete creator list and creator order;
- affiliations and authoritative ORCIDs, if available;
- funding or an explicit record that there is none;
- manuscript DOI and related identifiers, or explicit absence;
- source-code license scope;
- derived-data license and exact artifact scope;
- documentation license;
- dataset-specific redistribution evidence and approved fields;
- source/atlas/region/checksum attestation binding for any participant-level
  safe export;
- aggregate result content and PDF metadata;
- privacy scan, data dictionary, inventory, provenance, manifest, and checksums;
- clean source revision and final Zenodo form.

No value should be inferred from Git statistics, a copyright line, a dataset
name, or common practice.

Metadata, license, privacy, catalog, final-form, and publication-gate approvals
are rebuild-specific human records, not cryptographic signatures. Any change
to configuration, release content, repository revision, or generated catalogs
operationally invalidates them: reset the affected status to
`manual_required`, rebuild, and repeat the review. Aggregate-artifact approvals
additionally require the exact reviewed path and SHA-256 binding enforced by
the validator.

## Rebuilding a new version

1. Start from a reviewed clean repository commit.
2. Choose a semantic version deliberately and update `pyproject.toml` and
   `project.version` together.
3. Update `release.archive_basename` to
   `spd_connectome_benchmark_v<version>`.
4. Re-review policies, metadata allowlists, source bindings, licenses, and
   approvals; never carry them forward automatically.
5. Generate a fresh safe export in a new empty controlled directory.
6. Build into a new empty output directory.
7. Validate in structural and publication-ready modes.
8. Package, extract, and revalidate.
9. Compare the upload checklist manually.

A new Git tag, GitHub software release, Zenodo version, and related identifiers
are separate human-authorized operations. This workflow performs none of them.

After Zenodo publishes a version, preserve its frozen archive. Corrections
require a new reviewed version rather than silently replacing local files.
Relate versions only with identifiers that Zenodo or another authoritative
record has actually issued.

## GitHub, GitHub software releases, and Zenodo

- **GitHub repository:** evolving code, tests, configuration, and
  documentation.
- **GitHub software release:** an optional frozen source-code distribution,
  created only through a separate authorized workflow.
- **Zenodo dataset record:** the frozen, versioned research-resource archive
  produced here, limited to policy-approved materials.

A commit hash establishes code provenance but is not a DOI. A GitHub tag does
not grant data redistribution rights. A Zenodo dataset record does not
automatically license repository code or original datasets.

## Handoff summary

The operational sequence is:

1. run the [approval-binding dry run](#approval-binding-dry-run);
2. perform the [trusted internal export](#actual-trusted-export);
3. create a fresh [public release build](#stage-2-public-release-build);
4. run [structural and publication-ready validation](#validation);
5. create the [deterministic upload package](#deterministic-packaging);
6. complete the [human Zenodo review](#human-upload-procedure).

Stop if any command refuses the operation. A failing check represents an
unresolved input, policy, metadata, or review condition; it must be resolved at
its source and followed by a fresh build.
