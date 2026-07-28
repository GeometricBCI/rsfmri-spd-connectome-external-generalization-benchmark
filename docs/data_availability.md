# Data availability and redistribution

The GitHub repository contains software, tests, configuration, and
documentation. Participant-level data are never committed to Git.

The companion Zenodo workflow has a narrower purpose: it can package only the
derived files that are explicitly permitted by a dataset-specific policy,
bound to a reviewed source artifact, and accepted by the privacy and schema
validators. Permission to use a source dataset does not automatically grant
permission to redistribute every derived field.

## Dataset handling

| Dataset | Public-release handling |
|---|---|
| ABIDE | Confirmed derived-connectome scope under the recorded ABIDE/INDI terms. The current public metadata allowlist contains only a release-generated `sample_uid`. |
| COBRE | Confirmed derived-connectome scope under the recorded COBRE/FCP terms. The current public metadata allowlist contains only a release-generated `sample_uid`. |
| CamCAN | Confirmed participant-level derived-connectome scope under the project-held written permission. The current public metadata allowlist contains `sample_uid`, age, and sex. |
| ADNI | Participant-level release is forbidden. Researchers must obtain access independently from the source custodian. |
| ADNI-DOD | Participant-level release is forbidden. Researchers must obtain access independently under the governing terms. |
| OASIS-3 | Participant-level release is forbidden. Researchers must obtain access independently under the governing terms. |
| 1000BRAINS and unknown datasets | Non-public unless the policy and approval records are deliberately changed and reviewed. |

For ABIDE, COBRE, and CamCAN, “confirmed” describes the permitted
derived-data scope. A specific export is still blocked until its input checksum,
declared schema, atlas, region count, metadata columns, reviewer, and
non-sensitive evidence reference are bound together. The final Zenodo record
also requires a separate human publication review.

## Local reconstruction

Researchers working with the source datasets should:

1. obtain access from the original custodian;
2. review the current terms for the intended use;
3. keep source and prepared data outside the Git checkout;
4. reconstruct `atlas_<atlas>/<dataset>_X_y.pkl` with the appropriate local
   preparation path;
5. record a private checksum and provenance manifest;
6. inspect the resolved experiment with `run_benchmark.py --dry-run`;
7. run the benchmark only in the authorized environment.

The repository does not distribute credentials, data-use agreements,
participant lists, private source paths, or participant-data download links.
Prepared pickle files must be treated as trusted artifacts because
deserializing an untrusted pickle can execute code.

## What may be public

GitHub may contain:

- source code and tests;
- logical configuration examples;
- aggregate metrics and reviewed figures or tables;
- documentation that contains no participant information.

A reviewed Zenodo dataset release may additionally contain:

- ABIDE, COBRE, and CamCAN derived connectomes admitted by the current policy;
- release-generated `sample_uid` values;
- CamCAN age and sex, and no other participant metadata;
- manifests, checksums, data dictionaries, provenance, and license notices.

Neither public channel may contain:

- original, encoded, or hashed participant identifiers;
- raw MRI, NIfTI, DICOM, BIDS, or ROI time-series files;
- source filenames, local paths, credentials, or access agreements;
- exact participant-level benchmark split membership;
- unrestricted clinical text, questionnaire responses, or unapproved
  behavioural variables;
- participant-level ADNI, ADNI-DOD, OASIS-3, 1000BRAINS, or unknown-dataset
  materials.

## GitHub and Zenodo are separate releases

GitHub is the evolving software repository. Zenodo is a frozen, versioned
dataset record created only after licensing, privacy, provenance, authorship,
metadata, and checksum review. The release tools prepare and validate local
files; they do not publish a deposit, reserve a DOI, or grant redistribution
permission.
