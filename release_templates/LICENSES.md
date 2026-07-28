# License and terms inventory

Release version: `{{VERSION}}`

This inventory separates source-dataset terms, derived-data rights, software,
and documentation. No single Zenodo license silently replaces the
dataset-specific terms below.

## 1. Source dataset terms

CamCAN, ABIDE, and COBRE custodians retain their source-dataset rights and
terms. The project has confirmed permission to release the participant-level
derived functional-connectivity matrices described below. That confirmation
does not relicense raw source data or authorize material outside each explicit
scope.

ADNI, ADNI-DOD, OASIS-3, 1000BRAINS, and unknown datasets remain governed by
their fail-closed policy entries. This package grants no access to their source
data.

## 2. Dataset-specific derived-data rights

The record-level mixed-rights presentation is:

- License selection: {{DERIVED_DATA_LICENSE_IDENTIFIER}}
- Review status: {{DERIVED_DATA_LICENSE_APPROVAL_STATUS}}

If the selection is unresolved, a final reviewer must choose the appropriate
presentation. Even after that review, the record-level selection must not
override any section below.

### CamCAN-derived functional-connectivity data

- Release status: {{CAMCAN_RIGHTS_STATUS}}
- Dataset-specific license: {{CAMCAN_DATA_LICENSE}}
- Rights statement: {{CAMCAN_RIGHTS_STATEMENT}}
- Conditions: {{CAMCAN_CONDITIONS}}
- Required citation: {{CAMCAN_CITATIONS}}

Included when present in the manifest:

- participant-level Schaefer-100 ROI-to-ROI functional-connectivity matrices;
- release-safe `sample_uid`;
- age;
- sex.

Excluded:

- raw MRI and T1-weighted images;
- identifiable images;
- Home Interview variables;
- identifiable or other unapproved behavioural variables;
- `CCID`, original or hashed identifiers, source paths, and exact split
  membership.

The canonical CamCAN files are:

```text
data/camcan/connectomes/camcan_schaefer100_fc.npz
data/camcan/metadata/participants.tsv
data/camcan/data_dictionary.tsv
data/camcan/LICENSE.txt
```

The required cohort citation is Shafto et al. (2014), *The Cambridge Centre
for Ageing and Neuroscience (Cam-CAN) study protocol: a cross-sectional,
lifespan, multidisciplinary examination of healthy cognitive ageing*,
BMC Neurology 14:204,
<https://doi.org/10.1186/s12883-014-0204-1>.

### ABIDE-derived functional-connectivity data

- Release status: {{ABIDE_RIGHTS_STATUS}}
- Dataset-specific terms: {{ABIDE_DATA_LICENSE}}
- Rights statement: {{ABIDE_RIGHTS_STATEMENT}}
- Conditions: {{ABIDE_CONDITIONS}}
- Required citations and acknowledgements: {{ABIDE_CITATIONS}}

The confirmed scope covers participant-level derived functional-connectivity
matrices. Attribution, non-commercial-use, and applicable share-alike
conditions must be retained. Public participant metadata is limited to the
release-generated `sample_uid` until additional fields are explicitly
confirmed and added to the dataset-specific allowlist. Raw images, source
paths, original identifiers, unrestricted clinical text, questionnaire
responses, and arbitrary behavioural variables are excluded.

The
[official ABIDE I page](https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html)
describes non-commercial research use under a Creative Commons
Attribution-NonCommercial-ShareAlike license, but does not state a version.
The exact applicable title, version, URI, acknowledgements, and full
ABIDE/INDI citation text remain final metadata-review items.

### COBRE-derived functional-connectivity data

- Release status: {{COBRE_RIGHTS_STATUS}}
- Dataset-specific terms: {{COBRE_DATA_LICENSE}}
- Rights statement: {{COBRE_RIGHTS_STATEMENT}}
- Conditions: {{COBRE_CONDITIONS}}
- Required citations and attribution: {{COBRE_CITATIONS}}

The confirmed scope covers participant-level derived functional-connectivity
matrices under the confirmed COBRE/FCP redistribution conditions. Public
participant metadata is limited to the release-generated `sample_uid` until
additional fields are explicitly confirmed and allowlisted. Raw images,
source paths, original identifiers, unrestricted clinical text, questionnaire
responses, and arbitrary behavioural variables are excluded.

The
[official COBRE page](https://fcon_1000.projects.nitrc.org/indi/retro/cobre.html)
describes the data as available under a Creative Commons
Attribution-NonCommercial license, but does not state a version. The exact
applicable title, version, URI, attribution wording, and full COBRE/FCP/INDI
citation text remain final metadata-review items.

## 3. Software license

Repository source code is distributed under `BSD-3-Clause`, as documented in
the repository `LICENSE`. This dataset archive does not include executable
source files; it links to the separately maintained GitHub repository. If a
future archive adds scripts, their BSD license notice and third-party
attributions must be retained and reviewed. The software license does not
license any dataset.

## 4. Documentation license

- License selection: {{DOCUMENTATION_LICENSE_IDENTIFIER}}
- Approval status: `{{DOCUMENTATION_LICENSE_APPROVAL_STATUS}}`

The documentation license and its exact scope require final review. It must
not be inferred from either the software license or a dataset-specific data
license.

## Artifact-level mapping

Every uploaded file must resolve through the manifest's dataset, content,
access, and source categories to the applicable terms above. Unknown,
conflicting, or empty mappings block publication.

## No implied permissions

Deidentification, hashing, aggregation, repository visibility, or technical
convertibility does not by itself establish redistribution permission.
Confirmed permission and a valid source-bound artifact review are separate
requirements.

## Outstanding manual actions

Before publication, a designated human reviewer must:

- approve the Zenodo record-level mixed-rights presentation without replacing
  any dataset-specific condition;
- confirm the documentation-license identifier and scope;
- replace the draft ABIDE and COBRE labels with exact authoritative license,
  URI, attribution, acknowledgement, and citation wording;
- replace draft conversion attestations with authoritative, non-sensitive
  evidence references;
- approve the final manifest, checksums, archive, and Zenodo form.
