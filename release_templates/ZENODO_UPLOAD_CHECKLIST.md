{{SYNTHETIC_BANNER}}# Zenodo upload checklist

## Confirmed dataset permissions

Dataset permission confirmed for the three derived-data scopes below.

- [x] CamCAN derived-data sharing permission confirmed
- [x] ABIDE derived-data sharing conditions confirmed
- [x] COBRE derived-data sharing conditions confirmed

These confirmations authorize only the dataset-specific derived-connectome
scopes recorded in `LICENSES.md`. They do not replace source binding, privacy
review, or final human publication approval.

## Artifact and policy checks

- [ ] Confirm each real safe export is bound to the reviewed source SHA-256,
  atlas, and region count.
- [ ] Confirm every included metadata column matches its dataset-specific
  allowlist.
- [ ] Confirm raw MRI, NIfTI, DICOM, T1-weighted images, source paths, original
  identifiers, unrestricted clinical text, Home Interview variables,
  questionnaire responses, and arbitrary behavioural variables are absent.
- [ ] Confirm source-code, documentation, and all dataset-specific terms.
- [ ] Confirm the version and clean Git revision.
- [ ] Re-run release validation in publication-ready mode.
- [ ] Verify `{{ARCHIVE_NAME}}` against `SHA256SUMS.txt`.
- [ ] Review the Zenodo web form; do not assume the JSON populates it.
- [ ] Preview the draft record before publication.
- [ ] Record the DOI only after Zenodo creates it.

## Final approvals

- [ ] final metadata verification
- [ ] final author verification
- [ ] final Zenodo upload approval

This package builder does not call Zenodo, reserve a DOI, create a Git tag, or
publish a GitHub release.
