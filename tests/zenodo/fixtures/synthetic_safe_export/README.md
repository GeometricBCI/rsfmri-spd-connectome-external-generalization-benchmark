# Synthetic safe-export fixture

This directory documents the fixture shape used by `tests/zenodo`. It contains
no participant data, source identifiers, connectome binaries, or copied
research artifacts.

At test time, `conftest.py` and `_synthetic.py` generate a complete export under
pytest's temporary directory:

```text
.safe-export-root.json
export_manifest.json
datasets/
└── abide/
    ├── connectomes.npz
    ├── metadata.tsv
    └── splits.tsv
```

The logical dataset name is `abide`, but every matrix, release-safe sample UID,
metadata value, split, checksum, and source-binding digest is synthetic. The
connectomes are deterministic 100 × 100 SPD correlation matrices generated in
memory. No committed binary fixture is required.

`export_manifest.schema.json` records the narrow fixture contract without
providing a usable data manifest.
