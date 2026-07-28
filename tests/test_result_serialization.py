import json
import sys

import pytest

from spd_connectome_benchmark.results import (
    portable_result_reference,
    save_metrics_csv,
)


def test_result_serialization_rejects_wrong_fold_count(tmp_path):
    with pytest.raises(ValueError, match="expected 2"):
        save_metrics_csv(
            {"MAE": [1.0]},
            n_splits=2,
            out_csv=str(tmp_path / "bad.csv"),
            elapsed=0.1,
        )


def test_metadata_sidecar_redacts_paths(tmp_path, monkeypatch):
    private_path = "/restricted/source/participant-data"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_benchmark.py",
            "--data_root",
            private_path,
            "--results_folder=/private/results",
            "--harmonized_spd_cache_dirs",
            "/restricted/cache-one",
            "/restricted/cache-two",
            "--seed",
            "1",
        ],
    )
    monkeypatch.setenv("RSFMRI_SPD_DATA_ROOT", private_path)
    csv_path = tmp_path / "metrics.csv"

    save_metrics_csv(
        {"MAE": [1.0, 2.0]},
        n_splits=2,
        out_csv=str(csv_path),
        elapsed=0.5,
    )

    payload = json.loads((tmp_path / "metrics.csv.metadata.json").read_text())
    serialized = json.dumps(payload)
    assert payload["metadata_schema_version"] == 2
    assert "cwd" not in payload
    assert private_path not in serialized
    assert "/private/results" not in serialized
    assert "/restricted/cache-one" not in serialized
    assert "/restricted/cache-two" not in serialized
    assert payload["environment_is_set"]["RSFMRI_SPD_DATA_ROOT"] is True


def test_portable_result_reference_never_contains_output_root(tmp_path):
    output_root = tmp_path / "private-user-root"
    result_path = output_root / "cobre" / "metrics.csv"

    reference = portable_result_reference(result_path, output_root=output_root)

    assert reference == "cobre/metrics.csv"
    assert str(tmp_path) not in reference
