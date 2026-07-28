from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from spd_connectome_benchmark.analysis_outputs.dataset_description import (
    _build_dataset_artifacts,
    _reject_stale_identifier_outputs,
    _write_dataset_tables,
)


def test_dataset_description_writes_aggregates_without_identifiers(tmp_path):
    frame = pd.DataFrame(
        {
            "SubjectID": ["synthetic-a", "synthetic-b"],
            "TimeSeries": [np.ones((120, 4)), np.ones((125, 4))],
            "Age": [30.0, 40.0],
            "Sex": ["F", "M"],
            "Diagnosis": [0, 1],
        }
    )
    dataset_row, diagnosis_frame, qc_row, context_row = _build_dataset_artifacts(
        "cobre",
        frame,
    )

    _write_dataset_tables(
        tmp_path,
        [dataset_row],
        diagnosis_frame.to_dict("records"),
        [qc_row],
        [context_row],
    )

    generated_names = {
        path.relative_to(tmp_path).as_posix()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert "support/scan_level_metadata.csv" not in generated_names
    assert "support/abide_excluded_after_fetch.csv" not in generated_names
    assert generated_names == {
        "table1_dataset_summary.csv",
        "support/dataset_context.csv",
        "support/diagnosis_summary.csv",
        "support/qc_motion_summary.csv",
    }
    for path in tmp_path.rglob("*.csv"):
        columns = {column.lower() for column in pd.read_csv(path).columns}
        assert "subjectid" not in columns
        assert "subject_id" not in columns
        assert "scan_id" not in columns
        assert "session" not in columns


def test_dataset_description_rejects_stale_identifier_outputs(tmp_path):
    stale_path = tmp_path / "support" / "scan_level_metadata.csv"
    stale_path.parent.mkdir()
    stale_path.write_text("SubjectID\nsynthetic-only\n")

    with pytest.raises(RuntimeError, match="legacy participant-level tables"):
        _reject_stale_identifier_outputs(tmp_path)


def test_make_results_rejects_stale_identifier_outputs_before_any_subcommand(
    monkeypatch,
    tmp_path,
):
    make_results = pytest.importorskip("make_results")
    stale_path = tmp_path / "support" / "abide_excluded_after_fetch.csv"
    stale_path.parent.mkdir()
    stale_path.write_text("SubjectID\nsynthetic-only\n")
    ran = []
    args = Namespace(table_dir=tmp_path, run=lambda _: ran.append(True))
    monkeypatch.setattr(make_results, "parse_args", lambda argv: args)

    with pytest.raises(RuntimeError, match="legacy participant-level tables"):
        make_results.main(["lodo-tables"])

    assert ran == []
