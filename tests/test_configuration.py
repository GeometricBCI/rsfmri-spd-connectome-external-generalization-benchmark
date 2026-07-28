import pytest

from spd_connectome_benchmark.config import (
    PROJECT_ROOT,
    ensure_data_path_outside_project,
)
from spd_connectome_benchmark.configuration import (
    ConfigurationError,
    resolve_config,
)


def test_config_precedence_cli_over_yaml_over_environment(tmp_path):
    yaml_dir = tmp_path / "config"
    yaml_dir.mkdir()
    config_path = yaml_dir / "experiment.yaml"
    config_path.write_text(
        "\n".join(
            [
                "models: [ridge]",
                "folds: 4",
                "input_root: yaml-data",
                "output_root: yaml-results",
            ]
        )
    )
    environ = {
        "RSFMRI_SPD_MODELS": "dummy",
        "RSFMRI_SPD_FOLDS": "3",
        "RSFMRI_SPD_DATA_ROOT": "env-data",
        "RSFMRI_SPD_OUTPUT_ROOT": "env-results",
    }

    config = resolve_config(
        [
            "--config",
            str(config_path),
            "--model",
            "spdnet",
            "--folds",
            "5",
            "--input-root",
            "cli-data",
        ],
        environ=environ,
        cwd=tmp_path,
    )

    assert config.models == ("spdnet",)
    assert config.folds == 5
    assert config.input_root == (tmp_path / "cli-data").resolve()
    assert config.output_root == (yaml_dir / "yaml-results").resolve()
    assert config.sources["models"] == "cli"
    assert config.sources["output_root"] == "yaml"


def test_yaml_relative_paths_are_resolved_from_config_file(tmp_path):
    config_dir = tmp_path / "nested"
    config_dir.mkdir()
    config_path = config_dir / "run.yaml"
    config_path.write_text("input_root: data\noutput_root: output\n")

    config = resolve_config(
        ["--config", str(config_path)],
        environ={},
        cwd=tmp_path,
    )

    assert config.input_root == (config_dir / "data").resolve()
    assert config.output_root == (config_dir / "output").resolve()


def test_unknown_yaml_key_is_rejected(tmp_path):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("classification_model: xgb\n")

    with pytest.raises(ConfigurationError, match="Unknown configuration keys"):
        resolve_config(["--config", str(config_path)], environ={}, cwd=tmp_path)


def test_yaml_false_dry_run_is_not_treated_as_truthy_text(tmp_path):
    config_path = tmp_path / "run.yaml"
    config_path.write_text("dry_run: false\n")

    config = resolve_config(["--config", str(config_path)], environ={}, cwd=tmp_path)

    assert config.dry_run is False


def test_cli_can_override_yaml_dry_run(tmp_path):
    config_path = tmp_path / "run.yaml"
    config_path.write_text("dry_run: true\n")

    config = resolve_config(
        ["--config", str(config_path), "--no-dry-run"],
        environ={},
        cwd=tmp_path,
    )

    assert config.dry_run is False
    assert config.sources["dry_run"] == "cli"


def test_non_yaml_config_is_rejected_without_reading(monkeypatch, tmp_path):
    participant_path = tmp_path / "synthetic_X_y.pkl"
    participant_path.write_bytes(b"synthetic test marker")
    original_read_text = type(participant_path).read_text

    def guarded_read_text(path, *args, **kwargs):
        if path == participant_path:
            raise AssertionError("non-YAML participant file was opened")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(type(participant_path), "read_text", guarded_read_text)

    with pytest.raises(ConfigurationError, match=r"\.yaml or \.yml"):
        resolve_config(
            ["--config", str(participant_path), "--dry-run"],
            environ={},
            cwd=tmp_path,
        )


def test_defaults_do_not_inherit_unpassed_process_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("RSFMRI_SPD_DATA_ROOT", str(tmp_path / "process-data"))
    monkeypatch.setenv("RSFMRI_SPD_OUTPUT_ROOT", str(tmp_path / "process-output"))

    config = resolve_config([], environ={}, cwd=tmp_path)

    assert config.input_root == (PROJECT_ROOT.parent / "rsfmri_spd_data").resolve()
    assert config.output_root == (PROJECT_ROOT / "results" / "benchmark_csv").resolve()
    assert config.sources["input_root"] == "default"
    assert config.sources["output_root"] == "default"


def test_output_inside_repository_must_use_ignored_results_tree():
    with pytest.raises(ConfigurationError, match="must be under results"):
        resolve_config(
            ["--output-dir", str(PROJECT_ROOT / "custom-output")],
            environ={},
        )


def test_preparation_data_root_must_be_outside_repository():
    with pytest.raises(ValueError, match="outside the Git repository"):
        ensure_data_path_outside_project(PROJECT_ROOT / "data", label="test root")


def test_stable_input_root_must_be_outside_project():
    with pytest.raises(ConfigurationError, match="input_root must be outside"):
        resolve_config(
            ["--input-root", str(PROJECT_ROOT / "data")],
            environ={},
        )


def test_output_environment_roots_have_consistent_semantics(tmp_path):
    shared = resolve_config(
        [],
        environ={"RSFMRI_SPD_OUTPUT_ROOT": "shared-results"},
        cwd=tmp_path,
    )
    specific = resolve_config(
        [],
        environ={
            "RSFMRI_SPD_OUTPUT_ROOT": "shared-results",
            "RSFMRI_SPD_BENCHMARK_OUTPUT_ROOT": "exact-benchmark-output",
        },
        cwd=tmp_path,
    )

    assert shared.output_root == (tmp_path / "shared-results" / "benchmark_csv").resolve()
    assert specific.output_root == (tmp_path / "exact-benchmark-output").resolve()


@pytest.mark.parametrize("field", ["input_root", "output_root"])
@pytest.mark.parametrize("bad_value", ["null", "[]", "7", "''"])
def test_malformed_yaml_path_values_are_configuration_errors(
    tmp_path,
    field,
    bad_value,
):
    config_path = tmp_path / "invalid-path.yaml"
    config_path.write_text(f"{field}: {bad_value}\n")

    with pytest.raises(ConfigurationError, match="Path values"):
        resolve_config(["--config", str(config_path)], environ={}, cwd=tmp_path)


@pytest.mark.parametrize(
    ("input_root", "output_root"),
    [
        ("shared", "shared"),
        ("shared", "shared/output"),
        ("shared/input", "shared"),
    ],
)
def test_input_and_output_roots_must_not_overlap(
    tmp_path,
    input_root,
    output_root,
):
    with pytest.raises(ConfigurationError, match="must not be equal or contain"):
        resolve_config(
            [
                "--input-root",
                str(tmp_path / input_root),
                "--output-dir",
                str(tmp_path / output_root),
            ],
            environ={},
            cwd=tmp_path,
        )


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--target", "Diagnosis"], "chronological age"),
        (["--folds", "1"], "at least 2"),
        (["--model", "xgb"], "Unsupported model"),
        (["--datasets", "../private"], "Invalid dataset"),
        (["--atlas", "schaefer_400"], "Unsupported atlas"),
    ],
)
def test_unsupported_or_malformed_config_is_rejected(tmp_path, argv, message):
    with pytest.raises(ConfigurationError, match=message):
        resolve_config(argv, environ={}, cwd=tmp_path)
