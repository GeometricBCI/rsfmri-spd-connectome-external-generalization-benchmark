import json
import pickle
import subprocess
import sys
from types import SimpleNamespace

import spd_connectome_benchmark.benchmark_cli as benchmark_cli
from spd_connectome_benchmark.benchmark_cli import dispatch_run, main
from spd_connectome_benchmark.configuration import resolve_config
from spd_connectome_benchmark.config import PAPER_DATASETS


def _filename_placeholders(root):
    atlas_dir = root / "atlas_schaefer_100"
    atlas_dir.mkdir(parents=True)
    for dataset in PAPER_DATASETS:
        (atlas_dir / f"{dataset}_X_y.pkl").touch()


def test_dry_run_never_deserializes_or_creates_outputs(tmp_path, monkeypatch, capsys):
    input_root = tmp_path / "input"
    output_root = tmp_path / "not-created"
    _filename_placeholders(input_root)

    def forbidden_pickle_load(*args, **kwargs):
        raise AssertionError("dry run attempted to deserialize participant data")

    monkeypatch.setattr(pickle, "load", forbidden_pickle_load)

    status = main(
        [
            "--dry-run",
            "--input-root",
            str(input_root),
            "--output-dir",
            str(output_root),
            "--model",
            "ridge",
            "--cv",
            "both",
            "--harm",
            "both",
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert status == 0
    assert report["valid"] is True
    assert len(report["experiments"]) == 4
    assert report["side_effects"] == {
        "models_run": False,
        "output_directories_created": False,
        "participant_files_opened": False,
    }
    assert str(tmp_path) not in json.dumps(report)
    assert not output_root.exists()


def test_dry_run_reports_all_missing_inputs_without_creating_output(tmp_path, capsys):
    output_root = tmp_path / "not-created"

    status = main(
        [
            "--dry-run",
            "--input-root",
            str(tmp_path / "missing"),
            "--output-dir",
            str(output_root),
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert status == 2
    assert len(report["missing_inputs"]) == len(PAPER_DATASETS)
    assert not output_root.exists()


def test_dispatch_maps_stable_config_to_legacy_pooled_runner(tmp_path, monkeypatch):
    captured = {}

    def args_parser(argv):
        captured["argv"] = list(argv)
        return SimpleNamespace(
            N_SPLITS=4,
            DATASETS=["cobre", "adni"],
            atlas_name="schaefer_100",
            task="Age",
            debug=None,
            rng_seed=13,
            ts_metric="riemann",
            ridge_alphas=[0.1, 1.0],
            dummy_strategy="mean",
            no_make_tag=False,
            algorithms=["ridge"],
            log_level="INFO",
        )

    def run_pooled_age_benchmarks(**kwargs):
        captured["run_kwargs"] = kwargs

    fake_runner = SimpleNamespace(
        args_parser=args_parser,
        configure_logging=lambda level: captured.setdefault("log_level", level),
        run_pooled_age_benchmarks=run_pooled_age_benchmarks,
    )
    monkeypatch.setitem(sys.modules, "run_pooled_benchmark", fake_runner)
    config = resolve_config(
        [
            "--model",
            "ridge",
            "--cv",
            "lodo",
            "--folds",
            "4",
            "--harm",
            "harm",
            "--datasets",
            "cobre",
            "adni",
            "--seed",
            "9",
            "--data-shuffle-seed",
            "13",
            "--input-root",
            str(tmp_path / "input"),
            "--output-dir",
            str(tmp_path / "output"),
        ],
        environ={},
        cwd=tmp_path,
    )

    dispatch_run(config)

    argv = captured["argv"]
    assert argv[argv.index("--protocol") + 1] == "lodo"
    assert argv[argv.index("--N_SPLITS") + 1] == "4"
    assert argv[argv.index("--harm_mode") + 1] == "harm"
    assert argv[argv.index("--seed") + 1] == "9"
    assert captured["run_kwargs"]["algorithms"] == ("ridge",)
    manifests = list((tmp_path / "output").glob("run_config_*.json"))
    assert len(manifests) == 1
    manifest_text = manifests[0].read_text()
    assert str(tmp_path) not in manifest_text
    manifest = json.loads(manifest_text)
    assert manifest["selection"]["seed"] == 9
    assert manifest["resolved_execution_parameters"]["N_SPLITS"] == 4
    assert manifest["resolved_execution_parameters"]["ridge_alphas"] == [0.1, 1.0]
    assert manifest["code_state"]["package_version"] == "0.1.0"
    assert "git_commit" in manifest["code_state"]


def test_git_code_state_reports_tracked_and_untracked_changes(tmp_path, monkeypatch):
    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    tracked_path = tmp_path / "tracked.txt"
    tracked_path.write_text("initial\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "tracked.txt"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Codex Test",
            "-c",
            "user.email=codex-test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "initial",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    (tmp_path / "untracked.txt").write_text("untracked\n", encoding="utf-8")
    monkeypatch.setattr(benchmark_cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(benchmark_cli, "SOURCE_CHECKOUT_ROOT", tmp_path)

    untracked_only = benchmark_cli._git_code_state()

    assert untracked_only["tracked_worktree_dirty"] is False
    assert untracked_only["untracked_worktree_dirty"] is True
    assert untracked_only["worktree_dirty"] is True

    tracked_path.write_text("modified\n", encoding="utf-8")
    tracked_and_untracked = benchmark_cli._git_code_state()

    assert tracked_and_untracked["tracked_worktree_dirty"] is True
    assert tracked_and_untracked["untracked_worktree_dirty"] is True
    assert tracked_and_untracked["worktree_dirty"] is True
