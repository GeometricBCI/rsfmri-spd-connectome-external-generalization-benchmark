import subprocess
from pathlib import Path

import pytest


BLOCKED_FILENAMES = {
    "participants.tsv",
    "participants.json",
}
BLOCKED_SUFFIXES = (
    ".pkl",
    ".pickle",
    ".nii",
    ".nii.gz",
    ".dcm",
    ".ima",
    "_scans.tsv",
    "_sessions.tsv",
    "_events.tsv",
    "_phenotype.csv",
)
BLOCKED_PATH_COMPONENTS = {
    "sourcedata",
    "derivatives",
}


def test_git_tracks_no_participant_or_raw_neuroimaging_filenames():
    repository_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repository_root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        pytest.skip("Git worktree metadata is unavailable")

    tracked = [
        Path(raw_path.decode("utf-8"))
        for raw_path in completed.stdout.split(b"\0")
        if raw_path
    ]
    blocked = []
    for path in tracked:
        lowered = path.as_posix().lower()
        if (
            path.name.lower() in BLOCKED_FILENAMES
            or lowered.endswith(BLOCKED_SUFFIXES)
            or any(
                part.lower() in BLOCKED_PATH_COMPONENTS
                or part.lower().startswith(("sub-", "ses-"))
                for part in path.parts
            )
            or "phenotypic" in path.name.lower()
        ):
            blocked.append(path.as_posix())

    assert not blocked, "Tracked participant/raw-data filenames: " + ", ".join(blocked)


@pytest.mark.parametrize(
    "candidate",
    [
        "participants.tsv",
        "sub-0001/func/sub-0001_task-rest_events.tsv",
        "sourcedata/vendor/raw-source.bin",
        "derivatives/group/connectivity.tsv",
    ],
)
def test_gitignore_blocks_common_bids_and_source_data_paths(candidate):
    repository_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["git", "check-ignore", "--no-index", "--quiet", candidate],
        cwd=repository_root,
        check=False,
    )

    assert completed.returncode == 0, f"Sensitive path is not ignored: {candidate}"
