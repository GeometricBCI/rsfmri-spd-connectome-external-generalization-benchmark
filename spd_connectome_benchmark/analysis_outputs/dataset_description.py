"""Generate dataset description tables used by Paper Figure 2 and Table 1."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import pandas as pd

from spd_connectome_benchmark.config import DEFAULT_ATLAS_DIR, DEFAULT_RAW_DATA_DIR, DEFAULT_TABLES_DIR, PAPER_DATASETS


# Paper §2.1/§2.2 and Table 1 introduce the six benchmark datasets.
DATASETS = list(PAPER_DATASETS)
DEFAULT_RAW_ABIDE_DIR = DEFAULT_RAW_DATA_DIR / "ABIDE_pcp"
FORBIDDEN_IDENTIFIER_OUTPUTS = (
    Path("support/scan_level_metadata.csv"),
    Path("support/abide_excluded_after_fetch.csv"),
)


DATASET_CONTEXT = {
    "abide": {
        "research_positioning": "neurodevelopmental / psychiatric",
        "primary_population": "autism spectrum disorder and controls",
        "multi_center_or_single_center": "multi-center",
        "acquisition_notes": "site/scanner details not retained in processed table",
    },
    "adni": {
        "research_positioning": "aging / dementia",
        "primary_population": "CN / SMC / MCI / AD spectrum",
        "multi_center_or_single_center": "unavailable_from_processed_table",
        "acquisition_notes": "processed table retains sessions but not scanner protocol fields",
    },
    "adnidod": {
        "research_positioning": "aging / dementia",
        "primary_population": "older adults in ADNI-DOD cohort",
        "multi_center_or_single_center": "multi-center",
        "acquisition_notes": "site IDs retained, scanner protocol fields unavailable",
    },
    "camcan": {
        "research_positioning": "lifespan normative",
        "primary_population": "community-dwelling healthy adults",
        "multi_center_or_single_center": "unavailable_from_processed_table",
        "acquisition_notes": "processed table retains age/sex only",
    },
    "cobre": {
        "research_positioning": "psychiatric",
        "primary_population": "schizophrenia and controls",
        "multi_center_or_single_center": "unavailable_from_processed_table",
        "acquisition_notes": "processed table retains diagnosis but no scanner/site columns",
    },
    "oasis3": {
        "research_positioning": "aging / dementia",
        "primary_population": "aging and dementia spectrum",
        "multi_center_or_single_center": "unavailable_from_processed_table",
        "acquisition_notes": "processed table retains sessions but not scanner protocol fields",
    },
}


def _load_df(dataset: str, pkl_dir: Path) -> pd.DataFrame:
    with open(pkl_dir / f"{dataset}_X_y.pkl", "rb") as f:
        return pickle.load(f)


def _extract_abide_subject_id(path: Path) -> int | None:
    match = re.search(r"(\d{7})", path.name)
    return int(match.group(1)) if match else None


def _missing_abide_cache_selection(processed_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "step": "final benchmark sample",
                "n_scans": int(len(processed_df)),
                "n_excluded_from_previous": "unavailable",
                "criterion": "Processed ABIDE table available; raw ABIDE PCP cache not found.",
            }
        ]
    )


def _abide_functional_subjects(func_dir: Path) -> set[int]:
    func_paths = sorted(func_dir.glob("*_func_preproc.nii.gz"))
    return {
        subject_id
        for path in func_paths
        for subject_id in [_extract_abide_subject_id(path)]
        if subject_id is not None
    }


def _integer_id_set(series: pd.Series) -> set[int]:
    return set(pd.to_numeric(series, errors="coerce").dropna().astype(int))


def _abide_selection_table(
    n_phenotypic_rows: int,
    phenotypic_ids: set[int],
    matched_ids: set[int],
    processed_ids: set[int],
    excluded_after_fetch: list[int],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "step": "ABIDE I phenotypic table",
                "n_scans": int(n_phenotypic_rows),
                "n_excluded_from_previous": 0,
                "criterion": "Rows in Phenotypic_V1_0b_preprocessed1.csv.",
            },
            {
                "step": "PCP CPAC nofilt_noglobal functional files with matching phenotype",
                "n_scans": int(len(matched_ids)),
                "n_excluded_from_previous": int(len(phenotypic_ids - matched_ids)),
                "criterion": (
                    "Subjects retained by the local ABIDE PCP cache used by nilearn.fetch_abide_pcp; "
                    "these have a functional file and a phenotype row."
                ),
            },
            {
                "step": "final Schaefer-100 time-series table used in benchmark",
                "n_scans": int(len(processed_ids)),
                "n_excluded_from_previous": int(len(excluded_after_fetch)),
                "criterion": (
                    "Subjects retained in atlas_schaefer_100/abide_X_y.pkl after confound regression, "
                    "atlas extraction, and final processed-table construction."
                ),
            },
        ]
    )


def _abide_sample_selection(
    processed_df: pd.DataFrame,
    raw_abide_dir: Path,
) -> pd.DataFrame:
    """Reconstruct aggregate ABIDE counts without exporting identifiers."""
    phenotypic_path = raw_abide_dir / "Phenotypic_V1_0b_preprocessed1.csv"
    func_dir = raw_abide_dir / "cpac" / "nofilt_noglobal"
    if not phenotypic_path.exists() or not func_dir.exists():
        return _missing_abide_cache_selection(processed_df)

    phenotypic = pd.read_csv(phenotypic_path)
    functional_subjects = _abide_functional_subjects(func_dir)
    phenotypic_ids = _integer_id_set(phenotypic["SUB_ID"])
    matched_ids = phenotypic_ids & functional_subjects
    processed_ids = _integer_id_set(processed_df["SubjectID"])
    excluded_after_fetch = sorted(matched_ids - processed_ids)

    return _abide_selection_table(
        n_phenotypic_rows=len(phenotypic),
        phenotypic_ids=phenotypic_ids,
        matched_ids=matched_ids,
        processed_ids=processed_ids,
        excluded_after_fetch=excluded_after_fetch,
    )


def _age_stats(age: pd.Series) -> dict:
    age = pd.to_numeric(age, errors="coerce").dropna()
    return {
        "age_mean": float(age.mean()),
        "age_std": float(age.std(ddof=0)),
        "age_min": float(age.min()),
        "age_q1": float(age.quantile(0.25)),
        "age_median": float(age.median()),
        "age_q3": float(age.quantile(0.75)),
        "age_max": float(age.max()),
    }


def _sex_counts(df: pd.DataFrame) -> tuple[dict, str]:
    if "Sex" not in df.columns:
        return {}, "unavailable"
    counts = df["Sex"].fillna("missing").astype(str).value_counts(dropna=False).to_dict()
    return counts, json.dumps(counts, sort_keys=True)


def _site_summary(df: pd.DataFrame) -> tuple[str, int | str]:
    site_col = None
    for candidate in ("Site", "SITE", "SiteID"):
        if candidate in df.columns:
            site_col = candidate
            break
    if site_col is None:
        return "unavailable_from_processed_table", "unavailable"
    n_sites = int(df[site_col].nunique(dropna=True))
    return site_col, n_sites


def _diagnosis_mapping(dataset: str) -> tuple[dict | None, str]:
    if dataset == "abide":
        return {0: "control", 1: "ASD"}, "Mapped from DX_GROUP-derived binary label"
    if dataset == "adnidod":
        return {0: "PTSD-negative", 1: "PTSD-positive"}, "Mapped from ADNIDOD binary diagnosis code"
    if dataset == "cobre":
        return {0: "control", 1: "schizophrenia"}, "Mapped from COBRE diagnosis preprocessing rule"
    if dataset == "oasis3":
        return (
            {
                0.0: "CDR 0: cognitively normal",
                0.5: "CDR 0.5: very mild impairment",
                1.0: "CDR 1: mild dementia",
                2.0: "CDR 2: moderate dementia",
                3.0: "CDR 3: severe dementia",
                "missing": "missing",
            },
            "Mapped from OASIS-3 CDRTOT score",
        )
    return None, "Raw diagnosis values from processed table"


def _diagnosis_frame(dataset: str, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if dataset == "adni" and "Group" in df.columns:
        counts = df["Group"].fillna("missing").astype(str).value_counts().sort_index()
        for label, count in counts.items():
            rows.append(
                {
                    "dataset": dataset,
                    "diagnosis_variable": "Group",
                    "diagnosis_label": label,
                    "count": int(count),
                    "fraction_of_scans": float(count / len(df)),
                    "note": "ADNI group label retained in processed table",
                }
            )
        return pd.DataFrame(rows)

    if "Diagnosis" not in df.columns:
        return pd.DataFrame(
            [
                {
                    "dataset": dataset,
                    "diagnosis_variable": "not_available_in_processed_table",
                    "diagnosis_label": "no diagnosis label retained",
                    "count": int(len(df)),
                    "fraction_of_scans": 1.0,
                    "note": "No diagnosis column available in processed table",
                }
            ]
        )

    diagnosis = df["Diagnosis"]
    label_map, mapping_note = _diagnosis_mapping(dataset)

    diagnosis = diagnosis.apply(lambda x: "missing" if pd.isna(x) else x)
    counts = diagnosis.value_counts(dropna=False)
    for raw_label, count in counts.items():
        label = label_map.get(raw_label, raw_label) if label_map else raw_label
        rows.append(
            {
                "dataset": dataset,
                "diagnosis_variable": "Diagnosis",
                "diagnosis_label": str(label),
                "count": int(count),
                "fraction_of_scans": float(count / len(df)),
                "note": mapping_note,
            }
        )
    return pd.DataFrame(rows)


def _qc_motion_summary_row(dataset: str, timepoints: pd.Series) -> dict:
    return {
        "dataset": dataset,
        "available_motion_metric": "unavailable_from_processed_table",
        "mean_fd": "unavailable",
        "median_fd": "unavailable",
        "max_fd": "unavailable",
        "scrub_ratio": "unavailable",
        "excluded_scans_by_qc": "unavailable_from_processed_table",
        "mean_n_timepoints_after_processing": float(timepoints.mean()),
        "median_n_timepoints_after_processing": float(timepoints.median()),
        "min_n_timepoints_after_processing": int(timepoints.min()),
        "max_n_timepoints_after_processing": int(timepoints.max()),
        "qc_note": (
            "Processed tables retain final time-series length only; "
            "framewise displacement and exclusion logs are not preserved here."
        ),
    }


def _dataset_context_row(dataset: str, n_sites: int | str, longitudinal_flag: bool) -> dict:
    return {
        "dataset": dataset,
        **DATASET_CONTEXT[dataset],
        "site_count_from_processed_table": n_sites,
        "longitudinal_in_processed_table": longitudinal_flag,
    }


def _build_dataset_artifacts(
    dataset: str,
    df: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, dict, dict]:
    age = pd.to_numeric(df["Age"], errors="coerce")
    age_summary = _age_stats(age)
    sex_counts, sex_counts_json = _sex_counts(df)
    site_col, n_sites = _site_summary(df)
    scans_per_subject = df.groupby("SubjectID").size()
    timepoints = df["TimeSeries"].apply(len)

    n_longitudinal_subjects = int((scans_per_subject > 1).sum())
    longitudinal_flag = bool(n_longitudinal_subjects > 0)
    dataset_row = {
        "dataset": dataset,
        "n_subjects": int(df["SubjectID"].nunique()),
        "n_scans": int(len(df)),
        "n_sites": n_sites,
        "site_column": site_col,
        "sex_counts_json": sex_counts_json,
        "n_male_scans": int(sex_counts.get("M", 0)),
        "n_female_scans": int(sex_counts.get("F", 0)),
        "n_missing_sex_scans": int(sex_counts.get("missing", 0)),
        "multiple_scans_per_subject": longitudinal_flag,
        "n_longitudinal_subjects": n_longitudinal_subjects,
        "mean_scans_per_subject": float(scans_per_subject.mean()),
        **age_summary,
    }

    return (
        dataset_row,
        _diagnosis_frame(dataset, df),
        _qc_motion_summary_row(dataset, timepoints),
        _dataset_context_row(dataset, n_sites, longitudinal_flag),
    )


def _write_dataset_tables(
    out_dir: Path,
    dataset_rows: list[dict],
    diagnosis_rows: list[dict],
    qc_rows: list[dict],
    context_rows: list[dict],
) -> None:
    pd.DataFrame(dataset_rows).sort_values("dataset").to_csv(out_dir / "table1_dataset_summary.csv", index=False)
    support_dir = out_dir / "support"
    support_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(diagnosis_rows).sort_values(["dataset", "diagnosis_variable", "diagnosis_label"]).to_csv(
        support_dir / "diagnosis_summary.csv",
        index=False,
    )
    pd.DataFrame(qc_rows).sort_values("dataset").to_csv(support_dir / "qc_motion_summary.csv", index=False)
    pd.DataFrame(context_rows).sort_values("dataset").to_csv(support_dir / "dataset_context.csv", index=False)


def _write_dataset_readme(out_dir: Path) -> None:
    readme = (
        "Generated from processed dataset tables used by the benchmark.\n"
        "PDF-facing table: table1_dataset_summary.csv.\n"
        "Age, sex, diagnosis, site, longitudinal structure, and final time-series length are empirical.\n"
        "Acquisition parameters (TR, scan duration in seconds, spatial resolution) and scan-level motion metrics "
        "are not retained in the processed pkl tables and are therefore marked unavailable here.\n"
        "ABIDE sample accounting is additionally reconstructed from the local ABIDE PCP cache in "
        "abide_sample_selection.csv as aggregate counts only. No participant or scan identifiers are exported.\n"
    )
    (out_dir / "support" / "README.txt").write_text(readme)


def _reject_stale_identifier_outputs(out_dir: Path) -> None:
    """Fail closed when a reused output tree contains legacy identifier tables."""
    stale = [
        relative_path.as_posix()
        for relative_path in FORBIDDEN_IDENTIFIER_OUTPUTS
        if (out_dir / relative_path).exists()
    ]
    if stale:
        raise RuntimeError(
            "Refusing to reuse an output directory containing legacy "
            "participant-level tables: "
            + ", ".join(stale)
            + ". Move or securely remove them before generating aggregate tables."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper dataset description tables.")
    parser.add_argument("--pkl_dir", type=Path, default=DEFAULT_ATLAS_DIR, help="Directory with *_X_y.pkl files.")
    parser.add_argument(
        "--raw_abide_dir",
        type=Path,
        default=DEFAULT_RAW_ABIDE_DIR,
        help="Optional local ABIDE PCP cache used to reconstruct sample selection.",
    )
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_TABLES_DIR, help="Output directory for CSV tables.")
    return parser.parse_args()


def main(
    pkl_dir: Path = DEFAULT_ATLAS_DIR,
    raw_abide_dir: Path = DEFAULT_RAW_ABIDE_DIR,
    out_dir: Path = DEFAULT_TABLES_DIR,
) -> None:
    _reject_stale_identifier_outputs(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows = []
    diagnosis_rows = []
    qc_rows = []
    context_rows = []
    abide_sample_selection = None

    for dataset in DATASETS:
        df = _load_df(dataset, pkl_dir).copy()
        if dataset == "abide":
            abide_sample_selection = _abide_sample_selection(df, raw_abide_dir)

        dataset_row, diagnosis_frame, qc_row, context_row = _build_dataset_artifacts(dataset, df)
        dataset_rows.append(dataset_row)
        diagnosis_rows.extend(diagnosis_frame.to_dict("records"))
        qc_rows.append(qc_row)
        context_rows.append(context_row)

    _write_dataset_tables(out_dir, dataset_rows, diagnosis_rows, qc_rows, context_rows)
    if abide_sample_selection is not None:
        abide_sample_selection.to_csv(out_dir / "support" / "abide_sample_selection.csv", index=False)
    _write_dataset_readme(out_dir)


if __name__ == "__main__":
    cli_args = parse_args()
    main(
        pkl_dir=cli_args.pkl_dir,
        raw_abide_dir=cli_args.raw_abide_dir,
        out_dir=cli_args.out_dir,
    )
