"""Read benchmark result CSVs and reshape them for paper figures."""

from __future__ import annotations

import os
import re
from datetime import datetime

import pandas as pd

VARIANT_ALIASES = {
    "quarterdim": ("quarterdim",),
    "halfdim": ("halfdim",),
    "one": ("one", "base"),
    "base": ("one", "base"),
    "two": ("two", "2layer"),
    "2layer": ("two", "2layer"),
}


def _format_dataset_label(name: str, n: int, scans: int) -> str:
    return f"{name}\n{'n':>5} = {n}\n{'scans':>5} = {scans}"


def load_metric_folds(csv_path: str, metric: str = "MAE"):
    df = pd.read_csv(csv_path, index_col=0)
    if metric not in df.index:
        raise ValueError(f"{csv_path} missing metric '{metric}'. Available: {list(df.index)}")
    row = df.loc[metric]
    fold_cols = [c for c in row.index if str(c).startswith("R")]
    if not fold_cols:
        raise ValueError(f"{csv_path} has no fold columns R1..; columns={list(df.columns)}")
    return row[fold_cols].astype(float).tolist()

def build_long_df(csv_map, dataset_labels=None, metric="MAE"):
    """
    dataframe:
    Dataset, DatasetLabel, Model, Fold, MAE, NegMAE
    """
    rows = []
    for ds, models in csv_map.items():
        for model, path in models.items():
            folds = load_metric_folds(path, metric=metric)
            for k, v in enumerate(folds, start=1):
                rows.append({
                    "Dataset": ds,
                    "DatasetLabel": dataset_labels.get(ds, ds) if dataset_labels else ds,
                    "Model": model,
                    "Fold": f"R{k}",
                    "MAE": v,
                    "NegMAE": -v,
                })
    return pd.DataFrame(rows)

def build_long_df_metric(
    csv_map,
    dataset_labels=None,
    metric="MAE",
    negate=False,
    value_col="Value",
):
    rows = []
    for ds, models in csv_map.items():
        for model, path in models.items():
            folds = load_metric_folds(path, metric=metric)
            for k, v in enumerate(folds, start=1):
                value = -v if negate else v
                rows.append({
                    "Dataset": ds,
                    "DatasetLabel": dataset_labels.get(ds, ds) if dataset_labels else ds,
                    "Model": model,
                    "Fold": f"R{k}",
                    value_col: value,
                })
    return pd.DataFrame(rows)

def _parse_timestamp(name: str):
    match = re.search(r"\[(\d{4}-\d{2}-\d{2}[ _]\d{2}[:\-]\d{2}[:\-]\d{2})\]", name)
    if not match:
        return None
    stamp = match.group(1)
    for fmt in ("%Y-%m-%d_%H-%M-%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(stamp, fmt)
        except ValueError:
            continue
    return None

def _select_latest_csv(files, dataset, model):
    dataset = dataset.lower()
    model = model.lower()
    model_patterns = {
        "dummy": "dummy_agereg_",
        "ridge": "ridge_agereg_ts_",
        "spdnet": "spdnet_agereg_",
        # New runs use CorrVec_Ridge_AgeReg; legacy single-dataset runs used
        # Ridge_AgeReg_VecCorr. Both names refer to the paper's CorrVec model.
        "veccorrridge": ("corrvec_ridge_agereg", "ridge_agereg_veccorr"),
    }
    pattern = model_patterns.get(model, model)
    candidates = []
    for fname in files:
        lower = fname.lower()
        if f"]{dataset}_" not in lower:
            continue
        if isinstance(pattern, tuple):
            pattern_found = any(p in lower for p in pattern)
        else:
            pattern_found = pattern in lower
        if not pattern_found or "agereg" not in lower:
            continue
        ts = _parse_timestamp(fname)
        if ts is None:
            continue
        candidates.append((ts, fname))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]

def build_csv_map_from_results(
    results_root: str,
    dataset_order,
    models=("Dummy", "Ridge", "SPDNet"),
):
    csv_map = {}
    for ds in dataset_order:
        ds_dir = os.path.join(results_root, ds)
        if not os.path.isdir(ds_dir):
            continue
        files = [f for f in os.listdir(ds_dir) if f.endswith(".csv")]
        models_map = {}
        for model in models:
            latest = _select_latest_csv(files, ds, model)
            if latest is None:
                print(f"Warning: missing {model} results for dataset '{ds}' in {ds_dir}")
                continue
            models_map[model] = os.path.join(ds_dir, latest)
        if models_map:
            csv_map[ds] = models_map
    return csv_map

def load_metric_avg(csv_path: str, metric: str = "MAE") -> float:
    df = pd.read_csv(csv_path, index_col=0)
    if metric not in df.index:
        raise ValueError(f"{csv_path} missing metric '{metric}'. Available: {list(df.index)}")
    row = df.loc[metric]
    if "Avg" in row.index:
        return float(row["Avg"])
    fold_cols = [c for c in row.index if str(c).startswith("R")]
    if not fold_cols:
        raise ValueError(f"{csv_path} has no fold columns R1..; columns={list(df.columns)}")
    return float(row[fold_cols].astype(float).mean())

def load_metric_points(csv_path: str, metric: str = "MAE", point_prefix: str = "R"):
    df = pd.read_csv(csv_path, index_col=0)
    if metric not in df.index:
        raise ValueError(f"{csv_path} missing metric '{metric}'. Available: {list(df.index)}")
    row = df.loc[metric]
    point_cols = [c for c in row.index if str(c).startswith(point_prefix)]
    if not point_cols:
        raise ValueError(f"{csv_path} has no point columns with prefix '{point_prefix}'.")
    return row[point_cols].astype(float).tolist()

def _harm_match(fname: str, harm: bool | None, exp: str | None = None) -> bool:
    if harm is None:
        return True
    lower = fname.lower()
    has_harm = lower.endswith("_harm.csv")
    if harm:
        return has_harm
    if exp is None:
        return not has_harm
    return lower.endswith(f"_{exp}.csv")

def _select_latest_all_age_csv(files, model, experiment, harm: bool | None = None):
    model = model.lower()
    exp = experiment.lower()
    model_patterns = {
        "dummy": "dummy_agereg_",
        "ridge": "ridge_agereg_ts_",
        "spdnet": "spdnet_agereg_",
        "veccorrridge": "corrvec_ridge_agereg_",
    }
    pattern = model_patterns.get(model, f"_{model}_")
    candidates = []
    for fname in files:
        lower = fname.lower()
        if pattern not in lower or "agereg" not in lower:
            continue
        if f"_{exp}" not in lower:
            continue
        if not _harm_match(fname, harm, exp=exp):
            continue
        ts = _parse_timestamp(fname)
        if ts is None:
            continue
        candidates.append((ts, fname))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]

def _select_latest_all_age_csv_variant(
    files,
    model,
    experiment,
    harm: bool | None = None,
    variant: str | None = None,
):
    model = model.lower()
    exp = experiment.lower()
    candidates = []
    for fname in files:
        lower = fname.lower()
        if f"_{model}_" not in lower or "agereg" not in lower:
            continue
        if f"_{exp}" not in lower:
            continue
        has_harm = "_harm" in lower
        if harm is True and not has_harm:
            continue
        if harm is False and has_harm:
            continue
        if variant is not None:
            variant_tokens = VARIANT_ALIASES.get(variant.lower(), (variant.lower(),))
            if not any(f"_{token}" in lower for token in variant_tokens):
                continue
        ts = _parse_timestamp(fname)
        if ts is None:
            continue
        candidates.append((ts, fname))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]

def build_all_age_avg_df(
    results_root: str,
    experiments=("LODO", "GKF5"),
    models=("Dummy", "Ridge", "SPDNet"),
    metric="MAE",
    negate=False,
    harm: bool | None = None,
):
    rows = []
    files = [f for f in os.listdir(results_root) if f.endswith(".csv")]
    for exp in experiments:
        for model in models:
            latest = _select_latest_all_age_csv(files, model, exp, harm=harm)
            if latest is None:
                print(f"Warning: missing {model} results for experiment '{exp}' in {results_root}")
                continue
            path = os.path.join(results_root, latest)
            value = load_metric_avg(path, metric=metric)
            if negate:
                value = -value
            rows.append({
                "Experiment": exp,
                "Model": model,
                "Value": value,
            })
    return pd.DataFrame(rows)

def build_all_age_points_df(
    results_root: str,
    experiment: str,
    models=("Dummy", "Ridge", "SPDNet"),
    metric="MAE",
    negate=False,
    harm: bool | None = None,
):
    rows = []
    files = [f for f in os.listdir(results_root) if f.endswith(".csv")]
    if experiment.upper() == "GKF5":
        point_prefix = "R"
    elif experiment.upper() == "LODO":
        point_prefix = "TEST_"
    else:
        raise ValueError("experiment must be 'GKF5' or 'LODO'")

    for model in models:
        latest = _select_latest_all_age_csv(files, model, experiment, harm=harm)
        if latest is None:
            print(f"Warning: missing {model} results for experiment '{experiment}' in {results_root}")
            continue
        path = os.path.join(results_root, latest)
        points = load_metric_points(path, metric=metric, point_prefix=point_prefix)
        for idx, value in enumerate(points, start=1):
            rows.append({
                "Experiment": experiment,
                "Model": model,
                "Point": f"P{idx}",
                "Value": -value if negate else value,
            })
    if not rows:
        return pd.DataFrame(columns=["Experiment", "Model", "Point", "Value"])
    return pd.DataFrame(rows)

def build_all_age_points_df_variant(
    results_root: str,
    experiment: str,
    model: str,
    metric="MAE",
    negate=False,
    harm: bool | None = None,
    variant: str | None = None,
):
    rows = []
    files = [f for f in os.listdir(results_root) if f.endswith(".csv")]
    if experiment.upper() == "GKF5":
        point_prefix = "R"
    elif experiment.upper() == "LODO":
        point_prefix = "TEST_"
    else:
        raise ValueError("experiment must be 'GKF5' or 'LODO'")

    latest = _select_latest_all_age_csv_variant(
        files,
        model=model,
        experiment=experiment,
        harm=harm,
        variant=variant,
    )
    if latest is None:
        print(
            f"Warning: missing {model} results for experiment '{experiment}' in {results_root} "
            f"(variant={variant}, harm={harm})"
        )
        return pd.DataFrame(columns=["Experiment", "Model", "Point", "Value"])
    path = os.path.join(results_root, latest)
    points = load_metric_points(path, metric=metric, point_prefix=point_prefix)
    for idx, value in enumerate(points, start=1):
        rows.append({
            "Experiment": experiment,
            "Model": model,
            "Point": f"P{idx}",
            "Value": -value if negate else value,
        })
    return pd.DataFrame(rows)
