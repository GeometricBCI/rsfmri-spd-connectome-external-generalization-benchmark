import os
import re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

def _rename_legend_labels(labels):
    renamed = []
    for label in labels:
        if isinstance(label, str) and label.startswith("Ridge"):
            renamed.append(f"TS Ridge{label[len('Ridge'):]}")
        else:
            renamed.append(label)
    return renamed

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
    candidates = []
    for fname in files:
        lower = fname.lower()
        if f"]{dataset}_" not in lower:
            continue
        if f"_{model}_" not in lower or "agereg" not in lower:
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
    candidates = []
    for fname in files:
        lower = fname.lower()
        if f"_{model}_" not in lower or "agereg" not in lower:
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
        if variant == "2layer":
            if "_2layer" not in lower:
                continue
        elif variant == "base":
            if "_2layer" in lower:
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

def plot_all_age_avg_bar(
    df,
    order=("Dummy", "Ridge", "SPDNet"),
    palette=None,
    title="ALL Age Regression",
    ylabel="Metric (↑)",
    save_dir="tables_out",
    save_name="benchmark_ALL_Age.png",
    figsize=(5.5, 4),
    dpi=300,
):
    os.makedirs(save_dir, exist_ok=True)

    if palette is None:
        palette = {
            "Dummy":  "#4C78A8",
            "Ridge":  "#F58518",
            "SPDNet": "#54A24B",
        }

    model_order = [m for m in order if m in df["Model"].unique()]
    if not model_order:
        raise ValueError("No models found in df.")

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    sns.barplot(
        data=df,
        x="Experiment",
        y="Value",
        hue="Model",
        hue_order=model_order,
        palette=palette,
        ax=ax,
    )
    ax.set_title(title, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    labels = _rename_legend_labels(labels)
    ax.legend(
        handles,
        labels,
        title="Regressor",
        frameon=True,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    plt.tight_layout()
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print("Saved figure to:", save_path)
    plt.show()

def plot_all_age_box(
    df,
    order=("Dummy", "Ridge", "SPDNet"),
    palette=None,
    title="ALL Age Regression",
    ylabel="Metric (↑)",
    save_dir="tables_out",
    save_name="benchmark_ALL_Age.png",
    figsize=(5.5, 4),
    dpi=300,
    show_points=True,
    point_size=3.0,
    box_width=0.9,
    jitter=0.12,
    hue_offset_factor=0.85,
    seed=0,
    ax=None,
    show_legend=True,
    zero_baseline=False,
):
    if palette is None:
        palette = {
            "Dummy":  "#4C78A8",
            "Ridge":  "#F58518",
            "SPDNet": "#54A24B",
        }

    model_order = [m for m in order if m in df["Model"].unique()]
    if not model_order:
        raise ValueError("No models found in df.")

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)

    standalone = False
    if ax is None:
        standalone = True
        os.makedirs(save_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
    exp_order = list(df["Experiment"].unique())
    sns.boxplot(
        data=df,
        x="Experiment",
        y="Value",
        hue="Model",
        order=exp_order,
        hue_order=model_order,
        palette=palette,
        width=box_width,
        linewidth=1.1,
        fliersize=0,
        ax=ax,
    )

    if show_points:
        sns.stripplot(
            data=df,
            x="Experiment",
            y="Value",
            hue="Model",
            order=exp_order,
            hue_order=model_order,
            dodge=True,
            color="white",
            edgecolor="black",
            size=point_size,
            alpha=0.95,
            linewidth=0.8,
            jitter=jitter,
            ax=ax,
        )

    if hue_offset_factor < 1.0:
        _compress_hue_offsets(ax, factor=hue_offset_factor)

    ax.set_title(title, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    if zero_baseline:
        ax.axhline(0.0, color="#6b7280", linewidth=1.0, alpha=0.8, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", alpha=0.25)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        handles = handles[:len(model_order)]
        labels = _rename_legend_labels(labels[:len(model_order)])
        ax.legend(
            handles,
            labels,
            title="Regressor",
            frameon=True,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )
    elif ax.get_legend() is not None:
        ax.get_legend().remove()

    if standalone:
        plt.tight_layout()
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print("Saved figure to:", save_path)
        plt.show()


def plot_all_age_box_grid(
    plots,
    title="ALL Age Regression",
    save_dir="tables_out",
    save_name="benchmark_ALL_Age_grid.png",
    figsize=(16, 4),
    dpi=300,
    legend_title="Regressor",
):
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)

    fig, axes = plt.subplots(1, len(plots), figsize=figsize, dpi=dpi)
    if len(plots) == 1:
        axes = [axes]
    axes = axes.ravel()

    for ax, cfg in zip(axes, plots):
        plot_all_age_box(
            cfg["df"],
            order=cfg.get("order", ("Dummy", "Ridge", "SPDNet")),
            palette=cfg.get("palette"),
            title=cfg.get("title", ""),
            ylabel=cfg.get("ylabel", "Metric (↑)"),
            figsize=figsize,
            dpi=dpi,
            show_points=cfg.get("show_points", True),
            point_size=cfg.get("point_size", 2.0),
            box_width=cfg.get("box_width", 1.0),
            jitter=cfg.get("jitter", 0.12),
            hue_offset_factor=cfg.get("hue_offset_factor", 0.85),
            ax=ax,
            show_legend=False,
            zero_baseline=cfg.get("zero_baseline", False),
        )
        ax.set_xlabel("")
        ax.set_xticks([])

    handles, labels = axes[0].get_legend_handles_labels()
    seen = set()
    uniq = []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        uniq.append((h, l))
    handles, labels = zip(*uniq) if uniq else ([], [])
    labels = _rename_legend_labels(labels)
    fig.legend(
        handles,
        labels,
        title=legend_title,
        frameon=True,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=len(labels) if labels else 1,
        columnspacing=1.2,
        handletextpad=0.6,
    )

    if title:
        fig.suptitle(title, y=1.06)
    plt.tight_layout()
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print("Saved figure to:", save_path)
    plt.show()

def _compress_hue_offsets(ax, factor=0.85):
    """
    Pull hue-dodged elements closer to their category centers.
    factor=1.0 keeps original positions; smaller pulls closer.
    """
    centers = np.array(ax.get_xticks(), dtype=float)
    if centers.size == 0:
        return

    def _closest_center(x):
        return centers[np.argmin(np.abs(centers - x))]

    for patch in ax.patches:
        if not hasattr(patch, "set_x"):
            continue
        x = patch.get_x()
        w = patch.get_width()
        center = _closest_center(x + 0.5 * w)
        new_x = center + (x - center) * factor
        patch.set_x(new_x)

    for line in ax.lines:
        xdata = line.get_xdata()
        if xdata is None or len(xdata) == 0:
            continue
        center = _closest_center(np.mean(xdata))
        line.set_xdata(center + (np.array(xdata) - center) * factor)

    for coll in ax.collections:
        offsets = getattr(coll, "get_offsets", None)
        if offsets is None:
            continue
        offs = np.array(coll.get_offsets(), copy=True)
        if offs.size == 0:
            continue
        xs = offs[:, 0]
        centers_sel = np.array([_closest_center(x) for x in xs])
        offs[:, 0] = centers_sel + (xs - centers_sel) * factor
        coll.set_offsets(offs)

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

def plot_paper_style_box(
    csv_map,
    dataset_labels=None,
    order=("Dummy", "Ridge", "SPDNet"),
    palette=None,
    title="Age Regression Benchmark",
    save_dir="tables_out",
    save_name="benchmark_Age_paper.png",
    figsize=(10, 3.6),       
    dpi=300,
    legend_pos="right",      # "right" or "top" or "none"
    show_points=True,
    point_size=3.0,
    seed=0,
):
    os.makedirs(save_dir, exist_ok=True)

    if palette is None:
        palette = {
            "Dummy":  "#4C78A8",
            "Ridge":  "#F58518",
            "SPDNet": "#54A24B",
        }

    datasets = list(csv_map.keys())
    model_order = [m for m in order if any(m in csv_map[ds] for ds in datasets)]
    if not model_order:
        raise ValueError("No models found in csv_map.")

    df = build_long_df(csv_map, dataset_labels=dataset_labels, metric="MAE")

    ds_order = [dataset_labels.get(ds, ds) if dataset_labels else ds for ds in datasets]
    df["DatasetLabel"] = pd.Categorical(df["DatasetLabel"], categories=ds_order, ordered=True)
    df["Model"] = pd.Categorical(df["Model"], categories=model_order, ordered=True)

    # --- seaborn paper style ---
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    sns.boxplot(
        data=df,
        x="DatasetLabel",
        y="NegMAE",
        hue="Model",
        order=ds_order,
        hue_order=model_order,
        palette=palette,
        width=0.72,
        linewidth=1.1,
        fliersize=0,
        ax=ax,
    )

    if show_points:
        rng = np.random.RandomState(seed)
        sns.stripplot(
                    data=df,
                    x="DatasetLabel",
                    y="NegMAE",
                    hue="Model",
                    order=ds_order,
                    hue_order=model_order,
                    dodge=True,
                    color="white",          
                    edgecolor="black",  
                    size=point_size,
                    alpha=0.95,
                    linewidth=0.8,
                    ax=ax,
                )

    handles, labels = ax.get_legend_handles_labels()
    handles = handles[:len(model_order)]
    labels = _rename_legend_labels(labels[:len(model_order)])

    ax.legend_.remove() if ax.get_legend() is not None else None

    if legend_pos == "right":
        ax.legend(
            handles, labels,
            title="Regressor",
            frameon=True,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )
    elif legend_pos == "top":
        ax.legend(
            handles, labels,
            title=None,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(model_order),
            columnspacing=1.2,
            handletextpad=0.6,
        )
    elif legend_pos == "none":
        pass
    else:
        raise ValueError("legend_pos must be one of: 'right', 'top', 'none'")

    # labels & title
    ax.set_title(title, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("Negative MAE (↑)")

    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", alpha=0.25)

    ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()

    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print("Saved figure to:", save_path)
    plt.show()

def plot_paper_style_box_row(
    csv_map,
    dataset_labels=None,
    order=("Dummy", "Ridge", "SPDNet"),
    palette=None,
    title="Age Regression Benchmark",
    save_dir="tables_out",
    save_name="benchmark_Age_all.png",
    figsize=(18, 4),
    dpi=300,
    show_points=True,
    point_size=3.0,
    seed=0,
    box_width=0.35,
    strip_jitter=0.0,
    wspace=0.2,
):
    os.makedirs(save_dir, exist_ok=True)

    if palette is None:
        palette = {
            "Dummy":  "#4C78A8",
            "Ridge":  "#F58518",
            "SPDNet": "#54A24B",
        }

    datasets = list(csv_map.keys())
    model_order = [m for m in order if any(m in csv_map[ds] for ds in datasets)]
    if not model_order:
        raise ValueError("No models found in csv_map.")

    df = build_long_df(csv_map, dataset_labels=dataset_labels, metric="MAE")

    # --- seaborn paper style ---
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    sns.despine(top=True, right=True)

    fig, axes = plt.subplots(1, len(datasets), figsize=figsize, dpi=dpi, sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    for i, (ax, ds) in enumerate(zip(axes, datasets)):
        label = dataset_labels.get(ds, ds) if dataset_labels else ds
        df_ds = df[df["Dataset"] == ds]

        sns.boxplot(
            data=df_ds,
            x="DatasetLabel",
            y="NegMAE",
            hue="Model",
            order=[label],
            hue_order=model_order,
            palette=palette,
            width=box_width,
            linewidth=1.1,
            fliersize=0,
            ax=ax,
        )

        if show_points:
            sns.stripplot(
                data=df_ds,
                x="DatasetLabel",
                y="NegMAE",
                hue="Model",
                order=[label],
                hue_order=model_order,
                dodge=True,
                color="white",
                edgecolor="black",
                size=point_size,
                alpha=0.95,
                linewidth=0.8,
                jitter=strip_jitter,
                ax=ax,
            )

        ax.set_title(label, pad=8)
        ax.set_xlabel("")
        ax.set_xticks([])
        ax.set_ylabel("Negative MAE (↑)" if i == 0 else "")
        ax.tick_params(axis="y", pad=2)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", alpha=0.25)

        if ax.get_legend() is not None:
            ax.get_legend().remove()

    handles, labels = axes[0].get_legend_handles_labels()
    handles = handles[:len(model_order)]
    labels = _rename_legend_labels(labels[:len(model_order)])
    fig.legend(
        handles,
        labels,
        title="Regressor",
        frameon=True,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=len(model_order),
        columnspacing=1.2,
        handletextpad=0.6,
    )

    if title:
        fig.suptitle(title, y=1.12)
    plt.tight_layout()
    fig.subplots_adjust(wspace=wspace)

    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print("Saved figure to:", save_path)
    plt.show()

def plot_paper_style_box_row_metric(
    csv_map,
    dataset_labels=None,
    order=("Dummy", "Ridge", "SPDNet"),
    palette=None,
    title="Age Regression Benchmark",
    save_dir="tables_out",
    save_name="benchmark_metric.png",
    figsize=(18, 4),
    dpi=300,
    show_points=True,
    point_size=3.0,
    seed=0,
    metric="MAE",
    negate=False,
    ylabel="Metric (↑)",
    align_zero=False,
    box_width=0.35,
    strip_jitter=0.0,
    wspace=0.2,
):
    os.makedirs(save_dir, exist_ok=True)

    if palette is None:
        palette = {
            "Dummy":  "#4C78A8",
            "Ridge":  "#F58518",
            "SPDNet": "#54A24B",
        }

    datasets = list(csv_map.keys())
    model_order = [m for m in order if any(m in csv_map[ds] for ds in datasets)]
    if not model_order:
        raise ValueError("No models found in csv_map.")

    df = build_long_df_metric(
        csv_map,
        dataset_labels=dataset_labels,
        metric=metric,
        negate=negate,
        value_col="Value",
    )

    zero_frac = None
    if align_zero:
        data_min = df["Value"].min()
        data_max = df["Value"].max()
        if data_min < 0 < data_max:
            zero_frac = -data_min / (data_max - data_min)
        elif data_min >= 0:
            zero_frac = 0.0
        else:
            zero_frac = 1.0

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    sns.despine(top=True, right=True)

    fig, axes = plt.subplots(1, len(datasets), figsize=figsize, dpi=dpi, sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    rng = np.random.RandomState(seed)
    for i, (ax, ds) in enumerate(zip(axes, datasets)):
        label = dataset_labels.get(ds, ds) if dataset_labels else ds
        df_ds = df[df["Dataset"] == ds]

        sns.boxplot(
            data=df_ds,
            x="DatasetLabel",
            y="Value",
            hue="Model",
            order=[label],
            hue_order=model_order,
            palette=palette,
            width=box_width,
            linewidth=1.1,
            fliersize=0,
            ax=ax,
        )

        if show_points:
            sns.stripplot(
                data=df_ds,
                x="DatasetLabel",
                y="Value",
                hue="Model",
                order=[label],
                hue_order=model_order,
                dodge=True,
                color="white",
                edgecolor="black",
                size=point_size,
                alpha=0.95,
                linewidth=0.8,
                jitter=strip_jitter,
                ax=ax,
            )

        ax.set_title(label, pad=8)
        ax.set_xlabel("")
        ax.set_xticks([])
        ax.set_ylabel(ylabel if i == 0 else "")
        ax.tick_params(axis="y", pad=2)
        ax.axhline(0.0, color="#6b7280", linewidth=1.0, alpha=0.8, zorder=0)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", alpha=0.25)

        if align_zero and zero_frac is not None:
            vmin = df_ds["Value"].min()
            vmax = df_ds["Value"].max()
            if zero_frac == 0.0:
                span = max(vmax, 1e-6)
                pad = 0.05 * span
                ax.set_ylim(-pad, span + pad)
            elif zero_frac == 1.0:
                span = max(-vmin, 1e-6)
                pad = 0.05 * span
                ax.set_ylim(-span - pad, pad)
            else:
                span = max((0.0 - vmin) / zero_frac, (vmax - 0.0) / (1.0 - zero_frac))
                pad = 0.05 * span
                ymin = -zero_frac * span - pad
                ymax = (1.0 - zero_frac) * span + pad
                ax.set_ylim(ymin, ymax)

        if ax.get_legend() is not None:
            ax.get_legend().remove()

    handles, labels = axes[0].get_legend_handles_labels()
    handles = handles[:len(model_order)]
    labels = _rename_legend_labels(labels[:len(model_order)])
    fig.legend(
        handles,
        labels,
        title="Regressor",
        frameon=True,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=len(model_order),
        columnspacing=1.2,
        handletextpad=0.6,
    )

    if title:
        fig.suptitle(title, y=1.12)
    plt.tight_layout()
    fig.subplots_adjust(wspace=wspace)

    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print("Saved figure to:", save_path)
    plt.show()

def plot_paper_style_box_two_rows(
    csv_map,
    dataset_labels=None,
    order=("Dummy", "Ridge", "SPDNet"),
    palette=None,
    title_top="",
    title_bottom="",
    save_dir="tables_out",
    save_name="benchmark_Age_all_combined.png",
    figsize=(10, 7),
    dpi=300,
    show_points=True,
    point_size=3.0,
    metric_bottom="R2",
    negate_bottom=False,
    ylabel_top="Negative MAE (↑)",
    ylabel_bottom="R2 (↑)",
    align_zero=False,
    box_width=0.8,
    strip_jitter=0.0,
    wspace=0.6,
    hspace=0.08,
):
    os.makedirs(save_dir, exist_ok=True)

    if palette is None:
        palette = {
            "Dummy":  "#4C78A8",
            "Ridge":  "#F58518",
            "SPDNet": "#54A24B",
        }

    datasets = list(csv_map.keys())
    model_order = [m for m in order if any(m in csv_map[ds] for ds in datasets)]
    if not model_order:
        raise ValueError("No models found in csv_map.")

    df_top = build_long_df(csv_map, dataset_labels=dataset_labels, metric="MAE")
    df_bottom = build_long_df_metric(
        csv_map,
        dataset_labels=dataset_labels,
        metric=metric_bottom,
        negate=negate_bottom,
        value_col="Value",
    )

    if align_zero:
        data_min = df_bottom["Value"].min()
        data_max = df_bottom["Value"].max()
        if data_min < 0 < data_max:
            zero_frac = -data_min / (data_max - data_min)
        elif data_min >= 0:
            zero_frac = 0.0
        else:
            zero_frac = 1.0
    else:
        zero_frac = None

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    sns.despine(top=True, right=True)

    fig, axes = plt.subplots(2, len(datasets), figsize=figsize, dpi=dpi, sharey=False)
    if len(datasets) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, ds in enumerate(datasets):
        label = dataset_labels.get(ds, ds) if dataset_labels else ds

        ax_top = axes[0, col]
        df_ds_top = df_top[df_top["Dataset"] == ds]
        sns.boxplot(
            data=df_ds_top,
            x="DatasetLabel",
            y="NegMAE",
            hue="Model",
            order=[label],
            hue_order=model_order,
            palette=palette,
            width=box_width,
            linewidth=1.1,
            fliersize=0,
            ax=ax_top,
        )
        if show_points:
            sns.stripplot(
                data=df_ds_top,
                x="DatasetLabel",
                y="NegMAE",
                hue="Model",
                order=[label],
                hue_order=model_order,
                dodge=True,
                color="white",
                edgecolor="black",
                size=point_size,
                alpha=0.95,
                linewidth=0.8,
                jitter=strip_jitter,
                ax=ax_top,
            )
        ax_top.set_title(label, pad=8)
        ax_top.set_xlabel("")
        ax_top.set_xticks([])
        ax_top.set_ylabel(ylabel_top if col == 0 else "")
        ax_top.tick_params(axis="y", pad=2)
        ax_top.grid(axis="x", visible=False)
        ax_top.grid(axis="y", alpha=0.25)
        if ax_top.get_legend() is not None:
            ax_top.get_legend().remove()

        ax_bot = axes[1, col]
        df_ds_bottom = df_bottom[df_bottom["Dataset"] == ds]
        sns.boxplot(
            data=df_ds_bottom,
            x="DatasetLabel",
            y="Value",
            hue="Model",
            order=[label],
            hue_order=model_order,
            palette=palette,
            width=box_width,
            linewidth=1.1,
            fliersize=0,
            ax=ax_bot,
        )
        if show_points:
            sns.stripplot(
                data=df_ds_bottom,
                x="DatasetLabel",
                y="Value",
                hue="Model",
                order=[label],
                hue_order=model_order,
                dodge=True,
                color="white",
                edgecolor="black",
                size=point_size,
                alpha=0.95,
                linewidth=0.8,
                jitter=strip_jitter,
                ax=ax_bot,
            )
        ax_bot.set_title("")
        ax_bot.set_xlabel("")
        ax_bot.set_xticks([])
        ax_bot.set_ylabel(ylabel_bottom if col == 0 else "")
        ax_bot.tick_params(axis="y", pad=2)
        ax_bot.axhline(0.0, color="#6b7280", linewidth=1.0, alpha=0.8, zorder=0)
        ax_bot.grid(axis="x", visible=False)
        ax_bot.grid(axis="y", alpha=0.25)

        if align_zero and zero_frac is not None:
            vmin = df_ds_bottom["Value"].min()
            vmax = df_ds_bottom["Value"].max()
            if zero_frac == 0.0:
                span = max(vmax, 1e-6)
                pad = 0.05 * span
                ax_bot.set_ylim(-pad, span + pad)
            elif zero_frac == 1.0:
                span = max(-vmin, 1e-6)
                pad = 0.05 * span
                ax_bot.set_ylim(-span - pad, pad)
            else:
                span = max((0.0 - vmin) / zero_frac, (vmax - 0.0) / (1.0 - zero_frac))
                pad = 0.05 * span
                ymin = -zero_frac * span - pad
                ymax = (1.0 - zero_frac) * span + pad
                ax_bot.set_ylim(ymin, ymax)

        if ax_bot.get_legend() is not None:
            ax_bot.get_legend().remove()

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles = handles[:len(model_order)]
    labels = _rename_legend_labels(labels[:len(model_order)])
    fig.legend(
        handles,
        labels,
        title="Regressor",
        frameon=True,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=len(model_order),
        columnspacing=1.0,
        handletextpad=0.5,
    )

    if title_top:
        fig.text(0.5, 0.98, title_top, ha="center", va="top")
    if title_bottom:
        fig.text(0.5, 0.50, title_bottom, ha="center", va="top")

    plt.tight_layout()
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print("Saved figure to:", save_path)
    plt.show()


if __name__ == "__main__":
    RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "results_AgeReg")
    DATASET_ORDER = ["1000brains", "abide", "adni", "adnidod", "camcan", "cobre", "oasis3"]
    DATASET_LABELS = {
        "1000brains": "1000BRAINS",
        "abide": "ABIDE",
        "adni": "ADNI",
        "adnidod": "ADNIDOD",
        "camcan": "Cam-CAN",
        "cobre": "COBRE",
        "oasis3": "OASIS-3",
    }

    CSV_MAP = build_csv_map_from_results(RESULTS_ROOT, DATASET_ORDER, models=("Dummy", "Ridge", "SPDNet"))

    plot_paper_style_box_row(
        csv_map=CSV_MAP,
        dataset_labels=DATASET_LABELS,
        order=("Dummy", "Ridge", "SPDNet"),
        title="",
        save_dir="tables_out",
        save_name="benchmark_Age_all.png",
        figsize=(10, 4),
        show_points=True,
        point_size=2.5,
        dpi=300,
        box_width=0.8,
        strip_jitter=0.0,
        wspace=0.6,
    )

    plot_paper_style_box_row_metric(
        csv_map=CSV_MAP,
        dataset_labels=DATASET_LABELS,
        order=("Dummy", "Ridge", "SPDNet"),
        title="",
        save_dir="tables_out",
        save_name="benchmark_Age_all_R2.png",
        figsize=(10, 4),
        show_points=True,
        point_size=2.5,
        dpi=300,
        metric="R2",
        negate=False,
        ylabel="R2 (↑)",
        align_zero=True,
        box_width=0.8,
        strip_jitter=0.0,
        wspace=0.6,
    )

    plot_paper_style_box_two_rows(
        csv_map=CSV_MAP,
        dataset_labels=DATASET_LABELS,
        order=("Dummy", "Ridge", "SPDNet"),
        title_top="",
        title_bottom="",
        save_dir="tables_out",
        save_name="benchmark_Age_all_combined.png",
        figsize=(10, 7),
        show_points=True,
        point_size=2.5,
        dpi=300,
        metric_bottom="R2",
        negate_bottom=False,
        ylabel_top="Negative MAE (↑)",
        ylabel_bottom="R2 (↑)",
        align_zero=True,
        box_width=0.8,
        strip_jitter=0.0,
        wspace=0.6,
        hspace=0.08,
    )

    ALL_RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "results_ALL_AgeReg", "01-14")

    def _with_harm_label(df, tag: str):
        if df.empty:
            return df
        out = df.copy()
        out["Model"] = out["Model"].astype(str) + f" ({tag})"
        return out

    def _adjust_harmonization_labels(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        out = out[out["Model"] != "Dummy (original)"]
        out["Model"] = out["Model"].replace(
            {
                "Dummy (harmonization)": "Dummy",
            }
        )
        return out

def _build_all_age_plots_combined(tag: str = "combined"):
        df_gkf5_negmae = pd.concat(
            [
                _with_harm_label(
                    build_all_age_points_df(
                        ALL_RESULTS_ROOT,
                        experiment="GKF5",
                        models=("Dummy", "Ridge", "SPDNet"),
                        metric="MAE",
                        negate=True,
                        harm=False,
                    ),
                    "original",
                ),
                _with_harm_label(
                    build_all_age_points_df(
                        ALL_RESULTS_ROOT,
                        experiment="GKF5",
                        models=("Dummy", "Ridge", "SPDNet"),
                        metric="MAE",
                        negate=True,
                        harm=True,
                    ),
                    "harmonization",
                ),
            ],
            ignore_index=True,
        )
        df_gkf5_negmae = _adjust_harmonization_labels(df_gkf5_negmae)
        df_gkf5_r2 = pd.concat(
            [
                _with_harm_label(
                    build_all_age_points_df(
                        ALL_RESULTS_ROOT,
                        experiment="GKF5",
                        models=("Dummy", "Ridge", "SPDNet"),
                        metric="R2",
                        negate=False,
                        harm=False,
                    ),
                    "original",
                ),
                _with_harm_label(
                    build_all_age_points_df(
                        ALL_RESULTS_ROOT,
                        experiment="GKF5",
                        models=("Dummy", "Ridge", "SPDNet"),
                        metric="R2",
                        negate=False,
                        harm=True,
                    ),
                    "harmonization",
                ),
            ],
            ignore_index=True,
        )
        df_gkf5_r2 = _adjust_harmonization_labels(df_gkf5_r2)
        df_lodo_negmae = pd.concat(
            [
                _with_harm_label(
                    build_all_age_points_df(
                        ALL_RESULTS_ROOT,
                        experiment="LODO",
                        models=("Dummy", "Ridge", "SPDNet"),
                        metric="MAE",
                        negate=True,
                        harm=False,
                    ),
                    "original",
                ),
                _with_harm_label(
                    build_all_age_points_df(
                        ALL_RESULTS_ROOT,
                        experiment="LODO",
                        models=("Dummy", "Ridge", "SPDNet"),
                        metric="MAE",
                        negate=True,
                        harm=True,
                    ),
                    "harmonization",
                ),
            ],
            ignore_index=True,
        )
        df_lodo_negmae = _adjust_harmonization_labels(df_lodo_negmae)
        df_lodo_r2 = pd.concat(
            [
                _with_harm_label(
                    build_all_age_points_df(
                        ALL_RESULTS_ROOT,
                        experiment="LODO",
                        models=("Dummy", "Ridge", "SPDNet"),
                        metric="R2",
                        negate=False,
                        harm=False,
                    ),
                    "original",
                ),
                _with_harm_label(
                    build_all_age_points_df(
                        ALL_RESULTS_ROOT,
                        experiment="LODO",
                        models=("Dummy", "Ridge", "SPDNet"),
                        metric="R2",
                        negate=False,
                        harm=True,
                    ),
                    "harmonization",
                ),
            ],
            ignore_index=True,
        )
        df_lodo_r2 = _adjust_harmonization_labels(df_lodo_r2)

        model_order = [
            "Dummy",
            "Ridge (original)",
            "Ridge (harmonization)",
            "SPDNet (original)",
            "SPDNet (harmonization)",
        ]
        palette = {
            "Dummy": "#7FA9D9",
            "Ridge (original)": "#F58518",
            "Ridge (harmonization)": "#F8B26A",
            "SPDNet (original)": "#7CCB7A",
            "SPDNet (harmonization)": "#4DAF74",
        }
        all_ds_label = "ADNI, OASIS-3, ADNIDOD, 1000BRAINS, ABIDE, Cam-CAN, COBRE"
        plot_all_age_box_grid(
            plots=[
                {
                    "df": df_gkf5_negmae,
                    "title": "GroupKFold (NegMAE)",
                    "ylabel": "Negative MAE (↑)",
                    "point_size": 3.5,
                    "box_width": 0.8,
                    "jitter": 0.10,
                    "hue_offset_factor": 1.0,
                    "order": model_order,
                    "palette": palette,
                },
                {
                    "df": df_lodo_negmae,
                    "title": "LODO (NegMAE)",
                    "ylabel": "Negative MAE (↑)",
                    "point_size": 3.5,
                    "box_width": 0.8,
                    "jitter": 0.10,
                    "hue_offset_factor": 1.0,
                    "order": model_order,
                    "palette": palette,
                },
                {
                    "df": df_gkf5_r2,
                    "title": "GroupKFold (R2)",
                    "ylabel": "R2 (↑)",
                    "point_size": 3.5,
                    "zero_baseline": True,
                    "box_width": 0.8,
                    "jitter": 0.10,
                    "hue_offset_factor": 1.0,
                    "order": model_order,
                    "palette": palette,
                },
                {
                    "df": df_lodo_r2,
                    "title": "LODO (R2)",
                    "ylabel": "R2 (↑)",
                    "point_size": 3.5,
                    "zero_baseline": True,
                    "box_width": 0.8,
                    "jitter": 0.10,
                    "hue_offset_factor": 1.0,
                    "order": model_order,
                    "palette": palette,
                },
            ],
            title="",
            save_dir="tables_out",
            save_name=f"benchmark_ALL_Age_grid_{tag}.png",
            figsize=(12, 5.5),
            dpi=300,
        )

_build_all_age_plots_combined()

ALL_RESULTS_ROOT_0131 = os.path.join(os.path.dirname(__file__), "results_ALL_AgeReg", "01-31")

def _build_all_age_plots_bimaps_0131(tag: str = "bimaps_0131"):
    df_gkf5_negmae = pd.concat(
        [
            build_all_age_points_df_variant(
                ALL_RESULTS_ROOT_0131,
                experiment="GKF5",
                model="SPDNet",
                metric="MAE",
                negate=True,
                harm=False,
                variant="base",
            ).assign(Model="one (BiMap+ReEig)(original)"),
            build_all_age_points_df_variant(
                ALL_RESULTS_ROOT_0131,
                experiment="GKF5",
                model="SPDNet",
                metric="MAE",
                negate=True,
                harm=True,
                variant="base",
            ).assign(Model="one (BiMap+ReEig)(harmonization)"),
            build_all_age_points_df_variant(
                ALL_RESULTS_ROOT_0131,
                experiment="GKF5",
                model="SPDNet",
                metric="MAE",
                negate=True,
                harm=False,
                variant="2layer",
            ).assign(Model="two (BiMap+ReEig)s(original)"),
            build_all_age_points_df_variant(
                ALL_RESULTS_ROOT_0131,
                experiment="GKF5",
                model="SPDNet",
                metric="MAE",
                negate=True,
                harm=True,
                variant="2layer",
            ).assign(Model="two (BiMap+ReEig)s(harmonization)"),
        ],
        ignore_index=True,
    )

    df_lodo_negmae = pd.concat(
        [
            build_all_age_points_df_variant(
                ALL_RESULTS_ROOT_0131,
                experiment="LODO",
                model="SPDNet",
                metric="MAE",
                negate=True,
                harm=False,
                variant="base",
            ).assign(Model="one (BiMap+ReEig)(original)"),
            build_all_age_points_df_variant(
                ALL_RESULTS_ROOT_0131,
                experiment="LODO",
                model="SPDNet",
                metric="MAE",
                negate=True,
                harm=True,
                variant="base",
            ).assign(Model="one (BiMap+ReEig)(harmonization)"),
            build_all_age_points_df_variant(
                ALL_RESULTS_ROOT_0131,
                experiment="LODO",
                model="SPDNet",
                metric="MAE",
                negate=True,
                harm=False,
                variant="2layer",
            ).assign(Model="two (BiMap+ReEig)s(original)"),
            build_all_age_points_df_variant(
                ALL_RESULTS_ROOT_0131,
                experiment="LODO",
                model="SPDNet",
                metric="MAE",
                negate=True,
                harm=True,
                variant="2layer",
            ).assign(Model="two (BiMap+ReEig)s(harmonization)"),
        ],
        ignore_index=True,
    )

    df_gkf5_r2 = pd.concat(
        [
            build_all_age_points_df_variant(
                ALL_RESULTS_ROOT_0131,
                experiment="GKF5",
                model="SPDNet",
                metric="R2",
                negate=False,
                harm=False,
                variant="base",
            ).assign(Model="one (BiMap+ReEig)(original)"),
            build_all_age_points_df_variant(
                ALL_RESULTS_ROOT_0131,
                experiment="GKF5",
                model="SPDNet",
                metric="R2",
                negate=False,
                harm=True,
                variant="base",
            ).assign(Model="one (BiMap+ReEig)(harmonization)"),
            build_all_age_points_df_variant(
                ALL_RESULTS_ROOT_0131,
                experiment="GKF5",
                model="SPDNet",
                metric="R2",
                negate=False,
                harm=False,
                variant="2layer",
            ).assign(Model="two (BiMap+ReEig)s(original)"),
            build_all_age_points_df_variant(
                ALL_RESULTS_ROOT_0131,
                experiment="GKF5",
                model="SPDNet",
                metric="R2",
                negate=False,
                harm=True,
                variant="2layer",
            ).assign(Model="two (BiMap+ReEig)s(harmonization)"),
        ],
        ignore_index=True,
    )

    df_lodo_r2 = pd.concat(
        [
            build_all_age_points_df_variant(
                ALL_RESULTS_ROOT_0131,
                experiment="LODO",
                model="SPDNet",
                metric="R2",
                negate=False,
                harm=False,
                variant="base",
            ).assign(Model="one (BiMap+ReEig)(original)"),
            build_all_age_points_df_variant(
                ALL_RESULTS_ROOT_0131,
                experiment="LODO",
                model="SPDNet",
                metric="R2",
                negate=False,
                harm=True,
                variant="base",
            ).assign(Model="one (BiMap+ReEig)(harmonization)"),
            build_all_age_points_df_variant(
                ALL_RESULTS_ROOT_0131,
                experiment="LODO",
                model="SPDNet",
                metric="R2",
                negate=False,
                harm=False,
                variant="2layer",
            ).assign(Model="two (BiMap+ReEig)s(original)"),
            build_all_age_points_df_variant(
                ALL_RESULTS_ROOT_0131,
                experiment="LODO",
                model="SPDNet",
                metric="R2",
                negate=False,
                harm=True,
                variant="2layer",
            ).assign(Model="two (BiMap+ReEig)s(harmonization)"),
        ],
        ignore_index=True,
    )

    model_order = [
        "one (BiMap+ReEig)(original)",
        "one (BiMap+ReEig)(harmonization)",
        "two (BiMap+ReEig)s(original)",
        "two (BiMap+ReEig)s(harmonization)",
    ]
    palette = {
        "one (BiMap+ReEig)(original)": "#7CCB7A",
        "one (BiMap+ReEig)(harmonization)": "#4DAF74",
        "two (BiMap+ReEig)s(original)": "#8FA7E0",
        "two (BiMap+ReEig)s(harmonization)": "#5778C8",
    }
    plot_all_age_box_grid(
        plots=[
            {
                "df": df_gkf5_negmae,
                "title": "GroupKFold (NegMAE)",
                "ylabel": "Negative MAE (↑)",
                "point_size": 3.5,
                "box_width": 0.8,
                "jitter": 0.10,
                "hue_offset_factor": 1.0,
                "order": model_order,
                "palette": palette,
            },
            {
                "df": df_lodo_negmae,
                "title": "LODO (NegMAE)",
                "ylabel": "Negative MAE (↑)",
                "point_size": 3.5,
                "box_width": 0.8,
                "jitter": 0.10,
                "hue_offset_factor": 1.0,
                "order": model_order,
                "palette": palette,
            },
            {
                "df": df_gkf5_r2,
                "title": "GroupKFold (R2)",
                "ylabel": "R2 (↑)",
                "point_size": 3.5,
                "zero_baseline": True,
                "box_width": 0.8,
                "jitter": 0.10,
                "hue_offset_factor": 1.0,
                "order": model_order,
                "palette": palette,
            },
            {
                "df": df_lodo_r2,
                "title": "LODO (R2)",
                "ylabel": "R2 (↑)",
                "point_size": 3.5,
                "zero_baseline": True,
                "box_width": 0.8,
                "jitter": 0.10,
                "hue_offset_factor": 1.0,
                "order": model_order,
                "palette": palette,
            },
        ],
        title="",
        save_dir="tables_out",
        save_name=f"benchmark_ALL_Age_grid_{tag}.png",
        figsize=(12, 5.5),
        dpi=300,
        legend_title="SPDNet",
    )

_build_all_age_plots_bimaps_0131()
