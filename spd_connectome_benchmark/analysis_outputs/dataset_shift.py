"""Generate descriptive dataset-shift figures used by Paper Figure 2."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle

from spd_connectome_benchmark.config import (
    DEFAULT_ATLAS_DIR,
    DEFAULT_FIGURES_DIR,
    DEFAULT_TABLES_DIR,
    PAPER_DATASETS,
)


# Paper Figure 2 / §2.1: six datasets in ascending scan-count order.
DATASETS = list(PAPER_DATASETS)
DISPLAY_NAMES = {
    "abide": "ABIDE",
    "adni": "ADNI",
    "adnidod": "ADNIDOD",
    "camcan": "Cam-CAN",
    "cobre": "COBRE",
    "oasis3": "OASIS-3",
}
COLORS = {
    "abide": "#0F6D8C",
    "adni": "#E67E22",
    "adnidod": "#C0392B",
    "camcan": "#2D8F6F",
    "cobre": "#7D4E9D",
    "oasis3": "#4361A8",
}
BACKGROUND = "#FFFFFF"
GRID = "#D9D9D9"
TEXT = "#1E1E1E"
RIDGE_SCALE = 2.2
RIDGE_BASE_LIFT = 0.10
RIDGE_LABEL_LIFT = 0.20
PKL_DIR = DEFAULT_ATLAS_DIR
TABLE_DIR = DEFAULT_TABLES_DIR
FIGURE_DIR = DEFAULT_FIGURES_DIR


def _rgba(color: str, alpha: float) -> tuple:
    return mcolors.to_rgba(color, alpha=alpha)


def _set_theme() -> None:
    sns.set_theme(style="white", context="paper", font_scale=1.08)
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "axes.edgecolor": TEXT,
            "axes.labelcolor": TEXT,
            "text.color": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "grid.color": GRID,
            "grid.alpha": 0.35,
            "axes.titleweight": "semibold",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.framealpha": 0.95,
        }
    )


def _load_data() -> tuple[pd.DataFrame, dict]:
    age_rows = []
    meta = {}
    for ds in DATASETS:
        with open(PKL_DIR / f"{ds}_X_y.pkl", "rb") as f:
            df = pickle.load(f)
        meta[ds] = df.copy()
        age = pd.to_numeric(df["Age"], errors="coerce")
        for value in age.dropna():
            age_rows.append({"dataset": ds, "dataset_label": DISPLAY_NAMES[ds], "age": float(value)})
    return pd.DataFrame(age_rows), meta


def _style_axis(ax, grid_axis: str = "x") -> None:
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#665F55")
    ax.spines["bottom"].set_color("#665F55")
    ax.grid(axis=grid_axis, alpha=0.28, linewidth=0.8)
    ax.grid(axis="both" if grid_axis == "none" else ("y" if grid_axis == "x" else "x"), visible=False)


def _compute_summary_frames(age_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset_summary = pd.read_csv(TABLE_DIR / "table1_dataset_summary.csv")
    dataset_summary = dataset_summary.set_index("dataset").loc[DATASETS].reset_index()
    diagnosis_summary = pd.read_csv(TABLE_DIR / "support" / "diagnosis_summary.csv")
    dataset_summary["dataset_label"] = dataset_summary["dataset"].map(DISPLAY_NAMES)
    dataset_summary["sex_female_frac"] = dataset_summary["n_female_scans"] / dataset_summary["n_scans"]
    dataset_summary["sex_male_frac"] = dataset_summary["n_male_scans"] / dataset_summary["n_scans"]

    lodo_rows = []
    for test_ds in DATASETS:
        test_age = age_df.loc[age_df["dataset"] == test_ds, "age"]
        train_age = age_df.loc[age_df["dataset"] != test_ds, "age"]
        lodo_rows.append(
            {
                "dataset": test_ds,
                "dataset_label": DISPLAY_NAMES[test_ds],
                "test_mean_age": float(test_age.mean()),
                "train_mean_age": float(train_age.mean()),
                "mean_gap": float(test_age.mean() - train_age.mean()),
            }
        )
    lodo_gap = pd.DataFrame(lodo_rows)
    return dataset_summary, diagnosis_summary, lodo_gap


def _dataset_order_by_scan_count(age_df: pd.DataFrame) -> list[str]:
    counts = age_df.groupby("dataset", observed=False).size().reindex(DATASETS)
    return counts.sort_values(kind="stable").index.tolist()


def _format_diag_label(label: str) -> str:
    cdr_label_map = {
        "CDR 0: cognitively normal": "cognitively\nnormal",
        "CDR 0.5: very mild impairment": "very mild\nimpairment",
        "CDR 1: mild dementia": "mild\ndementia",
        "CDR 2: moderate dementia": "moderate",
        "CDR 3: severe dementia": "severe",
    }
    if label in cdr_label_map:
        return cdr_label_map[label]
    return label


def _diagnosis_progression_sort_key(label: str) -> tuple[int, str]:
    text = str(label).strip().lower()
    progression_order = {
        "control": 0,
        "cn": 0,
        "cdr 0: cognitively normal": 0,
        "ptsd-negative": 0,
        "smc": 1,
        "cdr 0.5: very mild impairment": 1,
        "asd": 2,
        "mci": 2,
        "cdr 1: mild dementia": 2,
        "schizophrenia": 2,
        "ptsd-positive": 2,
        "ad": 3,
        "cdr 2: moderate dementia": 3,
        "cdr 3: severe dementia": 4,
        "missing": 5,
        "no diagnosis label retained": 5,
    }
    return progression_order.get(text, 6), text


def _ordered_diagnosis_columns(columns: list[str]) -> list[str]:
    labels = [str(col) for col in columns]
    global_priority = [
        "CN",
        "SMC",
        "MCI",
        "AD",
        "CDR 0: cognitively normal",
        "CDR 0.5: very mild impairment",
        "CDR 1: mild dementia",
        "CDR 2: moderate dementia",
        "CDR 3: severe dementia",
        "control",
        "ASD",
        "PTSD-negative",
        "PTSD-positive",
        "schizophrenia",
        "no diagnosis label retained",
    ]
    priority_index = {label: idx for idx, label in enumerate(global_priority)}
    return sorted(
        [label for label in labels if label != "missing"],
        key=lambda label: (priority_index.get(label, len(global_priority)), _diagnosis_progression_sort_key(label)),
    )


def _ordered_diagnosis_columns_for_dataset(dataset: str, columns: list[str]) -> list[str]:
    labels = [str(col) for col in columns]
    dataset_name = str(dataset).strip().lower()
    if dataset_name == "adni":
        priority = ["CN", "SMC", "MCI", "AD"]
        priority_index = {label: idx for idx, label in enumerate(priority)}
        return sorted(labels, key=lambda label: (priority_index.get(label, len(priority)), label))
    if dataset_name == "oasis3":
        priority = [
            "CDR 0: cognitively normal",
            "CDR 0.5: very mild impairment",
            "CDR 1: mild dementia",
            "CDR 2: moderate dementia",
            "CDR 3: severe dementia",
        ]
        priority_index = {label: idx for idx, label in enumerate(priority)}
        return [
            label
            for label in sorted(labels, key=lambda label: (priority_index.get(label, len(priority)), label))
            if label != "missing"
        ]
    return _ordered_diagnosis_columns(labels)


def _age_stat_text_y(dataset: str, stat_kind: str, base_y: float) -> float:
    ds = str(dataset).strip().lower()
    if ds == "cobre" and stat_kind in {"min", "max"}:
        return base_y - 0.05
    if ds == "camcan" and stat_kind in {"min", "max"}:
        return base_y - 0.05
    if ds == "abide" and stat_kind == "max":
        return base_y - 0.05
    if ds == "oasis3" and stat_kind in {"min", "max"}:
        return base_y - 0.05
    if ds == "adni" and stat_kind in {"min", "max"}:
        return base_y - 0.05
    if ds == "adnidod" and stat_kind == "min":
        return base_y + 0.05
    return base_y


def _draw_ridgeline_density(ax, dataset: str, ages: np.ndarray, offset: float, linewidth: float) -> np.ndarray:
    kde = sns.kdeplot(
        x=ages,
        bw_adjust=0.9,
        fill=False,
        cut=0,
        clip=(0, 105),
        ax=ax,
        color=COLORS[dataset],
        linewidth=0,
    )
    x, y = kde.lines[-1].get_data()
    kde.lines[-1].remove()
    y = y * RIDGE_SCALE
    ax.fill_between(x, offset, y + offset + RIDGE_BASE_LIFT, color=_rgba(COLORS[dataset], 0.30))
    ax.plot(x, y + offset + RIDGE_BASE_LIFT, color=COLORS[dataset], linewidth=linewidth)
    return y


def _draw_ridgeline_quartiles(
    ax,
    dataset: str,
    ages: np.ndarray,
    offset: float,
    density: np.ndarray,
    *,
    median_scale: float,
    median_linewidth: float,
    quartile_linewidth: float,
) -> tuple[float, float, float]:
    q1, median, q3 = np.quantile(ages, [0.25, 0.5, 0.75])
    ax.plot(
        [median, median],
        [offset + RIDGE_BASE_LIFT, offset + density.max() * median_scale + RIDGE_BASE_LIFT],
        color=COLORS[dataset],
        linewidth=median_linewidth,
        linestyle="--",
    )
    ax.plot(
        [q1, q3],
        [offset + 0.03, offset + 0.03],
        color=COLORS[dataset],
        linewidth=quartile_linewidth,
        solid_capstyle="round",
    )
    return float(q1), float(median), float(q3)


def _draw_age_ridgelines(ax_top, age_df: pd.DataFrame, ordered_datasets: list[str]) -> None:
    for idx, ds in enumerate(ordered_datasets[::-1]):
        sub = age_df[age_df["dataset"] == ds]["age"].to_numpy()
        offset = idx * 1.0
        y = _draw_ridgeline_density(ax_top, ds, sub, offset, linewidth=3.0)
        ax_top.hlines(offset, 0, 105, color=_rgba(COLORS[ds], 0.26), linewidth=1.0)
        ax_top.text(
            106.2,
            offset + RIDGE_LABEL_LIFT,
            DISPLAY_NAMES[ds],
            va="center",
            ha="left",
            fontsize=10,
            color=COLORS[ds],
            fontweight="semibold",
        )
        _draw_ridgeline_quartiles(
            ax_top,
            ds,
            sub,
            offset,
            y,
            median_scale=0.95,
            median_linewidth=1.8,
            quartile_linewidth=4.0,
        )

    ax_top.set_xlim(0, 112)
    ax_top.set_ylim(-0.2, len(DATASETS) - 0.1)
    ax_top.set_yticks([])
    ax_top.set_xlabel("")
    ax_top.set_ylabel("")
    ax_top.set_title("Age Shift Across Datasets", loc="left", pad=10)
    ax_top.text(
        0.0,
        1.02,
        "Ridgeline densities emphasize the severe age mismatch of ABIDE relative to the aging cohorts.",
        transform=ax_top.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#5E564C",
    )
    _style_axis(ax_top, grid_axis="x")


def _draw_age_box_strip(
    ax_bottom,
    age_df: pd.DataFrame,
    ordered_datasets: list[str],
    ordered_labels: list[str],
) -> None:
    sns.boxplot(
        data=age_df,
        x="age",
        y="dataset_label",
        order=ordered_labels,
        orient="h",
        palette=[_rgba(COLORS[d], 0.75) for d in ordered_datasets],
        width=0.56,
        linewidth=1.0,
        fliersize=0,
        ax=ax_bottom,
    )
    sns.stripplot(
        data=age_df.sample(min(len(age_df), 4500), random_state=0),
        x="age",
        y="dataset_label",
        order=ordered_labels,
        orient="h",
        palette=[_rgba(COLORS[d], 0.35) for d in ordered_datasets],
        size=1.8,
        jitter=0.19,
        alpha=0.18,
        ax=ax_bottom,
    )
    ax_bottom.set_xlabel("Age")
    ax_bottom.set_ylabel("")
    ax_bottom.set_title("Age Spread and Overlap", loc="left", pad=8)
    _style_axis(ax_bottom, grid_axis="x")


def _plot_age_distributions(age_df: pd.DataFrame) -> None:
    _set_theme()
    ordered_datasets = _dataset_order_by_scan_count(age_df)
    ordered_labels = [DISPLAY_NAMES[d] for d in ordered_datasets]
    fig = plt.figure(figsize=(12.5, 9.5), dpi=300, constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.0])
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bottom = fig.add_subplot(gs[1, 0])

    _draw_age_ridgelines(ax_top, age_df, ordered_datasets)
    _draw_age_box_strip(ax_bottom, age_df, ordered_datasets, ordered_labels)

    fig.savefig(TABLE_DIR / "dataset_shift_age_distributions.pdf", bbox_inches="tight")
    plt.close(fig)


def _draw_composition_counts(ax_counts, dataset_summary: pd.DataFrame) -> None:
    y = np.arange(len(DATASETS))
    scans = dataset_summary["n_scans"].to_numpy()
    subjects = dataset_summary["n_subjects"].to_numpy()
    ax_counts.barh(y, scans, color="#D9C6A5", height=0.58, label="scans")
    ax_counts.barh(y, subjects, color="#355C7D", height=0.34, label="subjects")
    for i, (s1, s2) in enumerate(zip(subjects, scans)):
        ax_counts.text(
            s2 + scans.max() * 0.012,
            i,
            f"{int(s1)}/{int(s2)}",
            va="center",
            ha="left",
            fontsize=9,
            color="#4B443A",
        )
    ax_counts.set_yticks(y, [DISPLAY_NAMES[d] for d in DATASETS])
    ax_counts.invert_yaxis()
    ax_counts.set_xlabel("Count")
    ax_counts.set_title("Subjects and Scans", loc="left")
    ax_counts.legend(frameon=True, loc="lower right")
    _style_axis(ax_counts, grid_axis="x")


def _draw_composition_sex_balance(ax_sex, dataset_summary: pd.DataFrame) -> None:
    y = np.arange(len(DATASETS))
    scans = dataset_summary["n_scans"].to_numpy()
    female = dataset_summary["n_female_scans"].to_numpy() / scans
    male = dataset_summary["n_male_scans"].to_numpy() / scans
    ax_sex.barh(y, male, color="#2D8F6F", height=0.55, label="Male")
    ax_sex.barh(y, female, left=male, color="#D95F5F", height=0.55, label="Female")
    for i, (m, f) in enumerate(zip(male, female)):
        ax_sex.text(
            0.01,
            i,
            f"{m*100:.0f}% M",
            va="center",
            ha="left",
            fontsize=8.8,
            color="white",
            fontweight="semibold",
        )
        ax_sex.text(
            min(m + f - 0.01, 0.99),
            i,
            f"{f*100:.0f}% F",
            va="center",
            ha="right",
            fontsize=8.8,
            color="white",
            fontweight="semibold",
        )
    ax_sex.set_xlim(0, 1)
    ax_sex.set_yticks(y, [DISPLAY_NAMES[d] for d in DATASETS])
    ax_sex.invert_yaxis()
    ax_sex.set_xlabel("Fraction of scans")
    ax_sex.set_title("Sex Balance", loc="left")
    ax_sex.legend(frameon=True, loc="lower right")
    _style_axis(ax_sex, grid_axis="x")


def _draw_composition_diagnosis(ax_diag, diagnosis_summary: pd.DataFrame) -> None:
    diag = diagnosis_summary.copy()
    diag["dataset"] = pd.Categorical(diag["dataset"], DATASETS, ordered=True)
    diag["dataset_label"] = diag["dataset"].map(DISPLAY_NAMES)
    diag["diagnosis_label"] = diag["diagnosis_label"].astype(str)
    diag_pivot = diag.pivot_table(
        index="dataset_label",
        columns="diagnosis_label",
        values="fraction_of_scans",
        aggfunc="sum",
        fill_value=0.0,
        observed=False,
    ).reindex([DISPLAY_NAMES[d] for d in DATASETS])
    diag_pivot = diag_pivot.reindex(columns=_ordered_diagnosis_columns_for_dataset("adni", list(diag_pivot.columns)))
    diag_colors = sns.color_palette("blend:#F4EBD8,#7A5195", n_colors=max(3, diag_pivot.shape[1]))
    left = np.zeros(len(diag_pivot))
    for color, column in zip(diag_colors, diag_pivot.columns):
        vals = diag_pivot[column].to_numpy()
        ax_diag.barh(
            diag_pivot.index,
            vals,
            left=left,
            color=color,
            edgecolor=BACKGROUND,
            linewidth=1.2,
            label=column,
        )
        for i, value in enumerate(vals):
            if value >= 0.095:
                ax_diag.text(left[i] + value / 2, i, column, ha="center", va="center", fontsize=8.2, color=TEXT)
        left += vals
    ax_diag.invert_yaxis()
    ax_diag.set_xlim(0, 1)
    ax_diag.set_xlabel("Fraction of scans")
    ax_diag.set_ylabel("")
    ax_diag.set_title("Diagnostic Composition", loc="left")
    ax_diag.legend(
        title="Diagnosis",
        ncol=5,
        fontsize=8,
        title_fontsize=9,
        frameon=True,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.24),
    )
    _style_axis(ax_diag, grid_axis="x")


def _plot_dataset_composition(_meta: dict) -> None:
    _set_theme()
    age_df, _ = _load_data()
    dataset_summary, diagnosis_summary, _ = _compute_summary_frames(age_df)

    fig = plt.figure(figsize=(13.5, 10.5), dpi=300, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.1], width_ratios=[1.0, 1.1])
    ax_counts = fig.add_subplot(gs[0, 0])
    ax_sex = fig.add_subplot(gs[0, 1])
    ax_diag = fig.add_subplot(gs[1, :])

    _draw_composition_counts(ax_counts, dataset_summary)
    _draw_composition_sex_balance(ax_sex, dataset_summary)
    _draw_composition_diagnosis(ax_diag, diagnosis_summary)

    fig.savefig(TABLE_DIR / "dataset_shift_composition.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_age_mean_gap_matrix(age_df: pd.DataFrame) -> None:
    _set_theme()
    means = {ds: age_df.loc[age_df["dataset"] == ds, "age"].mean() for ds in DATASETS}
    mat = np.array([[means[test] - means[ref] for ref in DATASETS] for test in DATASETS], dtype=float)

    fig, ax = plt.subplots(figsize=(9.2, 7.2), dpi=300, constrained_layout=True)
    cmap = sns.color_palette("blend:#21618C,#FFFFFF,#C0392B", as_cmap=True)
    sns.heatmap(
        mat,
        ax=ax,
        cmap=cmap,
        center=0.0,
        annot=True,
        fmt=".1f",
        linewidths=1.0,
        linecolor="#D9D9D9",
        cbar_kws={"label": "Held-out mean age minus reference mean age"},
        square=True,
        annot_kws={"fontsize": 10, "color": "#111111", "fontweight": "semibold"},
    )
    ax.set_xticklabels([DISPLAY_NAMES[d] for d in DATASETS], rotation=28, ha="right")
    ax.set_yticklabels([f"TEST {DISPLAY_NAMES[d]}" for d in DATASETS], rotation=0)
    ax.set_title("Dataset-by-Dataset Mean Age Gap", loc="left", pad=10)
    ax.set_xlabel("Reference dataset")
    ax.set_ylabel("Held-out dataset")

    abide_row = DATASETS.index("abide")
    ax.add_patch(Rectangle((0, abide_row), len(DATASETS), 1, fill=False, edgecolor="#111111", linewidth=2.6))
    ax.text(
        0.02,
        1.03,
        "ABIDE is the dominant outlier row: its mean age is dramatically younger than every aging cohort.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#5E564C",
    )

    fig.savefig(TABLE_DIR / "dataset_shift_age_mean_gap_matrix.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_lodo_train_test_age_gap(age_df: pd.DataFrame) -> None:
    _set_theme()
    rows = []
    for test_ds in DATASETS:
        test_age = age_df.loc[age_df["dataset"] == test_ds, "age"]
        train_age = age_df.loc[age_df["dataset"] != test_ds, "age"]
        rows.append(
            {
                "held_out_dataset": DISPLAY_NAMES[test_ds],
                "test_mean_age": float(test_age.mean()),
                "train_mean_age": float(train_age.mean()),
                "mean_gap": float(test_age.mean() - train_age.mean()),
            }
        )
    df = pd.DataFrame(rows)
    df = df.set_index("held_out_dataset").loc[[DISPLAY_NAMES[d] for d in DATASETS]].reset_index()

    fig, ax = plt.subplots(figsize=(10.8, 5.6), dpi=300, constrained_layout=True)
    x = np.arange(len(df))
    width = 0.36
    train_color = "#D9C6A5"
    test_colors = ["#C0392B" if name == "ABIDE" else "#355C7D" for name in df["held_out_dataset"]]
    ax.bar(
        x - width / 2,
        df["train_mean_age"],
        width=width,
        color=train_color,
        edgecolor="#8C7B68",
        linewidth=0.8,
        label="Pooled train mean age",
    )
    ax.bar(
        x + width / 2,
        df["test_mean_age"],
        width=width,
        color=test_colors,
        edgecolor="white",
        linewidth=0.8,
        label="Held-out test mean age",
    )
    for i, row in enumerate(df.itertuples(index=False)):
        txt_color = "#C0392B" if row.held_out_dataset == "ABIDE" else "#444444"
        ax.text(
            i + width / 2,
            row.test_mean_age + 1.0,
            f"{row.mean_gap:+.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=txt_color,
            fontweight="semibold",
        )

    ax.set_xticks(x, df["held_out_dataset"])
    ax.set_xlabel("")
    ax.set_ylabel("Mean age")
    ax.set_title("LODO Train vs Test Mean Age", loc="left")
    ax.text(
        0.0,
        1.02,
        "Numbers above test bars indicate held-out minus pooled-train mean age.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#5E564C",
    )
    ax.legend(frameon=True, loc="upper right")
    _style_axis(ax, grid_axis="y")

    fig.savefig(TABLE_DIR / "dataset_shift_lodo_train_test_age_gap.pdf", bbox_inches="tight")
    plt.close(fig)


def _overview_figure():
    fig = plt.figure(figsize=(15, 9), dpi=300, constrained_layout=True)
    gridspec = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0], width_ratios=[1.25, 1.0, 1.0])
    return (
        fig,
        fig.add_subplot(gridspec[:, 0]),
        fig.add_subplot(gridspec[0, 1]),
        fig.add_subplot(gridspec[0, 2]),
        fig.add_subplot(gridspec[1, 1:]),
    )


def _draw_overview_ridge_labels(
    ax_ridge,
    dataset: str,
    ages: np.ndarray,
    offset: float,
    density: np.ndarray,
    median: float,
) -> None:
    min_age = float(np.min(ages))
    max_age = float(np.max(ages))
    stat_y = offset + 0.19
    ax_ridge.text(
        min_age,
        _age_stat_text_y(dataset, "min", stat_y),
        f"{min_age:.1f}",
        ha="center",
        va="bottom",
        fontsize=9.1,
        color="#000000",
        fontweight="normal",
    )
    ax_ridge.text(
        median,
        offset + density.max() * 1.03 + RIDGE_BASE_LIFT,
        f"{median:.1f}",
        ha="center",
        va="bottom",
        fontsize=9.4,
        color="#000000",
        fontweight="normal",
    )
    ax_ridge.text(
        max_age,
        _age_stat_text_y(dataset, "max", stat_y),
        f"{max_age:.1f}",
        ha="center",
        va="bottom",
        fontsize=9.1,
        color="#000000",
        fontweight="normal",
    )
    ax_ridge.text(
        1.2,
        offset + RIDGE_LABEL_LIFT + 0.16,
        DISPLAY_NAMES[dataset],
        va="center",
        ha="left",
        fontsize=9.5,
        color="#000000",
        fontweight="normal",
    )


def _plot_overview_age_ridges(ax_ridge, age_df: pd.DataFrame, ordered_datasets: list[str]) -> None:
    for idx, ds in enumerate(ordered_datasets[::-1]):
        sub = age_df[age_df["dataset"] == ds]["age"].to_numpy()
        offset = idx * 1.0
        density = _draw_ridgeline_density(ax_ridge, ds, sub, offset, linewidth=2.8)
        _, median, _ = _draw_ridgeline_quartiles(
            ax_ridge,
            ds,
            sub,
            offset,
            density,
            median_scale=0.94,
            median_linewidth=1.6,
            quartile_linewidth=3.8,
        )
        _draw_overview_ridge_labels(ax_ridge, ds, sub, offset, density, median)
    ax_ridge.set_xlim(0, 105)
    ax_ridge.set_yticks([])
    ax_ridge.set_xlabel("Age")
    ax_ridge.set_title("A. Age Distributions", loc="left")
    _style_axis(ax_ridge, "x")


def _plot_overview_subject_scan_counts(
    ax_counts,
    ordered_summary: pd.DataFrame,
    ordered_datasets: list[str],
    ordered_labels: list[str],
) -> None:
    y = np.arange(len(ordered_datasets))
    scans = ordered_summary["n_scans"].to_numpy()
    subjects = ordered_summary["n_subjects"].to_numpy()
    row_colors = [COLORS[d] for d in ordered_datasets]
    count_scan_height = 0.56
    label_row_lift = 0.06
    ax_counts.barh(
        y,
        scans,
        color=[_rgba(c, 0.18) for c in row_colors],
        edgecolor=[_rgba(c, 0.55) for c in row_colors],
        linewidth=1.0,
        height=0.56,
        label="scans",
    )
    ax_counts.barh(
        y,
        subjects,
        color=[_rgba(c, 0.58) for c in row_colors],
        edgecolor="white",
        linewidth=0.9,
        height=0.32,
        label="subjects",
    )
    counts_xpad = float(scans.max()) * 0.015
    for yi, subj, scan in zip(y, subjects, scans):
        count_label = f"{int(subj)}/{int(scan)}"
        count_x = scan - counts_xpad
        count_ha = "right"
        count_color = TEXT
        if scan < scans.max() * 0.16:
            count_x = scan + counts_xpad
            count_ha = "left"
        ax_counts.text(
            count_x,
            yi - count_scan_height / 2 - label_row_lift,
            count_label,
            va="bottom",
            ha=count_ha,
            fontsize=8.5,
            color=count_color,
        )
    ax_counts.set_yticks(y, ordered_labels, fontsize=9)
    ax_counts.invert_yaxis()
    ax_counts.set_title("B. Subjects / Scans", loc="left")
    ax_counts.set_xlabel("Scan count")
    count_legend = [
        Patch(facecolor="#D9D9D9", edgecolor="#6E6E6E", linewidth=1.0, label="scans"),
        Patch(facecolor="#8A8A8A", edgecolor="#2F2F2F", linewidth=1.0, label="subjects"),
    ]
    ax_counts.legend(handles=count_legend, frameon=True, fontsize=8, loc="upper right")
    _style_axis(ax_counts, "x")


def _plot_overview_sex_balance(
    ax_sex,
    ordered_summary: pd.DataFrame,
    ordered_datasets: list[str],
    ordered_labels: list[str],
) -> None:
    y = np.arange(len(ordered_datasets))
    scans = ordered_summary["n_scans"].to_numpy()
    row_colors = [COLORS[d] for d in ordered_datasets]
    female_counts = ordered_summary["n_female_scans"].to_numpy()
    male_counts = ordered_summary["n_male_scans"].to_numpy()
    sex_height = 0.56
    label_row_lift = 0.06
    ax_sex.barh(
        y,
        male_counts,
        color=[_rgba(c, 0.55) for c in row_colors],
        edgecolor="white",
        linewidth=0.9,
        height=0.56,
        label="M",
    )
    ax_sex.barh(
        y,
        female_counts,
        left=male_counts,
        color=[_rgba(c, 0.22) for c in row_colors],
        edgecolor=[_rgba(c, 0.50) for c in row_colors],
        linewidth=0.9,
        height=0.56,
        label="F",
    )
    sex_xpad = float(scans.max()) * 0.015
    for yi, male_count, female_count, total in zip(y, male_counts, female_counts, scans):
        sex_label = f"{int(male_count)}/{int(female_count)}"
        sex_x = total - sex_xpad
        sex_ha = "right"
        if total < scans.max() * 0.16:
            sex_x = total + sex_xpad
            sex_ha = "left"
        ax_sex.text(
            sex_x,
            yi - sex_height / 2 - label_row_lift,
            sex_label,
            va="bottom",
            ha=sex_ha,
            fontsize=8.5,
            color=TEXT,
        )
    ax_sex.set_yticks(y, ordered_labels, fontsize=9)
    ax_sex.invert_yaxis()
    ax_sex.set_xlim(0, scans.max() * 1.02)
    ax_sex.set_title("C. Sex Balance (Scans)", loc="left")
    sex_legend = [
        Patch(facecolor="#D9D9D9", edgecolor="#6E6E6E", linewidth=1.0, label="F"),
        Patch(facecolor="#8A8A8A", edgecolor="#2F2F2F", linewidth=1.0, label="M"),
    ]
    ax_sex.legend(handles=sex_legend, frameon=True, fontsize=8, loc="upper right")
    ax_sex.set_xlabel("Scan count")
    _style_axis(ax_sex, "x")


def _plot_overview_diagnosis(ax_diag, diagnosis_summary: pd.DataFrame, ordered_datasets: list[str]) -> None:
    diag = diagnosis_summary.copy()
    diag["diagnosis_label"] = diag["diagnosis_label"].astype(str)
    diag_datasets = [ds for ds in ordered_datasets if ds != "camcan"]
    diag_labels = [DISPLAY_NAMES[d] for d in diag_datasets]
    y_positions = np.arange(len(diag_datasets))
    diag_total_counts = []
    top_label_rows: list[tuple[float, str]] = []
    for row_idx, ds in enumerate(diag_datasets):
        ds_diag = diag.loc[diag["dataset"] == ds, ["diagnosis_label", "count"]].copy()
        ordered_columns = _ordered_diagnosis_columns_for_dataset(ds, ds_diag["diagnosis_label"].tolist())
        ds_diag = ds_diag.set_index("diagnosis_label").reindex(ordered_columns).fillna(0.0).reset_index()
        diag_total_counts.append(float(ds_diag["count"].sum()))
        n_diag = max(1, len(ds_diag))
        alpha_levels = np.linspace(0.18, 0.58, n_diag)
        left = 0.0
        row_labels: list[str] = []
        for idx_col, row in enumerate(ds_diag.itertuples(index=False)):
            column = str(row.diagnosis_label)
            value = float(row.count)
            if value <= 0:
                continue
            color = _rgba(COLORS[ds], float(alpha_levels[idx_col]))
            ax_diag.barh(
                y_positions[row_idx],
                value,
                left=left,
                color=color,
                edgecolor=BACKGROUND,
                linewidth=1.0,
                height=0.8,
            )
            display_label = _format_diag_label(column)
            if display_label:
                row_labels.append(f"{display_label.replace(chr(10), ' ')} ({int(round(value))})")
            left += value
        if row_labels:
            top_label_rows.append((y_positions[row_idx], " / ".join(row_labels)))
    ax_diag.set_yticks(y_positions, diag_labels, fontsize=9)
    max_diag_count = max(diag_total_counts) if diag_total_counts else 1.0
    for y_pos, label_text in top_label_rows:
        ax_diag.text(
            0.01 * max_diag_count,
            y_pos - 0.39,
            label_text,
            ha="left",
            va="bottom",
            fontsize=9.1,
            color=TEXT,
            fontweight="normal",
        )
    ax_diag.invert_yaxis()
    ax_diag.set_xlim(0, max_diag_count * 1.02)
    ax_diag.set_title("D. Diagnosis Composition (Scans)", loc="left")
    ax_diag.set_xlabel("Scan count")
    ax_diag.set_ylim(len(diag_datasets) - 0.5, -0.92)
    _style_axis(ax_diag, "x")


def _plot_overview(age_df: pd.DataFrame) -> None:
    _set_theme()
    dataset_summary, diagnosis_summary, _ = _compute_summary_frames(age_df)
    ordered_datasets = _dataset_order_by_scan_count(age_df)
    ordered_labels = [DISPLAY_NAMES[d] for d in ordered_datasets]
    ordered_summary = dataset_summary.set_index("dataset").loc[ordered_datasets].reset_index()

    fig, ax_ridge, ax_counts, ax_sex, ax_diag = _overview_figure()
    _plot_overview_age_ridges(ax_ridge, age_df, ordered_datasets)
    _plot_overview_subject_scan_counts(ax_counts, ordered_summary, ordered_datasets, ordered_labels)
    _plot_overview_sex_balance(ax_sex, ordered_summary, ordered_datasets, ordered_labels)
    _plot_overview_diagnosis(ax_diag, diagnosis_summary, ordered_datasets)

    fig.savefig(FIGURE_DIR / "figure2_dataset_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper dataset-shift figures.")
    parser.add_argument("--pkl_dir", type=Path, default=DEFAULT_ATLAS_DIR, help="Directory with *_X_y.pkl files.")
    parser.add_argument(
        "--table_dir",
        type=Path,
        default=DEFAULT_TABLES_DIR,
        help="Directory for input/output tables.",
    )
    parser.add_argument(
        "--figure_dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Directory for generated figure outputs.",
    )
    parser.add_argument(
        "--include_components",
        action="store_true",
        help="Also write intermediate component plots that are not standalone PDF figures.",
    )
    return parser.parse_args()


def main(
    pkl_dir: Path = DEFAULT_ATLAS_DIR,
    table_dir: Path = DEFAULT_TABLES_DIR,
    figure_dir: Path | str = DEFAULT_FIGURES_DIR,
    include_components: bool = False,
) -> None:
    global PKL_DIR, TABLE_DIR, FIGURE_DIR
    PKL_DIR = pkl_dir
    TABLE_DIR = table_dir
    FIGURE_DIR = Path(figure_dir)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    age_df, meta = _load_data()
    if include_components:
        _plot_age_distributions(age_df)
        _plot_dataset_composition(meta)
        _plot_age_mean_gap_matrix(age_df)
        _plot_lodo_train_test_age_gap(age_df)
    _plot_overview(age_df)


if __name__ == "__main__":
    cli_args = parse_args()
    main(
        pkl_dir=cli_args.pkl_dir,
        table_dir=cli_args.table_dir,
        figure_dir=cli_args.figure_dir,
        include_components=cli_args.include_components,
    )
