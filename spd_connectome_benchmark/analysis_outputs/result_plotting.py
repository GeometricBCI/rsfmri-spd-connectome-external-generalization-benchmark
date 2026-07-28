"""Reusable plotting primitives for benchmark result figures."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from spd_connectome_benchmark.config import DEFAULT_FIGURES_DIR
from spd_connectome_benchmark.analysis_outputs.result_io import (
    build_long_df,
    build_long_df_metric,
)
from spd_connectome_benchmark.analysis_outputs.plot_style import (
    rename_legend_labels as _rename_legend_labels,
)


def _seeded_stripplot(*, seed: int, **kwargs):
    """Draw a strip plot deterministically without changing global RNG state."""
    random_state = np.random.get_state()
    np.random.seed(seed)
    try:
        return sns.stripplot(**kwargs)
    finally:
        np.random.set_state(random_state)


def plot_paper_style_box(
    csv_map,
    dataset_labels=None,
    order=("Dummy", "Ridge", "SPDNet"),
    palette=None,
    title="Age Regression Benchmark",
    save_dir=DEFAULT_FIGURES_DIR,
    save_name="benchmark_Age_paper.pdf",
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
        _seeded_stripplot(
                    seed=seed,
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
        # Caller requested a legend-free panel for multi-axis figure layouts.
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
    save_dir=DEFAULT_FIGURES_DIR,
    save_name="figure3_within_dataset_negmae.pdf",
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
            _seeded_stripplot(
                seed=seed,
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

        title_text = ax.set_title(label, pad=8, fontfamily="monospace")
        title_text.set_multialignment("left")
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
    save_dir=DEFAULT_FIGURES_DIR,
    save_name="benchmark_metric.pdf",
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
            _seeded_stripplot(
                seed=seed,
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

        title_text = ax.set_title(label, pad=8, fontfamily="monospace")
        title_text.set_multialignment("left")
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
    save_dir=DEFAULT_FIGURES_DIR,
    save_name="figure3_within_dataset_negmae_r2.pdf",
    figsize=(10, 7),
    dpi=300,
    show_points=True,
    point_size=3.0,
    metric_bottom="R2",
    negate_bottom=False,
    ylabel_top="Negative MAE (↑)",
    ylabel_bottom="R² (↑)",
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
        title_text = ax_top.set_title(label, pad=8, fontfamily="monospace")
        title_text.set_multialignment("left")
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
