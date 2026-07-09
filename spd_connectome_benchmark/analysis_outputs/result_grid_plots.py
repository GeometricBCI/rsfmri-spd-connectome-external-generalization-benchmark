"""Generic grid and box plotting helpers for benchmark result figures."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from spd_connectome_benchmark.config import DEFAULT_FIGURES_DIR
from spd_connectome_benchmark.analysis_outputs.plot_style import (
    align_strip_points_to_box_centers,
    compress_hue_offsets,
    rename_legend_labels,
)


def plot_all_age_avg_bar(
    df,
    order=("Dummy", "Ridge", "SPDNet"),
    palette=None,
    title="ALL Age Regression",
    subtitle=None,
    ylabel="Metric (↑)",
    save_dir=DEFAULT_FIGURES_DIR,
    save_name="benchmark_ALL_Age.pdf",
    figsize=(5.5, 4),
    dpi=300,
):
    os.makedirs(save_dir, exist_ok=True)

    if palette is None:
        palette = {
            "Dummy": "#4C78A8",
            "Ridge": "#F58518",
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
    ax.set_title(title, pad=20 if subtitle else 10)
    if subtitle:
        ax.text(
            0.5,
            1.015,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
            color="#4b5563",
        )
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    labels = rename_legend_labels(labels)
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
    subtitle=None,
    ylabel="Metric (↑)",
    save_dir=DEFAULT_FIGURES_DIR,
    save_name="benchmark_ALL_Age.pdf",
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
    smart_ylim=True,
):
    if palette is None:
        palette = {
            "Dummy": "#4C78A8",
            "Ridge": "#F58518",
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
        compress_hue_offsets(ax, factor=hue_offset_factor)
    align_strip_points_to_box_centers(ax)

    ax.set_title(title, pad=20 if subtitle else 10)
    if subtitle:
        ax.text(
            0.5,
            1.015,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
            color="#4b5563",
        )
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    draw_zero = False
    if smart_ylim and "Value" in df:
        values = pd.to_numeric(df["Value"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not values.empty:
            vmin = float(values.min())
            vmax = float(values.max())
            span = vmax - vmin
            pad = max(abs(vmin) * 0.05, 0.1) if span <= 0 else span * 0.12
            near_zero = min(abs(vmin), abs(vmax)) <= max(span * 0.25, 1e-12)
            draw_zero = bool(zero_baseline and (vmin <= 0.0 <= vmax or near_zero))
            ymin = min(vmin, 0.0) if draw_zero else vmin
            ymax = max(vmax, 0.0) if draw_zero else vmax
            ax.set_ylim(ymin - pad, ymax + pad)
    else:
        draw_zero = bool(zero_baseline)

    if draw_zero:
        ax.axhline(0.0, color="#6b7280", linewidth=1.0, alpha=0.8, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", alpha=0.25)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        handles = handles[: len(model_order)]
        labels = rename_legend_labels(labels[: len(model_order)])
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
    save_dir=DEFAULT_FIGURES_DIR,
    save_name="figure4_pooled_benchmark_5panel_row_narrow_tall.pdf",
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
            subtitle=cfg.get("subtitle"),
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
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        uniq.append((handle, label))
    handles, labels = zip(*uniq) if uniq else ([], [])
    labels = rename_legend_labels(labels)
    fig.legend(
        handles,
        labels,
        title=legend_title,
        frameon=True,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.045),
        ncol=min(len(labels), 3) if labels else 1,
        columnspacing=1.2,
        handletextpad=0.6,
    )

    if title:
        fig.suptitle(title, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.895])
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print("Saved figure to:", save_path)
    plt.show()


def plot_all_age_box_grid_wrapped(
    plots,
    title="ALL Age Regression",
    save_dir=DEFAULT_FIGURES_DIR,
    save_name="figure4_pooled_benchmark_5panel_row_narrow_tall.pdf",
    ncols=3,
    figsize=(15, 12),
    dpi=300,
    legend_title="Regressor",
    legend_ncol=None,
):
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)

    nplots = len(plots)
    nrows = int(np.ceil(nplots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
    axes = np.asarray(axes).ravel()

    for ax, cfg in zip(axes, plots):
        plot_all_age_box(
            cfg["df"],
            order=cfg.get("order", ("Dummy", "Ridge", "SPDNet")),
            palette=cfg.get("palette"),
            title=cfg.get("title", ""),
            subtitle=cfg.get("subtitle"),
            ylabel=cfg.get("ylabel", "Metric"),
            show_points=cfg.get("show_points", True),
            point_size=cfg.get("point_size", 2.5),
            box_width=cfg.get("box_width", 0.8),
            jitter=cfg.get("jitter", 0.10),
            hue_offset_factor=cfg.get("hue_offset_factor", 1.0),
            ax=ax,
            show_legend=False,
            zero_baseline=cfg.get("zero_baseline", False),
        )
        ax.set_xlabel("")
        ax.set_xticks([])

    for ax in axes[nplots:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    seen = set()
    uniq = []
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        uniq.append((handle, label))
    handles, labels = zip(*uniq) if uniq else ([], [])
    labels = rename_legend_labels(labels)
    fig.legend(
        handles,
        labels,
        title=legend_title,
        frameon=True,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=legend_ncol or (min(len(labels), 3) if labels else 1),
        columnspacing=1.2,
        handletextpad=0.6,
    )

    if title:
        fig.suptitle(title, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.89])
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print("Saved figure to:", save_path)
    plt.show()
