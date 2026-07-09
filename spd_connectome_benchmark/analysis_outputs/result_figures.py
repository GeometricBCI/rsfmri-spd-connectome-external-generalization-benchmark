"""Build paper benchmark result figures from saved CSV outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from spd_connectome_benchmark.config import (
    DEFAULT_ABLATION_RESULTS_DIR,
    DEFAULT_FIGURES_DIR,
    DEFAULT_POOLED_RESULTS_DIR,
    DEFAULT_SINGLE_RESULTS_DIR,
    PAPER_DATASETS,
)
from spd_connectome_benchmark.analysis_outputs.result_io import (
    _format_dataset_label,
    build_all_age_points_df,
    build_all_age_points_df_variant,
    build_csv_map_from_results,
)
from spd_connectome_benchmark.analysis_outputs.result_plotting import (
    plot_all_age_box_grid,
    plot_all_age_box_grid_wrapped,
    plot_paper_style_box_row,
    plot_paper_style_box_row_metric,
)

DEFAULT_RESULT_FIGURES_DIR = DEFAULT_FIGURES_DIR

SINGLE_DATASET_RESULTS_ROOT = str(DEFAULT_SINGLE_RESULTS_DIR)
ALL_RESULTS_ROOT = str(DEFAULT_POOLED_RESULTS_DIR)
ALL_RESULTS_ROOT_0423 = str(DEFAULT_ABLATION_RESULTS_DIR)
RESULT_FIGURES_DIR = str(DEFAULT_RESULT_FIGURES_DIR)


def configure_result_paths(
    single_results_dir: Path | str = DEFAULT_SINGLE_RESULTS_DIR,
    pooled_results_dir: Path | str = DEFAULT_POOLED_RESULTS_DIR,
    ablation_results_dir: Path | str = DEFAULT_ABLATION_RESULTS_DIR,
    result_figure_dir: Path | str = DEFAULT_RESULT_FIGURES_DIR,
) -> None:
    """Set result directories used by the plotting helpers."""
    global SINGLE_DATASET_RESULTS_ROOT, ALL_RESULTS_ROOT, ALL_RESULTS_ROOT_0423, RESULT_FIGURES_DIR
    SINGLE_DATASET_RESULTS_ROOT = str(single_results_dir)
    ALL_RESULTS_ROOT = str(pooled_results_dir)
    ALL_RESULTS_ROOT_0423 = str(ablation_results_dir)
    RESULT_FIGURES_DIR = str(result_figure_dir)


def _build_single_dataset_plots():
    """Generate Figure 3-style within-dataset plots explicitly."""
    results_root = SINGLE_DATASET_RESULTS_ROOT
    # Paper Methods 2.1 / Figure 3: ascending scan-count order.
    DATASET_ORDER = list(PAPER_DATASETS)
    DATASET_LABELS = {
        "abide": _format_dataset_label("ABIDE", 843, 843),
        "adni": _format_dataset_label("ADNI", 936, 1997),
        "adnidod": _format_dataset_label("ADNIDOD", 134, 190),
        "camcan": _format_dataset_label("Cam-CAN", 652, 652),
        "cobre": _format_dataset_label("COBRE", 143, 143),
        "oasis3": _format_dataset_label("OASIS-3", 1035, 1792),
    }
    MODEL_ORDER = ("Dummy", "VecCorrRidge", "Ridge", "SPDNet")
    MODEL_PALETTE = {
        "Dummy": "#4C78A8",
        "VecCorrRidge": "#E45756",
        "Ridge": "#F58518",
        "SPDNet": "#54A24B",
    }

    CSV_MAP = build_csv_map_from_results(
        results_root,
        DATASET_ORDER,
        models=MODEL_ORDER,
    )

    plot_paper_style_box_row(
        csv_map=CSV_MAP,
        dataset_labels=DATASET_LABELS,
        order=MODEL_ORDER,
        palette=MODEL_PALETTE,
        title="",
        save_dir=RESULT_FIGURES_DIR,
        save_name="figure3_within_dataset_negmae.pdf",
        figsize=(11, 4),
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
        order=MODEL_ORDER,
        palette=MODEL_PALETTE,
        title="",
        save_dir=RESULT_FIGURES_DIR,
        save_name="figure3_within_dataset_r2.pdf",
        figsize=(11, 4),
        show_points=True,
        point_size=2.5,
        dpi=300,
        metric="R2",
        negate=False,
        ylabel="R² (↑)",
        align_zero=True,
        box_width=0.8,
        strip_jitter=0.0,
        wspace=0.6,
    )

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
    out = out[~out["Model"].astype(str).str.startswith("Dummy")]
    out["Model"] = out["Model"].replace(
        {
            "Dummy (harmonization)": "Dummy",
        }
    )
    return out

ALL_COMBINED_MODEL_ORDER = [
    "VecCorrRidge (original)",
    "VecCorrRidge (harmonization)",
    "Ridge (original)",
    "Ridge (harmonization)",
    "SPDNet (original)",
    "SPDNet (harmonization)",
]
ALL_COMBINED_PALETTE = {
    "VecCorrRidge (original)": "#E45756",
    "VecCorrRidge (harmonization)": "#F08A83",
    "Ridge (original)": "#F58518",
    "Ridge (harmonization)": "#F8B26A",
    "SPDNet (original)": "#7CCB7A",
    "SPDNet (harmonization)": "#4DAF74",
}
ALL_COMBINED_MODELS = ("VecCorrRidge", "Ridge", "SPDNet")

def _build_all_age_metric_points(experiment: str, metric: str, negate: bool = False) -> pd.DataFrame:
    df = pd.concat(
        [
            _with_harm_label(
                build_all_age_points_df(
                    ALL_RESULTS_ROOT,
                    experiment=experiment,
                    models=ALL_COMBINED_MODELS,
                    metric=metric,
                    negate=negate,
                    harm=False,
                ),
                "original",
            ),
            _with_harm_label(
                build_all_age_points_df(
                    ALL_RESULTS_ROOT,
                    experiment=experiment,
                    models=ALL_COMBINED_MODELS,
                    metric=metric,
                    negate=negate,
                    harm=True,
                ),
                "harmonization",
            ),
        ],
        ignore_index=True,
    )
    return _adjust_harmonization_labels(df)

def _dummy_metric_summary(experiment: str, metric: str, negate: bool = False) -> str:
    df = build_all_age_points_df(
        ALL_RESULTS_ROOT,
        experiment=experiment,
        models=("Dummy",),
        metric=metric,
        negate=negate,
        harm=False,
    )
    values = pd.to_numeric(df.get("Value", pd.Series(dtype=float)), errors="coerce").dropna()
    if values.empty:
        return "Dummy: undefined"
    return f"Dummy mean={values.mean():.2f}, std={values.std(ddof=0):.2f}"

def _all_age_metric_plot(experiment, metric, negate, title, ylabel, zero_baseline):
    return {
        "df": _build_all_age_metric_points(experiment, metric, negate=negate),
        "title": title,
        "subtitle": _dummy_metric_summary(experiment, metric, negate=negate),
        "ylabel": ylabel,
        "point_size": 3.0,
        "zero_baseline": zero_baseline,
        "box_width": 0.8,
        "jitter": 0.10,
        "hue_offset_factor": 1.0,
        "order": ALL_COMBINED_MODEL_ORDER,
        "palette": ALL_COMBINED_PALETTE,
    }

def _build_all_age_plots_pooled_all_metrics(tag: str = "pooled_all_metrics"):
    metric_specs = [
        ("GKF5", "MAE", True, "GroupKFold (NegMAE)", "Negative MAE (↑)", False),
        ("LODO", "MAE", True, "LODO (NegMAE)", "Negative MAE (↑)", False),
        ("GKF5", "R2", False, "GroupKFold (R²)", "R² (↑)", True),
        ("LODO", "R2", False, "LODO (R²)", "R² (↑)", True),
        ("LODO", "Spearman_rho", False, "LODO (Spearman rho)", "Spearman rho (↑)", True),
    ]
    plots = [
        _all_age_metric_plot(experiment, metric, negate, plot_title, ylabel, zero_baseline)
        for experiment, metric, negate, plot_title, ylabel, zero_baseline in metric_specs
    ]

    plot_all_age_box_grid_wrapped(
        plots=plots,
        title="",
        save_dir=RESULT_FIGURES_DIR,
        save_name=f"{tag}.pdf",
        ncols=5,
        figsize=(16.0, 4.2),
        dpi=300,
        legend_ncol=6,
    )

def _build_bimap_variant_points_df(metric: str, experiment: str, negate: bool = False):
    variant_labels = [
        ("quarterdim", "quarterdim (BiMap+ReEig)"),
        ("halfdim", "halfdim (BiMap+ReEig)"),
        ("one", "one (BiMap+ReEig)"),
        ("two", "two (BiMap+ReEig)s"),
    ]
    frames = []
    for variant, label in variant_labels:
        frames.append(
            build_all_age_points_df_variant(
                ALL_RESULTS_ROOT_0423,
                experiment=experiment,
                model="SPDNet",
                metric=metric,
                negate=negate,
                harm=False,
                variant=variant,
            ).assign(Model=f"{label}(original)")
        )
        frames.append(
            build_all_age_points_df_variant(
                ALL_RESULTS_ROOT_0423,
                experiment=experiment,
                model="SPDNet",
                metric=metric,
                negate=negate,
                harm=True,
                variant=variant,
            ).assign(Model=f"{label}(harmonization)")
        )
    return pd.concat(frames, ignore_index=True)

def _build_all_age_plots_bimaps_0423(tag: str = "bimaps_0423"):
    df_gkf5_negmae = _build_bimap_variant_points_df("MAE", experiment="GKF5", negate=True)
    df_lodo_negmae = _build_bimap_variant_points_df("MAE", experiment="LODO", negate=True)
    df_gkf5_r2 = _build_bimap_variant_points_df("R2", experiment="GKF5")
    df_lodo_r2 = _build_bimap_variant_points_df("R2", experiment="LODO")
    df_lodo_spearman = _build_bimap_variant_points_df("Spearman_rho", experiment="LODO")

    model_order = [
        "quarterdim (BiMap+ReEig)(original)",
        "quarterdim (BiMap+ReEig)(harmonization)",
        "halfdim (BiMap+ReEig)(original)",
        "halfdim (BiMap+ReEig)(harmonization)",
        "one (BiMap+ReEig)(original)",
        "one (BiMap+ReEig)(harmonization)",
        "two (BiMap+ReEig)s(original)",
        "two (BiMap+ReEig)s(harmonization)",
    ]
    palette = {
        "quarterdim (BiMap+ReEig)(original)": "#C3A6E8",
        "quarterdim (BiMap+ReEig)(harmonization)": "#8E63C7",
        "halfdim (BiMap+ReEig)(original)": "#8FA7E0",
        "halfdim (BiMap+ReEig)(harmonization)": "#5778C8",
        "one (BiMap+ReEig)(original)": "#7CCB7A",
        "one (BiMap+ReEig)(harmonization)": "#4DAF74",
        "two (BiMap+ReEig)s(original)": "#A0A0A0",
        "two (BiMap+ReEig)s(harmonization)": "#5F5F5F",
    }
    plot_all_age_box_grid_wrapped(
        plots=[
            {
                "df": df_gkf5_negmae,
                "title": "GroupKFold (NegMAE)",
                "ylabel": "Negative MAE (↑)",
                "point_size": 3.5,
                "box_width": 0.78,
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
                "box_width": 0.78,
                "jitter": 0.10,
                "hue_offset_factor": 1.0,
                "order": model_order,
                "palette": palette,
            },
            {
                "df": df_gkf5_r2,
                "title": "GroupKFold (R²)",
                "ylabel": "R² (↑)",
                "point_size": 3.5,
                "zero_baseline": True,
                "box_width": 0.78,
                "jitter": 0.10,
                "hue_offset_factor": 1.0,
                "order": model_order,
                "palette": palette,
            },
            {
                "df": df_lodo_r2,
                "title": "LODO (R²)",
                "ylabel": "R² (↑)",
                "point_size": 3.5,
                "zero_baseline": True,
                "box_width": 0.78,
                "jitter": 0.10,
                "hue_offset_factor": 1.0,
                "order": model_order,
                "palette": palette,
            },
            {
                "df": df_lodo_spearman,
                "title": "LODO (Spearman $\\rho$)",
                "ylabel": "Spearman $\\rho$ (↑)",
                "point_size": 3.5,
                "zero_baseline": True,
                "box_width": 0.78,
                "jitter": 0.10,
                "hue_offset_factor": 1.0,
                "order": model_order,
                "palette": palette,
            },
        ],
        title="",
        save_dir=RESULT_FIGURES_DIR,
        save_name=f"{tag}.pdf",
        ncols=5,
        figsize=(16.0, 4.2),
        dpi=300,
        legend_title="SPDNet",
        legend_ncol=8,
    )


def build_all_result_figures() -> None:
    """Generate only the new benchmark result figures Figure 4 and Figure 5."""
    _build_single_dataset_plots()
    _build_all_age_plots_pooled_all_metrics(tag="figure4_pooled_benchmark_5panel_row_narrow_tall")
    _build_all_age_plots_bimaps_0423(tag="figure5_spdnet_ablation_5panel_row")
