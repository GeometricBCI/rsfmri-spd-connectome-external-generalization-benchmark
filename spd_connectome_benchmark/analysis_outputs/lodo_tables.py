"""Generate per-held-out-dataset summaries for Paper Table 3."""

import argparse
from pathlib import Path
import re

import pandas as pd

from spd_connectome_benchmark.config import DEFAULT_POOLED_RESULTS_DIR, DEFAULT_TABLES_DIR, PAPER_DATASETS

METRICS = ["MAE", "R2", "Spearman_rho", "age_bias_slope"]
# Paper Methods 2.1 / Figure order: ascending scan count.
DATASET_ORDER = list(PAPER_DATASETS)


def parse_model_and_harm(path: Path) -> tuple[str, str]:
    """Map result filenames to the model labels used in Paper Table 3."""
    name = path.name
    harm = "harmonized" if "_harm.csv" in name else "original"
    if "_SPDNet_" in name:
        model = "SPDNet"
    elif "_CorrVec_Ridge_" in name:
        model = "CorrVec"
    elif "_Ridge_AgeReg_TS_" in name:
        model = "Ridge"
    elif "_Dummy_AgeReg_" in name:
        model = "Dummy"
    else:
        model = "Unknown"
    return model, harm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper LODO per-dataset summary tables.")
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_POOLED_RESULTS_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_TABLES_DIR)
    return parser.parse_args()


def _collect_lodo_metric_rows(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("*LODO*.csv")):
        df = pd.read_csv(path, index_col=0)
        model, harm = parse_model_and_harm(path)
        test_cols = [c for c in df.columns if c.startswith("TEST_")]
        for col in test_cols:
            held_out = col.removeprefix("TEST_")
            row = {
                "held_out_dataset": held_out,
                "model": model,
                "harmonization": harm,
            }
            for metric in METRICS:
                row[metric] = float(df.loc[metric, col]) if metric in df.index else float("nan")
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No LODO result CSV files found in {results_dir}")
    return rows


def _long_lodo_metrics_table(rows: list[dict]) -> pd.DataFrame:
    long_df = pd.DataFrame(rows)
    long_df["held_out_dataset"] = pd.Categorical(long_df["held_out_dataset"], DATASET_ORDER, ordered=True)
    long_df["model"] = pd.Categorical(long_df["model"], ["Dummy", "CorrVec", "Ridge", "SPDNet"], ordered=True)
    long_df["harmonization"] = pd.Categorical(long_df["harmonization"], ["original", "harmonized"], ordered=True)
    long_df = long_df.sort_values(["held_out_dataset", "model", "harmonization"]).reset_index(drop=True)
    return long_df.rename(
        columns={
            "R2": "R_squared",
            "Spearman_rho": "Spearman_rank_correlation",
        }
    )


def _wide_lodo_metrics_table(long_df: pd.DataFrame) -> pd.DataFrame:
    wide_df = long_df.pivot(
        index="held_out_dataset",
        columns=["model", "harmonization"],
        values=["MAE", "R_squared", "Spearman_rank_correlation", "age_bias_slope"],
    )
    return wide_df.sort_index(axis=1, level=[0, 1, 2])


def _lodo_metric_extrema_table(long_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for held_out, sub in long_df.groupby("held_out_dataset", observed=False):
        if pd.isna(held_out):
            continue
        best_mae = sub.loc[sub["MAE"].idxmin()]
        best_r2 = sub.loc[sub["R_squared"].idxmax()]
        best_spearman = sub.loc[sub["Spearman_rank_correlation"].idxmax()]
        most_extreme_bias = sub.loc[sub["age_bias_slope"].abs().idxmax()]
        summary_rows.append(
            {
                "held_out_dataset": held_out,
                "lowest_MAE_model": f"{best_mae['model']} ({best_mae['harmonization']})",
                "lowest_MAE": best_mae["MAE"],
                "highest_R2_model": f"{best_r2['model']} ({best_r2['harmonization']})",
                "highest_R2": best_r2["R_squared"],
                "highest_Spearman_model": f"{best_spearman['model']} ({best_spearman['harmonization']})",
                "highest_Spearman": best_spearman["Spearman_rank_correlation"],
                "largest_abs_age_bias_slope_model": (
                    f"{most_extreme_bias['model']} ({most_extreme_bias['harmonization']})"
                ),
                "largest_abs_age_bias_slope": most_extreme_bias["age_bias_slope"],
            }
        )
    return pd.DataFrame(summary_rows)


def main(
    results_dir: Path = DEFAULT_POOLED_RESULTS_DIR,
    out_dir: Path = DEFAULT_TABLES_DIR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    long_df = _long_lodo_metrics_table(_collect_lodo_metric_rows(results_dir))
    wide_df = _wide_lodo_metrics_table(long_df)
    extrema_df = _lodo_metric_extrema_table(long_df)

    wide_df.to_csv(out_dir / "table3_lodo_per_dataset_metrics.csv")
    support_dir = out_dir / "support"
    support_dir.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(support_dir / "lodo_per_dataset_metrics_long.csv", index=False)
    extrema_df.to_csv(support_dir / "lodo_per_dataset_metric_extrema.csv", index=False)


if __name__ == "__main__":
    cli_args = parse_args()
    main(results_dir=cli_args.results_dir, out_dir=cli_args.out_dir)
