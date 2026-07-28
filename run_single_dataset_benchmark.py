"""Run the paper's within-dataset GroupKFold age-regression benchmark.

This script implements the single-dataset part of Methods 2.4-2.7:
OAS-based regularized SPD correlation connectomes, subject-level GroupKFold,
CorrVec, Tangent-Space Ridge, Dummy, and one-block SPDNet.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler

from pyriemann.tangentspace import TangentSpace

from spd_connectome_benchmark.config import (
    DEFAULT_SINGLE_RESULTS_DIR,
    DEFAULT_SINGLE_WEIGHTS_DIR,
    PAPER_DATASETS,
)
from spd_connectome_benchmark.connectomes import (
    estimate_connectome_matrices,
    vectorize_correlation_matrices,
)
from spd_connectome_benchmark.models.spd import SPDNetRegressor
from spd_connectome_benchmark.results import portable_result_reference
from spd_connectome_benchmark.benchmark_tools.runtime import (
    EarlyStopping,
    MatrixRegressionDataset,
    ensure_nonempty_training_batches,
    evaluate_regression_loss,
    load_age_timeseries,
    save_metrics_csv,
    split_train_validation_by_group,
    timestamp_tag,
)
from spd_connectome_benchmark.benchmark_tools.cli import (
    add_connectome_args,
    add_data_root_arg,
    add_device_arg,
    add_logging_args,
    add_ridge_args,
    add_split_args,
    add_spdnet_head_args,
    add_spdnet_optimization_args,
    finalize_single_dataset_runtime,
)
from spd_connectome_benchmark.benchmark_tools.logging import configure_logging


def fit_ridge_regressor(args, Z_tr, y_tr, groups_tr, alphas):
    """Fit Ridge with group-aware inner CV over alpha values."""
    n_groups_tr = len(np.unique(groups_tr))
    inner_k = min(args.ridge_inner_splits, n_groups_tr)

    # Original single-dataset code standardized tangent features before Ridge.
    # The paper does not describe this scaler explicitly; see the alignment
    # report before comparing these numbers against the paper text.
    base = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(random_state=args.seed)),
    ])

    if inner_k < 2:
        best_alpha = float(alphas[0])
        base.set_params(ridge__alpha=best_alpha)
        base.fit(Z_tr, y_tr)
        return base, best_alpha

    inner_cv = GroupKFold(n_splits=inner_k)
    gs = GridSearchCV(
        estimator=base,
        param_grid={"ridge__alpha": list(alphas)},
        cv=inner_cv,
        scoring="neg_mean_absolute_error",
        n_jobs=args.ridge_n_jobs,
    )
    gs.fit(Z_tr, y_tr, groups=groups_tr)
    return gs.best_estimator_, float(gs.best_params_["ridge__alpha"])


def run_spdnet_age_regression(args, X, y, subject_ids, splits, device, dataset_name: str):
    """Run the within-dataset SPDNet age-regression benchmark."""
    alg_name = "SPDNet_AgeReg"
    fold_mae, fold_rmse, fold_r2, fold_mse = [], [], [], []
    t0 = time.time()

    for kf_iter, (train_idx, test_idx) in enumerate(splits, start=1):
        # Paper Methods 2.5.2/2.7: group-aware 10% validation split.
        tr_idx, va_idx = split_train_validation_by_group(
            train_idx, groups=subject_ids, val_size=args.val_size, seed=args.seed
        )

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        ensure_nonempty_training_batches(
            len(X_tr),
            args.train_batch_size,
            drop_last=True,
        )

        train_loader = DataLoader(
            MatrixRegressionDataset(
                torch.from_numpy(X_tr).float(),
                torch.from_numpy(y_tr).float(),
            ),
            batch_size=args.train_batch_size,
            shuffle=True,
            pin_memory=True,
            # Source behavior: incomplete final training batch is dropped. The
            # paper states the batch size but does not spell out drop_last.
            drop_last=True,
        )
        # Training batches are shuffled under the run-level RNG, without a
        # fold-specific DataLoader generator seed.
        val_loader = DataLoader(
            MatrixRegressionDataset(
                torch.from_numpy(X_va).float(),
                torch.from_numpy(y_va).float(),
            ),
            batch_size=args.test_batch_size,
            shuffle=False,
            pin_memory=True,
        )
        test_loader = DataLoader(
            MatrixRegressionDataset(
                torch.from_numpy(X_te).float(),
                torch.from_numpy(y_te).float(),
            ),
            batch_size=args.test_batch_size,
            shuffle=False,
            pin_memory=True,
        )

        model = SPDNetRegressor(
            dims=(X.shape[1], X.shape[2]),
            out_dim=1,
            fc_layer_no=args.fc_layer_no,
            fc_hidden_dim=args.fc_hidden_dim,
            fc_dropout=args.fc_dropout,
            # Source behavior for the within-dataset script; pooled runs keep
            # the default float32 LogEig path.
            logeig_double_precision=True,
        ).to(device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=args.initial_lr,
            weight_decay=args.weight_decay,
        )
        criterion = nn.MSELoss()

        ckpt_dir = Path(args.weights_folder_root) / dataset_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"{alg_name}_fold{kf_iter}.pt"

        early = EarlyStopping(ckpt_path, patience=args.patience, verbose=True)

        print(f"\n===== {dataset_name} | SPDNet Fold {kf_iter}/{len(splits)} =====")
        print(
            f"Train groups: {len(np.unique(subject_ids[tr_idx]))} | "
            f"Val groups: {len(np.unique(subject_ids[va_idx]))} | "
            f"Test groups: {len(np.unique(subject_ids[test_idx]))}"
        )

        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_total = 0

            for bx, by in train_loader:
                bx = bx.to(device, non_blocking=True)
                by = by.to(device, dtype=torch.float32, non_blocking=True)

                optimizer.zero_grad()
                pred = model(bx).squeeze(-1)
                loss = criterion(pred, by)
                loss.backward()
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()

                epoch_loss += float(loss.item()) * by.size(0)
                n_total += by.size(0)

            train_mse = epoch_loss / max(n_total, 1)
            val_mse, val_mae, val_rmse, val_r2 = evaluate_regression_loss(
                model,
                val_loader,
                device,
                criterion,
            )

            print(
                f"[{dataset_name} | {alg_name} | Fold {kf_iter} | Epoch {epoch}/{args.epochs}] "
                f"TrainMSE {train_mse:.4f} | ValMSE {val_mse:.4f} | "
                f"MAE {val_mae:.4f} | RMSE {val_rmse:.4f} | R2 {val_r2:.4f}"
            )

            early(val_mse, model)
            if early.early_stop:
                print("Early stopping triggered. Loading best model...")
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                break

        # If early stopping never triggers, evaluate the final epoch model.
        test_mse, test_mae, test_rmse, test_r2 = evaluate_regression_loss(
            model,
            test_loader,
            device,
            criterion,
        )
        fold_mae.append(float(test_mae))
        fold_rmse.append(float(test_rmse))
        fold_r2.append(float(test_r2))
        fold_mse.append(float(test_mse))

        print(
            f"[{dataset_name} | {alg_name} | Fold {kf_iter}] "
            f"TestMSE {test_mse:.4f} | TestMAE {test_mae:.4f} | "
            f"TestRMSE {test_rmse:.4f} | TestR2 {test_r2:.4f}"
        )

    elapsed = time.time() - t0

    out_csv = os.path.join(
        args.results_folder_root,
        dataset_name,
        f"[{timestamp_tag()}]{dataset_name}_{alg_name}_epochs{args.epochs}.csv",
    )
    df = save_metrics_csv(
        metrics={"MAE": fold_mae, "RMSE": fold_rmse, "R2": fold_r2, "MSE": fold_mse},
        n_splits=len(splits),
        out_csv=out_csv,
        elapsed=elapsed,
    )
    return df, out_csv


def run_tangent_space_ridge_age_regression(
    args,
    X,
    y,
    subject_ids,
    splits,
    dataset_name: str,
    ts_metric="riemann",
    alphas=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0),
):
    """Run Paper §2.5.1 Tangent-Space Ridge on one dataset."""
    alg_name = f"Ridge_AgeReg_TS_{ts_metric}"
    X = X.astype(np.float64)
    y = y.astype(np.float64)

    fold_mae, fold_rmse, fold_r2, fold_mse = [], [], [], []
    t0 = time.time()

    for kf_iter, (train_idx, test_idx) in enumerate(splits, start=1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        g_tr = subject_ids[train_idx]

        print(f"\n===== {dataset_name} | {alg_name} Fold {kf_iter}/{len(splits)} =====")
        print("Fitting TangentSpace ...")
        # Paper Methods 2.5.1: tangent reference is fit on the full outer
        # training split before inner Ridge alpha selection.
        ts = TangentSpace(metric=ts_metric, tsupdate=False)
        Z_tr = ts.fit_transform(X_tr)
        Z_te = ts.transform(X_te)

        model, best_alpha = fit_ridge_regressor(args, Z_tr, y_tr, g_tr, alphas)

        pred = model.predict(Z_te)

        mse = float(mean_squared_error(y_te, pred))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_te, pred))
        r2 = float(r2_score(y_te, pred))

        fold_mse.append(mse)
        fold_rmse.append(rmse)
        fold_mae.append(mae)
        fold_r2.append(r2)
        print(
            f"[{dataset_name} | {alg_name} | Fold {kf_iter}] alpha={best_alpha:g} | "
            f"MAE {mae:.4f} | RMSE {rmse:.4f} | R2 {r2:.4f}"
        )

    elapsed = time.time() - t0

    out_csv = os.path.join(
        args.results_folder_root,
        dataset_name,
        f"[{timestamp_tag()}]{dataset_name}_{alg_name}.csv",
    )
    df = save_metrics_csv(
        metrics={"MAE": fold_mae, "RMSE": fold_rmse, "R2": fold_r2, "MSE": fold_mse},
        n_splits=len(splits),
        out_csv=out_csv,
        elapsed=elapsed,
    )
    return df, out_csv


def run_corrvec_ridge_age_regression(
    args,
    X,
    y,
    subject_ids,
    splits,
    dataset_name: str,
    alphas=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0),
):
    """Run Paper §2.5.3 CorrVec Ridge on one dataset."""
    alg_name = "CorrVec_Ridge_AgeReg"
    X = X.astype(np.float64)
    y = y.astype(np.float64)

    fold_mae, fold_rmse, fold_r2, fold_mse = [], [], [], []
    t0 = time.time()

    for kf_iter, (train_idx, test_idx) in enumerate(splits, start=1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        g_tr = subject_ids[train_idx]

        print(f"\n===== {dataset_name} | {alg_name} Fold {kf_iter}/{len(splits)} =====")
        print("Vectorizing correlation matrices ...")
        # CorrVec uses the isometric off-diagonal upper-triangle convention.
        Z_tr = vectorize_correlation_matrices(X_tr, include_diagonal=False)
        Z_te = vectorize_correlation_matrices(X_te, include_diagonal=False)

        model, best_alpha = fit_ridge_regressor(args, Z_tr, y_tr, g_tr, alphas)
        pred = model.predict(Z_te)

        mse = float(mean_squared_error(y_te, pred))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_te, pred))
        r2 = float(r2_score(y_te, pred))

        fold_mse.append(mse)
        fold_rmse.append(rmse)
        fold_mae.append(mae)
        fold_r2.append(r2)
        print(
            f"[{dataset_name} | {alg_name} | Fold {kf_iter}] alpha={best_alpha:g} | "
            f"MAE {mae:.4f} | RMSE {rmse:.4f} | R2 {r2:.4f}"
        )

    elapsed = time.time() - t0

    out_csv = os.path.join(
        args.results_folder_root,
        dataset_name,
        f"[{timestamp_tag()}]{dataset_name}_{alg_name}.csv",
    )
    df = save_metrics_csv(
        metrics={"MAE": fold_mae, "RMSE": fold_rmse, "R2": fold_r2, "MSE": fold_mse},
        n_splits=len(splits),
        out_csv=out_csv,
        elapsed=elapsed,
    )
    return df, out_csv


# ---------------------------------------------------------------------
# Dummy baseline
# ---------------------------------------------------------------------
def run_dummy_age_baseline(args, y, splits, dataset_name: str, strategy="mean"):
    """Run the age-regression dummy baseline on one dataset."""
    alg_name = f"Dummy_AgeReg_{strategy}"
    y = y.astype(np.float64)

    fold_mae, fold_rmse, fold_r2, fold_mse = [], [], [], []
    t0 = time.time()

    for kf_iter, (train_idx, test_idx) in enumerate(splits, start=1):
        y_tr = y[train_idx]
        y_te = y[test_idx]

        model = DummyRegressor(strategy=strategy)
        model.fit(np.zeros((len(train_idx), 1)), y_tr)
        pred = model.predict(np.zeros((len(test_idx), 1)))

        mse = float(mean_squared_error(y_te, pred))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_te, pred))
        r2 = float(r2_score(y_te, pred))

        fold_mse.append(mse)
        fold_rmse.append(rmse)
        fold_mae.append(mae)
        fold_r2.append(r2)
        print(
            f"[{dataset_name} | {alg_name} | Fold {kf_iter}] "
            f"MAE {mae:.4f} | RMSE {rmse:.4f} | R2 {r2:.4f}"
        )

    elapsed = time.time() - t0

    out_csv = os.path.join(
        args.results_folder_root,
        dataset_name,
        f"[{timestamp_tag()}]{dataset_name}_{alg_name}.csv",
    )
    df = save_metrics_csv(
        metrics={"MAE": fold_mae, "RMSE": fold_rmse, "R2": fold_r2, "MSE": fold_mse},
        n_splits=len(splits),
        out_csv=out_csv,
        elapsed=elapsed,
    )
    return df, out_csv


# ---------------------------------------------------------------------
# Runner per dataset
# ---------------------------------------------------------------------
def run_dataset(args, dataset_name: str, atlas_name: str = "schaefer_100"):
    print("\n" + "=" * 80)
    print(f"Running dataset: {dataset_name}")
    print("=" * 80)

    subject_ids, ts, y, _ = load_age_timeseries(
        dataset_name,
        atlas_name,
        "Age",
        debug=None,
        rng=np.random.RandomState(args.seed_data_shuffle),
        data_root=Path(args.data_root),
    )

    n_samples = len(ts)
    n_subj = len(np.unique(subject_ids))
    print(f"Loaded N={n_samples} samples, S={n_subj} unique subjects.")

    X = estimate_connectome_matrices(ts, normalize=True, n_jobs=args.cov_jobs, eps=args.cov_eps)

    gkf = GroupKFold(n_splits=args.N_SPLITS)
    splits = list(gkf.split(X, y, groups=subject_ids))

    results = {}

    if args.run_spdnet:
        df_spd, spd_csv = run_spdnet_age_regression(
            args,
            X,
            y,
            subject_ids,
            splits,
            args.device,
            dataset_name,
        )
        results["SPDNet"] = spd_csv

    if args.run_ridge:
        df_ridge, ridge_csv = run_tangent_space_ridge_age_regression(
            args,
            X,
            y,
            subject_ids,
            splits,
            dataset_name,
            ts_metric=args.ts_metric,
            alphas=tuple(args.ridge_alphas),
        )
        results["TangentSpaceRidge"] = ridge_csv

    if args.run_vec_corr_ridge:
        df_vec, vec_csv = run_corrvec_ridge_age_regression(
            args,
            X,
            y,
            subject_ids,
            splits,
            dataset_name,
            alphas=tuple(args.ridge_alphas),
        )
        results["CorrVecRidge"] = vec_csv

    if args.run_dummy:
        df_dummy, dummy_csv = run_dummy_age_baseline(
            args,
            y,
            splits,
            dataset_name,
            strategy=args.dummy_strategy,
        )
        results["Dummy"] = dummy_csv

    return results


def args_parser(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Run Paper §2.7 within-dataset GroupKFold age regression. "
            "Experiment defaults preserve the original single-dataset script."
        )
    )

    add_device_arg(p)
    add_logging_args(p)
    add_data_root_arg(p)

    p.add_argument(
        "--datasets",
        type=str,
        default="camcan",
        help=(
            "Comma-separated datasets. Original default is camcan; paper "
            f"uses {', '.join(PAPER_DATASETS)}."
        ),
    )
    p.add_argument(
        "--atlas",
        type=str,
        default="schaefer_100",
        help="Atlas name. Paper uses Schaefer-100.",
    )

    add_split_args(p)
    p.add_argument("--seed_data_shuffle", type=int, default=42)

    add_connectome_args(p)

    p.add_argument("--weights_folder_root", type=str, default=str(DEFAULT_SINGLE_WEIGHTS_DIR))
    p.add_argument("--results_folder_root", type=str, default=str(DEFAULT_SINGLE_RESULTS_DIR))

    p.add_argument("--run_spdnet", action="store_true", default=True)
    add_spdnet_optimization_args(
        p,
        train_batch_size=100,
        test_batch_size=100,
        train_batch_help="Original single-dataset default; paper pooled setting uses 1024.",
        test_batch_help="Original single-dataset default; paper pooled setting uses 1024.",
    )
    add_spdnet_head_args(p)

    p.add_argument("--run_ridge", action="store_true", default=False)
    p.add_argument("--ts_metric", type=str, default="riemann", choices=["logeuclid", "riemann"])
    add_ridge_args(p, include_n_jobs=True)
    p.add_argument("--run_vec_corr_ridge", action="store_true", default=False)
    p.add_argument("--only_vec_corr_ridge", action="store_true", default=False)


    p.add_argument("--run_dummy", action="store_true", default=False)
    p.add_argument("--dummy_strategy", type=str, default="mean", choices=["mean", "median"])

    # Defaults preserve the original code; CLI overrides make paper-like runs
    # possible without editing constants.
    args = p.parse_args(argv)

    if args.only_vec_corr_ridge:
        args.run_spdnet = False
        args.run_ridge = False
        args.run_vec_corr_ridge = True
        args.run_dummy = False

    finalize_single_dataset_runtime(args)

    return args


def main(argv=None) -> int:
    args = args_parser(argv)
    configure_logging(args.log_level)
    print("Using device:", args.device)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    all_results = {}
    had_failures = False

    for ds in datasets:
        try:
            res = run_dataset(args, ds, atlas_name=args.atlas)
            all_results[ds] = res
        except Exception as e:
            print(f"[ERROR] Dataset {ds} failed: {e}")
            all_results[ds] = {"error": type(e).__name__}
            had_failures = True

    summary_rows = []
    for ds, res in all_results.items():
        row = {"Dataset": ds}
        row.update(
            {
                name: (
                    value
                    if name == "error"
                    else portable_result_reference(
                        value,
                        output_root=args.results_folder_root,
                    )
                )
                for name, value in res.items()
            }
        )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.results_folder_root, f"[{timestamp_tag()}]SUMMARY_paths.csv")
    Path(args.results_folder_root).mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print("\nSaved summary path table:", summary_path)
    print(summary_df)
    return 1 if had_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
