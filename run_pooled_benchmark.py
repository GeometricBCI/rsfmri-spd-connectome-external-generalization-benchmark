"""Run the pooled age-regression benchmark with paper-alignment notes.

This entry point covers Paper §2.7 pooled GroupKFold and LODO experiments,
including the §2.7.1 split-local harmonization protocol. Original experiment
defaults are preserved; output paths use the cleaned ``results/`` layout.
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
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor

from pyriemann.tangentspace import TangentSpace

from spd_connectome_benchmark.config import (
    DEFAULT_POOLED_DATASETS,
    DEFAULT_POOLED_RESULTS_DIR,
    DEFAULT_POOLED_WEIGHTS_DIR,
)
from spd_connectome_benchmark.connectomes import (
    estimate_connectome_matrices,
    vectorize_correlation_upper,
)
from spd_connectome_benchmark.models.spd import SPDNetRegressor
from spd_connectome_benchmark.benchmark_tools.runtime import (
    EarlyStopping,
    MatrixRegressionDataset,
    append_fold_metrics,
    compute_age_regression_metrics,
    evaluate_regression_model,
    format_metrics_for_log,
    init_fold_metrics,
    load_pooled_age_timeseries,
    make_groupkfold_splits,
    make_lodo_splits,
    save_protocol_metrics_csv,
    select_metrics_for_protocol,
    split_train_validation_by_group,
)
from spd_connectome_benchmark.benchmark_tools.harmonization import (
    load_or_harmonize_features,
    load_or_harmonize_spd_matrices,
)
from spd_connectome_benchmark.benchmark_tools.cli import (
    add_common_data_args,
    add_connectome_args,
    add_device_arg,
    add_logging_args,
    add_pooled_dataset_args,
    add_ridge_args,
    add_split_args,
    add_spdnet_head_args,
    add_spdnet_optimization_args,
    resolve_torch_device,
)
from spd_connectome_benchmark.benchmark_tools.logging import configure_logging


def train_spdnet_fold(
    args,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    kf_iter: int,
    device: torch.device,
    protocol_tag: str,
):
    """Train and evaluate one pooled SPDNet fold from Paper §2.5.2/§2.7."""
    alg_name = "SPDNet_AgeReg"

    # Paper Methods 2.5.2/2.7: reserve 10% of outer-train subject groups.
    tr_idx, va_idx = split_train_validation_by_group(
        train_idx,
        groups=groups,
        val_size=args.val_size,
        seed=args.seed,
    )

    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    train_dataset = MatrixRegressionDataset(
        torch.from_numpy(X_tr).float(),
        torch.from_numpy(y_tr).float(),
    )
    val_dataset = MatrixRegressionDataset(
        torch.from_numpy(X_va).float(),
        torch.from_numpy(y_va).float(),
    )
    test_dataset = MatrixRegressionDataset(
        torch.from_numpy(X_te).float(),
        torch.from_numpy(y_te).float(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        pin_memory=True,
        # Source behavior: incomplete final training batch is dropped. The
        # paper states the batch size but does not spell out drop_last.
        drop_last=True,
    )
    # Paper limitation note: training order is random at the run level, without
    # a separate fold-specific DataLoader generator seed.
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        pin_memory=True,
    )

    # Main SPDNet: one full-rank BiMap(P->P)+ReEig block, then LogEig+MLP.
    model = SPDNetRegressor(
        dims=(args.P, args.P),
        out_dim=1,
        fc_layer_no=args.fc_layer_no,
        fc_hidden_dim=args.fc_hidden_dim,
        fc_dropout=args.fc_dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    weights_folder = Path(args.weights_folder_path)
    weights_folder.mkdir(parents=True, exist_ok=True)
    safe_protocol = protocol_tag.replace("/", "_")
    ckpt_path = weights_folder / f"{alg_name}_{safe_protocol}_fold{kf_iter}.pt"

    early_stopper = EarlyStopping(
        path_w=ckpt_path,
        patience=args.patience,
        verbose=True,
        delta=0.0,
    )

    print(f"\n===== SPDNet Fold {kf_iter}/{args.N_SPLITS} =====")
    print(
        f"Train groups: {len(np.unique(groups[tr_idx]))} | "
        f"Val groups: {len(np.unique(groups[va_idx]))} | "
        f"Test groups: {len(np.unique(groups[test_idx]))}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_total = 0

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            t_batch = time.time()
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, dtype=torch.float32, non_blocking=True)

            optimizer.zero_grad()
            pred = model(batch_x).squeeze(-1)
            loss = criterion(pred, batch_y)
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            epoch_loss += float(loss.item()) * batch_y.size(0)
            n_total += batch_y.size(0)

            if batch_idx % 1 == 0:
                print(
                    f"Epoch {epoch} Batch {batch_idx}/{len(train_loader)} "
                    f"loss={loss.item():.6f} time={time.time() - t_batch:.2f}s"
                )

        train_loss = epoch_loss / max(n_total, 1)
        val_loss, val_mae, val_rmse, val_r2, _, _ = evaluate_regression_model(
            model,
            val_loader,
            device,
            criterion,
        )

        print(
            f"[{alg_name} | Fold {kf_iter} | Epoch {epoch}/{args.epochs}] "
            f"TrainMSE {train_loss:.6f} | ValMSE {val_loss:.6f} | "
            f"MAE {val_mae:.4f} | RMSE {val_rmse:.4f} | R2 {val_r2:.4f}"
        )

        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered! Loading best model...")
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            break

    # If early stopping never triggers, evaluate the final epoch model, as
    # stated in Methods 2.5.2.
    test_loss, test_mae, test_rmse, test_r2, test_pred, test_y = (
        evaluate_regression_model(
            model,
            test_loader,
            device,
            criterion,
        )
    )
    test_metrics = compute_age_regression_metrics(test_y, test_pred)
    print(
        f"[{alg_name} | Fold {kf_iter}] "
        f"TestMSE {test_loss:.6f} | TestMAE {test_mae:.4f} | "
        f"TestRMSE {test_rmse:.4f} | TestR2 {test_r2:.4f}"
    )
    selected_metrics = select_metrics_for_protocol(test_metrics, protocol_tag)
    print(f"[{alg_name} | Fold {kf_iter}] {format_metrics_for_log(selected_metrics)}")

    return selected_metrics, alg_name



def run_pooled_age_benchmarks(
    args,
    datasets=DEFAULT_POOLED_DATASETS,
    atlas_name="schaefer_100",
    task="Age",
    debug=None,
    rng_seed=42,
    ts_metric="riemann",
    ridge_alphas=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0),
    dummy_strategy="mean",
    make_tag=True,
    algorithms=("spdnet", "ridge", "dummy"),
):
    def _run_spdnet(
        protocol_tag,
        splits,
        fold_names,
        X,
        y,
        subject_ids,
        dataset_ids,
        harm_mode,
        apply_harm_to_test,
    ):
        def _run_one(label):
            fold_metrics = init_fold_metrics(protocol_tag)
            t0 = time.time()

            print(f"[{tag}_SPDNet_{label}_{protocol_tag}] Starting folds...")
            for kf_iter, (train_idx, test_idx) in enumerate(splits, start=1):
                X_use = X
                if label == "harm":
                    fold_tag = f"{tag}_SPDNet_{protocol_tag}_fold{kf_iter}"
                    save_dir = os.path.join(args.results_folder, "harmonized_spd")
                    X_use = load_or_harmonize_spd_matrices(
                        X=X,
                        y=y,
                        dataset_ids=dataset_ids,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        apply_harm_to_test=apply_harm_to_test,
                        ts_metric=ts_metric,
                        save_dir=save_dir,
                        fold_tag=fold_tag,
                    )
                metrics, alg_name = train_spdnet_fold(
                    args=args,
                    X=X_use,
                    y=y,
                    groups=subject_ids,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    kf_iter=kf_iter,
                    device=device,
                    protocol_tag=protocol_tag,
                )
                append_fold_metrics(fold_metrics, metrics)

            elapsed = time.time() - t0

            suffix = "_harm" if label == "harm" else ""
            csv_name = (
                f"{timestamp}{tag}_SPDNet_AgeReg_epochs{args.epochs}_"
                f"{protocol_tag}{suffix}.csv"
            )
            csv_path = os.path.join(args.results_folder, csv_name)
            return save_protocol_metrics_csv(fold_metrics, fold_names, elapsed, csv_path)

        if harm_mode in ("none", "both"):
            _run_one("noharm")
        if harm_mode in ("harm", "both"):
            _run_one("harm")
        return None

    def _run_tangent_space_ridge(
        protocol_tag,
        splits,
        fold_names,
        X,
        y,
        subject_ids,
        dataset_ids,
        harm_mode,
        apply_harm_to_test,
    ):
        fold_metrics = init_fold_metrics(protocol_tag)
        fold_metrics_h = init_fold_metrics(protocol_tag)
        t0 = time.time()

        X64 = X.astype(np.float32)
        y64 = y.astype(np.float32)

        for kf_iter, (train_idx, test_idx) in enumerate(splits, start=1):
            X_tr, y_tr = X64[train_idx], y64[train_idx]
            X_te, y_te = X64[test_idx], y64[test_idx]
            g_tr = subject_ids[train_idx]

            # Paper Methods 2.5.1: fit the tangent reference on the full outer
            # training split, then tune Ridge alpha on these fixed features.
            ts = TangentSpace(metric=ts_metric, tsupdate=False)
            Z_tr = ts.fit_transform(X_tr)
            Z_te = ts.transform(X_te)

            cov_train = np.stack([dataset_ids[train_idx], y_tr], axis=1)
            cov_test = np.stack([dataset_ids[test_idx], y_te], axis=1)

            if harm_mode in ("harm", "both"):
                fold_tag = f"{tag}_Ridge_TS_{ts_metric}_{protocol_tag}_fold{kf_iter}"
                save_dir = os.path.join(args.results_folder, "harmonized_features")
                Z_tr_h, Z_te_h = load_or_harmonize_features(
                    Z_tr,
                    Z_te,
                    cov_train,
                    cov_test,
                    apply_harm_to_test=apply_harm_to_test,
                    save_dir=save_dir,
                    fold_tag=fold_tag,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    feature_kind="ridge_tangent",
                    feature_metric=ts_metric,
                )

            # This is intentionally not fully nested for the tangent reference:
            # only Ridge alpha is selected inside the group-aware inner CV.
            n_groups_tr = len(np.unique(g_tr))
            inner_k = min(args.ridge_inner_splits, n_groups_tr)

            pipe = Pipeline([("ridge", Ridge(random_state=args.seed))])

            def _fit_predict(Z_train, Z_test, label):
                if inner_k < 2:
                    best_alpha = float(ridge_alphas[0])
                    pipe.set_params(ridge__alpha=best_alpha)
                    pipe.fit(Z_train, y_tr)
                    model = pipe
                else:
                    inner_cv = GroupKFold(n_splits=inner_k)
                    gs = GridSearchCV(
                        estimator=pipe,
                        param_grid={"ridge__alpha": list(ridge_alphas)},
                        cv=inner_cv,
                        scoring="neg_mean_absolute_error",
                        n_jobs=-1,
                    )
                    gs.fit(Z_train, y_tr, groups=g_tr)
                    model = gs.best_estimator_
                    best_alpha = float(gs.best_params_["ridge__alpha"])

                pred = model.predict(Z_test)
                metrics = compute_age_regression_metrics(y_te, pred)
                selected_metrics = select_metrics_for_protocol(metrics, protocol_tag)
                print(
                    f"[{tag}_Ridge_TS_{ts_metric}_{label}_{protocol_tag} | "
                    f"Fold {kf_iter}/{len(splits)}]"
                    f" alpha={best_alpha:g} | {format_metrics_for_log(selected_metrics)}"
                )
                return selected_metrics

            if harm_mode in ("none", "both"):
                metrics = _fit_predict(Z_tr, Z_te, "noharm")
                append_fold_metrics(fold_metrics, metrics)

            if harm_mode in ("harm", "both"):
                metrics_h = _fit_predict(Z_tr_h, Z_te_h, "harm")
                append_fold_metrics(fold_metrics_h, metrics_h)

        elapsed = time.time() - t0

        if harm_mode in ("none", "both"):
            csv_name = f"{timestamp}{tag}_Ridge_AgeReg_TS_{ts_metric}_{protocol_tag}.csv"
            csv_path = os.path.join(args.results_folder, csv_name)
            save_protocol_metrics_csv(fold_metrics, fold_names, elapsed, csv_path)
        if harm_mode in ("harm", "both"):
            csv_name = f"{timestamp}{tag}_Ridge_AgeReg_TS_{ts_metric}_{protocol_tag}_harm.csv"
            csv_path = os.path.join(args.results_folder, csv_name)
            save_protocol_metrics_csv(fold_metrics_h, fold_names, elapsed, csv_path)
        return None

    def _run_corrvec_ridge(
        protocol_tag,
        splits,
        fold_names,
        X,
        y,
        subject_ids,
        dataset_ids,
        harm_mode,
        apply_harm_to_test,
    ):
        fold_metrics = init_fold_metrics(protocol_tag)
        fold_metrics_h = init_fold_metrics(protocol_tag)
        t0 = time.time()

        X64 = X.astype(np.float32)
        y64 = y.astype(np.float32)

        for kf_iter, (train_idx, test_idx) in enumerate(splits, start=1):
            X_tr, y_tr = X64[train_idx], y64[train_idx]
            X_te, y_te = X64[test_idx], y64[test_idx]
            g_tr = subject_ids[train_idx]

            # Paper Methods 2.5.3: CorrVec uses off-diagonal upper-triangle
            # correlations and no sqrt(2) tangent weighting.
            Z_tr = vectorize_correlation_upper(X_tr)
            Z_te = vectorize_correlation_upper(X_te)

            cov_train = np.stack([dataset_ids[train_idx], y_tr], axis=1)
            cov_test = np.stack([dataset_ids[test_idx], y_te], axis=1)

            if harm_mode in ("harm", "both"):
                fold_tag = f"{tag}_CorrVec_Ridge_{protocol_tag}_fold{kf_iter}"
                save_dir = os.path.join(args.results_folder, "harmonized_features")
                Z_tr_h, Z_te_h = load_or_harmonize_features(
                    Z_tr,
                    Z_te,
                    cov_train,
                    cov_test,
                    apply_harm_to_test=apply_harm_to_test,
                    save_dir=save_dir,
                    fold_tag=fold_tag,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    feature_kind="corr_upper",
                    feature_metric="upper_triangle",
                )

            n_groups_tr = len(np.unique(g_tr))
            inner_k = min(args.ridge_inner_splits, n_groups_tr)

            pipe = Pipeline([("ridge", Ridge(random_state=args.seed))])

            def _fit_predict(Z_train, Z_test, label):
                if inner_k < 2:
                    best_alpha = float(ridge_alphas[0])
                    pipe.set_params(ridge__alpha=best_alpha)
                    pipe.fit(Z_train, y_tr)
                    model = pipe
                else:
                    inner_cv = GroupKFold(n_splits=inner_k)
                    gs = GridSearchCV(
                        estimator=pipe,
                        param_grid={"ridge__alpha": list(ridge_alphas)},
                        cv=inner_cv,
                        scoring="neg_mean_absolute_error",
                        n_jobs=-1,
                    )
                    gs.fit(Z_train, y_tr, groups=g_tr)
                    model = gs.best_estimator_
                    best_alpha = float(gs.best_params_["ridge__alpha"])

                pred = model.predict(Z_test)
                metrics = compute_age_regression_metrics(y_te, pred)
                selected_metrics = select_metrics_for_protocol(metrics, protocol_tag)
                print(
                    f"[{tag}_CorrVec_Ridge_{label}_{protocol_tag} | Fold {kf_iter}/{len(splits)}]"
                    f" alpha={best_alpha:g} | {format_metrics_for_log(selected_metrics)}"
                )
                return selected_metrics

            if harm_mode in ("none", "both"):
                metrics = _fit_predict(Z_tr, Z_te, "noharm")
                append_fold_metrics(fold_metrics, metrics)

            if harm_mode in ("harm", "both"):
                metrics_h = _fit_predict(Z_tr_h, Z_te_h, "harm")
                append_fold_metrics(fold_metrics_h, metrics_h)

        elapsed = time.time() - t0

        if harm_mode in ("none", "both"):
            csv_name = f"{timestamp}{tag}_CorrVec_Ridge_AgeReg_{protocol_tag}.csv"
            csv_path = os.path.join(args.results_folder, csv_name)
            save_protocol_metrics_csv(fold_metrics, fold_names, elapsed, csv_path)
        if harm_mode in ("harm", "both"):
            csv_name = f"{timestamp}{tag}_CorrVec_Ridge_AgeReg_{protocol_tag}_harm.csv"
            csv_path = os.path.join(args.results_folder, csv_name)
            save_protocol_metrics_csv(fold_metrics_h, fold_names, elapsed, csv_path)
        return None

    def _run_dummy(protocol_tag, splits, fold_names, y, harm_mode):
        def _run_one(label):
            fold_metrics = init_fold_metrics(protocol_tag)
            t0 = time.time()

            y64 = y.astype(np.float32)

            for kf_iter, (train_idx, test_idx) in enumerate(splits, start=1):
                y_tr = y64[train_idx]
                y_te = y64[test_idx]

                model = DummyRegressor(strategy=dummy_strategy)
                model.fit(np.zeros((len(train_idx), 1)), y_tr)
                pred = model.predict(np.zeros((len(test_idx), 1)))

                metrics = compute_age_regression_metrics(y_te, pred)
                selected_metrics = select_metrics_for_protocol(metrics, protocol_tag)
                append_fold_metrics(fold_metrics, selected_metrics)

                print(
                    f"[{tag}_Dummy_{dummy_strategy}_{label}_{protocol_tag} | "
                    f"Fold {kf_iter}/{len(splits)}] "
                    f"{format_metrics_for_log(selected_metrics)}"
                )

            elapsed = time.time() - t0

            suffix = "_harm" if label == "harm" else ""
            csv_name = f"{timestamp}{tag}_Dummy_AgeReg_{dummy_strategy}_{protocol_tag}{suffix}.csv"
            csv_path = os.path.join(args.results_folder, csv_name)
            return save_protocol_metrics_csv(fold_metrics, fold_names, elapsed, csv_path)

        if harm_mode in ("none", "both"):
            _run_one("noharm")
        if harm_mode in ("harm", "both"):
            _run_one("harm")
        return None


    os.makedirs(args.results_folder, exist_ok=True)
    os.makedirs(args.weights_folder_path, exist_ok=True)

    # Original behavior: do not add an extra fold-specific DataLoader seed here.
    rng = np.random.RandomState(rng_seed)

    device = resolve_torch_device(args.no_cuda)
    print("Using device:", device)

    subject_ids, dataset_ids, ts, y = load_pooled_age_timeseries(
        datasets=datasets,
        atlas_name=atlas_name,
        task=task,
        debug=debug,
        rng_seed=rng_seed,
        data_root=Path(args.data_root),
    )

    print(f"\n[POOLED] N samples: {len(ts)}")
    print(f"[POOLED] Unique subject groups: {len(np.unique(subject_ids))}")
    print(f"[POOLED] Datasets: {list(pd.unique(dataset_ids))}")

    X = estimate_connectome_matrices(ts, normalize=True, n_jobs=args.cov_jobs, eps=args.cov_eps)
    args.P = X.shape[1]
    assert X.shape[1] == X.shape[2], "X must be (N,P,P)"

    if make_tag:
        tag = "ALL-" + "-".join(list(pd.unique(dataset_ids)))
    else:
        tag = "ALL"

    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())
    algos = {a.lower() for a in algorithms}

    protocol_choice = args.protocol.lower()

    # ---------------------------
    # Protocol 1: GroupKFold
    # ---------------------------
    if protocol_choice in ("gkf", "both"):
        protocol1 = f"GKF{args.N_SPLITS}"
        splits_gkf, fold_names_gkf = make_groupkfold_splits(X, y, subject_ids, args.N_SPLITS)

        print(f"\n==================== {protocol1} ====================")
        if "spdnet" in algos:
            _ = _run_spdnet(
                protocol1,
                splits_gkf,
                fold_names_gkf,
                X,
                y,
                subject_ids,
                dataset_ids,
                args.harm_mode,
                apply_harm_to_test=True,
            )
        if "ridge" in algos:
            _ = _run_tangent_space_ridge(
                protocol1,
                splits_gkf,
                fold_names_gkf,
                X,
                y,
                subject_ids,
                dataset_ids,
                args.harm_mode,
                apply_harm_to_test=True,
            )
        if "corr_ridge" in algos:
            _ = _run_corrvec_ridge(
                protocol1,
                splits_gkf,
                fold_names_gkf,
                X,
                y,
                subject_ids,
                dataset_ids,
                args.harm_mode,
                apply_harm_to_test=True,
            )
        if "dummy" in algos:
            _ = _run_dummy(protocol1, splits_gkf, fold_names_gkf, y, args.harm_mode)

    # ---------------------------
    # Protocol 2: LODO
    # ---------------------------
    if protocol_choice in ("lodo", "both"):
        protocol2 = "LODO"
        splits_lodo, fold_names_lodo = make_lodo_splits(dataset_ids)

        print(f"\n==================== {protocol2} ====================")
        if "spdnet" in algos:
            _ = _run_spdnet(
                protocol2,
                splits_lodo,
                fold_names_lodo,
                X,
                y,
                subject_ids,
                dataset_ids,
                args.harm_mode,
                apply_harm_to_test=False,
            )
        if "ridge" in algos:
            _ = _run_tangent_space_ridge(
                protocol2,
                splits_lodo,
                fold_names_lodo,
                X,
                y,
                subject_ids,
                dataset_ids,
                args.harm_mode,
                apply_harm_to_test=False,
            )
        if "corr_ridge" in algos:
            _ = _run_corrvec_ridge(
                protocol2,
                splits_lodo,
                fold_names_lodo,
                X,
                y,
                subject_ids,
                dataset_ids,
                args.harm_mode,
                apply_harm_to_test=False,
            )
        if "dummy" in algos:
            _ = _run_dummy(protocol2, splits_lodo, fold_names_lodo, y, args.harm_mode)

    print("\n############ ALL DONE ############")


def args_parser(argv=None):

    parser = argparse.ArgumentParser(
        description=(
            "Run Paper §2.7 pooled GroupKFold/LODO age regression. "
            "Experiment defaults preserve the original source code; use "
            "--algorithms spdnet ridge corr_ridge dummy for the full paper "
            "model set."
        )
    )

    add_device_arg(parser)
    add_logging_args(parser)
    add_spdnet_optimization_args(parser, train_batch_size=1024, test_batch_size=1024)
    add_pooled_dataset_args(parser)
    add_split_args(parser)

    parser.add_argument("--weights_folder_path", type=str, default=str(DEFAULT_POOLED_WEIGHTS_DIR))
    parser.add_argument("--results_folder", type=str, default=str(DEFAULT_POOLED_RESULTS_DIR))

    add_spdnet_head_args(parser)
    add_connectome_args(parser)

    add_ridge_args(parser, include_n_jobs=False)

    add_common_data_args(parser, atlas_arg="--atlas_name")
    parser.add_argument("--ts_metric", type=str, default="riemann")
    parser.add_argument("--dummy_strategy", type=str, default="mean")
    parser.add_argument("--no_make_tag", action="store_true", default=False)
    parser.add_argument(
        "--harm_mode",
        type=str,
        choices=["none", "harm", "both"],
        default="none",
        help="Harmonization for ridge-style vector features: none, harm, or both.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["spdnet", "ridge", "dummy"],
        choices=["spdnet", "ridge", "corr_ridge", "dummy"],
        help=(
            "Algorithms to run. corr_ridge is available for paper matching "
            "but is not an original default."
        ),
    )
    parser.add_argument(
        "--protocol",
        type=str,
        choices=["gkf", "lodo", "both"],
        default="both",
        help="Cross-validation protocol: gkf, lodo, or both.",
    )

    args = parser.parse_args(argv)

    return args


def main():
    args = args_parser()
    configure_logging(args.log_level)

    print("############ Start Age Regression (POOLED DATASETS) ############")
    print("Datasets:", args.DATASETS)

    run_pooled_age_benchmarks(
        args=args,
        datasets=tuple(args.DATASETS),
        atlas_name=args.atlas_name,
        task=args.task,
        debug=args.debug,
        rng_seed=args.rng_seed,
        ts_metric=args.ts_metric,
        ridge_alphas=tuple(args.ridge_alphas),
        dummy_strategy=args.dummy_strategy,
        make_tag=not args.no_make_tag,
        algorithms=tuple(args.algorithms),
    )


if __name__ == "__main__":
    main()
