"""Run the paper's pooled-dataset SPDNet configuration ablation.

This script reuses the same pooled-dataset setup as
``run_pooled_benchmark.py``
and evaluates the four Figure 5 variants:
    - quarterdim: dims=(P, P//4)
    - halfdim:    dims=(P, P//2)
    - one:        dims=(P, P)
    - two:        dims=(P, P, P, P)

Legacy CLI names ``base`` and ``2layer`` are accepted as aliases for ``one``
and ``two`` so older result folders remain readable.
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

from pyriemann.tangentspace import TangentSpace

from spd_connectome_benchmark.config import (
    DEFAULT_ABLATION_RESULTS_DIR,
    DEFAULT_ABLATION_WEIGHTS_DIR,
    DEFAULT_POOLED_DATASETS,
)
from spd_connectome_benchmark.connectomes import estimate_connectome_matrices
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
from spd_connectome_benchmark.benchmark_tools.cli import (
    add_common_data_args,
    add_connectome_args,
    add_device_arg,
    add_logging_args,
    add_pooled_dataset_args,
    add_split_args,
    add_spdnet_head_args,
    add_spdnet_optimization_args,
    resolve_torch_device,
)
from spd_connectome_benchmark.benchmark_tools.harmonization import harmonize_tangent_features
from spd_connectome_benchmark.benchmark_tools.logging import configure_logging


PAPER_SPDNET_VARIANTS = ("quarterdim", "halfdim", "one", "two")
ORIGINAL_SPDNET_VARIANTS = ("quarterdim", "halfdim", "base", "2layer")
SPDNET_VARIANT_ALIASES = {
    "quarterdim": "quarterdim",
    "halfdim": "halfdim",
    "one": "one",
    "base": "one",
    "two": "two",
    "2layer": "two",
}
LEGACY_VARIANT_ALIASES = {
    "quarterdim": ("quarterdim",),
    "halfdim": ("halfdim",),
    "one": ("one", "base"),
    "two": ("two", "2layer"),
}


def canonical_spdnet_variant(variant_name: str) -> str:
    """Return the paper label for an SPDNet ablation variant."""
    key = variant_name.lower()
    if key not in SPDNET_VARIANT_ALIASES:
        raise ValueError(f"Unknown SPDNet variant: {variant_name}")
    return SPDNET_VARIANT_ALIASES[key]


def train_spdnet_ablation_fold(
    args,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    kf_iter: int,
    device: torch.device,
    protocol_tag: str,
    spd_dims: tuple[int, ...],
    variant_tag: str,
):
    """Train and evaluate one fold for a Figure 5 SPDNet ablation variant."""
    alg_name = f"SPDNet_AgeReg_{variant_tag}"

    # Paper Methods 2.5.2/2.7: group-aware 10% validation split.
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
    # Training batches are shuffled under the run-level RNG, without a
    # fold-specific generator seed, matching the limitation stated in the paper.
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

    model = SPDNetRegressor(
        dims=spd_dims,
        out_dim=1,
        fc_layer_no=args.fc_layer_no,
        fc_hidden_dim=args.fc_hidden_dim,
        fc_dropout=args.fc_dropout,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.initial_lr,
        weight_decay=args.weight_decay,
    )
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

    print(f"\n===== {alg_name} Fold {kf_iter}/{args.N_SPLITS} =====")
    print(
        f"dims={spd_dims} | Train groups: {len(np.unique(groups[tr_idx]))} | "
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

        val_loss, _, _, _, val_preds, val_ys = evaluate_regression_model(
            model,
            val_loader,
            device,
            criterion,
        )
        val_metrics = compute_age_regression_metrics(val_ys, val_preds)

        print(
            f"[{alg_name} | Fold {kf_iter} | Epoch {epoch}/{args.epochs}] "
            f"TrainMSE {train_loss:.6f} | ValMSE {val_loss:.6f} | "
            f"MAE {val_metrics['MAE']:.4f} | "
            f"RMSE {val_metrics['RMSE']:.4f} | R2 {val_metrics['R2']:.4f}"
        )

        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered! Loading best model...")
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            break

    # If early stopping never triggers, evaluate the final epoch checkpoint.
    test_loss, _, _, _, test_pred, test_y = evaluate_regression_model(
        model,
        test_loader,
        device,
        criterion,
    )

    test_metrics = compute_age_regression_metrics(test_y, test_pred)
    selected_metrics = select_metrics_for_protocol(test_metrics, protocol_tag)

    print(
        f"[{alg_name} | Fold {kf_iter}] "
        f"TestMSE {test_loss:.6f} | {format_metrics_for_log(selected_metrics)}"
    )

    return selected_metrics


def run_pooled_spdnet_ablation_benchmarks(
    args,
    datasets=DEFAULT_POOLED_DATASETS,
    atlas_name="schaefer_100",
    task="Age",
    debug=None,
    rng_seed=42,
):
    def _harmonize_spd_for_fold(
        X,
        y,
        dataset_ids,
        train_idx,
        test_idx,
        apply_harm_to_test: bool,
        ts_metric: str,
        save_dir: str,
        fold_tag: str,
        legacy_fold_tags=(),
        cache_dirs=(),
    ):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = os.path.join(save_dir, f"{fold_tag}_harmonized_spd.npz")
        search_dirs = []
        for cache_dir in (save_dir, *cache_dirs):
            if cache_dir and cache_dir not in search_dirs:
                search_dirs.append(cache_dir)

        def _candidate_paths(tag):
            filename = f"{tag}_harmonized_spd.npz"
            suffix = f"_{tag.split('_SPDNet_', 1)[1]}_harmonized_spd.npz"
            candidates = []
            for cache_dir in search_dirs:
                cache_path = Path(cache_dir)
                exact_path = cache_path / filename
                if exact_path.exists():
                    candidates.append(exact_path)
                if cache_path.exists():
                    candidates.extend(sorted(cache_path.glob(f"*{suffix}")))
            unique_candidates = []
            seen = set()
            for path in candidates:
                resolved = str(path)
                if resolved in seen:
                    continue
                seen.add(resolved)
                unique_candidates.append(path)
            return unique_candidates

        def _load_cached(cache_path):
            cached = np.load(cache_path)
            cached_train_idx = cached["train_idx"]
            cached_test_idx = cached["test_idx"]
            if not np.array_equal(cached_train_idx, train_idx) or not np.array_equal(
                cached_test_idx,
                test_idx,
            ):
                print("Skip mismatched harmonized SPD cache:", cache_path)
                return None

            X_h = X.copy()
            X_h[cached_train_idx] = cached["X_tr"]
            X_h[cached_test_idx] = cached["X_te"]
            print("Loaded harmonized SPD cache:", cache_path)
            return X_h

        for candidate_path in _candidate_paths(fold_tag):
            X_cached = _load_cached(candidate_path)
            if X_cached is not None:
                if Path(candidate_path) != Path(save_path):
                    cached = np.load(candidate_path)
                    np.savez_compressed(
                        save_path,
                        X_tr=cached["X_tr"],
                        X_te=cached["X_te"],
                        train_idx=cached["train_idx"],
                        test_idx=cached["test_idx"],
                        reference=cached["reference"],
                    )
                    print("Copied harmonized SPD cache to local shared cache:", save_path)
                return X_cached

        for legacy_fold_tag in legacy_fold_tags:
            for legacy_path in _candidate_paths(legacy_fold_tag):
                X_cached = _load_cached(legacy_path)
                if X_cached is None:
                    continue
                cached = np.load(legacy_path)
                np.savez_compressed(
                    save_path,
                    X_tr=cached["X_tr"],
                    X_te=cached["X_te"],
                    train_idx=cached["train_idx"],
                    test_idx=cached["test_idx"],
                    reference=cached["reference"],
                )
                print("Copied legacy harmonized SPD cache to shared cache:", save_path)
                return X_cached

        X_tr = X[train_idx]
        X_te = X[test_idx]
        y_tr = y[train_idx]
        y_te = y[test_idx]

        # Paper Methods 2.6: tangent reference is estimated on outer-train only.
        ts = TangentSpace(metric=ts_metric, tsupdate=False)
        Z_tr = ts.fit_transform(X_tr)
        Z_te = ts.transform(X_te)

        cov_train = np.stack([dataset_ids[train_idx], y_tr], axis=1)
        cov_test = np.stack([dataset_ids[test_idx], y_te], axis=1)

        Z_tr_h, Z_te_h = harmonize_tangent_features(
            Z_tr,
            Z_te,
            cov_train,
            cov_test,
            apply_harm_to_test=apply_harm_to_test,
        )

        X_tr_h = ts.inverse_transform(Z_tr_h)
        X_te_h = ts.inverse_transform(Z_te_h) if apply_harm_to_test else X_te

        X_h = X.copy()
        X_h[train_idx] = X_tr_h
        X_h[test_idx] = X_te_h

        np.savez_compressed(
            save_path,
            X_tr=X_tr_h,
            X_te=X_te_h,
            train_idx=train_idx,
            test_idx=test_idx,
            reference=ts.reference_,
        )
        print("Saved harmonized SPD matrices:", save_path)
        return X_h

    def _load_harmonized_spd_cache(X, train_idx, test_idx, cache_path):
        cached = np.load(cache_path)
        cached_train_idx = cached["train_idx"]
        cached_test_idx = cached["test_idx"]
        if not np.array_equal(cached_train_idx, train_idx) or not np.array_equal(
            cached_test_idx,
            test_idx,
        ):
            raise ValueError(f"Harmonized SPD cache split mismatch: {cache_path}")

        X_h = X.copy()
        X_h[cached_train_idx] = cached["X_tr"]
        X_h[cached_test_idx] = cached["X_te"]
        return X_h

    def _build_harmonized_spd_cache(
        protocol_tag,
        splits,
        X,
        y,
        dataset_ids,
        apply_harm_to_test,
    ):
        if args.harm_mode not in ("harm", "both"):
            return {}

        cache_paths = {}
        save_dir = os.path.join(args.results_folder, "harmonized_spd")
        cache_dirs = list(args.harmonized_spd_cache_dirs or [])

        print(f"[{protocol_tag}] Preparing shared harmonized SPD cache once per fold...")
        for kf_iter, (train_idx, test_idx) in enumerate(splits, start=1):
            fold_tag = f"{tag}_SPDNet_{protocol_tag}_fold{kf_iter}"
            legacy_fold_tags = [
                f"{tag}_SPDNet_{name}_{protocol_tag}_fold{kf_iter}"
                for names in LEGACY_VARIANT_ALIASES.values()
                for name in names
            ]
            _harmonize_spd_for_fold(
                X=X,
                y=y,
                dataset_ids=dataset_ids,
                train_idx=train_idx,
                test_idx=test_idx,
                apply_harm_to_test=apply_harm_to_test,
                ts_metric=args.ts_metric,
                save_dir=save_dir,
                fold_tag=fold_tag,
                legacy_fold_tags=legacy_fold_tags,
                cache_dirs=cache_dirs,
            )
            cache_paths[kf_iter] = os.path.join(
                save_dir,
                f"{fold_tag}_harmonized_spd.npz",
            )

        return cache_paths

    def _get_variant_dims(P, variant_name):
        variant_name = canonical_spdnet_variant(variant_name)
        if variant_name == "one":
            return (P, P)
        if variant_name == "two":
            return (P, P, P, P)
        if variant_name == "halfdim":
            return (P, max(2, P // 2))
        if variant_name == "quarterdim":
            return (P, max(2, P // 4))
        raise ValueError(f"Unknown SPDNet variant: {variant_name}")

    def _run_variant(
        protocol_tag,
        splits,
        fold_names,
        X,
        y,
        subject_ids,
        variant_name,
        harmonized_cache_paths=None,
    ):
        output_variant_name = variant_name
        canonical_variant_name = canonical_spdnet_variant(variant_name)
        spd_dims = _get_variant_dims(args.P, canonical_variant_name)
        harmonized_cache_paths = harmonized_cache_paths or {}

        def _run_one(label):
            fold_metrics = init_fold_metrics(protocol_tag)
            t0 = time.time()

            print(
                f"[{tag}_SPDNet_{output_variant_name}_{label}_{protocol_tag}] "
                f"Starting folds with dims={spd_dims}..."
            )
            for kf_iter, (train_idx, test_idx) in enumerate(splits, start=1):
                X_use = X
                if label == "harm":
                    if kf_iter not in harmonized_cache_paths:
                        raise RuntimeError(
                            "Missing shared harmonized SPD cache for "
                            f"{protocol_tag} fold {kf_iter}."
                        )
                    X_use = _load_harmonized_spd_cache(
                        X=X,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        cache_path=harmonized_cache_paths[kf_iter],
                    )

                metrics = train_spdnet_ablation_fold(
                    args=args,
                    X=X_use,
                    y=y,
                    groups=subject_ids,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    kf_iter=kf_iter,
                    device=device,
                    protocol_tag=protocol_tag,
                    spd_dims=spd_dims,
                    variant_tag=output_variant_name,
                )
                append_fold_metrics(fold_metrics, metrics)

            elapsed = time.time() - t0
            harm_suffix = "_harm" if label == "harm" else ""
            csv_name = (
                f"{timestamp}{tag}_SPDNet_AgeReg_epochs{args.epochs}_"
                f"{protocol_tag}{harm_suffix}_{output_variant_name}.csv"
            )
            csv_path = os.path.join(args.results_folder, csv_name)
            return save_protocol_metrics_csv(fold_metrics, fold_names, elapsed, csv_path)

        if args.harm_mode in ("none", "both"):
            _run_one("noharm")
        if args.harm_mode in ("harm", "both"):
            _run_one("harm")

    os.makedirs(args.results_folder, exist_ok=True)
    os.makedirs(args.weights_folder_path, exist_ok=True)

    # Original behavior: keep DataLoader shuffling under the ambient run state,
    # without adding a fold-specific generator seed here.
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

    X = estimate_connectome_matrices(
        ts,
        normalize=True,
        n_jobs=args.cov_jobs,
        eps=args.cov_eps,
    )
    args.P = X.shape[1]
    assert X.shape[1] == X.shape[2], "X must be (N,P,P)"

    tag = "ALL-" + "-".join(list(pd.unique(dataset_ids))) if not args.no_make_tag else "ALL"
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())

    protocol_choice = args.protocol.lower()
    if args.cache_only and args.harm_mode == "none":
        args.harm_mode = "harm"
        print(
            "[cache_only] harm_mode=none is not useful for cache generation; "
            "using harm_mode=harm."
        )

    if protocol_choice in ("gkf", "both"):
        protocol1 = f"GKF{args.N_SPLITS}"
        splits_gkf, fold_names_gkf = make_groupkfold_splits(
            X,
            y,
            subject_ids,
            args.N_SPLITS,
        )
        print(f"\n==================== {protocol1} ====================")
        harmonized_cache_gkf = _build_harmonized_spd_cache(
            protocol1,
            splits_gkf,
            X,
            y,
            dataset_ids,
            apply_harm_to_test=True,
        )
        if not args.cache_only:
            for variant_name in args.spdnet_variants:
                _run_variant(
                    protocol1,
                    splits_gkf,
                    fold_names_gkf,
                    X,
                    y,
                    subject_ids,
                    variant_name=variant_name,
                    harmonized_cache_paths=harmonized_cache_gkf,
                )

    if protocol_choice in ("lodo", "both"):
        protocol2 = "LODO"
        splits_lodo, fold_names_lodo = make_lodo_splits(dataset_ids)
        print(f"\n==================== {protocol2} ====================")
        harmonized_cache_lodo = _build_harmonized_spd_cache(
            protocol2,
            splits_lodo,
            X,
            y,
            dataset_ids,
            apply_harm_to_test=False,
        )
        if not args.cache_only:
            for variant_name in args.spdnet_variants:
                _run_variant(
                    protocol2,
                    splits_lodo,
                    fold_names_lodo,
                    X,
                    y,
                    subject_ids,
                    variant_name=variant_name,
                    harmonized_cache_paths=harmonized_cache_lodo,
                )

    if args.cache_only:
        print("\n############ HARMONIZED SPD CACHE BUILD DONE ############")
        return

    print("\n############ SPDNet ABLATIONS DONE ############")


def args_parser(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run Paper Figure 5 SPDNet configuration ablation on pooled "
            "GroupKFold/LODO splits. Experiment defaults preserve original "
            "variant labels."
        )
    )

    add_device_arg(parser)
    add_logging_args(parser)
    add_spdnet_optimization_args(
        parser,
        train_batch_size=1024,
        test_batch_size=1024,
    )
    add_pooled_dataset_args(parser)
    add_split_args(parser)

    parser.add_argument(
        "--weights_folder_path",
        type=str,
        default=str(DEFAULT_ABLATION_WEIGHTS_DIR),
    )
    parser.add_argument("--results_folder", type=str, default=str(DEFAULT_ABLATION_RESULTS_DIR))
    parser.add_argument(
        "--harmonized_spd_cache_dirs",
        nargs="*",
        default=[],
        help="Extra directories to search for reusable harmonized SPD .npz caches.",
    )

    add_spdnet_head_args(parser)
    add_connectome_args(parser)

    add_common_data_args(parser, atlas_arg="--atlas_name")
    parser.add_argument("--ts_metric", type=str, default="riemann")
    parser.add_argument("--no_make_tag", action="store_true", default=False)
    parser.add_argument(
        "--harm_mode",
        type=str,
        choices=["none", "harm", "both"],
        default="none",
        help="Whether to run original SPD matrices, harmonized SPD matrices, or both.",
    )
    parser.add_argument(
        "--cache_only",
        action="store_true",
        default=False,
        help=(
            "Only build/reuse harmonized SPD caches for the selected protocol, "
            "then exit without training."
        ),
    )
    parser.add_argument(
        "--protocol",
        type=str,
        choices=["gkf", "lodo", "both"],
        default="both",
        help="Cross-validation protocol: gkf, lodo, or both.",
    )
    parser.add_argument(
        "--spdnet_variants",
        nargs="+",
        choices=["quarterdim", "halfdim", "one", "two", "base", "2layer"],
        default=list(ORIGINAL_SPDNET_VARIANTS),
        help=(
            "SPDNet ablation variants. Original labels are base/2layer; "
            "paper labels are one/two."
        ),
    )

    args = parser.parse_args(argv)
    return args


def main():
    args = args_parser()
    configure_logging(args.log_level)
    print("############ Start SPDNet Ablation Study (POOLED DATASETS) ############")
    print("Datasets:", args.DATASETS)
    print("Variants:", args.spdnet_variants)

    run_pooled_spdnet_ablation_benchmarks(
        args=args,
        datasets=tuple(args.DATASETS),
        atlas_name=args.atlas_name,
        task=args.task,
        debug=args.debug,
        rng_seed=args.rng_seed,
    )


if __name__ == "__main__":
    main()
