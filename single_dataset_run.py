"""
Section 3.1:

Run (GroupKFold) age regression on each rs-fMRI dataset using dummy, TS ridge, and SPDNet models.

This script loads one dataset's time series, estimates covariance/FC matrices, trains/evaluates regressors with cross-validation.
"""

import argparse
import os
import time
from typing import Tuple, Dict, List
from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor

from pyriemann.tangentspace import TangentSpace

from sklearn.covariance import OAS
from joblib import delayed, Parallel

from torch_riemannian.modules import (
    ReEig,
    LogEig,
    BiMap,
)

home_dir = os.path.expanduser("~")


# ---------------------------------------------------------------------
#  Utils
# ---------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, path_w, patience=7, verbose=False, delta=0.0, trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = float(delta)
        self.path = str(path_w)
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -float(val_loss)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {float(val_loss):.6f}). Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = float(val_loss)


class MyDataset(Dataset):
    def __init__(self, data: torch.Tensor, labels: torch.Tensor):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


def now_tag():
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def load_data(dataset, atlas_name, task, debug, rng):
    file_path = os.path.join(home_dir, "dataset/dataset", f"atlas_{atlas_name}", f"{dataset}_X_y.pkl")
    with open(file_path, "rb") as f:
        df = pickle.load(f)

    df.reset_index(drop=True, inplace=True)

    if task not in df.columns:
        warnings.warn(f"Task {task} not available in dataset {dataset}")
        return None, None, None, None

    idx = rng.permutation(np.arange(len(df)))
    df = df.iloc[idx].reset_index(drop=True)

    subject_ids = df["SubjectID"].values
    ts = df["TimeSeries"].values
    ts = [np.array(t) for t in ts]

    if task == "Age":
        y = df[task].values.astype("float32")
        y_type = "continuous"
    else:
        raise ValueError("This script is for Age regression only.")

    if debug:
        N_SUBJ_MAX = 50
        if len(ts) > N_SUBJ_MAX:
            idx = rng.choice(len(ts), N_SUBJ_MAX, replace=False)
        subject_ids = subject_ids[idx]
        ts = [ts[i] for i in idx]
        y = y[idx]

    return subject_ids, ts, y, y_type


def cov_est(ts, normalize=True, n_jobs=-1, eps=1e-5):
    """
    OAS covariance + correlation normalization + SPD jitter.
    Returns (N,P,P)
    """
    if isinstance(ts, np.ndarray) and ts.ndim == 2:
        ts = [ts]
    elif not isinstance(ts, (list, tuple)):
        raise ValueError("ts must be list of arrays or a single 2D array")

    def _cov_est_single(t):
        try:
            C = OAS(store_precision=False).fit(t).covariance_
        except Exception:
            C = np.cov(t, rowvar=False)

        C = 0.5 * (C + C.T)
        C = C + np.eye(C.shape[0]) * eps
        return C

    cov_list = Parallel(n_jobs=n_jobs)(delayed(_cov_est_single)(t) for t in ts)
    cov = np.stack(cov_list, axis=0)

    if normalize:
        diag = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
        diag = np.maximum(diag, np.sqrt(eps))
        denom = diag[:, :, None] * diag[:, None, :]
        cov = cov / denom

        cov = 0.5 * (cov + np.transpose(cov, (0, 2, 1)))
        cov = cov + np.eye(cov.shape[1]) * eps

    return cov


def group_train_val_split(train_idx, groups, val_size=0.2, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    sub_train, sub_val = next(gss.split(train_idx, groups=groups[train_idx]))
    train_sub_idx = train_idx[sub_train]
    val_sub_idx = train_idx[sub_val]
    return train_sub_idx, val_sub_idx


def save_metrics_csv(metrics: Dict[str, List[float]], n_splits: int, out_csv: str, elapsed: float):
    cols = [f"R{i}" for i in range(1, n_splits + 1)] + ["Avg", "Time(sec)"]
    df = pd.DataFrame(columns=cols)
    for k, vals in metrics.items():
        df.loc[k] = vals + [float(np.mean(vals))] + [elapsed]
    Path(os.path.dirname(out_csv)).mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=True)
    return df


# ---------------------------------------------------------------------
#  SPDNet
# ---------------------------------------------------------------------
class SPDNet(nn.Module):
    def __init__(
        self,
        dims: Tuple[int, int] = (100, 100),
        out_dim: int = 1,
        fc_layer_no: int = 1,
        fc_hidden_dim: int = 100,
        fc_dropout: float = 0.5,
    ):
        super().__init__()
        self.dims = dims
        self.out_dim = out_dim
        self.fc_layer_no = fc_layer_no
        self.fc_hidden_dim = fc_hidden_dim
        self.fc_dropout = fc_dropout

        triu_idx = torch.triu_indices(self.dims[-1], self.dims[-1])
        self.register_buffer("triu_i", triu_idx[0])
        self.register_buffer("triu_j", triu_idx[1])

        self.BiMap_Block = self._make_bimap_block(layer_num=len(self.dims) // 2)
        self.reeig_guard = ReEig()
        self.logeig = LogEig(dim=self.dims[-1])
        self.fc = self._make_fc_block()

    def _make_bimap_block(self, layer_num: int) -> nn.Sequential:
        layers = []
        if layer_num > 1:
            for i in range(layer_num - 1):
                dim_in, dim_out = self.dims[2 * i], self.dims[2 * i + 1]
                layers.append(BiMap(in_features=dim_in, out_features=dim_out))
                layers.append(ReEig())
        dim_in, dim_out = self.dims[-2], self.dims[-1]
        layers.append(BiMap(in_features=dim_in, out_features=dim_out))
        layers.append(ReEig())
        return nn.Sequential(*layers)

    def _vecSPD(self, X):
        return X[:, self.triu_i, self.triu_j]

    def _make_fc_block(self):
        in_dim = self.dims[-1] * (self.dims[-1] + 1) // 2
        layers = [
            nn.Linear(in_dim, self.fc_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.fc_hidden_dim),
            nn.Dropout(self.fc_dropout),
        ]
        for _ in range(self.fc_layer_no - 1):
            layers += [
                nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(self.fc_hidden_dim),
                nn.Dropout(self.fc_dropout),
            ]
        layers.append(nn.Linear(self.fc_hidden_dim, self.out_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.BiMap_Block(x)
        x = 0.5 * (x + x.transpose(-1, -2))
        x = self.reeig_guard(x)

        x = x.double()
        x = self.logeig(x)
        x = x.float()

        x = self._vecSPD(x)
        return self.fc(x)


def eval_regression_torch(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_n = 0
    preds_all, y_all = [], []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, dtype=torch.float32, non_blocking=True)

            pred = model(batch_x).squeeze(-1)
            loss = criterion(pred, batch_y)

            total_loss += float(loss.item()) * batch_y.size(0)
            total_n += batch_y.size(0)

            preds_all.append(pred.detach().cpu().numpy())
            y_all.append(batch_y.detach().cpu().numpy())

    avg_loss = total_loss / max(total_n, 1)
    preds = np.concatenate(preds_all, axis=0)
    ys = np.concatenate(y_all, axis=0)

    if not np.isfinite(preds).all():
        bad = np.where(~np.isfinite(preds))[0][:10]
        raise ValueError(f"Predictions contain NaN/Inf, idx: {bad}, values: {preds[bad]}")

    mae = mean_absolute_error(ys, preds)
    rmse = np.sqrt(mean_squared_error(ys, preds))
    r2 = r2_score(ys, preds)
    return avg_loss, mae, rmse, r2


def run_spdnet_age(args, X, y, subject_ids, splits, device, dataset_name: str):
    alg_name = "SPDNet_AgeReg"
    fold_mae, fold_rmse, fold_r2, fold_mse = [], [], [], []
    t0 = time.time()

    for kf_iter, (train_idx, test_idx) in enumerate(splits, start=1):
        tr_idx, va_idx = group_train_val_split(train_idx, groups=subject_ids, val_size=args.val_size, seed=args.seed)

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        train_loader = DataLoader(
            MyDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float()),
            batch_size=args.train_batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            MyDataset(torch.from_numpy(X_va).float(), torch.from_numpy(y_va).float()),
            batch_size=args.test_batch_size,
            shuffle=False,
            pin_memory=True,
        )
        test_loader = DataLoader(
            MyDataset(torch.from_numpy(X_te).float(), torch.from_numpy(y_te).float()),
            batch_size=args.test_batch_size,
            shuffle=False,
            pin_memory=True,
        )

        model = SPDNet(
            dims=(X.shape[1], X.shape[2]),
            out_dim=1,
            fc_layer_no=args.fc_layer_no,
            fc_hidden_dim=args.fc_hidden_dim,
            fc_dropout=args.fc_dropout,
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)
        criterion = nn.MSELoss()

        ckpt_dir = Path(args.weights_folder_root) / dataset_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"{alg_name}_fold{kf_iter}.pt"

        early = EarlyStopping(ckpt_path, patience=args.patience, verbose=True)

        print(f"\n===== {dataset_name} | SPDNet Fold {kf_iter}/{len(splits)} =====")
        print(f"Train groups: {len(np.unique(subject_ids[tr_idx]))} | Val groups: {len(np.unique(subject_ids[va_idx]))} | Test groups: {len(np.unique(subject_ids[test_idx]))}")

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
            val_mse, val_mae, val_rmse, val_r2 = eval_regression_torch(model, val_loader, device, criterion)

            print(
                f"[{dataset_name} | {alg_name} | Fold {kf_iter} | Epoch {epoch}/{args.epochs}] "
                f"TrainMSE {train_mse:.4f} | ValMSE {val_mse:.4f} | MAE {val_mae:.4f} | RMSE {val_rmse:.4f} | R2 {val_r2:.4f}"
            )

            early(val_mse, model)
            if early.early_stop:
                print("Early stopping triggered. Loading best model...")
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                break

        test_mse, test_mae, test_rmse, test_r2 = eval_regression_torch(model, test_loader, device, criterion)
        fold_mae.append(float(test_mae))
        fold_rmse.append(float(test_rmse))
        fold_r2.append(float(test_r2))
        fold_mse.append(float(test_mse))

        print(
            f"[{dataset_name} | {alg_name} | Fold {kf_iter}] "
            f"TestMSE {test_mse:.4f} | TestMAE {test_mae:.4f} | TestRMSE {test_rmse:.4f} | TestR2 {test_r2:.4f}"
        )

    elapsed = time.time() - t0

    out_csv = os.path.join(
        args.results_folder_root,
        dataset_name,
        f"[{now_tag()}]{dataset_name}_{alg_name}_epochs{args.epochs}.csv",
    )
    df = save_metrics_csv(
        metrics={"MAE": fold_mae, "RMSE": fold_rmse, "R2": fold_r2, "MSE": fold_mse},
        n_splits=len(splits),
        out_csv=out_csv,
        elapsed=elapsed,
    )
    return df, out_csv


def run_ridge_ts_age(
    args, 
    X, 
    y, 
    subject_ids, 
    splits, 
    dataset_name: str,
    ts_metric="riemann",
    alphas=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0),
):
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
        ts = TangentSpace(metric=ts_metric, tsupdate=False)
        Z_tr = ts.fit_transform(X_tr)
        Z_te = ts.transform(X_te)

        n_groups_tr = len(np.unique(g_tr))
        inner_k = min(args.ridge_inner_splits, n_groups_tr)

        base = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(random_state=args.seed)),
        ])

        if inner_k < 2:
            best_alpha = float(alphas[0])
            base.set_params(ridge__alpha=best_alpha)
            base.fit(Z_tr, y_tr)
            model = base
        else:
            inner_cv = GroupKFold(n_splits=inner_k)
            gs = GridSearchCV(
                estimator=base,
                param_grid={"ridge__alpha": list(alphas)},
                cv=inner_cv,
                scoring="neg_mean_absolute_error",
                n_jobs=args.ridge_n_jobs,
            )
            gs.fit(Z_tr, y_tr, groups=g_tr)
            model = gs.best_estimator_
            best_alpha = float(gs.best_params_["ridge__alpha"])

        pred = model.predict(Z_te)

        mse  = float(mean_squared_error(y_te, pred))
        rmse = float(np.sqrt(mse))
        mae  = float(mean_absolute_error(y_te, pred))
        r2   = float(r2_score(y_te, pred))

        fold_mse.append(mse); fold_rmse.append(rmse); fold_mae.append(mae); fold_r2.append(r2)
        print(f"[{dataset_name} | {alg_name} | Fold {kf_iter}] alpha={best_alpha:g} | MAE {mae:.4f} | RMSE {rmse:.4f} | R2 {r2:.4f}")

    elapsed = time.time() - t0

    out_csv = os.path.join(
        args.results_folder_root,
        dataset_name,
        f"[{now_tag()}]{dataset_name}_{alg_name}.csv",
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
def run_dummy_age(args, y, splits, dataset_name: str, strategy="mean"):
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

        fold_mse.append(mse); fold_rmse.append(rmse); fold_mae.append(mae); fold_r2.append(r2)
        print(f"[{dataset_name} | {alg_name} | Fold {kf_iter}] MAE {mae:.4f} | RMSE {rmse:.4f} | R2 {r2:.4f}")

    elapsed = time.time() - t0

    out_csv = os.path.join(
        args.results_folder_root,
        dataset_name,
        f"[{now_tag()}]{dataset_name}_{alg_name}.csv",
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

    subject_ids, ts, y, _ = load_data(
        dataset_name,
        atlas_name,
        "Age",
        debug=None,
        rng=np.random.RandomState(args.seed_data_shuffle),
    )

    n_samples = len(ts)
    n_subj = len(np.unique(subject_ids))
    print(f"Loaded N={n_samples} samples, S={n_subj} unique subjects.")

    X = cov_est(ts, normalize=True, n_jobs=args.cov_jobs, eps=args.cov_eps)

    gkf = GroupKFold(n_splits=args.N_SPLITS)
    splits = list(gkf.split(X, y, groups=subject_ids))

    results = {}

    if args.run_spdnet:
        df_spd, spd_csv = run_spdnet_age(args, X, y, subject_ids, splits, args.device, dataset_name)
        results["SPDNet"] = spd_csv

    if args.run_ridge:
        df_ridge, ridge_csv = run_ridge_ts_age(
            args, X, y, subject_ids, splits, dataset_name,
            ts_metric=args.ts_metric,
            alphas=tuple(args.ridge_alphas),
        )
        results["RidgeTS"] = ridge_csv

    if args.run_dummy:
        df_dummy, dummy_csv = run_dummy_age(args, y, splits, dataset_name, strategy=args.dummy_strategy)
        results["Dummy"] = dummy_csv

    return results


def args_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--no-cuda", action="store_true", default=False)

    p.add_argument("--datasets", type=str, default="camcan")
    p.add_argument("--atlas", type=str, default="schaefer_100")

    p.add_argument("--N_SPLITS", type=int, default=5)
    p.add_argument("--val_size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--seed_data_shuffle", type=int, default=42)

    p.add_argument("--cov_jobs", type=int, default=-1)
    p.add_argument("--cov_eps", type=float, default=1e-5)

    p.add_argument("--weights_folder_root", type=str, default="model_paras_AgeReg")
    p.add_argument("--results_folder_root", type=str, default="results_AgeReg")

    p.add_argument("--run_spdnet", action="store_true", default=True)
    p.add_argument("--initial_lr", type=float, default=1e-2)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--train_batch_size", type=int, default=100)
    p.add_argument("--test_batch_size", type=int, default=100)
    p.add_argument("--fc_layer_no", type=int, default=1)
    p.add_argument("--fc_hidden_dim", type=int, default=100)
    p.add_argument("--fc_dropout", type=float, default=0.5)
    p.add_argument("--clip_grad", type=float, default=1.0)

    p.add_argument("--run_ridge", action="store_true", default=False)
    p.add_argument("--ts_metric", type=str, default="riemann", choices=["logeuclid", "riemann"])
    p.add_argument("--ridge_inner_splits", type=int, default=5)
    p.add_argument("--ridge_n_jobs", type=int, default=1) 
    p.add_argument("--ridge_alphas", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3])


    p.add_argument("--run_dummy", action="store_true", default=False)
    p.add_argument("--dummy_strategy", type=str, default="mean", choices=["mean", "median"])

    args = p.parse_args(args=[])

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")

    return args


def main():
    args = args_parser()
    print("Using device:", args.device)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    all_results = {}

    for ds in datasets:
        try:
            res = run_dataset(args, ds, atlas_name=args.atlas)
            all_results[ds] = res
        except Exception as e:
            print(f"[ERROR] Dataset {ds} failed: {e}")
            all_results[ds] = {"error": str(e)}

    summary_rows = []
    for ds, res in all_results.items():
        row = {"Dataset": ds}
        row.update(res)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.results_folder_root, f"[{now_tag()}]SUMMARY_paths.csv")
    Path(args.results_folder_root).mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print("\nSaved summary path table:", summary_path)
    print(summary_df)


if __name__ == "__main__":
    main()
