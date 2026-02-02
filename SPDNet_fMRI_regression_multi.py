import argparse
import os
import time
from typing import Tuple
from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor

from pyriemann.tangentspace import TangentSpace
from neuroHarmonize import harmonizationLearn, harmonizationApply

from sklearn.covariance import OAS
from joblib import delayed, Parallel

from torch_riemannian.torch_riemannian.modules import (
    ReEig,
    BrooksBatchNorm,
    LogEig,
    BiMap,
    SPDBatchNorm,
)


home_dir = os.path.expanduser("~")


# ---------------------------------------------------------------------
#  Utils
# ---------------------------------------------------------------------
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
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


class RIEHarmonizer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names, covariates_names,
                 eb=True, smooth_terms=None, smooth_term_bounds=(None, None)):
        self.feature_names = feature_names
        # Ensure a list so pandas treats this as multiple columns, not a tuple key.
        self.covariates_names = list(covariates_names)
        self.eb = eb
        self.smooth_terms = smooth_terms or []
        self.smooth_term_bounds = smooth_term_bounds
        self.model = None

    def fit(self, X, y=None):
        data = X[self.feature_names].to_numpy()
        covars = X[self.covariates_names]
        self.model, _ = harmonizationLearn(
            data, covars,
            eb=self.eb,
            smooth_terms=self.smooth_terms,
            smooth_term_bounds=self.smooth_term_bounds
        )
        return self

    def transform(self, X, y=None):
        data = X[self.feature_names].to_numpy()
        covars = X[self.covariates_names]
        return harmonizationApply(data, covars, self.model)


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
    Robust covariance estimation using OAS + stable correlation normalization.
    Returns (N, P, P).
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

        # force symmetry + jitter
        C = 0.5 * (C + C.T)
        C = C + np.eye(C.shape[0]) * eps
        return C

    cov_list = Parallel(n_jobs=n_jobs)(delayed(_cov_est_single)(t) for t in ts)
    cov = np.stack(cov_list, axis=0)

    if normalize:
        # IMPORTANT: clamp diag only (do NOT clamp denom element-wise)
        diag = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))  # (N, P)
        diag = np.maximum(diag, np.sqrt(eps))               # clamp diag
        denom = diag[:, :, None] * diag[:, None, :]
        cov = cov / denom

        cov = 0.5 * (cov + np.transpose(cov, (0, 2, 1)))    # enforce symmetry again
        cov = cov + np.eye(cov.shape[1]) * eps              # jitter again

    return cov


def group_train_val_split(train_idx, groups, val_size=0.2, seed=42):
    """
    Split train_idx into train_sub_idx and val_sub_idx using group-aware split.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    sub_train, sub_val = next(gss.split(train_idx, groups=groups[train_idx]))
    train_sub_idx = train_idx[sub_train]
    val_sub_idx = train_idx[sub_val]
    return train_sub_idx, val_sub_idx


# ---------------------------------------------------------------------
#  SPDNet Regressor
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
        self.reeig_guard = ReEig()           # extra guard before LogEig
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
        #layers.append(BrooksBatchNorm(num_features=dim_out))
        layers.append(ReEig())  # keep SPD after BiMap
        layers.append(SPDBatchNorm(num_features=dim_out))
        layers.append(ReEig())  # keep SPD after BiMap
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
        # (B, P, P)
        x = self.BiMap_Block(x)

        # symmetry + SPD guard (avoid NaN in log)
        x = 0.5 * (x + x.transpose(-1, -2))
        x = self.reeig_guard(x)

        # log map in double for stability
        #x = x.double()
        x = self.logeig(x)
        #x = x.float()

        x = self._vecSPD(x)
        return self.fc(x)  # (B, 1)


def eval_regression(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_n = 0

    preds_all = []
    y_all = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, dtype=torch.float32, non_blocking=True)

            pred = model(batch_x).squeeze(-1)  # (B,)
            loss = criterion(pred, batch_y)

            total_loss += float(loss.item()) * batch_y.size(0)
            total_n += batch_y.size(0)

            preds_all.append(pred.detach().cpu().numpy())
            y_all.append(batch_y.detach().cpu().numpy())

    avg_loss = total_loss / max(total_n, 1)
    preds = np.concatenate(preds_all, axis=0)
    ys = np.concatenate(y_all, axis=0)

    # safety
    if not np.isfinite(preds).all():
        bad = np.where(~np.isfinite(preds))[0][:10]
        raise ValueError(f"Predictions contain NaN/Inf, example indices: {bad}, values: {preds[bad]}")

    mae = mean_absolute_error(ys, preds)
    rmse = np.sqrt(mean_squared_error(ys, preds))
    r2 = r2_score(ys, preds)

    return avg_loss, mae, rmse, r2


def train_one_fold(
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
    alg_name = "SPDNet_AgeReg"

    # ----- train/val split (group-aware) -----
    tr_idx, va_idx = group_train_val_split(train_idx, groups=groups, val_size=args.val_size, seed=args.seed)

    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    train_dataset = MyDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float())
    val_dataset   = MyDataset(torch.from_numpy(X_va).float(), torch.from_numpy(y_va).float())
    test_dataset  = MyDataset(torch.from_numpy(X_te).float(), torch.from_numpy(y_te).float())

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,  # avoid tiny last batch instability
    )
    val_loader   = DataLoader(val_dataset,  batch_size=args.test_batch_size, shuffle=False, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True)

    # ----- model -----
    model = SPDNet(
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
    print(f"Train groups: {len(np.unique(groups[tr_idx]))} | Val groups: {len(np.unique(groups[va_idx]))} | Test groups: {len(np.unique(groups[test_idx]))}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_total = 0

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            t_batch = time.time()
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, dtype=torch.float32, non_blocking=True)

            optimizer.zero_grad()
            pred = model(batch_x).squeeze(-1)  # (B,)
            loss = criterion(pred, batch_y)
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            epoch_loss += float(loss.item()) * batch_y.size(0)
            n_total += batch_y.size(0)

            if batch_idx % 1 == 0:
                print(f"Epoch {epoch} Batch {batch_idx}/{len(train_loader)} "
                    f"loss={loss.item():.6f} time={time.time()-t_batch:.2f}s")

        train_loss = epoch_loss / max(n_total, 1)
        val_loss, val_mae, val_rmse, val_r2 = eval_regression(model, val_loader, device, criterion)

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

    test_loss, test_mae, test_rmse, test_r2 = eval_regression(model, test_loader, device, criterion)
    print(
        f"[{alg_name} | Fold {kf_iter}] "
        f"TestMSE {test_loss:.6f} | TestMAE {test_mae:.4f} | TestRMSE {test_rmse:.4f} | TestR2 {test_r2:.4f}"
    )

    return test_mae, test_rmse, test_r2, test_loss, alg_name



def run_merged_age_benchmarks(
    args,
    datasets=("1000brains", "abide", "adni", "adnidod", "camcan", "cobre", "oasis3"),
    atlas_name="schaefer_100",
    task="Age",
    debug=None,
    rng_seed=42,
    ts_metric="riemann",
    ridge_alphas=(1e-2, 1e-1, 1.0, 10.0, 100.0),
    dummy_strategy="mean",
    make_tag=True,
    algorithms=("spdnet", "ridge", "dummy"),
):
    """
    Run protocols on merged datasets (controlled by args.protocol):
      1) GroupKFold (group by subject_ids)
      2) Leave-One-Dataset-Out (LODO)

    For each protocol, run selected algorithms:
      - SPDNet (your train_one_fold)
      - Ridge on TangentSpace features
      - DummyRegressor (mean/median)

    Output:
      Up to 12 CSVs written to args.results_folder (depends on harm_mode).
    """

    # ---------------------------
    # Helpers
    # ---------------------------
    def _load_data_multi():
        all_subject_ids = []
        all_dataset_ids = []
        all_ts = []
        all_y = []

        rng = np.random.RandomState(rng_seed)

        for ds in datasets:
            subject_ids, ts, y, y_type = load_data(
                dataset=ds,
                atlas_name=atlas_name,
                task=task,
                debug=debug,
                rng=rng,
            )
            if subject_ids is None:
                print(f"[WARN] Skip dataset {ds} (load failed or task missing).")
                continue

            # 防止跨数据集 subject_id 冲突：加前缀
            subject_ids = np.array([f"{ds}_{sid}" for sid in subject_ids], dtype=object)
            dataset_ids = np.array([ds] * len(subject_ids), dtype=object)

            all_subject_ids.append(subject_ids)
            all_dataset_ids.append(dataset_ids)
            all_ts.extend([np.array(t) for t in ts])
            all_y.append(y.astype("float32"))

            print(f"[OK] Loaded {len(ts)} samples from {ds}.")

        if len(all_ts) == 0:
            raise RuntimeError("No dataset loaded successfully.")

        subject_ids = np.concatenate(all_subject_ids, axis=0)
        dataset_ids = np.concatenate(all_dataset_ids, axis=0)
        y = np.concatenate(all_y, axis=0).astype("float32")

        # 全局 shuffle（对齐）
        idx = rng.permutation(np.arange(len(all_ts)))
        subject_ids = subject_ids[idx]
        dataset_ids = dataset_ids[idx]
        y = y[idx]
        all_ts = [all_ts[i] for i in idx]

        return subject_ids, dataset_ids, all_ts, y

    def _make_groupkfold_splits(X, y, subject_ids, n_splits):
        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(X, y, groups=subject_ids))
        fold_names = [f"R{i}" for i in range(1, len(splits) + 1)]
        return splits, fold_names

    def _make_lodo_splits(dataset_ids):
        uniq = list(pd.unique(dataset_ids))
        splits, fold_names = [], []
        for ds in uniq:
            test_idx = np.where(dataset_ids == ds)[0]
            train_idx = np.where(dataset_ids != ds)[0]
            if len(test_idx) == 0 or len(train_idx) == 0:
                continue
            splits.append((train_idx, test_idx))
            fold_names.append(f"TEST_{ds}")
        return splits, fold_names

    def _save_metrics_csv(fold_mae, fold_rmse, fold_r2, fold_mse, fold_names, elapsed, csv_path):
        cols = list(fold_names) + ["Avg", "Time(sec)"]
        df = pd.DataFrame(columns=cols)
        df.loc["MAE"]  = list(fold_mae)  + [float(np.mean(fold_mae))]  + [elapsed]
        df.loc["RMSE"] = list(fold_rmse) + [float(np.mean(fold_rmse))] + [elapsed]
        df.loc["R2"]   = list(fold_r2)   + [float(np.mean(fold_r2))]   + [elapsed]
        df.loc["MSE"]  = list(fold_mse)  + [float(np.mean(fold_mse))]  + [elapsed]
        df.to_csv(csv_path, index=True)
        print("Saved:", csv_path)
        return df

    def _harmonize_ts_features(
        Z_tr,
        Z_te,
        cov_train,
        cov_test,
        apply_harm_to_test: bool,
        covar_cols=("SITE", "age"),
    ):
        feat_cols = [f"cov_{i}" for i in range(Z_tr.shape[1])]
        df_train = pd.concat(
            [pd.DataFrame(cov_train, columns=covar_cols),
             pd.DataFrame(Z_tr, columns=feat_cols)],
            axis=1
        )
        df_test = pd.concat(
            [pd.DataFrame(cov_test, columns=covar_cols),
             pd.DataFrame(Z_te, columns=feat_cols)],
            axis=1
        )

        smooth_terms = ["age"] if "age" in covar_cols else []
        harmonizer = RIEHarmonizer(
            feature_names=feat_cols,
            covariates_names=covar_cols,
            eb=True,
            smooth_terms=smooth_terms,
            smooth_term_bounds=(0, 120),
        )
        harmonizer.fit(df_train)
        Z_tr_h = harmonizer.transform(df_train)
        if apply_harm_to_test:
            Z_te_h = harmonizer.transform(df_test)
        else:
            Z_te_h = Z_te

        Z_tr_h = np.nan_to_num(Z_tr_h, nan=0.0, posinf=0.0, neginf=0.0)
        Z_te_h = np.nan_to_num(Z_te_h, nan=0.0, posinf=0.0, neginf=0.0)
        return Z_tr_h, Z_te_h

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
    ):
        X_tr = X[train_idx]
        X_te = X[test_idx]
        y_tr = y[train_idx]
        y_te = y[test_idx]

        ts = TangentSpace(metric=ts_metric, tsupdate=False)
        Z_tr = ts.fit_transform(X_tr)
        Z_te = ts.transform(X_te)

        cov_train = np.stack([dataset_ids[train_idx], y_tr], axis=1)
        cov_test = np.stack([dataset_ids[test_idx], y_te], axis=1)

        Z_tr_h, Z_te_h = _harmonize_ts_features(
            Z_tr, Z_te, cov_train, cov_test, apply_harm_to_test=apply_harm_to_test
        )

        X_tr_h = ts.inverse_transform(Z_tr_h)
        if apply_harm_to_test:
            X_te_h = ts.inverse_transform(Z_te_h)
        else:
            X_te_h = X_te

        X_h = X.copy()
        X_h[train_idx] = X_tr_h
        X_h[test_idx] = X_te_h

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = os.path.join(save_dir, f"{fold_tag}_harmonized_spd.npz")
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

    def _run_spdnet(protocol_tag, splits, fold_names, X, y, subject_ids, dataset_ids, harm_mode, apply_harm_to_test):
        def _run_one(label):
            fold_mae, fold_rmse, fold_r2, fold_mse = [], [], [], []
            t0 = time.time()

            print(f"[{tag}_SPDNet_{label}_{protocol_tag}] Starting folds...")
            for kf_iter, (train_idx, test_idx) in enumerate(splits, start=1):
                X_use = X
                if label == "harm":
                    fold_tag = f"{tag}_SPDNet_{protocol_tag}_fold{kf_iter}"
                    save_dir = os.path.join(args.results_folder, "harmonized_spd")
                    X_use = _harmonize_spd_for_fold(
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
                mae, rmse, r2, mse, alg_name = train_one_fold(
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
                fold_mae.append(mae)
                fold_rmse.append(rmse)
                fold_r2.append(r2)
                fold_mse.append(mse)

            elapsed = time.time() - t0

            suffix = "_harm" if label == "harm" else ""
            csv_name = f"{timestamp}{tag}_SPDNet_AgeReg_epochs{args.epochs}_{protocol_tag}{suffix}.csv"
            csv_path = os.path.join(args.results_folder, csv_name)
            return _save_metrics_csv(fold_mae, fold_rmse, fold_r2, fold_mse, fold_names, elapsed, csv_path)

        if harm_mode in ("none", "both"):
            _run_one("noharm")
        if harm_mode in ("harm", "both"):
            _run_one("harm")
        return None

    def _run_ridge(protocol_tag, splits, fold_names, X, y, subject_ids, dataset_ids, harm_mode, apply_harm_to_test):
        fold_mae, fold_rmse, fold_r2, fold_mse = [], [], [], []
        fold_mae_h, fold_rmse_h, fold_r2_h, fold_mse_h = [], [], [], []
        t0 = time.time()

        X64 = X.astype(np.float32)
        y64 = y.astype(np.float32)

        for kf_iter, (train_idx, test_idx) in enumerate(splits, start=1):
            X_tr, y_tr = X64[train_idx], y64[train_idx]
            X_te, y_te = X64[test_idx], y64[test_idx]
            g_tr = subject_ids[train_idx]

            # Tangent features (fit on train only)
            ts = TangentSpace(metric=ts_metric, tsupdate=False)
            Z_tr = ts.fit_transform(X_tr)
            Z_te = ts.transform(X_te)

            cov_train = np.stack([dataset_ids[train_idx], y_tr], axis=1)
            cov_test = np.stack([dataset_ids[test_idx], y_te], axis=1)

            if harm_mode in ("harm", "both"):
                Z_tr_h, Z_te_h = _harmonize_ts_features(
                    Z_tr, Z_te, cov_train, cov_test, apply_harm_to_test=apply_harm_to_test
                )

            # group-aware inner CV for alpha
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
                mse = float(mean_squared_error(y_te, pred))
                rmse = float(np.sqrt(mse))
                mae = float(mean_absolute_error(y_te, pred))
                r2 = float(r2_score(y_te, pred))
                print(
                    f"[{tag}_Ridge_TS_{ts_metric}_{label}_{protocol_tag} | Fold {kf_iter}/{len(splits)}]"
                    f" alpha={best_alpha:g} | MAE {mae:.4f} | RMSE {rmse:.4f} | R2 {r2:.4f}"
                )
                return mae, rmse, r2, mse

            if harm_mode in ("none", "both"):
                mae, rmse, r2, mse = _fit_predict(Z_tr, Z_te, "noharm")
                fold_mse.append(mse)
                fold_rmse.append(rmse)
                fold_mae.append(mae)
                fold_r2.append(r2)

            if harm_mode in ("harm", "both"):
                mae_h, rmse_h, r2_h, mse_h = _fit_predict(Z_tr_h, Z_te_h, "harm")
                fold_mse_h.append(mse_h)
                fold_rmse_h.append(rmse_h)
                fold_mae_h.append(mae_h)
                fold_r2_h.append(r2_h)

        elapsed = time.time() - t0

        if harm_mode in ("none", "both"):
            csv_name = f"{timestamp}{tag}_Ridge_AgeReg_TS_{ts_metric}_{protocol_tag}.csv"
            csv_path = os.path.join(args.results_folder, csv_name)
            _save_metrics_csv(fold_mae, fold_rmse, fold_r2, fold_mse, fold_names, elapsed, csv_path)
        if harm_mode in ("harm", "both"):
            csv_name = f"{timestamp}{tag}_Ridge_AgeReg_TS_{ts_metric}_{protocol_tag}_harm.csv"
            csv_path = os.path.join(args.results_folder, csv_name)
            _save_metrics_csv(fold_mae_h, fold_rmse_h, fold_r2_h, fold_mse_h, fold_names, elapsed, csv_path)
        return None

    def _run_dummy(protocol_tag, splits, fold_names, y, harm_mode):
        def _run_one(label):
            fold_mae, fold_rmse, fold_r2, fold_mse = [], [], [], []
            t0 = time.time()

            y64 = y.astype(np.float32)

            for kf_iter, (train_idx, test_idx) in enumerate(splits, start=1):
                y_tr = y64[train_idx]
                y_te = y64[test_idx]

                model = DummyRegressor(strategy=dummy_strategy)
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

                print(f"[{tag}_Dummy_{dummy_strategy}_{label}_{protocol_tag} | Fold {kf_iter}/{len(splits)}] MAE {mae:.4f} | RMSE {rmse:.4f} | R2 {r2:.4f}")

            elapsed = time.time() - t0

            suffix = "_harm" if label == "harm" else ""
            csv_name = f"{timestamp}{tag}_Dummy_AgeReg_{dummy_strategy}_{protocol_tag}{suffix}.csv"
            csv_path = os.path.join(args.results_folder, csv_name)
            return _save_metrics_csv(fold_mae, fold_rmse, fold_r2, fold_mse, fold_names, elapsed, csv_path)

        if harm_mode in ("none", "both"):
            _run_one("noharm")
        if harm_mode in ("harm", "both"):
            _run_one("harm")
        return None

    # ---------------------------
    # Start
    # ---------------------------
    os.makedirs(args.results_folder, exist_ok=True)
    os.makedirs(args.weights_folder_path, exist_ok=True)

    rng = np.random.RandomState(rng_seed)

    # device from your outer script
    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device:", device)

    # load merged
    subject_ids, dataset_ids, ts, y = _load_data_multi()

    print(f"\n[MERGED] N samples: {len(ts)}")
    print(f"[MERGED] Unique subject groups: {len(np.unique(subject_ids))}")
    print(f"[MERGED] Datasets: {list(pd.unique(dataset_ids))}")

    # cov features
    X = cov_est(ts, normalize=True, n_jobs=args.cov_jobs, eps=args.cov_eps)
    args.P = X.shape[1]
    assert X.shape[1] == X.shape[2], "X must be (N,P,P)"

    # naming tag
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
        splits_gkf, fold_names_gkf = _make_groupkfold_splits(X, y, subject_ids, args.N_SPLITS)

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
            _ = _run_ridge(
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
        splits_lodo, fold_names_lodo = _make_lodo_splits(dataset_ids)

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
            _ = _run_ridge(
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


def args_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--initial_lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train_batch_size", type=int, default=1024)
    parser.add_argument("--test_batch_size", type=int, default=1024)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)

    parser.add_argument(
        "--DATASETS",
        nargs="+",
        default=["1000brains", "abide", "adni", "adnidod", "camcan", "cobre", "oasis3"],
        help="List of datasets to merge and run together.",
    )

    parser.add_argument("--N_SPLITS", type=int, default=5)

    parser.add_argument("--weights_folder_path", type=str, default="model_paras_ALL_AgeReg")
    parser.add_argument("--results_folder", type=str, default="results_ALL_AgeReg")

    parser.add_argument("--fc_layer_no", type=int, default=1)
    parser.add_argument("--fc_hidden_dim", type=int, default=100)
    parser.add_argument("--fc_dropout", type=float, default=0.5)

    # val split inside each fold
    parser.add_argument("--val_size", type=float, default=0.1)

    # cov estimation
    parser.add_argument("--cov_jobs", type=int, default=-1)
    parser.add_argument("--cov_eps", type=float, default=1e-5)

    # grad clip
    parser.add_argument("--clip_grad", type=float, default=1.0)

    # Ridge alpha grid
    parser.add_argument("--ridge_inner_splits", type=int, default=5)
    parser.add_argument("--ridge_alphas", nargs="+", type=float, default=[1e-2, 1e-1, 1.0, 10.0, 100.0])

    # data/config options
    parser.add_argument("--atlas_name", type=str, default="schaefer_100")
    parser.add_argument("--task", type=str, default="Age")
    parser.add_argument("--debug", type=int, default=None)
    parser.add_argument("--rng_seed", type=int, default=42)
    parser.add_argument("--ts_metric", type=str, default="riemann")
    parser.add_argument("--dummy_strategy", type=str, default="mean")
    parser.add_argument("--no_make_tag", action="store_true", default=False)
    parser.add_argument(
        "--harm_mode",
        type=str,
        choices=["none", "harm", "both"],
        default="none",
        help="Harmonization for Ridge TS features: none, harm, or both.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["spdnet", "ridge", "dummy"],
        choices=["spdnet", "ridge", "dummy"],
        help="Algorithms to run: spdnet, ridge, dummy.",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        choices=["gkf", "lodo", "both"],
        default="both",
        help="Cross-validation protocol: gkf, lodo, or both.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = args_parser()

    print("############ Start Age Regression (MERGED DATASETS) ############")
    print("Datasets:", args.DATASETS)

    run_merged_age_benchmarks(
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
