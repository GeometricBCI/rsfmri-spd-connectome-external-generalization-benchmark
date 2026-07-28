"""Dataset loading, training helpers, and metric compatibility exports."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import Dataset

from spd_connectome_benchmark.config import (
    DEFAULT_DATA_ROOT,
    DEFAULT_DEBUG_SAMPLE_COUNT,
)
from spd_connectome_benchmark.data_contract import validate_timeseries
from spd_connectome_benchmark.datasets import (
    ATLAS_REGISTRY,
    canonical_atlas_name,
    canonical_dataset_name,
)
from spd_connectome_benchmark.splits import (
    make_groupkfold_splits as _make_groupkfold_splits,
    make_lodo_splits as _make_lodo_splits,
    split_train_validation_by_group as _split_train_validation_by_group,
)
from spd_connectome_benchmark.results import (
    save_metrics_csv as _save_metrics_csv,
    save_protocol_metrics_csv as _save_protocol_metrics_csv,
    save_run_metadata_sidecar as _save_run_metadata_sidecar,
    timestamp_tag as _timestamp_tag,
)
from spd_connectome_benchmark.training import (
    ensure_nonempty_training_batches as _ensure_nonempty_training_batches,
    set_global_random_seed as _set_global_random_seed,
)

# Backward-compatible public names for existing entry points and external users.
make_groupkfold_splits = _make_groupkfold_splits
make_lodo_splits = _make_lodo_splits
split_train_validation_by_group = _split_train_validation_by_group
save_metrics_csv = _save_metrics_csv
save_protocol_metrics_csv = _save_protocol_metrics_csv
save_run_metadata_sidecar = _save_run_metadata_sidecar
timestamp_tag = _timestamp_tag
ensure_nonempty_training_batches = _ensure_nonempty_training_batches
set_global_random_seed = _set_global_random_seed


@dataclass(frozen=True)
class LoadedAgeDataset:
    """Prepared age-regression scans with split-safe grouping metadata."""

    subject_ids: np.ndarray
    dataset_ids: np.ndarray
    timeseries: list[np.ndarray]
    targets: np.ndarray


def load_age_timeseries(
    dataset: str,
    atlas_name: str,
    task: str,
    debug: int | None,
    rng: np.random.RandomState,
    data_root: Path | None = None,
):
    """Load one processed dataset and return scan rows for age regression.

    Paper §2.1/§2.7 use chronological age as the shared target. This loader
    returns scan-level time series, age targets, and subject-level groups.
    Rows are shuffled before splitting so GroupKFold receives a stable but
    non-dataset-sorted scan order. The returned groups are subject identifiers;
    callers that pool datasets should prefix them with the dataset name.
    """
    dataset = canonical_dataset_name(dataset)
    atlas_name = canonical_atlas_name(atlas_name)
    expected_regions = ATLAS_REGISTRY[atlas_name].n_regions
    data_root = Path(data_root or DEFAULT_DATA_ROOT).expanduser()
    file_path = data_root / f"atlas_{atlas_name}" / f"{dataset}_X_y.pkl"
    if not file_path.exists():
        raise FileNotFoundError(
            f"Prepared dataset file not found: {file_path}. "
            "Reconstruct it from independently authorized local sources with "
            "prepare_fmri_datasets.py, pass --data_root, or set "
            "RSFMRI_SPD_DATA_ROOT. No processed participant-data archive is "
            "distributed by this repository."
        )

    with open(file_path, "rb") as f:
        df = pickle.load(f)

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Prepared dataset must deserialize to a pandas DataFrame.")
    missing_columns = sorted({"SubjectID", "TimeSeries"} - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Prepared dataset is missing required columns: "
            + ", ".join(missing_columns)
        )
    df = df.reset_index(drop=True)
    if task not in df.columns:
        raise ValueError(
            f"Prepared dataset {dataset!r} is missing required target column "
            f"{task!r}."
        )

    row_order = rng.permutation(np.arange(len(df)))
    df = df.iloc[row_order].reset_index(drop=True)

    subject_ids = df["SubjectID"].values
    if any(pd.isna(subject_id) or not str(subject_id).strip() for subject_id in subject_ids):
        raise ValueError("Prepared dataset contains an empty SubjectID.")
    timeseries = list(
        validate_timeseries(
            [np.asarray(t) for t in df["TimeSeries"].values],
            expected_regions=expected_regions,
        )
    )

    if task == "Age":
        y = df[task].values.astype("float32")
        if not np.isfinite(y).all():
            raise ValueError("Age target contains NaN or infinity.")
        y_type = "continuous"
    else:
        raise ValueError("This benchmark module currently supports Age regression only.")

    if debug:
        debug_count = int(debug) if int(debug) > 0 else DEFAULT_DEBUG_SAMPLE_COUNT
        if len(timeseries) > debug_count:
            debug_idx = rng.choice(len(timeseries), debug_count, replace=False)
        else:
            debug_idx = np.arange(len(timeseries))
        subject_ids = subject_ids[debug_idx]
        timeseries = [timeseries[i] for i in debug_idx]
        y = y[debug_idx]

    return subject_ids, timeseries, y, y_type


def load_pooled_age_timeseries(
    datasets: Sequence[str],
    atlas_name: str,
    task: str,
    debug: int | None,
    rng_seed: int,
    data_root: Path | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    """Compatibility wrapper returning pooled data as a tuple."""
    dataset = load_pooled_age_dataset(
        datasets=datasets,
        atlas_name=atlas_name,
        task=task,
        debug=debug,
        rng_seed=rng_seed,
        data_root=data_root,
        verbose=verbose,
    )
    return dataset.subject_ids, dataset.dataset_ids, dataset.timeseries, dataset.targets


def load_pooled_age_dataset(
    datasets: Sequence[str],
    atlas_name: str,
    task: str,
    debug: int | None,
    rng_seed: int,
    data_root: Path | None = None,
    verbose: bool = True,
) -> LoadedAgeDataset:
    """Load and shuffle pooled age-regression scans from multiple datasets.

    Paper §2.7 requires subject-level grouping after pooling. Subject
    identifiers are prefixed with their dataset name before pooling so
    nominally identical raw IDs from different datasets cannot collide.
    """
    rng = np.random.RandomState(rng_seed)
    all_subject_ids: list[np.ndarray] = []
    all_dataset_ids: list[np.ndarray] = []
    all_timeseries: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for raw_dataset in datasets:
        dataset = canonical_dataset_name(raw_dataset)
        subject_ids, timeseries, targets, _ = load_age_timeseries(
            dataset=dataset,
            atlas_name=atlas_name,
            task=task,
            debug=debug,
            rng=rng,
            data_root=data_root,
        )
        prefixed_subject_ids = np.array([f"{dataset}_{sid}" for sid in subject_ids], dtype=object)
        dataset_ids = np.array([dataset] * len(prefixed_subject_ids), dtype=object)

        all_subject_ids.append(prefixed_subject_ids)
        all_dataset_ids.append(dataset_ids)
        all_timeseries.extend([np.asarray(t) for t in timeseries])
        all_targets.append(targets.astype("float32"))

        if verbose:
            print(f"[OK] Loaded {len(timeseries)} samples from {dataset}.")

    if not all_timeseries:
        raise RuntimeError("No dataset loaded successfully.")

    subject_ids = np.concatenate(all_subject_ids, axis=0)
    dataset_ids = np.concatenate(all_dataset_ids, axis=0)
    y = np.concatenate(all_targets, axis=0).astype("float32")

    order = rng.permutation(np.arange(len(all_timeseries)))
    subject_ids = subject_ids[order]
    dataset_ids = dataset_ids[order]
    y = y[order]
    all_timeseries = [all_timeseries[i] for i in order]

    return LoadedAgeDataset(
        subject_ids=subject_ids,
        dataset_ids=dataset_ids,
        timeseries=all_timeseries,
        targets=y,
    )


class MatrixRegressionDataset(Dataset):
    """Tiny Dataset wrapper for SPD matrices and scalar regression targets."""

    def __init__(self, matrices: torch.Tensor, labels: torch.Tensor):
        self.matrices = matrices
        self.labels = labels

    def __getitem__(self, index):
        return self.matrices[index], self.labels[index]

    def __len__(self):
        return len(self.matrices)


class EarlyStopping:
    """Track validation loss and persist the best model state for one fold.

    Supplementary Methods specify patience 10 and evaluation of the best
    validation checkpoint only if early stopping triggers. If the full epoch
    budget is reached, the final epoch model is evaluated.
    """

    def __init__(
        self,
        path_w=None,
        patience: int = 7,
        verbose: bool = False,
        delta: float = 0.0,
        trace_func=print,
        checkpoint_path=None,
    ):
        self.patience = int(patience)
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = float(delta)
        self.path = str(checkpoint_path or path_w)
        self.trace_func = trace_func

    def __call__(self, val_loss, model) -> None:
        score = -float(val_loss)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            return

        if score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return

        self.best_score = score
        self.save_checkpoint(val_loss, model)
        self.counter = 0

    def save_checkpoint(self, val_loss, model) -> None:
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {float(val_loss):.6f}). "
                "Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = float(val_loss)


def evaluate_regression_model(model, loader, device, criterion):
    """Evaluate a torch regressor and return loss, core metrics, and arrays."""
    model.eval()
    total_loss = 0.0
    total_n = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, dtype=torch.float32, non_blocking=True)

            pred = model(batch_x).squeeze(-1)
            loss = criterion(pred, batch_y)

            total_loss += float(loss.item()) * batch_y.size(0)
            total_n += batch_y.size(0)
            predictions.append(pred.detach().cpu().numpy())
            targets.append(batch_y.detach().cpu().numpy())

    avg_loss = total_loss / max(total_n, 1)
    y_pred = np.concatenate(predictions, axis=0)
    y_true = np.concatenate(targets, axis=0)

    if not np.isfinite(y_pred).all():
        bad_idx = np.where(~np.isfinite(y_pred))[0][:10]
        raise ValueError(
            f"Predictions contain NaN/Inf, example indices: {bad_idx}, values: {y_pred[bad_idx]}"
        )

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return avg_loss, mae, rmse, r2, y_pred, y_true


def evaluate_regression_loss(model, loader, device, criterion):
    """Evaluate a torch regressor when predictions are not needed by caller."""
    avg_loss, mae, rmse, r2, _, _ = evaluate_regression_model(model, loader, device, criterion)
    return avg_loss, mae, rmse, r2


def safe_correlation(x, y, method: str = "pearson") -> float:
    """Return a finite correlation or NaN for degenerate inputs."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.size < 2 or y.size < 2:
        return np.nan
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return np.nan
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan

    return float(pd.Series(x).corr(pd.Series(y), method=method))


def compute_age_regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute benchmark metrics for age prediction and brain-age gap.

    Paper §2.8 reports NegMAE/R2 and adds Spearman rho plus age-bias slope for
    LODO. CSVs retain RMSE, MSE, Pearson, and BAG-age correlation as diagnostic
    extras.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    brain_age_gap = y_pred - y_true

    mse = float(mean_squared_error(y_true, y_pred))
    metrics = {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
        "MSE": mse,
        "Pearson_r": safe_correlation(y_true, y_pred, method="pearson"),
        "Spearman_rho": safe_correlation(y_true, y_pred, method="spearman"),
        "corr(BAG,age)": safe_correlation(brain_age_gap, y_true, method="pearson"),
        "age_bias_slope": np.nan,
    }

    if y_true.size >= 2 and np.isfinite(brain_age_gap).all() and not np.allclose(y_true, y_true[0]):
        metrics["age_bias_slope"] = float(np.polyfit(y_true, brain_age_gap, deg=1)[0])

    return metrics


def metric_names_for_protocol(protocol_tag: str) -> tuple[str, ...]:
    """Return metrics saved for each benchmark protocol.

    Paper figures use NegMAE/R2 for GroupKFold and add LODO Spearman and
    age-bias slope. Pearson and BAG-age correlation are retained in CSVs as
    diagnostic extras.
    """
    protocol = str(protocol_tag).upper()
    if protocol.startswith("GKF"):
        return ("MAE", "R2", "Pearson_r")
    if protocol == "LODO":
        return ("MAE", "R2", "Pearson_r", "Spearman_rho", "corr(BAG,age)", "age_bias_slope")
    return tuple(compute_age_regression_metrics([0.0, 1.0], [0.0, 1.0]).keys())


def select_metrics_for_protocol(metrics: Mapping[str, float], protocol_tag: str) -> dict[str, float]:
    return {name: float(metrics[name]) for name in metric_names_for_protocol(protocol_tag)}


def format_metrics_for_log(metrics: Mapping[str, float]) -> str:
    return " | ".join(f"{name.replace('_', ' ')} {value:.4f}" for name, value in metrics.items())


def init_fold_metrics(protocol_tag: str) -> dict[str, list[float]]:
    return {name: [] for name in metric_names_for_protocol(protocol_tag)}


def append_fold_metrics(fold_metrics: dict[str, list[float]], metrics: Mapping[str, float]) -> None:
    for name in fold_metrics:
        fold_metrics[name].append(float(metrics[name]))
