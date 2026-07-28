"""Prepare benchmark rs-fMRI time-series tables from raw/local sources.

Use this script only after independently obtaining authorized source datasets.
It reconstructs trusted local ``*_X_y.pkl`` files from those sources and local
fMRIPrep outputs; no processed participant-data archive is distributed here.
"""

import argparse
import json
import os
import pickle
import re
import warnings
from pathlib import Path

import nibabel as nib
import nilearn
import numpy as np
import pandas as pd
import tabulate
from nilearn.datasets._utils import (
    fetch_files,
    get_dataset_descr,
    get_dataset_dir,
)
from nilearn.image import high_variance_confounds
from nilearn.interfaces.fmriprep import load_confounds_strategy
from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker
from sklearn.covariance import OAS
from sklearn.utils import Bunch

from joblib import Parallel, delayed

from spd_connectome_benchmark.config import (
    DEFAULT_ADNI_ADNIDOD_RAW_DIR,
    DEFAULT_DATA_ROOT,
    DEFAULT_OASIS3_RAW_DIR,
    DEFAULT_RAW_DATA_DIR,
    PAPER_DATASETS,
    ensure_data_path_outside_project,
)
from spd_connectome_benchmark.datasets import canonical_atlas_name

AVAILABLE_ATLASES  = ["msdl_39", "schaefer_100"]

AVAILABLE_DATASETS = list(PAPER_DATASETS)


PATH_RAW_DATA              = DEFAULT_RAW_DATA_DIR
PATH_RAW_DATA_ADNI_ADNIDOD = DEFAULT_ADNI_ADNIDOD_RAW_DIR
PATH_RAW_DATA_OASIS3       = DEFAULT_OASIS3_RAW_DIR


# Criteria used to remove time-series
MAX_NULL_REGIONS = 0
MIN_LEN = 100
MIN_COND = 10
MAX_COND = 1e6

# Remove first SAMPLE_MASK_START_IDX data points from the time series
# for some datasets
SAMPLE_MASK_START_IDX = 5


def fetch_adni_adnidod_oasis3(n_subjects=10, data_dir=None):
    # Load resting state data
    # Get all subject ids
    subjects_all = [d for d in data_dir.glob("sub-*") if d.is_dir()]
    subjects_all = subjects_all[:n_subjects] if n_subjects else subjects_all
    subjects_all = [sub.name for sub in subjects_all]

    # Get fMRI paths
    data = list()

    for subject in subjects_all:
        subject = subject.split("-")[-1]
        path_subject = data_dir / f"sub-{subject}"
        sessions = [d for d in path_subject.glob("ses-*") if d.is_dir()]
        sessions = [ses.name for ses in sessions]
        for session in sessions:
            session = session.split("-")[-1]
            path_session = path_subject / f"ses-{session}" / "func"
            filename = f"sub-{subject}_ses-{session}_task-rest_*space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
            fmri = list(path_session.glob(filename))
            if fmri:
                data.append(
                    {
                        "SubjectID": subject,
                        "Session": session,
                        "fmri_path": fmri[0],
                    }
                )

    fmri_data = pd.DataFrame(data)

    # try to transform Session into int
    try:
        fmri_data["Session"] = fmri_data["Session"].astype(int)
    except ValueError:
        pass

    return fmri_data


def get_nifti_masker(atlas_name):
    atlas_name = canonical_atlas_name(atlas_name)
    if atlas_name == "schaefer_100":
        atlas  = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=100).maps
        masker = NiftiLabelsMasker(atlas, standardize=True, detrend=True).fit()
    elif atlas_name == "msdl_39":
        atlas  = nilearn.datasets.fetch_atlas_msdl().maps
        masker = NiftiMapsMasker(atlas, standardize=True, detrend=True).fit()
    else:
        raise ValueError(
            f"Only 'schaefer_100' and 'msdl_39' are supported, got {atlas_name}"
        )
    return masker


def extract_timeseries_adni_adnidod_oasis3(fmri_data, atlas_name, n_jobs):
    # Extract confounds

    def _try_load_confounds_strategy(fmri_path, denoise_strategy):
        try:
            confounds, sample_mask = load_confounds_strategy(fmri_path, denoise_strategy)
        except ValueError:
            warnings.warn(f"Could not load confounds for {fmri_path}")
            confounds, sample_mask = None, None
        return confounds, sample_mask

    res = Parallel(n_jobs=n_jobs)(
        delayed(_try_load_confounds_strategy)(str(f), denoise_strategy="simple")
        for f in fmri_data["fmri_path"]
    )
    confounds   = [r[0] for r in res]
    sample_mask = [r[1] for r in res]

    # Compute Global Signal Regression
    confounds   = global_signal_regression(fmri_data["fmri_path"], confounds)

    # Post process masker to remove first data points
    sample_mask = postprocess_sample_mask(sample_mask,fmri_paths=fmri_data["fmri_path"])

    fmri_data["Confounds"]  = confounds
    fmri_data["SampleMask"] = sample_mask

    # Extract timeseries
    masker = get_nifti_masker(atlas_name)

    def try_transform(fmri, conf, sample_mask):
        try:
            res = masker.transform(fmri, confounds=conf, sample_mask=sample_mask).astype("float64")
        except Exception as e:
            print(e)
            res = None
        return res

    ts = Parallel(n_jobs=n_jobs)(
        delayed(try_transform)(f, c, s_m)
        for f, c, s_m in zip(
            fmri_data["fmri_path"],
            fmri_data["Confounds"],
            fmri_data["SampleMask"],
        )
    )
    
    fmri_data["TimeSeries"] = ts
    fmri_data.dropna(inplace=True)

    return fmri_data


def fetch_cobre(n_subjects=10, data_dir=None, url=None, verbose=1):
    """Download the legacy NIAK-preprocessed COBRE release from Figshare.

    This compatibility helper was adapted from Nilearn's deprecated
    ``fetch_cobre`` implementation as it existed in 2021. It performs network
    requests and returns local paths to functional images and confound files,
    together with the phenotypic table. Callers are responsible for reviewing
    the current upstream access and use conditions before downloading.

    Parameters
    ----------
    n_subjects : int or None
        Maximum number of participants to return; ``None`` requests all
        available participants.
    data_dir : path-like or None
        Local download/cache directory.
    url : str or None
        Figshare article API endpoint. The historical endpoint is used when
        omitted.
    verbose : int
        Verbosity passed to the Nilearn download helpers.
    """
    if url is None:
        # Here we use the file that provides URL for all others
        url = "https://api.figshare.com/v2/articles/4197885"
        
    dataset_name = "cobre"
    data_dir     = get_dataset_dir(dataset_name, data_dir=data_dir, verbose=verbose)
    fdescr       = get_dataset_descr(dataset_name)

    # First, fetch the file that references all individual URLs
    files = fetch_files(data_dir, [("4197885", url, {})], verbose=verbose)[0]
    files = json.load(open(files, "r"))
    files = files["files"]
    # Index files by name
    files_ = {}
    for f in files:
        files_[f["name"]] = f
    files = files_

    # Fetch the phenotypic file and load it
    csv_name_gz   = "phenotypic_data.tsv.gz"
    csv_name      = os.path.splitext(csv_name_gz)[0]
    csv_file_phen = fetch_files(
        data_dir,
        [
            (
                csv_name,
                files[csv_name_gz]["download_url"],
                {
                    "md5": files[csv_name_gz].get("md5", None),
                    "move": csv_name_gz,
                    "uncompress": True,
                },
            )
        ],
        verbose=verbose,
    )[0]

    # Load file in filename to numpy arrays
    names = [
        "ID",
        "Current Age",
        "Gender",
        "Handedness",
        "Subject Type",
        "Diagnosis",
        "Frames OK",
        "FD",
        "FD Scrubbed",
    ]

    csv_array_phen = np.genfromtxt(
        csv_file_phen,
        names=names,
        delimiter="\t",
        dtype=None,
        encoding="utf-8",
        skip_header=1
    )
    csv_array_phen.dtype.names = tuple(name.lower() for name in csv_array_phen.dtype.names)

    # Check number of subjects
    max_subjects = len(csv_array_phen)
    if n_subjects is None:
        n_subjects = max_subjects

    if n_subjects > max_subjects:
        warnings.warn("Warning: there are only %d subjects" % max_subjects)
        n_subjects = max_subjects

    sz_count = list(csv_array_phen["subject_type"]).count("Patient")
    ct_count = list(csv_array_phen["subject_type"]).count("Control")

    n_sz = np.round(float(n_subjects) / max_subjects * sz_count).astype(int)
    n_ct = np.round(float(n_subjects) / max_subjects * ct_count).astype(int)

    # First, restrict the csv files to the adequate number of subjects
    sz_ids = csv_array_phen[csv_array_phen["subject_type"] == "Patient"]["id"][:n_sz]
    ct_ids = csv_array_phen[csv_array_phen["subject_type"] == "Control"]["id"][:n_ct]
    ids    = np.hstack([sz_ids, ct_ids])
    csv_array_phen = csv_array_phen[np.isin(csv_array_phen["id"], ids)]

    # Call fetch_files once per subject.

    func = []
    con  = []
    for i in ids:
        f = "fmri_00" + str(i) + ".nii.gz"
        c_gz = "fmri_00" + str(i) + ".tsv.gz"
        c = os.path.splitext(c_gz)[0]

        f, c = fetch_files(
            data_dir,
            [
                (
                    f,
                    files[f]["download_url"],
                    {"md5": files[f].get("md5", None), "move": f},
                ),
                (
                    c,
                    files[c_gz]["download_url"],
                    {
                        "md5": files[c_gz].get("md5", None),
                        "move": c_gz,
                        "uncompress": True,
                    },
                ),
            ],
            verbose=verbose,
        )
        func.append(f)
        con.append(c)

    # Fetch the complementary files
    keys_con  = "keys_confounds.json"
    keys_phen = "keys_phenotypic_data.json"

    csv_keys_con, csv_keys_phen = fetch_files(
        data_dir,
        [
            (
                keys_con,
                files[keys_con]["download_url"],
                {"md5": files[keys_con].get("md5", None), "move": keys_con},
            ),
            (
                keys_phen,
                files[keys_phen]["download_url"],
                {"md5": files[keys_phen].get("md5", None), "move": keys_phen},
            ),
        ],
        verbose=verbose,
    )

    files_keys_con  = open(csv_keys_con, "r").read()
    files_keys_phen = open(csv_keys_phen, "r").read()

    return Bunch(
        func=func,
        confounds=con,
        phenotypic=csv_array_phen,
        description=fdescr,
        desc_con=files_keys_con,
        desc_phenotypic=files_keys_phen,
    )


def global_signal_regression(fmri_path, confounds=None):
    if confounds is None:
        raise ValueError("Global signal regression requires confound entries.")
    if len(confounds) != len(fmri_path):
        raise ValueError(
            "Global signal regression requires one confound entry per fMRI image."
        )

    # Determine type of var confounds
    first_confounds = next(
        (entry for entry in confounds if entry is not None),
        None,
    )
    if first_confounds is None:
        raise ValueError(
            "Global signal regression cannot run because all confound entries "
            "are missing."
        )
    if isinstance(first_confounds, np.ndarray):
        confound_kind = "array"
    elif isinstance(first_confounds, pd.DataFrame):
        confound_kind = "dataframe"
    else:
        raise TypeError(
            "Confound entries must be NumPy arrays or pandas DataFrames."
        )
    for entry in confounds:
        if entry is None:
            continue
        if confound_kind == "array" and not isinstance(entry, np.ndarray):
            raise TypeError("All non-missing confound entries must have one type.")
        if confound_kind == "dataframe" and not isinstance(entry, pd.DataFrame):
            raise TypeError("All non-missing confound entries must have one type.")

    # Add global signal to confounds
    for i, f in enumerate(fmri_path):
        gsr = np.mean(nib.load(str(f)).get_fdata(), axis=(0, 1, 2))
        if confound_kind == "array":
            if confounds[i] is None:
                confounds[i] = gsr
            else:
                if len(confounds[i]) != len(gsr):
                    raise ValueError(
                        f"Confound entry at index {i} has {len(confounds[i])} "
                        f"rows; the fMRI image has {len(gsr)} time points."
                    )
                confounds[i] = np.column_stack((gsr, confounds[i]))
        else:
            if confounds[i] is None:
                confounds[i] = pd.DataFrame(gsr, columns=["global_signal"])
            else:
                if len(confounds[i]) != len(gsr):
                    raise ValueError(
                        f"Confound entry at index {i} has {len(confounds[i])} "
                        f"rows; the fMRI image has {len(gsr)} time points."
                    )
                confounds[i]["global_signal"] = gsr

    return confounds


def filter_data(
    df,
    min_len=MIN_LEN,
    min_cond=MIN_COND,
    max_cond=MAX_COND,
    max_null_regions=MAX_NULL_REGIONS
):
    """Keep time series that satisfy length, conditioning, and null-ROI limits.

    A retained series has at least ``min_len`` time points, an OAS covariance
    condition number between ``min_cond`` and ``max_cond``, and no more than
    ``max_null_regions`` regions whose full time course is zero.
    """
    # Minimum length
    mask = df["TimeSeries"].apply(len) >= min_len

    # OAS conditioning
    cov = [OAS().fit(t).covariance_ for t in df["TimeSeries"]]

    def compute_cond(c):
        eigvals = np.linalg.eigvalsh(c)
        return eigvals.max() / eigvals.min()

    cond = np.array([compute_cond(c) for c in cov])
    mask = mask & (cond >= min_cond) & (cond <= max_cond)

    # A zero column norm identifies a region with no temporal variation.
    null_regions = df["TimeSeries"].apply(
        lambda ts: np.sum(np.linalg.norm(ts, axis=0) == 0)
    )
    mask         = mask & (null_regions <= max_null_regions)

    if np.sum(mask) != len(mask):
        warnings.warn(
            f"Removing {np.sum(~mask)} time series with less than {min_len} time points, "
            f"conditioning outside [{min_cond}, {max_cond}], or more than {max_null_regions} null regions."
            f" Percentage removed: {np.sum(~mask) / len(mask) * 100:.0f}%"
        )

    return df[mask]


def postprocess_sample_mask(sample_mask, fmri_paths, start_idx=SAMPLE_MASK_START_IDX):
    """Remove pre-start samples and reject entries with no retained time points."""
    fmri_paths = list(fmri_paths)
    new_sample_mask = list()
    if sample_mask is None:
        sample_mask = [None] * len(fmri_paths)
    else:
        sample_mask = list(sample_mask)
        if len(sample_mask) != len(fmri_paths):
            raise ValueError(
                "Sample masks and fMRI paths must contain the same number of entries."
            )

    for mask_index, (mask, f) in enumerate(zip(sample_mask, fmri_paths)):
        if mask is None:
            # Load the data
            data = nib.load(f).get_fdata()

            # Get length of the time series
            length = data.shape[-1]

            # Create a new mask
            new_mask = np.arange(start_idx, length)
        else:
            # Remove the first START_IDX data points
            mask = np.asarray(mask)
            if mask.ndim != 1:
                raise ValueError(
                    f"Sample mask at index {mask_index} must be one-dimensional."
                )
            if mask.size == 0:
                raise ValueError(
                    f"Sample mask at index {mask_index} is empty."
                )
            retained_indices = np.flatnonzero(mask >= start_idx)
            if retained_indices.size == 0:
                raise ValueError(
                    f"Sample mask at index {mask_index} retains no time points "
                    f"at or after start_idx={start_idx}."
                )
            new_mask = mask[retained_indices[0]:]

        if new_mask.size == 0:
            raise ValueError(
                f"Sample mask at index {mask_index} retains no time points "
                f"at or after start_idx={start_idx}."
            )

        new_sample_mask.append(new_mask)

    return new_sample_mask


def get_data(dataset, atlas_name, n_subjects=None, exist_ok=True, n_jobs=1, verbose=True):

    dataset = dataset.lower()
    assert dataset in AVAILABLE_DATASETS, f"Unknown dataset: {dataset}"

    # Check if dataset is already downloaded and processed
    path = DEFAULT_DATA_ROOT / f"atlas_{atlas_name}" / f"{dataset}_X_y.pkl"

    if path.exists() and exist_ok:
        with open(path, "rb") as f:
            df = pickle.load(f)
    else:
        if dataset == "abide":
            data              = nilearn.datasets.fetch_abide_pcp(data_dir=PATH_RAW_DATA, n_subjects=n_subjects)
            # Extract high variance confounds + gsr
            confounds = Parallel(n_jobs=n_jobs)(
                delayed(high_variance_confounds)(f) for f in data["func_preproc"]
            )
            confounds         = global_signal_regression(data["func_preproc"], confounds)
            data["Confounds"] = confounds

            def extract_subject_id(path):
                match = re.search(r'(\d{7})', path)
                return int(match.group(0)) if match else None

            subject_id        = [extract_subject_id(Path(f).stem) for f in data["func_preproc"]]

            # Extract timeseries
            masker            = get_nifti_masker(atlas_name)
            ts = Parallel(n_jobs=n_jobs)(
                delayed(masker.transform)(f, confounds=c)
                for f, c in zip(data["func_preproc"], data["Confounds"])
            )

            # Create ts_df, a dataframe with one column and
            # one full timeseries per row
            ts_df             = pd.DataFrame({"SUB_ID": subject_id, "TimeSeries": ts})

            # Get phenotypics
            phenotypic        = pd.DataFrame(data.phenotypic)
            phenotypic        = phenotypic.drop(columns=["i", "Unnamed: 0"]).reset_index(drop=True)

            # Merge phenotypic with timeseries
            df                = pd.merge(phenotypic, ts_df, on="SUB_ID", how="inner")

            # Create diagnosis: 0 for control, 1 for autism
            df["Diagnosis"]   = df["DX_GROUP"].replace({2: 0, 1: 1})

            # SEX
            df["SEX"]         = df["SEX"].replace({1: "M", 2: "F"})

            # Select columns
            df = df[["SUB_ID", "SITE_ID", "AGE_AT_SCAN", "SEX", "Diagnosis", "TimeSeries"]]

            df = df.rename(
                columns={
                    "SUB_ID": "SubjectID",
                    "SITE_ID": "Site",
                    "AGE_AT_SCAN": "Age",
                    "SEX": "Sex",
                }
            )

        elif dataset == "adni":
            path_data = PATH_RAW_DATA_ADNI_ADNIDOD / "ADNI_preprocessed_V2"

            # Get fmri resting state data
            fmri_data = fetch_adni_adnidod_oasis3(n_subjects=n_subjects, data_dir=path_data)

            # Extract time series
            fmri_data = extract_timeseries_adni_adnidod_oasis3(fmri_data, atlas_name, n_jobs=n_jobs)

            # Phetenoypic file is in the same directory as the downloaded zip file
            path_adni_phenotypic = path_data / "ADNI_phenotypic.csv"
            if not path_adni_phenotypic.exists():
                raise FileNotFoundError(f"Phenotypic file not found: {path_adni_phenotypic}")
            phenotypic = pd.read_csv(path_adni_phenotypic)
            phenotypic = phenotypic[["SubjectID", "Session", "ScanDate", "Age", "Sex", "Diagnosis", "Group"]]

            # Merge phenotypic and timeseries data
            # AD == Alzheimer's disease
            # CN == control
            # EMCI == Early Mild Cognitive Impairment
            # LMCI == Late Mild Cognitive Impairment
            # MCI == Mild Cognitive Impairment
            # SMC == Significant Memory Concern
            df = pd.merge(phenotypic, fmri_data, on=["SubjectID", "Session"])

            # Select columns
            df = df[["SubjectID", "Session", "ScanDate", "Age", "Sex", "Diagnosis", "Group", "TimeSeries"]]

        elif dataset == "adnidod":
            path_data = PATH_RAW_DATA_ADNI_ADNIDOD / "ADNIDOD_preprocessed"

            # Get fmri resting state data
            fmri_data = fetch_adni_adnidod_oasis3(n_subjects=n_subjects, data_dir=path_data)

            # Extract time series
            fmri_data = extract_timeseries_adni_adnidod_oasis3(fmri_data, atlas_name, n_jobs=n_jobs)

            # Load Phetenoypic file
            path_adnidod_phenotypic = path_data / "ADNIDOD_phenotypic.csv"
            if not path_adnidod_phenotypic.exists():
                raise FileNotFoundError(
                    f"Phenotypic file not found: {path_adnidod_phenotypic}"
                )
            phenotypic = pd.read_csv(path_adnidod_phenotypic, dtype={"SubjectID": str})

            # Merge phenotypic with timeseries using SubjectID and Session
            df = pd.merge(phenotypic, fmri_data, on=["SubjectID", "Session"], how="inner")

            # Select and rename columns
            df = df[
                [
                    "SubjectID",
                    "Site",
                    "Session",
                    "ScanDate",
                    "Age",
                    "Gender",
                    "Diagnosis",
                    "TimeSeries",
                ]
            ]
            df = df.rename(columns={"Gender": "Sex"})
            df = df.replace({"Sex": {1: "M", 2: "F"}})

        elif dataset == "oasis3":
            path_data = PATH_RAW_DATA_OASIS3 / "OASIS3_preprocessed"

            # Get fmri resting state data
            fmri_data = fetch_adni_adnidod_oasis3(n_subjects=n_subjects, data_dir=path_data)

            # Extract time series
            fmri_data = extract_timeseries_adni_adnidod_oasis3(fmri_data, atlas_name, n_jobs=n_jobs)

            # Phetenoypic file is in the same directory as the downloaded zip file
            path_oasis3_phenotypic = path_data / "OASIS3_phenotypic.csv"
            if not path_oasis3_phenotypic.exists():
                raise FileNotFoundError(
                    f"Phenotypic file not found: {path_oasis3_phenotypic}"
                )
            phenotypic = pd.read_csv(path_oasis3_phenotypic)
            phenotypic = phenotypic.rename(columns={"CDRTOT": "Diagnosis"})
            phenotypic["Session"] = "d" + phenotypic["days_to_visit"].astype(str).str.zfill(4)
            phenotypic = phenotypic.replace({"Sex": {1: "M", 2: "F"}})

            # Merge phenotypic and timeseries data
            df = pd.merge(phenotypic, fmri_data, on=["SubjectID", "Session"])

            # Select columns
            df = df[["SubjectID", "Session", "Age", "Sex", "Diagnosis", "TimeSeries"]]

        elif dataset == "camcan":
            # Load resting state data
            # Get all subject ids
            path_raw_data_camcan      = PATH_RAW_DATA / "camcan1366/cc700/mri/pipeline/release004/"
            path_raw_data_camcan_fmri = path_raw_data_camcan / "data_fMRI/aamod_norm_write_dartel_00001"
            subjects_all = list(path_raw_data_camcan_fmri.glob("CC*"))
            subjects_all = subjects_all[:n_subjects] if n_subjects else subjects_all
            subjects_all = [sub.name for sub in subjects_all]

            # Get fMRI paths
            fmri_paths, subjects = list(), list()

            for subject in subjects_all:
                fmri_path = (
                    path_raw_data_camcan_fmri
                    / subject
                    / "Rest"
                ).glob(f"*{subject}*.nii")
                fmri_path = list(fmri_path)
                if len(fmri_path) >= 1:
                    fmri_path = fmri_path[0]
                    if fmri_path.exists():
                        fmri_paths.append(fmri_path)
                        subjects.append(subject)

            fmri_data = pd.DataFrame({"SubjectID": subjects, "fmri_path": fmri_paths})

            # Extract high variance confounds + gsr
            confounds = Parallel(n_jobs=n_jobs)(delayed(high_variance_confounds)(f) for f in fmri_data["fmri_path"])
            confounds = global_signal_regression(fmri_data["fmri_path"], confounds)
            fmri_data["Confounds"] = confounds

            # Post process masker to remove first data points
            sample_mask = postprocess_sample_mask(sample_mask=None, fmri_paths=fmri_data["fmri_path"])
            fmri_data["sample_mask"] = sample_mask

            # Extract timeseries
            masker = get_nifti_masker(atlas_name)

            def try_transform(fmri, conf, sample_mask):
                try:
                    res = masker.transform(
                        fmri,
                        confounds=conf,
                        sample_mask=sample_mask
                    ).astype("float64")
                except Exception as e:
                    print(e)
                    res = None
                return res

            ts = Parallel(n_jobs=n_jobs)(
                delayed(try_transform)(f, c, s)
                for f, c, s in zip(
                    fmri_data["fmri_path"],
                    fmri_data["Confounds"],
                    fmri_data["sample_mask"]
                )
            )
            fmri_data["TimeSeries"] = ts
            fmri_data.dropna(inplace=True, subset=["TimeSeries"])

            # Get phenotypic
            label_path = path_raw_data_camcan / "BIDS_20190411/epi_rest/participants.tsv"
            phenotypic = pd.read_csv(label_path, sep="\t")

            # Extract SubjectID
            phenotypic["participant_id"] = phenotypic["participant_id"].apply(
                lambda x: x.split("-")[1]
            )

            # Extract sex
            phenotypic = phenotypic.replace(
                {"gender_text": {"MALE": "M", "FEMALE": "F"}}
            )

            # Rename and select columns
            phenotypic = phenotypic.rename(
                columns={
                    "participant_id": "SubjectID",
                    "gender_text": "Sex",
                    "age": "Age",
                }
            )
            phenotypic = phenotypic[["SubjectID", "Age", "Sex"]]

            # Merge phenotypic with timeseries
            df = pd.merge(phenotypic, fmri_data, how="inner", on=["SubjectID"])
            df = df.drop(columns=["fmri_path", "Confounds", "sample_mask"])

        elif dataset == "cobre":
            data = fetch_cobre(data_dir=PATH_RAW_DATA, n_subjects=n_subjects)

            # Extract high variance confounds + gsr
            confounds = Parallel(n_jobs=n_jobs)(
                delayed(high_variance_confounds)(f) for f in data.func
            )
            confounds = global_signal_regression(data.func, confounds)
            data["Confounds"] = confounds

            # Extract timeseries
            masker = get_nifti_masker(atlas_name)
            ts = Parallel(n_jobs=n_jobs)(
                delayed(masker.transform)(f, confounds=c)
                for f, c in zip(data.func, data.Confounds)
            )

            # Create ts_df, a dataframe with one column and
            # one full timeseries per row
            ts_df = pd.DataFrame(columns=["TimeSeries"])
            ts_df["TimeSeries"] = ts

            # Get phenotypics
            phenotypic = pd.DataFrame(data.phenotypic)

            # Merge phenotypic with timeseries
            df = pd.concat([phenotypic, ts_df], axis=1)

            # Remove subjects with '295.70 bipolar type' or '295.70 depressed type' diagnosis
            df = df[~df["diagnosis"].str.contains("295.70 bipolar type")]
            df = df[~df["diagnosis"].str.contains("295.70 depressed type")]

            # Select and rename columns
            df = df[["id", "current_age", "gender", "diagnosis", "TimeSeries"]]
            df = df.rename(
                columns={
                    "id": "SubjectID",
                    "current_age": "Age",
                    "gender": "Sex",
                    "diagnosis": "Diagnosis",
                }
            )

            # Diagnoses: None --> 0, Schizophrenia --> 1
            df["Diagnosis"] = df["Diagnosis"].apply(lambda x: 0 if x == "None" else 1)

            # Sex
            df = df.replace({"Sex": {"Male": "M", "Female": "F"}})

        else:
            raise NotImplementedError(f"Dataset {dataset} not implemented")

        # Reset df index
        df = df.reset_index(drop=True)

        # Post-process bad time-series
        df = filter_data(df)

        save_dir  = DEFAULT_DATA_ROOT / f"atlas_{atlas_name}"
        file_path = save_dir / f"{dataset}_X_y.pkl"

        # Save df
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "wb") as f:
            pickle.dump(df, f)

    # Print infos about the dataset
    if verbose:
        print(
            tabulate.tabulate(
                [
                    ["Dataset", dataset],
                    ["Atlas", atlas_name],
                    ["Number of subjects", df["SubjectID"].nunique()],
                    ["Number of time series", len(df)],
                    [
                        "Number of time series per subject",
                        round(len(df) / df["SubjectID"].nunique(), 1),
                    ],
                    ["Number of brain regions", df["TimeSeries"].iloc[0].shape[1]],
                    [
                        "Number of classes (diagnosis)",
                        df["Diagnosis"].nunique() if "Diagnosis" in df else "N/A",
                    ],
                    ["Age range", f"min: {df['Age'].min()}, max: {df['Age'].max()}"],
                    ["Saved in", path] if not exist_ok else ["Loaded from", path],
                ]
            )
        )

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Prepare *_X_y.pkl benchmark files from raw/local rs-fMRI sources. "
            "Source data must be independently obtained under applicable terms."
        )
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Output root for atlas_<atlas_name>/*_X_y.pkl files.",
    )
    parser.add_argument(
        "--raw_data_dir",
        type=Path,
        default=None,
        help=(
            "Root for raw/source datasets. Defaults to <data_root>/raw_data "
            "or RSFMRI_SPD_RAW_DATA_DIR."
        ),
    )
    parser.add_argument(
        "--adni_adnidod_raw_dir",
        type=Path,
        default=None,
        help=(
            "Directory containing ADNI_preprocessed_V2 and "
            "ADNIDOD_preprocessed. Defaults to --raw_data_dir."
        ),
    )
    parser.add_argument(
        "--oasis3_raw_dir",
        type=Path,
        default=None,
        help="Directory containing OASIS3_preprocessed. Defaults to --raw_data_dir.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Dataset to prepare.",
        choices=AVAILABLE_DATASETS + ["all"],
    )
    parser.add_argument(
        "--atlas",
        type=str,
        default="schaefer_100",
        help="Atlas to extract.",
        choices=AVAILABLE_ATLASES,
    )

    args       = parser.parse_args()
    try:
        DEFAULT_DATA_ROOT = ensure_data_path_outside_project(
            args.data_root,
            label="--data_root",
        )
    except ValueError as exc:
        parser.error(str(exc))
    raw_data_dir = (
        args.raw_data_dir
        or Path(os.environ.get("RSFMRI_SPD_RAW_DATA_DIR", DEFAULT_DATA_ROOT / "raw_data"))
    )
    try:
        raw_data_dir = ensure_data_path_outside_project(
            raw_data_dir,
            label="--raw_data_dir",
        )
    except ValueError as exc:
        parser.error(str(exc))
    PATH_RAW_DATA = raw_data_dir
    adni_adnidod_raw_dir = (
        args.adni_adnidod_raw_dir
        or Path(os.environ.get("RSFMRI_SPD_ADNI_ADNIDOD_RAW_DIR", raw_data_dir))
    )
    oasis3_raw_dir = (
        args.oasis3_raw_dir
        or Path(os.environ.get("RSFMRI_SPD_OASIS3_RAW_DIR", raw_data_dir))
    )
    try:
        PATH_RAW_DATA_ADNI_ADNIDOD = ensure_data_path_outside_project(
            adni_adnidod_raw_dir,
            label="--adni_adnidod_raw_dir",
        )
        PATH_RAW_DATA_OASIS3 = ensure_data_path_outside_project(
            oasis3_raw_dir,
            label="--oasis3_raw_dir",
        )
    except ValueError as exc:
        parser.error(str(exc))
    debug      = args.debug
    datasets   = args.dataset.lower()
    datasets   = AVAILABLE_DATASETS if datasets == "all" else [datasets]
    atlas_name = args.atlas

    for d in datasets:
        print(f"Loading {d} dataset...")

        if debug:
            df = get_data(d, atlas_name, n_subjects=20, exist_ok=False, n_jobs=-1, verbose=True)
            print(f"Debug preparation complete: {d}, rows={len(df)}")

        else:
            get_data(d, atlas_name, exist_ok=False, n_jobs=1, verbose=True)
