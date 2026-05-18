"""Prepare benchmark rs-fMRI time-series tables from raw/local sources.

This script is optional when users download the processed ``*_X_y.pkl`` files
directly. It exists for users who need to rebuild those files from source
datasets and local fMRIPrep outputs.
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
from nilearn.datasets import fetch_atlas_schaefer_2018, fetch_atlas_msdl
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
)

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
    if atlas_name == "schaefer_100":
        atlas  = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=100).maps
        masker = NiftiLabelsMasker(atlas, standardize=True, detrend=True).fit()
    elif atlas_name == "msdl":
        atlas  = nilearn.datasets.fetch_atlas_msdl().maps
        masker = NiftiMapsMasker(atlas, standardize=True, detrend=True).fit()
    else:
        raise ValueError(f"Only 'schaefer_100' and 'msdl' are supported, got {atlas_name}")
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


# last fetch_cobre from Nilearn (2021)
def fetch_cobre(n_subjects=10, data_dir=None, url=None, verbose=1):
    """Fetch COBRE datasets preprocessed using NIAK 0.17 under CentOS
    version 6.3 with Octave version 4.0.2 and the Minc toolkit version 0.3.18.
    Downloads and returns COBRE preprocessed resting state fMRI datasets,
    covariates and phenotypic information such as demographic, clinical
    variables, measure of frame displacement FD (an average FD for all the time
    frames left after censoring).
    Each subject `fmri_XXXXXXX.nii.gz` is a 3D+t nifti volume (150 volumes).
    WARNING: no confounds were actually regressed from the data, so it can be
    done interactively by the user who will be able to explore different
    analytical paths easily.
    For each subject, there is `fmri_XXXXXXX.tsv` files which contains the
    covariates such as motion parameters, mean CSF signal that should to be
    regressed out of the functional data.
    `keys_confounds.json`: a json file, that describes each variable mentioned
    in the files `fmri_XXXXXXX.tsv.gz`. It also contains a list of time frames
    that have been removed from the time series by censoring for high motion.
    `phenotypic_data.tsv` contains the data of clinical variables that
    explained in `keys_phenotypic_data.json`
    .. versionadded:: 0.3
    Warnings
    --------
    'fetch_cobre' has been deprecated and will be removed in release 0.9.
    Parameters
    ----------
    n_subjects : int, optional
        The number of subjects to load from maximum of 146 subjects.
        By default, 10 subjects will be loaded. If n_subjects=None,
        all subjects will be loaded. Default=10.
    %(data_dir)s
    %(url)s
    %(verbose)s
    Returns
    -------
    data : Bunch
        Dictionary-like object, the attributes are:
        - 'func': string list
            Paths to Nifti images.
        - 'confounds': string list
            Paths to .tsv files of each subject, confounds.
        - 'phenotypic': numpy.recarray
            Contains data of clinical variables, sex, age, FD.
        - 'description': data description of the release and references.
        - 'desc_con': str
            description of the confounds variables
        - 'desc_phenotypic': str
            description of the phenotypic variables.
    Notes
    -----
    See `more information about datasets structure
    <https://figshare.com/articles/COBRE_preprocessed_with_NIAK_0_17_-_lightweight_release/4197885>`_
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
        raise NotImplementedError("No confounds provided")

    # Determine type of var confounds
    i = 0
    while confounds[i] is None:
        i += 1
    type_confounds = type(confounds[i])

    # Add global signal to confounds
    for i, f in enumerate(fmri_path):
        gsr = np.mean(nib.load(str(f)).get_fdata(), axis=(0, 1, 2))
        if type_confounds is np.ndarray:
            if confounds[i] is None:
                confounds[i] = gsr
            else:
                confounds[i] = np.column_stack((gsr, confounds[i]))
        else:
            assert type_confounds is pd.DataFrame
            if confounds[i] is None:
                confounds[i] = pd.DataFrame(gsr, columns=["global_signal"])
            else:
                confounds[i]["global_signal"] = gsr

    return confounds


def filter_data(
    df,
    min_len=MIN_LEN,
    min_cond=MIN_COND,
    max_cond=MAX_COND,
    max_null_regions=MAX_NULL_REGIONS
):
    """Filter data to keep only time series from df["TimeSeries"]) with
    1) a minium of min_len time points
    2) OAS conditioning between min_cond and max_cond
    3) a maximum of max_null_regions null regions
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

    # to know if a time series has a null region
    # check if there is a zero in np.linalg.norm(ts, axis=0) 
    null_regions = df["TimeSeries"].apply(lambda ts: np.sum(np.linalg.norm(ts, axis=0) == 0))
    mask         = mask & (null_regions <= max_null_regions)

    if np.sum(mask) != len(mask):
        warnings.warn(
            f"Removing {np.sum(~mask)} time series with less than {min_len} time points, "
            f"conditioning outside [{min_cond}, {max_cond}], or more than {max_null_regions} null regions."
            f"Percentage of removed time series: {np.sum(~mask) / len(mask) * 100:.0f}%"
        )

    # Filter data
    new_ts = df["TimeSeries"][mask]

    return df[mask]


def postprocess_sample_mask(sample_mask, fmri_paths, start_idx=SAMPLE_MASK_START_IDX):
    """Post process sample_mask to remove start_idx first data points"""
    new_sample_mask = list()
    if sample_mask is None:
        sample_mask = [None] * len(fmri_paths)

    for (mask, f) in zip(sample_mask, fmri_paths):
        if mask is None:
            # Load the data
            data = nib.load(f).get_fdata()

            # Get length of the time series
            length = data.shape[-1]

            # Create a new mask
            new_mask = np.arange(start_idx, length)
        else:
            # Remove the first START_IDX data points
            i = 0
            while mask[i] < start_idx:
                i += 1
            new_mask = mask[i:]

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
            "Skip this step if you downloaded the processed data archive."
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
    DEFAULT_DATA_ROOT = args.data_root.expanduser()
    raw_data_dir = (
        args.raw_data_dir
        or Path(os.environ.get("RSFMRI_SPD_RAW_DATA_DIR", DEFAULT_DATA_ROOT / "raw_data"))
    ).expanduser()
    PATH_RAW_DATA = raw_data_dir
    PATH_RAW_DATA_ADNI_ADNIDOD = (
        args.adni_adnidod_raw_dir
        or Path(os.environ.get("RSFMRI_SPD_ADNI_ADNIDOD_RAW_DIR", raw_data_dir))
    ).expanduser()
    PATH_RAW_DATA_OASIS3 = (
        args.oasis3_raw_dir
        or Path(os.environ.get("RSFMRI_SPD_OASIS3_RAW_DIR", raw_data_dir))
    ).expanduser()
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
