"""Shared constants, data loading / validation and pipeline construction.

Used by train_model.py, predict.py, app.py and the tests so that training and
inference always agree on the feature schema.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

TARGET = "label"            # 0 = normal, 1 = malicious
ATTACK_CAT = "attack_cat"   # target-derived -> never used as a feature
ID_COL = "id"               # row identifier -> never used as a feature
EXCLUDED_FROM_FEATURES = (ID_COL, TARGET, ATTACK_CAT)
CATEGORICAL_FEATURES = ("proto", "service", "state")
NORMAL_CATEGORY = "Normal"
LABEL_NAMES = {0: "Normal", 1: "Malicious"}

DOWNLOAD_HELP = (
    "Download UNSW-NB15 from "
    "https://www.kaggle.com/datasets/ucimachinelearning/unsw-nb15-dataset "
    "and place UNSW_NB15_testing-set.csv (or the training-set file) in the data/ folder."
)


class DataValidationError(ValueError):
    """Raised when the dataset or an uploaded file is not usable."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Dataset loading, validation, cleaning
# --------------------------------------------------------------------------- #
def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate a UNSW-NB15 style CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. {DOWNLOAD_HELP}")
    if path.stat().st_size == 0:
        raise DataValidationError(f"Dataset file is empty: {path}")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"Dataset file has no data: {path}") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataValidationError(f"Dataset file is not a valid CSV: {exc}") from exc
    validate_dataset(df)
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Check required columns and target-label sanity (raises DataValidationError)."""
    if df is None or df.empty:
        raise DataValidationError("Dataset contains no rows.")
    required = [TARGET, ATTACK_CAT, *CATEGORICAL_FEATURES]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"Dataset is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)[:15]}... "
            "The official UNSW-NB15 training/testing-set CSVs contain these columns."
        )
    labels = pd.to_numeric(df[TARGET], errors="coerce")
    bad = ~labels.isin([0, 1])
    if bad.any():
        raise DataValidationError(
            f"Target column '{TARGET}' must contain only 0/1; found "
            f"{int(bad.sum())} invalid values (e.g. {df.loc[bad, TARGET].head(3).tolist()})."
        )
    if labels.nunique() < 2:
        raise DataValidationError(
            f"Target column '{TARGET}' has a single class; both normal (0) and malicious (1) are required."
        )
    cat = df[ATTACK_CAT].astype(str).str.strip()
    inconsistent = (labels == 0) != (cat == NORMAL_CATEGORY)
    if inconsistent.any():
        raise DataValidationError(
            f"'{TARGET}' and '{ATTACK_CAT}' disagree on {int(inconsistent.sum())} rows "
            f"(label 0 must correspond exactly to '{NORMAL_CATEGORY}')."
        )


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Strip text, drop null targets / invalid numerics / duplicates. Returns (df, report)."""
    report = {"rows_in": int(len(df))}
    out = df.copy()
    out[ATTACK_CAT] = out[ATTACK_CAT].astype(str).str.strip()
    out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce")
    for c in CATEGORICAL_FEATURES:
        out[c] = out[c].astype(str).str.strip()
    num_cols = [c for c in out.columns if c not in EXCLUDED_FROM_FEATURES and c not in CATEGORICAL_FEATURES]
    out[num_cols] = out[num_cols].apply(pd.to_numeric, errors="coerce")
    out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan)
    report["missing_values_before_imputation"] = int(out[num_cols].isna().sum().sum())

    before = len(out)
    out = out.dropna(subset=[TARGET, ATTACK_CAT])
    report["dropped_null_target"] = int(before - len(out))

    before = len(out)
    dedup_cols = [c for c in out.columns if c != ID_COL]  # id is unique, so exclude it
    out = out.drop_duplicates(subset=dedup_cols)
    report["dropped_duplicates"] = int(before - len(out))

    out[TARGET] = out[TARGET].astype(int)
    report["rows_out"] = int(len(out))
    return out.reset_index(drop=True), report


def split_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_features, categorical_features) from a cleaned frame."""
    cats = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    nums = [c for c in df.columns if c not in EXCLUDED_FROM_FEATURES and c not in cats]
    return nums, cats


# --------------------------------------------------------------------------- #
# Pipeline construction
# --------------------------------------------------------------------------- #
def signed_log1p(x):
    """Compress heavy-tailed network counters while preserving sign."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=50,
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)],
        verbose_feature_names_out=True,
    )


def build_pipeline(estimator, numeric: list[str], categorical: list[str]) -> Pipeline:
    return Pipeline([("preprocess", make_preprocessor(numeric, categorical)), ("model", estimator)])
