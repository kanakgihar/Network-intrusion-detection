"""Prediction API for the saved NIDS model.

Usage (CLI):
    python predict.py path/to/flows.csv -o predictions.csv

Security note: joblib files are pickles and can execute code when loaded.
This module therefore (a) only loads files located inside the models directory,
and (b) verifies each artifact's SHA-256 against models/metadata.json before
loading. Never load a model file from an untrusted source.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from common import (
    LABEL_NAMES,
    MODELS_DIR,
    DataValidationError,
    sha256_file,
)

MAX_ROWS = 250_000
METADATA_FILE = "metadata.json"
SELECTED_MODEL_FILE = "nids_pipeline.joblib"
RF_MODEL_FILE = "random_forest_pipeline.joblib"


class ModelNotTrainedError(RuntimeError):
    """No trained model artifacts were found."""


class ArtifactIntegrityError(RuntimeError):
    """Artifact hash does not match metadata (tampered / corrupted file)."""


class IncompatibleArtifactError(RuntimeError):
    """Artifact was built with an incompatible library version or schema."""


class MissingColumnsError(DataValidationError):
    """Uploaded data lacks columns required by the model."""

    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__(
            f"Input is missing {len(missing)} required column(s): {missing[:12]}"
            + (" ..." if len(missing) > 12 else "")
        )


@dataclass
class ModelBundle:
    pipeline: object
    metadata: dict
    file_name: str

    @property
    def feature_order(self) -> list[str]:
        return self.metadata["feature_order"]

    @property
    def model_name(self) -> str:
        return self.metadata.get("selected_model", "unknown")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_metadata(models_dir: Path | str = MODELS_DIR) -> dict:
    path = Path(models_dir) / METADATA_FILE
    if not path.exists():
        raise ModelNotTrainedError(
            "No trained model found (models/metadata.json missing). "
            "Run `python train_model.py` first."
        )
    try:
        meta = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise IncompatibleArtifactError(f"metadata.json is corrupted: {exc}") from exc
    for key in ("feature_order", "numeric_features", "categorical_features", "artifacts", "sklearn_version"):
        if key not in meta:
            raise IncompatibleArtifactError(f"metadata.json is missing required key '{key}'.")
    return meta


def load_model(models_dir: Path | str = MODELS_DIR, file_name: str = SELECTED_MODEL_FILE) -> ModelBundle:
    """Load a verified pipeline from `models_dir`."""
    models_dir = Path(models_dir).resolve()
    meta = load_metadata(models_dir)

    path = (models_dir / file_name).resolve()
    if path.parent != models_dir:  # refuse path traversal / files outside models dir
        raise ArtifactIntegrityError("Refusing to load a model file outside the models directory.")
    if not path.exists():
        raise ModelNotTrainedError(f"Model file {file_name} not found. Run `python train_model.py` first.")

    expected = meta["artifacts"].get(file_name)
    if not expected:
        raise ArtifactIntegrityError(f"{file_name} is not registered in metadata.json; refusing to load it.")
    if sha256_file(path) != expected:
        raise ArtifactIntegrityError(
            f"SHA-256 mismatch for {file_name}: the file was modified or corrupted. Retrain the model."
        )

    trained = meta["sklearn_version"].split(".")[:2]
    current = sklearn.__version__.split(".")[:2]
    if trained != current:
        raise IncompatibleArtifactError(
            f"Model was trained with scikit-learn {meta['sklearn_version']} but "
            f"{sklearn.__version__} is installed. Retrain with `python train_model.py` "
            "or install the matching version from requirements.txt."
        )

    try:
        pipeline = joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - surface any unpickling problem clearly
        raise IncompatibleArtifactError(f"Could not load {file_name}: {exc}") from exc

    if not hasattr(pipeline, "predict_proba") or not hasattr(pipeline, "feature_names_in_"):
        raise IncompatibleArtifactError(f"{file_name} is not a fitted scikit-learn pipeline.")
    if list(pipeline.feature_names_in_) != list(meta["feature_order"]):
        raise IncompatibleArtifactError("Model feature names do not match metadata.json.")
    return ModelBundle(pipeline=pipeline, metadata=meta, file_name=file_name)


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #
def read_csv_safely(source, max_rows: int = MAX_ROWS) -> pd.DataFrame:
    """Read an uploaded/filesystem CSV with friendly errors."""
    try:
        df = pd.read_csv(source, nrows=max_rows + 1)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError("The file is empty or has no header row.") from exc
    except UnicodeDecodeError as exc:
        raise DataValidationError("The file is not UTF-8 text. Upload a plain CSV file.") from exc
    except pd.errors.ParserError as exc:
        raise DataValidationError(f"The file could not be parsed as CSV: {exc}") from exc
    if df.empty:
        raise DataValidationError("The file has a header but no data rows.")
    if len(df) > max_rows:
        raise DataValidationError(f"File has more than {max_rows:,} rows; split it into smaller files.")
    return df


def prepare_features(df: pd.DataFrame, metadata: dict) -> tuple[pd.DataFrame, dict]:
    """Validate and align an input frame to the model's training schema.

    Returns (X, info). Raises MissingColumnsError / DataValidationError.
    """
    if df is None or df.empty:
        raise DataValidationError("Input contains no rows.")
    required = list(metadata["feature_order"])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise MissingColumnsError(missing)

    X = df[required].copy()
    info: dict = {"ignored_columns": [c for c in df.columns if c not in required]}

    coerced = 0
    for col in metadata["numeric_features"]:
        original_na = X[col].isna()
        X[col] = pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        coerced += int((X[col].isna() & ~original_na).sum())
    info["non_numeric_values_treated_as_missing"] = coerced
    info["missing_numeric_values_imputed"] = int(X[metadata["numeric_features"]].isna().sum().sum())

    unseen = {}
    known_all = metadata.get("known_categories", {})
    for col in metadata["categorical_features"]:
        X[col] = X[col].astype("object").where(X[col].isna(), X[col].astype(str).str.strip())
        known = set(known_all.get(col, []))
        if known:
            n = int((~X[col].isna() & ~X[col].isin(known)).sum())
            if n:
                unseen[col] = n
    info["unseen_category_values"] = unseen
    info["missing_categorical_values_imputed"] = int(X[metadata["categorical_features"]].isna().sum().sum())
    return X, info


def risk_level(score: np.ndarray, threshold: float) -> np.ndarray:
    """Heuristic risk bands on the (uncalibrated) attack score."""
    high_cut = max(0.9, threshold)
    return np.where(score >= high_cut, "High", np.where(score >= threshold, "Medium", "Low"))


def predict_dataframe(df: pd.DataFrame, bundle: ModelBundle, threshold: float | None = None) -> pd.DataFrame:
    """Run the saved pipeline on new records.

    attack_score is the Random-Forest/LogReg model's score for class 'Malicious'.
    For tree ensembles it is a vote share, NOT a calibrated probability.
    """
    thr = float(bundle.metadata.get("threshold", 0.5) if threshold is None else threshold)
    if not 0.0 < thr < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1")
    X, info = prepare_features(df, bundle.metadata)
    score = bundle.pipeline.predict_proba(X)[:, 1]
    pred = (score >= thr).astype(int)
    out = df.reset_index(drop=True).copy()
    out.insert(0, "row_number", np.arange(1, len(out) + 1))
    out["prediction"] = [LABEL_NAMES[int(p)] for p in pred]
    out["attack_score"] = np.round(score, 4)
    out["risk_level"] = risk_level(score, thr)
    out.attrs["info"] = info
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Classify network flows with the trained NIDS model.")
    ap.add_argument("input_csv")
    ap.add_argument("-o", "--output", default="predictions.csv")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args(argv)
    try:
        bundle = load_model()
        df = read_csv_safely(args.input_csv)
        result = predict_dataframe(df, bundle, args.threshold)
    except (ModelNotTrainedError, ArtifactIntegrityError, IncompatibleArtifactError, DataValidationError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    result.to_csv(args.output, index=False)
    n_mal = int((result["prediction"] == "Malicious").sum())
    print(f"{len(result)} records classified: {n_mal} malicious, {len(result) - n_mal} normal -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
