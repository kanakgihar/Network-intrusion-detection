import ast
import gzip
import json
import shutil
import tempfile
import unittest
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from tests.helpers import ROOT, make_synthetic, NUMERIC, CATEGORIES

import common
from common import DataValidationError, clean_dataset, load_dataset, validate_dataset
import predict
from predict import (ArtifactIntegrityError, IncompatibleArtifactError, MissingColumnsError,
                     ModelNotTrainedError, load_model, predict_dataframe, prepare_features, read_csv_safely)
from train_model import run_training


class TmpMixin:
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestDataLoading(TmpMixin, unittest.TestCase):
    def write(self, df, name="d.csv"):
        p = self.tmp / name
        df.to_csv(p, index=False)
        return p

    def test_loads_valid_file(self):
        df = load_dataset(self.write(make_synthetic(200)))
        self.assertEqual(len(df), 200)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError) as cm:
            load_dataset(self.tmp / "nope.csv")
        self.assertIn("kaggle", str(cm.exception).lower())

    def test_zero_byte_file(self):
        p = self.tmp / "e.csv"; p.write_bytes(b"")
        with self.assertRaises(DataValidationError):
            load_dataset(p)

    def test_header_only_file(self):
        p = self.write(make_synthetic(10).iloc[0:0])
        with self.assertRaises(DataValidationError):
            load_dataset(p)

    def test_missing_target_column(self):
        with self.assertRaises(DataValidationError) as cm:
            validate_dataset(make_synthetic(50).drop(columns=["label"]))
        self.assertIn("label", str(cm.exception))

    def test_invalid_label_values(self):
        df = make_synthetic(50); df.loc[0, "label"] = 7
        with self.assertRaises(DataValidationError):
            validate_dataset(df)

    def test_single_class_target(self):
        df = make_synthetic(50); df["label"] = 1; df["attack_cat"] = "Generic"
        with self.assertRaises(DataValidationError):
            validate_dataset(df)

    def test_label_and_category_must_agree(self):
        df = make_synthetic(100); i = df.index[df.label == 1][0]; df.loc[i, "attack_cat"] = "Normal"
        with self.assertRaises(DataValidationError):
            validate_dataset(df)

    def test_cleaning_strips_dedups_and_handles_inf(self):
        df = make_synthetic(100)
        df.loc[0, "attack_cat"] = " " + df.loc[0, "attack_cat"] + " "
        df = pd.concat([df, df.iloc[[1, 2]].assign(id=[9001, 9002])], ignore_index=True)  # same features, new id
        df.loc[3, "dur"] = np.inf
        out, rep = clean_dataset(df)
        self.assertEqual(rep["dropped_duplicates"], 2)
        self.assertTrue(out["attack_cat"].str.strip().eq(out["attack_cat"]).all())
        self.assertTrue(out["dur"].isna().any())  # inf -> NaN, imputed later


class TestTrainingAndPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.csv = cls.tmp / "train.csv"
        make_synthetic(900, seed=1).to_csv(cls.csv, index=False)
        cls.models, cls.reports, cls.data = cls.tmp / "models", cls.tmp / "reports", cls.tmp / "data"
        cls.out = run_training(cls.csv, cls.models, cls.reports, cls.data, fast=True, make_plots=False)
        cls.bundle = load_model(cls.models)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # ---- training / artifacts ------------------------------------------------
    def test_artifacts_saved(self):
        for f in ("metadata.json", "nids_pipeline.joblib", "random_forest_pipeline.joblib"):
            self.assertTrue((self.models / f).exists(), f)
        for f in ("metrics.json", "dataset_summary.json", "feature_importance.csv", "test_sample.csv.gz"):
            self.assertTrue((self.reports / f).exists(), f)

    def test_metrics_are_valid_and_computed(self):
        m = json.loads((self.reports / "metrics.json").read_text())
        for model, vals in m["test"].items():
            for k in ("accuracy", "precision_malicious", "recall_malicious", "f1_malicious", "false_positive_rate", "roc_auc"):
                self.assertTrue(0.0 <= vals[k] <= 1.0, (model, k))
            (tn, fp), (fn, tp) = vals["confusion_matrix"]
            self.assertEqual(tn + fp + fn + tp, m["test_rows"])
            self.assertAlmostEqual(vals["false_positive_rate"], fp / (fp + tn), places=9)

    def test_no_leakage_columns_in_features(self):
        feats = self.bundle.feature_order
        for c in ("id", "label", "attack_cat"):
            self.assertNotIn(c, feats)
        self.assertEqual(set(feats), set(NUMERIC) | set(CATEGORIES))

    def test_split_sizes_add_up(self):
        m = json.loads((self.reports / "metrics.json").read_text())
        s = json.loads((self.reports / "dataset_summary.json").read_text())
        self.assertEqual(m["train_rows"] + m["test_rows"], s["records_processed"])

    def test_metadata_hashes_match(self):
        meta = json.loads((self.models / "metadata.json").read_text())
        for f, h in meta["artifacts"].items():
            self.assertEqual(common.sha256_file(self.models / f), h)

    # ---- prediction ----------------------------------------------------------
    def test_predict_valid_records(self):
        df = make_synthetic(60, seed=5, with_labels=False)
        res = predict_dataframe(df, self.bundle)
        self.assertEqual(len(res), 60)
        self.assertTrue(set(res["prediction"]) <= {"Normal", "Malicious"})
        self.assertTrue(res["attack_score"].between(0, 1).all())
        self.assertTrue(set(res["risk_level"]) <= {"Low", "Medium", "High"})
        direct = self.bundle.pipeline.predict_proba(df[self.bundle.feature_order])[:, 1]
        np.testing.assert_allclose(res["attack_score"], np.round(direct, 4))

    def test_learns_signal(self):
        df = make_synthetic(300, seed=11)
        res = predict_dataframe(df.drop(columns=["label", "attack_cat"]), self.bundle)
        acc = (res["prediction"].eq("Malicious").astype(int).values == df["label"].values).mean()
        self.assertGreater(acc, 0.8)  # synthetic data is easily separable

    def test_extra_columns_are_ignored_and_reported(self):
        df = make_synthetic(20, seed=2)  # includes id/label/attack_cat
        res = predict_dataframe(df, self.bundle)
        self.assertIn("label", res.attrs["info"]["ignored_columns"])

    def test_missing_columns_error_names_columns(self):
        df = make_synthetic(10, with_labels=False).drop(columns=["sttl", "proto"])
        with self.assertRaises(MissingColumnsError) as cm:
            predict_dataframe(df, self.bundle)
        self.assertEqual(sorted(cm.exception.missing), ["proto", "sttl"])

    def test_missing_values_are_imputed(self):
        df = make_synthetic(30, seed=3, with_labels=False)
        df.loc[0:4, "dur"] = np.nan; df.loc[5:9, "proto"] = np.nan
        res = predict_dataframe(df, self.bundle)
        self.assertFalse(res["attack_score"].isna().any())
        self.assertEqual(res.attrs["info"]["missing_numeric_values_imputed"], 5)
        self.assertEqual(res.attrs["info"]["missing_categorical_values_imputed"], 5)

    def test_unseen_categories_do_not_crash(self):
        df = make_synthetic(30, seed=4, with_labels=False)
        df.loc[0:9, "proto"] = "brand-new-proto"; df.loc[10:12, "state"] = "XYZ"
        res = predict_dataframe(df, self.bundle)
        self.assertEqual(len(res), 30)
        self.assertEqual(res.attrs["info"]["unseen_category_values"], {"proto": 10, "state": 3})

    def test_non_numeric_and_inf_values_are_treated_as_missing(self):
        df = make_synthetic(20, seed=6, with_labels=False).astype({"dur": object})
        df.loc[0, "dur"] = "abc"; df.loc[1, "rate"] = np.inf
        res = predict_dataframe(df, self.bundle)
        self.assertEqual(res.attrs["info"]["non_numeric_values_treated_as_missing"], 2)

    def test_empty_frame_rejected(self):
        with self.assertRaises(DataValidationError):
            predict_dataframe(make_synthetic(5, with_labels=False).iloc[0:0], self.bundle)

    def test_bad_threshold_rejected(self):
        with self.assertRaises(ValueError):
            predict_dataframe(make_synthetic(5, with_labels=False), self.bundle, threshold=1.5)

    def test_threshold_changes_flags(self):
        df = make_synthetic(200, seed=8, with_labels=False)
        lo = (predict_dataframe(df, self.bundle, 0.1)["prediction"] == "Malicious").sum()
        hi = (predict_dataframe(df, self.bundle, 0.9)["prediction"] == "Malicious").sum()
        self.assertGreaterEqual(lo, hi)

    # ---- artifact safety -----------------------------------------------------
    def clone_models(self):
        d = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, d, True)
        for f in self.models.iterdir():
            shutil.copy(f, d / f.name)
        return d

    def test_untrained_directory(self):
        d = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, d, True)
        with self.assertRaises(ModelNotTrainedError):
            load_model(d)

    def test_tampered_artifact_rejected(self):
        d = self.clone_models()
        with open(d / "nids_pipeline.joblib", "ab") as fh:
            fh.write(b"tamper")
        with self.assertRaises(ArtifactIntegrityError):
            load_model(d)

    def test_unregistered_or_outside_file_rejected(self):
        d = self.clone_models()
        shutil.copy(d / "nids_pipeline.joblib", d / "other.joblib")
        with self.assertRaises(ArtifactIntegrityError):
            load_model(d, "other.joblib")
        with self.assertRaises(ArtifactIntegrityError):
            load_model(d, "../evil.joblib")

    def test_sklearn_version_mismatch_rejected(self):
        d = self.clone_models()
        meta = json.loads((d / "metadata.json").read_text()); meta["sklearn_version"] = "0.20.0"
        (d / "metadata.json").write_text(json.dumps(meta))
        with self.assertRaises(IncompatibleArtifactError):
            load_model(d)

    def test_corrupt_or_incomplete_metadata_rejected(self):
        d = self.clone_models()
        (d / "metadata.json").write_text("{not json")
        with self.assertRaises(IncompatibleArtifactError):
            load_model(d)
        (d / "metadata.json").write_text(json.dumps({"feature_order": []}))
        with self.assertRaises(IncompatibleArtifactError):
            load_model(d)

    def test_feature_schema_mismatch_rejected(self):
        d = self.clone_models()
        meta = json.loads((d / "metadata.json").read_text()); meta["feature_order"] = meta["feature_order"][::-1]
        (d / "metadata.json").write_text(json.dumps(meta))
        with self.assertRaises(IncompatibleArtifactError):
            load_model(d)

    def test_non_pipeline_artifact_rejected(self):
        import joblib
        d = self.clone_models()
        joblib.dump({"not": "a model"}, d / "nids_pipeline.joblib")
        meta = json.loads((d / "metadata.json").read_text())
        meta["artifacts"]["nids_pipeline.joblib"] = common.sha256_file(d / "nids_pipeline.joblib")
        (d / "metadata.json").write_text(json.dumps(meta))
        with self.assertRaises(IncompatibleArtifactError):
            load_model(d)


class TestUploadReading(unittest.TestCase):
    def test_empty_upload(self):
        with self.assertRaises(DataValidationError):
            read_csv_safely(BytesIO(b""))

    def test_header_only_upload(self):
        with self.assertRaises(DataValidationError):
            read_csv_safely(StringIO("a,b,c\n"))

    def test_binary_garbage_upload(self):
        with self.assertRaises(DataValidationError):
            read_csv_safely(BytesIO(bytes(range(128, 256)) * 50))

    def test_malformed_csv_upload(self):
        with self.assertRaises(DataValidationError):
            read_csv_safely(StringIO('a,b\n1,2\n"unterminated,3\n4,5,6,7,8\n' * 3))

    def test_row_limit(self):
        with self.assertRaises(DataValidationError):
            read_csv_safely(StringIO("a\n" + "1\n" * 10), max_rows=5)

    def test_valid_upload(self):
        self.assertEqual(len(read_csv_safely(StringIO("a,b\n1,2\n3,4\n"))), 2)


@unittest.skipUnless((ROOT / "models" / "metadata.json").exists(), "real model not trained yet")
class TestShippedArtifacts(unittest.TestCase):
    """Checks that the committed model, reports, sample data and dashboard agree."""

    def test_shipped_model_predicts_demo_file(self):
        bundle = load_model(ROOT / "models")
        demo = pd.read_csv(ROOT / "data" / "sample_traffic.csv")
        self.assertEqual(list(demo.columns), bundle.feature_order)
        res = predict_dataframe(demo, bundle)
        self.assertEqual(len(res), len(demo))

    def test_dashboard_data_matches_model_features(self):
        bundle = load_model(ROOT / "models")
        sample = pd.read_csv(ROOT / "reports" / "test_sample.csv.gz", nrows=5)
        self.assertTrue(set(bundle.feature_order) <= set(sample.columns))
        imp = pd.read_csv(ROOT / "reports" / "feature_importance.csv")
        self.assertEqual(set(imp["feature"]), set(bundle.feature_order))

    def test_app_has_valid_syntax(self):
        ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
