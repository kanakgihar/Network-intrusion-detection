"""Train, evaluate and save the NIDS models.

    python train_model.py --data data/UNSW_NB15_testing-set.csv

Pipeline
  1. load + validate -> clean (strip text, drop null targets, drop duplicates)
  2. drop identifier / target-derived columns (id, attack_cat) from the features
  3. documented stratified train/test split (stratified on attack_cat)
  4. 3-fold stratified CV on the TRAIN split only -> select model by malicious-class F1
  5. refit both models on the full train split, evaluate ONCE on the held-out test split
  6. save pipelines (joblib) + SHA-256 metadata, metrics, figures and dashboard data
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from common import (
    ATTACK_CAT,
    DATA_DIR,
    ID_COL,
    LABEL_NAMES,
    MODELS_DIR,
    NORMAL_CATEGORY,
    REPORTS_DIR,
    TARGET,
    build_pipeline,
    clean_dataset,
    load_dataset,
    sha256_file,
    split_feature_columns,
)
from predict import RF_MODEL_FILE, SELECTED_MODEL_FILE


def default_data_path() -> Path:
    for name in ("UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"):
        if (DATA_DIR / name).exists():
            return DATA_DIR / name
    return DATA_DIR / "UNSW_NB15_testing-set.csv"


def make_models(seed: int, fast: bool = False):
    n_trees = 20 if fast else 100
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", C=1.0, solver="lbfgs", random_state=seed
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=18,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
    }


def binary_metrics(y_true, y_pred, score) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_malicious": precision_score(y_true, y_pred, zero_division=0),
        "recall_malicious": recall_score(y_true, y_pred, zero_division=0),
        "f1_malicious": f1_score(y_true, y_pred, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, score),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) else 0.0,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def aggregate_importance(pipe, numeric, categorical) -> pd.DataFrame:
    """Impurity importance of the RF, summed back onto the ORIGINAL columns."""
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    imp = pipe.named_steps["model"].feature_importances_
    rows: dict[str, float] = {}
    for name, val in zip(names, imp):
        if name.startswith("num__"):
            col = name[len("num__"):]
        else:
            body = name[len("cat__"):]
            col = next((c for c in categorical if body.startswith(c + "_")), body)
        rows[col] = rows.get(col, 0.0) + float(val)
    df = pd.DataFrame({"feature": list(rows), "impurity_importance": list(rows.values())})
    return df.sort_values("impurity_importance", ascending=False).reset_index(drop=True)


def run_training(
    data_path: Path | str,
    models_dir: Path | str = MODELS_DIR,
    reports_dir: Path | str = REPORTS_DIR,
    data_dir: Path | str = DATA_DIR,
    test_size: float = 0.2,
    seed: int = 42,
    fast: bool = False,
    make_plots: bool = True,
) -> dict:
    t0 = time.time()
    models_dir, reports_dir, data_dir = Path(models_dir), Path(reports_dir), Path(data_dir)
    fig_dir = reports_dir / "figures"
    for d in (models_dir, reports_dir, fig_dir, data_dir):
        d.mkdir(parents=True, exist_ok=True)

    raw = load_dataset(data_path)
    df, clean_report = clean_dataset(raw)
    numeric, categorical = split_feature_columns(df)
    feature_order = numeric + categorical
    print(f"[data] {clean_report}")
    print(f"[data] {len(numeric)} numeric + {len(categorical)} categorical features; excluded: {ID_COL}, {ATTACK_CAT}, {TARGET}")

    X = df[feature_order]
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te, cat_tr, cat_te = train_test_split(
        X, y, df[ATTACK_CAT], test_size=test_size, random_state=seed, stratify=df[ATTACK_CAT]
    )
    print(f"[split] train={len(X_tr):,} test={len(X_te):,} (stratified on {ATTACK_CAT}, seed={seed})")

    # ---- model selection: CV on training split only -------------------------
    models = make_models(seed, fast)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    scoring = {"f1": "f1", "precision": "precision", "recall": "recall", "roc_auc": "roc_auc"}
    cv_results = {}
    for name, est in models.items():
        res = cross_validate(build_pipeline(est, numeric, categorical), X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=1)
        cv_results[name] = {k: float(np.mean(res[f"test_{k}"])) for k in scoring}
        cv_results[name].update({f"{k}_std": float(np.std(res[f"test_{k}"])) for k in scoring})
        print(f"[cv] {name}: " + ", ".join(f"{k}={cv_results[name][k]:.4f}" for k in scoring))
    selected = max(cv_results, key=lambda n: cv_results[n]["f1"])
    print(f"[select] {selected} (highest mean CV F1 for the malicious class)")

    # ---- refit on full train split, evaluate once on test -------------------
    fitted, test_metrics, test_scores = {}, {}, {}
    for name, est in models.items():
        pipe = build_pipeline(est, numeric, categorical).fit(X_tr, y_tr)
        score = pipe.predict_proba(X_te)[:, 1]
        pred = (score >= 0.5).astype(int)
        fitted[name], test_scores[name] = pipe, score
        test_metrics[name] = binary_metrics(y_te, pred, score)
        print(f"[test] {name}: " + ", ".join(f"{k}={v:.4f}" for k, v in test_metrics[name].items() if k != "confusion_matrix"))

    sel_pipe, sel_score = fitted[selected], test_scores[selected]
    sel_pred = (sel_score >= 0.5).astype(int)
    class_report = classification_report(
        y_te, sel_pred, target_names=[LABEL_NAMES[0], LABEL_NAMES[1]], output_dict=True, zero_division=0
    )
    # detection rate per attack category (binary model; Normal row = specificity)
    cat_rows = []
    for c, idx in cat_te.groupby(cat_te).groups.items():
        mask = cat_te.index.isin(idx)
        n = int(mask.sum())
        correct = int((sel_pred[mask] == y_te.values[mask]).sum())
        cat_rows.append({"attack_cat": c, "test_records": n, "correctly_classified": correct, "rate": correct / n})
    per_cat = pd.DataFrame(cat_rows).sort_values("test_records", ascending=False)
    per_cat["metric"] = np.where(per_cat["attack_cat"] == NORMAL_CATEGORY, "specificity (true-negative rate)", "detection rate (recall)")

    # ---- explainability ----------------------------------------------------
    rf_pipe = fitted["Random Forest"]
    importance = aggregate_importance(rf_pipe, numeric, categorical)
    if not fast:
        rng = np.random.RandomState(seed)
        sub = rng.choice(len(X_te), size=min(5000, len(X_te)), replace=False)
        perm = permutation_importance(
            rf_pipe, X_te.iloc[sub], y_te.iloc[sub], scoring="f1", n_repeats=2, random_state=seed, n_jobs=1
        )
        pi = pd.DataFrame({"feature": X_te.columns, "permutation_importance_f1_drop": perm.importances_mean})
        importance = importance.merge(pi, on="feature", how="left")
    importance.to_csv(reports_dir / "feature_importance.csv", index=False)

    # ---- artifacts ---------------------------------------------------------
    joblib.dump(sel_pipe, models_dir / SELECTED_MODEL_FILE, compress=3)
    joblib.dump(rf_pipe, models_dir / RF_MODEL_FILE, compress=3)
    known_categories = {c: sorted(X_tr[c].dropna().astype(str).unique().tolist()) for c in categorical}
    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selected_model": selected,
        "threshold": 0.5,
        "label_names": {str(k): v for k, v in LABEL_NAMES.items()},
        "numeric_features": numeric,
        "categorical_features": categorical,
        "feature_order": feature_order,
        "known_categories": known_categories,
        "excluded_columns": [ID_COL, ATTACK_CAT, TARGET],
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "seed": seed,
        "test_size": test_size,
        "artifacts": {
            SELECTED_MODEL_FILE: sha256_file(models_dir / SELECTED_MODEL_FILE),
            RF_MODEL_FILE: sha256_file(models_dir / RF_MODEL_FILE),
        },
    }
    (models_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    metrics = {
        "selected_model": selected,
        "selection_criterion": "mean 3-fold CV F1 of the malicious class on the training split",
        "decision_threshold": 0.5,
        "cross_validation": cv_results,
        "test": test_metrics,
        "classification_report_selected": class_report,
        "per_attack_category": per_cat.to_dict(orient="records"),
        "train_rows": int(len(X_tr)),
        "test_rows": int(len(X_te)),
        "fast_mode": fast,
    }
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    summary = {
        "source_file": Path(data_path).name,
        "cleaning": clean_report,
        "records_processed": int(len(df)),
        "label_counts": {LABEL_NAMES[int(k)]: int(v) for k, v in y.value_counts().items()},
        "attack_cat_counts": {k: int(v) for k, v in df[ATTACK_CAT].value_counts().items()},
        "n_numeric_features": len(numeric),
        "n_categorical_features": len(categorical),
        "split": {"train": int(len(X_tr)), "test": int(len(X_te)), "stratified_on": ATTACK_CAT, "seed": seed},
    }
    (reports_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2))

    # held-out predictions sample for the dashboard (so deployment never needs the dataset)
    n_sample = min(6000, len(X_te))
    sample_idx = np.random.RandomState(seed).choice(len(X_te), size=n_sample, replace=False)
    ts = X_te.iloc[sample_idx].copy()
    ts[TARGET] = y_te.iloc[sample_idx].values
    ts[ATTACK_CAT] = cat_te.iloc[sample_idx].values
    ts["attack_score"] = np.round(sel_score[sample_idx], 4)
    ts["prediction"] = np.where(ts["attack_score"] >= 0.5, "Malicious", "Normal")
    ts.reset_index(drop=True).to_csv(reports_dir / "test_sample.csv.gz", index=False, compression="gzip")

    # small feature-only demo file users can upload to the app
    demo = X_te.iloc[sample_idx[: min(300, n_sample)]].reset_index(drop=True)
    demo.to_csv(data_dir / "sample_traffic.csv", index=False)

    if make_plots:
        make_figures(fig_dir, metrics, test_scores, y_te, importance, per_cat, summary, selected)

    print(f"[done] artifacts in {models_dir} ({time.time() - t0:.0f}s); selected={selected}")
    return {"metrics": metrics, "metadata": metadata, "summary": summary}


def make_figures(fig_dir, metrics, test_scores, y_te, importance, per_cat, summary, selected):
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
    # confusion matrix (selected)
    cm = np.array(metrics["test"][selected]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Normal", "Malicious"])
    ax.set_yticks([0, 1], ["Normal", "Malicious"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix - {selected} (test set)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(fig_dir / "confusion_matrix.png")
    plt.close(fig)
    # ROC
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, s in test_scores.items():
        fpr, tpr, _ = roc_curve(y_te, s)
        ax.plot(fpr, tpr, label=f"{name} (AUC={metrics['test'][name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "roc_curves.png")
    plt.close(fig)
    # model comparison
    keys = ["accuracy", "precision_malicious", "recall_malicious", "f1_malicious"]
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.38
    for i, name in enumerate(metrics["test"]):
        ax.bar(np.arange(len(keys)) + i * w, [metrics["test"][name][k] for k in keys], w, label=name)
    ax.set_xticks(np.arange(len(keys)) + w / 2, ["Accuracy", "Precision", "Recall", "F1"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Model comparison (test set, malicious = positive class)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "model_comparison.png")
    plt.close(fig)
    # feature importance
    top = importance.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh(top["feature"], top["impurity_importance"], color="#2a7de1")
    ax.set_xlabel("Impurity importance (summed over encoded columns)")
    ax.set_title("Random Forest - top 15 features")
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_importance.png")
    plt.close(fig)
    # class distribution
    cats = pd.Series(summary["attack_cat_counts"]).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(cats.index, cats.values, color="#e14b4b")
    ax.set_title("Records per traffic category (after cleaning)")
    fig.tight_layout()
    fig.savefig(fig_dir / "category_distribution.png")
    plt.close(fig)
    # per-category rate
    pc = per_cat.sort_values("rate")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(pc["attack_cat"], pc["rate"], color="#1b9e77")
    ax.set_xlim(0, 1.02)
    ax.set_title("Correct-classification rate per category (test set)")
    fig.tight_layout()
    fig.savefig(fig_dir / "per_category_rate.png")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train the NIDS models.")
    ap.add_argument("--data", default=str(default_data_path()))
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast", action="store_true", help="smaller RF, no permutation importance (for tests)")
    args = ap.parse_args(argv)
    run_training(args.data, test_size=args.test_size, seed=args.seed, fast=args.fast)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
