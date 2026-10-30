"""Network Intrusion Detection System - Streamlit dashboard.

Run with:  streamlit run app.py
All numbers shown come from files produced by train_model.py (models/, reports/).
Nothing here is simulated or live traffic.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import ATTACK_CAT, DATA_DIR, LABEL_NAMES, MODELS_DIR, REPORTS_DIR, TARGET, DataValidationError
from predict import (
    ArtifactIntegrityError,
    IncompatibleArtifactError,
    MissingColumnsError,
    ModelNotTrainedError,
    load_model,
    predict_dataframe,
    read_csv_safely,
    risk_level,
)

st.set_page_config(page_title="ML Network Intrusion Detection", page_icon="🛡️", layout="wide")

NORMAL_COLOR, MALICIOUS_COLOR, ACCENT = "#1fb6a6", "#ff4b5c", "#4da3ff"
MAX_UPLOAD_MB = 50

FEATURE_HELP = {
    "dur": "Total duration of the flow record",
    "proto": "Transaction protocol (tcp, udp, ...)",
    "service": "Network service (http, dns, ssh, ...; '-' if none)",
    "state": "Connection state / protocol indicator",
    "spkts": "Packets from source to destination",
    "dpkts": "Packets from destination to source",
    "sbytes": "Bytes from source to destination",
    "dbytes": "Bytes from destination to source",
    "rate": "Packet rate of the flow",
    "sttl": "Source-to-destination time-to-live value",
    "dttl": "Destination-to-source time-to-live value",
    "sload": "Source bits per second",
    "dload": "Destination bits per second",
    "sloss": "Source packets retransmitted or dropped",
    "dloss": "Destination packets retransmitted or dropped",
    "sinpkt": "Source inter-packet arrival time (ms)",
    "dinpkt": "Destination inter-packet arrival time (ms)",
    "sjit": "Source jitter (ms)",
    "djit": "Destination jitter (ms)",
    "swin": "Source TCP window advertisement",
    "dwin": "Destination TCP window advertisement",
    "stcpb": "Source TCP base sequence number",
    "dtcpb": "Destination TCP base sequence number",
    "tcprtt": "TCP round-trip time (SYN-ACK + ACK-DAT)",
    "synack": "Time between the SYN and SYN_ACK packets",
    "ackdat": "Time between the SYN_ACK and ACK packets",
    "smean": "Mean size of source packets",
    "dmean": "Mean size of destination packets",
    "trans_depth": "Depth of the HTTP request/response pipeline",
    "response_body_len": "Size of the HTTP response body",
    "ct_srv_src": "Recent connections with the same service and source address",
    "ct_state_ttl": "Connection count by state and TTL range",
    "ct_dst_ltm": "Recent connections to the same destination address",
    "ct_src_dport_ltm": "Recent connections from the same source to the same destination port",
    "ct_dst_sport_ltm": "Recent connections to the same destination from the same source port",
    "ct_dst_src_ltm": "Recent connections with the same source and destination addresses",
    "is_ftp_login": "Whether an FTP session used a login",
    "ct_ftp_cmd": "Number of FTP commands in the flow",
    "ct_flw_http_mthd": "Number of HTTP methods (GET/POST...) in the flow",
    "ct_src_ltm": "Recent connections from the same source address",
    "ct_srv_dst": "Recent connections with the same service and destination address",
    "is_sm_ips_ports": "Source and destination IPs and ports are identical",
}

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 1.6rem; max-width: 1250px; }}
      h1, h2, h3 {{ letter-spacing: .2px; }}
      .hero {{ background: linear-gradient(120deg,#0e1a2b 0%,#14263f 60%,#10303a 100%);
               border:1px solid #223a57; border-radius:14px; padding:1.2rem 1.5rem; margin-bottom:1rem; }}
      .hero h1 {{ margin:0; font-size:1.8rem; color:#e8f1ff; }}
      .hero p  {{ margin:.35rem 0 0 0; color:#a9bdd6; }}
      div[data-testid="stMetric"] {{ background:#0f1a2a; border:1px solid #1f3552; border-radius:12px; padding:.7rem .9rem; }}
      .pill {{ display:inline-block; padding:.1rem .6rem; border-radius:999px; font-size:.78rem; font-weight:600; }}
      .pill-n {{ background:{NORMAL_COLOR}22; color:{NORMAL_COLOR}; border:1px solid {NORMAL_COLOR}66; }}
      .pill-m {{ background:{MALICIOUS_COLOR}22; color:{MALICIOUS_COLOR}; border:1px solid {MALICIOUS_COLOR}66; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Small helpers (compatible with old and new Streamlit sizing APIs)
# --------------------------------------------------------------------------- #
def show_plot(fig: go.Figure, key: str | None = None) -> None:
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=45, b=10))
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_df(df: pd.DataFrame, **kwargs) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


@st.cache_data(show_spinner=False)
def read_json(name: str):
    p = REPORTS_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data(show_spinner=False)
def read_report_csv(name: str):
    p = REPORTS_DIR / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_resource(show_spinner="Loading verified model...")
def get_bundle():
    return load_model(MODELS_DIR)


def need_reports(*objs) -> bool:
    if any(o is None for o in objs):
        st.warning("Evaluation reports are missing. Run `python train_model.py` to generate them.")
        return False
    return True


# --------------------------------------------------------------------------- #
# Load artifacts (friendly failure if not trained)
# --------------------------------------------------------------------------- #
bundle, load_error = None, None
try:
    bundle = get_bundle()
except ModelNotTrainedError as exc:
    load_error = ("info", str(exc))
except (ArtifactIntegrityError, IncompatibleArtifactError) as exc:
    load_error = ("error", f"The model artifact cannot be used: {exc}")

metrics = read_json("metrics.json")
summary = read_json("dataset_summary.json")
importance = read_report_csv("feature_importance.csv")
test_sample = read_report_csv("test_sample.csv.gz")

# --------------------------------------------------------------------------- #
# Sidebar / navigation
# --------------------------------------------------------------------------- #
st.sidebar.title("🛡️ NIDS")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "CSV Traffic Analysis", "Security Analytics", "Model Explainability", "Project Information"],
)
st.sidebar.markdown("---")
if bundle:
    st.sidebar.success(f"Model loaded: **{bundle.model_name}**")
    st.sidebar.caption(f"Trained {bundle.metadata.get('created_utc', 'n/a')} · scikit-learn {bundle.metadata.get('sklearn_version')}")
else:
    st.sidebar.warning("No usable model loaded")
st.sidebar.caption("Defensive research project. Not a substitute for a production IDS.")

st.markdown(
    """<div class="hero"><h1>🛡️ ML Network Intrusion Detection System</h1>
    <p>Classifies network-flow records as <b>Normal</b> or <b>Potentially Malicious</b> using models trained on UNSW-NB15,
    with transparent evaluation and explainable results.</p></div>""",
    unsafe_allow_html=True,
)

if load_error and page in ("CSV Traffic Analysis",):
    (st.info if load_error[0] == "info" else st.error)(load_error[1])
    st.stop()
if load_error and page == "Overview":
    (st.info if load_error[0] == "info" else st.error)(load_error[1])


# --------------------------------------------------------------------------- #
# A. Overview
# --------------------------------------------------------------------------- #
if page == "Overview":
    if need_reports(metrics, summary):
        sel = metrics["selected_model"]
        m = metrics["test"][sel]
        c = st.columns(5)
        c[0].metric("Records processed", f"{summary['records_processed']:,}")
        c[1].metric("Accuracy", pct(m["accuracy"]))
        c[2].metric("Precision (malicious)", pct(m["precision_malicious"]))
        c[3].metric("Recall (malicious)", pct(m["recall_malicious"]))
        c[4].metric("F1 (malicious)", pct(m["f1_malicious"]))
        c = st.columns(3)
        c[0].metric("False-positive rate", pct(m["false_positive_rate"]))
        c[1].metric("ROC AUC", f"{m['roc_auc']:.4f}")
        c[2].metric("Selected model", sel)
        st.caption(
            f"Metrics are computed on a held-out test split of {metrics['test_rows']:,} records that were never used for "
            f"training or model selection (model chosen by 3-fold cross-validation on the {metrics['train_rows']:,}-record training split)."
        )
        left, right = st.columns(2)
        with left:
            lc = summary["label_counts"]
            fig = px.pie(names=list(lc), values=list(lc.values()), hole=0.55, color=list(lc),
                         color_discrete_map={"Normal": NORMAL_COLOR, "Malicious": MALICIOUS_COLOR},
                         title="Normal vs malicious traffic (cleaned dataset)")
            show_plot(fig, "ov_pie")
        with right:
            rows = []
            for name, mm in metrics["test"].items():
                rows.append({"Model": name, "Accuracy": pct(mm["accuracy"]), "Precision": pct(mm["precision_malicious"]),
                             "Recall": pct(mm["recall_malicious"]), "F1": pct(mm["f1_malicious"]),
                             "FPR": pct(mm["false_positive_rate"]), "ROC AUC": f"{mm['roc_auc']:.4f}",
                             "CV F1": f"{metrics['cross_validation'][name]['f1']:.4f}"})
            st.subheader("Model comparison")
            show_df(pd.DataFrame(rows))
            cl = summary["cleaning"]
            st.caption(f"Cleaning: {cl['rows_in']:,} input rows -> {cl['rows_out']:,} after removing "
                       f"{cl['dropped_duplicates']:,} duplicate records (and {cl['dropped_null_target']} rows with null targets).")


# --------------------------------------------------------------------------- #
# B. CSV Traffic Analysis
# --------------------------------------------------------------------------- #
elif page == "CSV Traffic Analysis":
    st.subheader("Analyse a CSV of network flows")
    with st.expander("Required columns", expanded=False):
        st.write(f"The file must contain these {len(bundle.feature_order)} columns (extra columns such as `id`, `label`, "
                 "`attack_cat` are ignored; they are never used for prediction):")
        st.code(", ".join(bundle.feature_order))
    up = st.file_uploader("Upload a CSV file", type=["csv"])
    demo_path = DATA_DIR / "sample_traffic.csv"
    use_demo = False
    if demo_path.exists():
        use_demo = st.button("Or use the bundled sample file (300 held-out flows, no labels)")
    threshold = st.slider("Decision threshold for 'Malicious'", 0.05, 0.95, float(bundle.metadata.get("threshold", 0.5)), 0.05,
                          help="Records with attack score >= threshold are flagged. Lower = more sensitive, more false alarms.")

    df_in, source_name = None, None
    try:
        if up is not None:
            if up.size > MAX_UPLOAD_MB * 1024 * 1024:
                raise DataValidationError(f"File is larger than {MAX_UPLOAD_MB} MB.")
            df_in, source_name = read_csv_safely(up), up.name
        elif use_demo:
            df_in, source_name = read_csv_safely(demo_path), demo_path.name
            st.session_state["demo_loaded"] = True
        elif st.session_state.get("demo_loaded") and "csv_df" in st.session_state:
            df_in, source_name = st.session_state["csv_df"], st.session_state["csv_name"]
    except MissingColumnsError as exc:
        st.error(str(exc))
    except DataValidationError as exc:
        st.error(f"Invalid file: {exc}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unexpected error while reading the file: {exc}")

    if df_in is not None:
        st.session_state["csv_df"], st.session_state["csv_name"] = df_in, source_name
        st.markdown(f"**{source_name}** - {len(df_in):,} rows × {df_in.shape[1]} columns")
        show_df(df_in.head(20))
        missing = [c for c in bundle.feature_order if c not in df_in.columns]
        if missing:
            st.error(f"Missing {len(missing)} required column(s): {missing}")
        elif st.button("Run prediction", type="primary"):
            try:
                with st.spinner("Scoring records..."):
                    res = predict_dataframe(df_in, bundle, threshold)
                st.session_state["last_result"] = res
                st.session_state["last_threshold"] = threshold
            except DataValidationError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Prediction failed: {exc}")

    res = st.session_state.get("last_result")
    if res is not None:
        info = res.attrs.get("info", {})
        n_mal = int((res["prediction"] == "Malicious").sum())
        c = st.columns(3)
        c[0].metric("Records", f"{len(res):,}")
        c[1].metric("Flagged malicious", f"{n_mal:,}")
        c[2].metric("Share flagged", pct(n_mal / len(res)))
        notes = []
        if info.get("unseen_category_values"):
            notes.append(f"Unseen category values (handled as 'infrequent'): {info['unseen_category_values']}")
        if info.get("non_numeric_values_treated_as_missing"):
            notes.append(f"{info['non_numeric_values_treated_as_missing']} non-numeric values were treated as missing.")
        if info.get("missing_numeric_values_imputed") or info.get("missing_categorical_values_imputed"):
            notes.append(f"Missing values imputed with training medians/modes: numeric={info.get('missing_numeric_values_imputed', 0)}, "
                         f"categorical={info.get('missing_categorical_values_imputed', 0)}.")
        for n in notes:
            st.warning(n)
        l, r = st.columns(2)
        with l:
            counts = res["prediction"].value_counts().reindex(["Normal", "Malicious"], fill_value=0)
            show_plot(px.bar(x=counts.index, y=counts.values, color=counts.index,
                             color_discrete_map={"Normal": NORMAL_COLOR, "Malicious": MALICIOUS_COLOR},
                             labels={"x": "Prediction", "y": "Records"}, title="Predicted classes"), "res_bar")
        with r:
            show_plot(px.histogram(res, x="attack_score", nbins=40, color="prediction",
                                   color_discrete_map={"Normal": NORMAL_COLOR, "Malicious": MALICIOUS_COLOR},
                                   title="Attack score distribution"), "res_hist")
        st.caption("**Attack score** is the model's output for the 'Malicious' class. For the Random Forest it is the share of trees voting "
                   "'malicious', **not a calibrated probability**: a score of 0.9 does not mean a 90% chance of an attack. Use it to rank records.")
        st.subheader("Results")
        show_df(res.sort_values("attack_score", ascending=False))
        st.download_button("Download results (CSV)", res.to_csv(index=False).encode("utf-8"),
                           file_name="nids_predictions.csv", mime="text/csv")


# --------------------------------------------------------------------------- #
# C. Security Analytics
# --------------------------------------------------------------------------- #
elif page == "Security Analytics":
    st.subheader("Security analytics")
    st.caption("Charts use the UNSW-NB15 dataset and the model's predictions on its held-out test split. This is recorded research data, not live traffic.")
    if need_reports(metrics, summary):
        t1, t2, t3, t4, t5 = st.tabs(["Traffic mix", "Feature distributions", "Evaluation", "Flagged records", "Your last upload"])
        with t1:
            lc, ac = summary["label_counts"], summary["attack_cat_counts"]
            l, r = st.columns(2)
            with l:
                show_plot(px.bar(x=list(lc), y=list(lc.values()), color=list(lc),
                                 color_discrete_map={"Normal": NORMAL_COLOR, "Malicious": MALICIOUS_COLOR},
                                 labels={"x": "", "y": "Records"}, title="Normal vs malicious"), "mix_bar")
            with r:
                a = pd.Series(ac).sort_values()
                show_plot(px.bar(x=a.values, y=a.index, orientation="h", labels={"x": "Records", "y": ""},
                                 color=[NORMAL_COLOR if k == "Normal" else MALICIOUS_COLOR for k in a.index],
                                 color_discrete_map="identity", title="Traffic category distribution"), "mix_cat")
            pc = pd.DataFrame(metrics["per_attack_category"])
            pc["rate"] = pc["rate"].astype(float)
            show_plot(px.bar(pc.sort_values("rate"), x="rate", y="attack_cat", orientation="h", range_x=[0, 1.02],
                             hover_data=["test_records", "metric"], labels={"rate": "Rate", "attack_cat": ""},
                             title="Per-category result on the test split (attack rows: detection rate; Normal: specificity)"), "mix_rate")
            st.caption("Categories with few records (e.g. Worms, Shellcode, Analysis, Backdoor) have small test counts, so their rates are noisy.")
        with t2:
            if test_sample is None:
                st.warning("reports/test_sample.csv.gz is missing. Run `python train_model.py`.")
            else:
                num = [f for f in bundle.metadata["numeric_features"]] if bundle else []
                cat = list(bundle.metadata["categorical_features"]) if bundle else []
                default = importance["feature"].iloc[0] if importance is not None and len(importance) else (num[0] if num else None)
                options = num + cat
                feat = st.selectbox("Feature", options, index=options.index(default) if default in options else 0,
                                    format_func=lambda f: f"{f} - {FEATURE_HELP.get(f, '')}")
                ts = test_sample.copy()
                ts["Actual"] = ts[TARGET].map(LABEL_NAMES)
                cmap = {"Normal": NORMAL_COLOR, "Malicious": MALICIOUS_COLOR}
                if feat in cat:
                    top = ts[feat].value_counts().head(12).index
                    g = ts[ts[feat].isin(top)].groupby([feat, "Actual"]).size().reset_index(name="count")
                    show_plot(px.bar(g, x=feat, y="count", color="Actual", barmode="group", color_discrete_map=cmap,
                                     title=f"{feat} by actual class (top 12 values)"), "dist_cat")
                else:
                    logx = st.checkbox("Use log10(1 + x) scale (helps heavy-tailed features)", value=True)
                    ts["_x"] = np.log10(1 + ts[feat].clip(lower=0)) if logx else ts[feat]
                    show_plot(px.histogram(ts, x="_x", color="Actual", nbins=60, barmode="overlay", opacity=0.65,
                                           color_discrete_map=cmap, labels={"_x": f"log10(1+{feat})" if logx else feat},
                                           title=f"Distribution of {feat} by actual class"), "dist_num")
                st.caption(f"Based on a random sample of {len(ts):,} held-out test records.")
        with t3:
            sel = metrics["selected_model"]
            cm = np.array(metrics["test"][sel]["confusion_matrix"])
            l, r = st.columns([1, 1.3])
            with l:
                fig = px.imshow(cm, text_auto=",", x=["Normal", "Malicious"], y=["Normal", "Malicious"],
                                color_continuous_scale="Blues", labels=dict(x="Predicted", y="Actual", color="Records"),
                                title=f"Confusion matrix - {sel}")
                show_plot(fig, "cm")
                tn, fp, fn, tp = cm.ravel()
                st.caption(f"False positives (normal flagged malicious): {fp:,} · False negatives (missed attacks): {fn:,}")
            with r:
                st.markdown("**Classification report (test split)**")
                rep = pd.DataFrame(metrics["classification_report_selected"]).T.reset_index().rename(columns={"index": "class"})
                for col in ("precision", "recall", "f1-score"):
                    rep[col] = rep[col].map(lambda v: f"{v:.4f}")
                rep["support"] = rep["support"].map(lambda v: f"{int(round(v)):,}")
                show_df(rep)
        with t4:
            if test_sample is None:
                st.warning("reports/test_sample.csv.gz is missing.")
            else:
                thr = st.slider("Flag threshold", 0.05, 0.95, 0.5, 0.05, key="flag_thr")
                ts = test_sample[test_sample["attack_score"] >= thr].copy()
                ts["risk_level"] = risk_level(ts["attack_score"].values, thr)
                ts["actual"] = ts[TARGET].map(LABEL_NAMES)
                lv = st.multiselect("Risk levels", ["High", "Medium"], default=["High", "Medium"])
                ts = ts[ts["risk_level"].isin(lv)].sort_values("attack_score", ascending=False)
                keycols = list(importance["feature"].head(5)) if importance is not None else []
                shown = ts[["attack_score", "risk_level", "actual", ATTACK_CAT, *keycols]].head(500)
                st.write(f"{len(ts):,} flagged records in the sample (showing up to 500). Columns `actual` and `{ATTACK_CAT}` are the dataset's ground truth, shown for validation.")
                show_df(shown)
                if len(ts):
                    prec = (ts[TARGET] == 1).mean()
                    st.caption(f"{pct(prec)} of the flagged sample records are truly malicious at this threshold.")
        with t5:
            res = st.session_state.get("last_result")
            if res is None:
                st.info("Run a prediction on the 'CSV Traffic Analysis' page to see analytics for your own file here.")
            else:
                flagged = res[res["prediction"] == "Malicious"].sort_values("attack_score", ascending=False)
                st.write(f"{len(flagged):,} of {len(res):,} uploaded records were flagged.")
                keycols = [c for c in (list(importance["feature"].head(5)) if importance is not None else []) if c in res.columns]
                show_df(flagged[["row_number", "attack_score", "risk_level", *keycols]].head(500))


# --------------------------------------------------------------------------- #
# D. Model Explainability
# --------------------------------------------------------------------------- #
elif page == "Model Explainability":
    st.subheader("Which features does the Random Forest rely on?")
    if need_reports(importance):
        top_n = st.slider("Number of features", 5, min(30, len(importance)), 15)
        top = importance.head(top_n).iloc[::-1]
        l, r = st.columns(2)
        with l:
            show_plot(px.bar(top, x="impurity_importance", y="feature", orientation="h",
                             title="Impurity importance (summed over encoded columns)",
                             color_discrete_sequence=[ACCENT]), "imp1")
        with r:
            if "permutation_importance_f1_drop" in importance.columns:
                p = importance.sort_values("permutation_importance_f1_drop", ascending=False).head(top_n).iloc[::-1]
                show_plot(px.bar(p, x="permutation_importance_f1_drop", y="feature", orientation="h",
                                 title="Permutation importance (drop in F1 when shuffled)",
                                 color_discrete_sequence=[MALICIOUS_COLOR]), "imp2")
            else:
                st.info("Permutation importance was not computed for this training run.")
        st.markdown("#### Most influential features")
        rows = []
        for _, rw in importance.head(8).iterrows():
            row = {"Feature": rw["feature"], "Meaning": FEATURE_HELP.get(rw["feature"], "-"),
                   "Impurity importance": f"{rw['impurity_importance']:.4f}"}
            if test_sample is not None and rw["feature"] in test_sample.columns and test_sample[rw["feature"]].dtype != object:
                med = test_sample.groupby(TARGET)[rw["feature"]].median()
                row["Median (normal)"] = f"{med.get(0, float('nan')):.4g}"
                row["Median (malicious)"] = f"{med.get(1, float('nan')):.4g}"
            rows.append(row)
        show_df(pd.DataFrame(rows))
        st.info(
            "**How to read this.** Importance shows how much the model *uses* a feature to separate the classes in this dataset. "
            "It is an association learned from data, **not evidence that the feature causes an attack**. Impurity importance can favour "
            "continuous or high-cardinality features, and correlated features share credit. Permutation importance (right) is measured "
            "on held-out data, but it can understate features whose information is duplicated by other (correlated) features, e.g. `sttl` and `ct_state_ttl`. Dataset-specific artefacts (for example TTL values that reflect how the "
            "UNSW testbed was generated) may inflate the importance of some features and may not hold on real networks."
        )


# --------------------------------------------------------------------------- #
# E. Project Information
# --------------------------------------------------------------------------- #
else:
    st.subheader("Project information")
    st.markdown("#### Architecture & workflow")
    st.graphviz_chart(
        """digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#14263f", fontcolor="white", color="#4da3ff"];
        edge [color="#7f9cc0"];
        "UNSW-NB15 CSV" -> "Clean & validate" -> "Stratified split" -> "Preprocessing pipeline\\n(impute, log, scale, one-hot)"
        -> "Logistic Regression\\nRandom Forest" -> "CV selection + test evaluation" -> "Joblib artifacts + SHA-256";
        "Joblib artifacts + SHA-256" -> "predict.py" -> "Streamlit dashboard";
        "Uploaded CSV" -> "predict.py"; }"""
    )
    st.markdown(
        """
**Dataset.** UNSW-NB15, created by the Australian Centre for Cyber Security (ACCS), UNSW Canberra. Moustafa, N. & Slay, J. (2015),
*UNSW-NB15: a comprehensive data set for network intrusion detection systems*, MilCIS 2015. Kaggle mirror:
https://www.kaggle.com/datasets/ucimachinelearning/unsw-nb15-dataset. Use is subject to the dataset's own terms.

**Algorithms.** Logistic Regression (linear baseline) and Random Forest (non-linear ensemble). Both use class weighting. The model is
selected by 3-fold cross-validated F1 on the malicious class using the training split only, then evaluated once on the held-out test split.

**Preprocessing.** Median imputation, signed `log1p` + standardisation for numeric features, most-frequent imputation + one-hot
encoding (rare/unseen categories grouped) for `proto`, `service`, `state`. `id`, `label` and `attack_cat` are excluded from the features
to prevent target leakage. Duplicate records were removed *before* splitting so identical rows cannot appear in both train and test.

**Limitations.**
- UNSW-NB15 is a synthetic testbed capture from 2015; real-network performance will be lower and attack styles have changed.
- Results come from one stratified split of one file; they are not a guarantee of future performance.
- The attack score is not a calibrated probability. The false-positive rate matters: at scale, even a few percent of false alarms is a lot of alerts.
- Flow-level features only; no payload inspection, no encrypted-traffic or concept-drift handling.

**Ethical use.** This is a defensive research/education tool. Use it only on traffic you own or are authorised to analyse, treat
flagged records as leads for human review rather than verdicts, and do not use it to take automated punitive action.

**Reproduce.**
```bash
pip install -r requirements.txt
# place UNSW_NB15_testing-set.csv (or training-set) in data/
python train_model.py --data data/UNSW_NB15_testing-set.csv
python -m unittest discover -s tests -v   # or: pytest
streamlit run app.py
```
"""
    )
    if summary:
        st.caption(f"Current artifacts were trained from `{summary['source_file']}` with seed {summary['split']['seed']}.")
