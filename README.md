# 🛡️ ML Network Intrusion Detection System (NIDS)

A defensive, machine-learning web application that classifies network-flow records as **Normal** or **Potentially Malicious**, shows honest evaluation metrics, and explains which features drive the model. Built with Python, scikit-learn and Streamlit on the **UNSW-NB15** dataset.

> Research / education project (B.Tech CSE - AI & ML). It is **not** a production IDS. See *Limitations & ethical use*.

| Dashboard overview | Security analytics | Explainability |
|---|---|---|
| _screenshot placeholder_ `docs/overview.png` | _screenshot placeholder_ `docs/analytics.png` | _screenshot placeholder_ `docs/explainability.png` |

## Results (actual, from `python train_model.py`)

Data: `UNSW_NB15_testing-set.csv` (175,341 rows) -> **107,740** rows after removing 67,601 duplicate records (identical features, ignoring the row `id`). Stratified 80/20 split (seed 42): 86,192 train / 21,548 test. Test rows were never used for training or model selection.

| Model (test split, malicious = positive) | Accuracy | Precision | Recall | F1 | False-positive rate | ROC AUC | CV F1 (train only) |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 91.02% | 88.35% | 95.23% | 91.66% | 13.52% | 0.9739 | 0.9160 |
| **Random Forest (selected)** | **93.17%** | **91.33%** | **95.93%** | **93.57%** | **9.80%** | **0.9856** | **0.9326** |

Random Forest confusion matrix (rows = actual, columns = predicted): Normal -> [9,361 correct, 1,017 false alarms]; Malicious -> [455 missed, 10,715 detected].
Per-category correct-classification rates on the test split: Reconnaissance 99.9%, Backdoor 100%\*, Worms 100%\*, DoS 99.6%, Generic 99.6%, Exploits 99.5%, Shellcode 99.5%, **Fuzzers 88.2%, Analysis 86.2%, Normal 90.2%**. (\*very small test counts: 307 and 26.) Full numbers: `reports/metrics.json`.

Interpretation: the model finds ~96% of malicious flows but raises a false alarm on ~10% of normal flows - far too many for real deployment. Results come from one synthetic 2015 testbed and one split; do not expect them to transfer to real networks.

## Features

- **Overview** - dataset size, real test metrics, class mix, model comparison
- **CSV Traffic Analysis** - upload flows, column validation, preview, adjustable threshold, risk levels, downloadable results
- **Security Analytics** - traffic/attack-category charts, per-category results, feature distributions, confusion matrix, classification report, flagged-record table
- **Model Explainability** - impurity + permutation importance with plain-language caveats (association, not causation)
- **Project Information** - architecture, dataset attribution, algorithms, limitations, reproduction steps

## Project structure

```
network-intrusion-detection/
├── app.py                 # Streamlit dashboard
├── train_model.py         # cleaning, split, CV selection, evaluation, artifacts, plots
├── predict.py             # verified model loading + prediction (also a CLI)
├── common.py              # schema constants, data validation/cleaning, pipeline factory
├── requirements.txt       # pinned to the training environment
├── .streamlit/config.toml # theme + upload limit
├── data/                  # raw CSV goes here (git-ignored); sample_traffic.csv is committed
├── models/                # nids_pipeline.joblib, random_forest_pipeline.joblib, metadata.json (SHA-256)
├── reports/               # metrics.json, dataset_summary.json, feature_importance.csv, test_sample.csv.gz, figures/
└── tests/                 # 40 unit tests (synthetic data; no dataset needed)
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) get the data (not committed): https://www.kaggle.com/datasets/ucimachinelearning/unsw-nb15-dataset
#    put UNSW_NB15_testing-set.csv (or the training-set file) into data/

python train_model.py --data data/UNSW_NB15_testing-set.csv   # ~2-5 min on a laptop
python -m unittest discover -s tests -t . -v                  # or: pytest
streamlit run app.py
python predict.py data/sample_traffic.csv -o predictions.csv  # CLI prediction
```

The repository already contains a trained model and reports, so `streamlit run app.py` works without the dataset. Retrain only if you want to reproduce or change the results. If you change scikit-learn versions, retrain (the loader refuses mismatched versions).

## How it works

1. **Load & validate** - required columns, 0/1 target, `label` ↔ `attack_cat` consistency.
2. **Clean** - strip text, replace ±inf, drop null targets, **drop duplicates before splitting** (otherwise identical flows land in both train and test).
3. **Leakage control** - `id`, `label` and `attack_cat` are never features (39 numeric + 3 categorical remain).
4. **Preprocess** (scikit-learn `ColumnTransformer`, fitted on train only) - median impute → signed log1p → standardise; most-frequent impute → one-hot (`proto`, `service`, `state`; rare/unseen values grouped).
5. **Models** - Logistic Regression baseline and Random Forest, both class-weighted.
6. **Selection** - 3-fold stratified CV (malicious-class F1) on the training split only; test set scored once.
7. **Artifacts** - fitted pipelines saved with Joblib; `metadata.json` stores feature schema, library versions and SHA-256 hashes. `predict.py` refuses files outside `models/`, unregistered files, hash mismatches and incompatible scikit-learn versions (Joblib files are pickles - never load model files from untrusted sources).

The *attack score* shown in the app is the model output for the malicious class. For the Random Forest it is a vote share, **not a calibrated probability**.

## Publish to GitHub

```bash
cd network-intrusion-detection
git init -b main
git add .
git commit -m "Network intrusion detection system"
# create an EMPTY repo on github.com (no README/.gitignore), then:
git remote add origin https://github.com/<your-username>/network-intrusion-detection.git
git push -u origin main
```
Check `git status` first: the raw dataset in `data/` must not be staged (it is git-ignored). Never commit credentials.

## Deploy on Streamlit Community Cloud

1. Push the repo to GitHub (above) - public, or private with Streamlit authorised for your account.
2. Go to https://share.streamlit.io and sign in with GitHub; authorise access to the repo when prompted.
3. Click **Create app → Deploy a public app from GitHub**.
4. Repository: `<your-username>/network-intrusion-detection`, branch `main`, main file path `app.py`.
5. Open **Advanced settings** and choose **Python 3.12**.
6. Click **Deploy**. The first build installs `requirements.txt` (a few minutes). The app needs no dataset: it uses the committed `models/`, `reports/` and `data/sample_traffic.csv`.
7. Test it: open the URL, go to *CSV Traffic Analysis* and use the bundled sample file.

If the build fails with a scikit-learn version message, either keep the pinned `requirements.txt` or retrain locally and commit the new `models/` and `reports/`.

## Limitations & ethical use

- UNSW-NB15 is a synthetic 2015 testbed capture; real traffic, modern attacks and encrypted traffic behave differently.
- One file, one stratified split, one seed - no confidence intervals across runs or temporal validation.
- ~10% false-positive rate on normal traffic; strongly TTL-related features (`sttl`, `ct_state_ttl`) dominate and may reflect how the testbed was generated.
- Flow-level features only; no payload inspection, no concept-drift handling.
- Use only on traffic you own or are authorised to analyse. Treat flags as leads for human review, never as grounds for automated punitive action. No offensive capability is implemented.

## Attribution

Moustafa, N., & Slay, J. (2015). *UNSW-NB15: a comprehensive data set for network intrusion detection systems.* MilCIS 2015, UNSW Canberra (ACCS). Kaggle mirror: https://www.kaggle.com/datasets/ucimachinelearning/unsw-nb15-dataset. Follow the dataset's own terms of use.
