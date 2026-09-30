# Transaction Classification — EY Data Science Challenge

Supervised classifier for `ClassificationLabel` (Column H) from Columns A–G, with a Streamlit dashboard:
**Tab 1** walks through exploration → preprocessing → split → model selection → hold-out results;
**Tab 2** runs the full pipeline live on an uploaded file and explains any row on demand with an AI agent.

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python train.py            # ~2 min: writes data/ and artifacts/
.venv/bin/streamlit run app.py       # opens http://localhost:8501
```

OpenAI key for the explainer (any one of):
`export OPENAI_API_KEY=sk-...` · copy `.streamlit/secrets.toml.example` → `.streamlit/secrets.toml` · paste in the sidebar.
Without a key the explainer falls back to a rule-based narrative built from the same evidence.

**Live demo:** Tab 2 → upload `data/holdout_test.csv` (Columns A–G). Upload
`data/holdout_test_with_labels.csv` instead to also see live accuracy against the true labels.

## Project layout

| File | Purpose |
|---|---|
| `src/pipeline.py` | Label cleaning, schema coercion, feature engineering, sklearn preprocessor, macro-F1 definition and the Category_2 cut-off rule (shared by training and inference); `DROPPED_COLS` lists the columns the model ignores |
| `train.py` | Dedupe → split → repeated-CV comparison of 4 models → tuning → Category_2 cut-off → unseen-vendor test → hold-out check → artifacts |
| `src/explain.py` | Occlusion attributions, nearest-neighbour precedent, class profiles, OpenAI tool-calling agent |
| `app.py` | Streamlit dashboard |
| `artifacts/` | `model.joblib` (fitted end-to-end pipeline), `metrics.json`, `train_log.txt` |
| `data/` | `train.csv`, `holdout_test.csv`, `holdout_test_with_labels.csv` |

## Key decisions (and why)

**Data issues found**
- 11 label spellings (`Category 3`, `Categry_6`, `category_1`, `Category _3`…) → 6 classes via regex on the digit.
- 679 exact duplicate rows → removed **before** splitting; otherwise copies of test rows sit in training (leakage).
- Severe imbalance: Category_1 ≈ 89%; Categories 3–6 have 2–23 rows each.
- Col2 IDs are near-unique and some were mangled by Excel into scientific notation (`4.80Z+11`).
- 153 missing Col4; amounts stored as text with thousands separators; negative amounts (credits).

**Columns not used by the model: Col2 (reference ID) and Col5 (posting date).**
We trained with and without them. Without them, CV macro-F1 rose from 0.743 to 0.784 and accuracy stayed
level (0.947 → 0.946) in a single 5-fold run, so we left them out. Uploads still need all seven columns, and both columns still
appear in the dashboard's Data exploration charts. The dashboard shows a note saying they are ignored.

**Features** — everything inside one sklearn `Pipeline`, so the upload path runs identical code
(6,217 model features in total):
- Col1/Col4/Col6 (anonymised word tokens): TF-IDF, whitespace tokens, uni+bigrams (6,206 features; Col4 alone is 4,371).
- Col3: parsed float → signed log1p, negative & round-number flags. Outliers kept (they are real).
- Col7: one-hot (values seen only once are grouped). Unknown categories ignored at inference.
- Word counts for Col1/Col4/Col6 and a Col4-missing flag. Numeric features are median-filled and standard-scaled.

**Split & validation**
- Stratified 90/10 hold-out (seed 42), set aside before anything else and scored once at the end.
- Everything else uses the 90%: 5-fold stratified CV, **repeated 3 times** with different shuffles. Each training
  row gets a prediction from a model that never saw it, and each repeat is scored once over all rows.
  Averaging per-fold scores was too noisy: a fold holds 0–1 rows of Category_5, so one row moved macro-F1 by ~0.1.
- Every model and setting sees the same splits, so they are compared repeat by repeat.
- Selection on **macro-F1** (always predicting Category_1 already gives 89% accuracy). One definition
  everywhere: the average F1 over the categories present in the rows being scored.

**Results** (cross-validated on the 90% training split, average of 3 repeats)

| Model | CV accuracy | CV macro-F1 |
|---|---|---|
| **Logistic Regression (balanced, C=10)** | 0.946 | **0.771** |
| Random Forest | 0.952 | 0.591 |
| Hist Gradient Boosting | 0.936 | 0.586 |
| Soft-voting ensemble | 0.950 | 0.615 |

Logistic regression wins on every repeat. Tuning tried C = 1, 3, 10, 30, 100; C = 10 was best (C = 30 within 0.002).

**Category_1 vs Category_2 cut-off.** Most mistakes are Category_1 and Category_2 being confused, and the model
leans toward Category_2. It now picks Category_2 only when that category's probability is **at least 75%**;
otherwise it takes the next most likely category. Other categories are unaffected. On the training rows this cuts
Category_1 → 2 mistakes from 163 to 94 per repeat, while Category_2 → 1 mistakes rise from 58 to 92.
To avoid grading the cut-off on the rows it was tuned on, the headline scores choose it inside each fold.
Measured that way, it adds **+0.010 macro-F1 and +0.4 pts accuracy**: a small, honest gain.
Turning down the class weighting instead was worse because it also hurt the rare categories.
The dashboard has a slider showing the trade-off, in case one kind of mistake costs the client more.

**Final model, cross-validated:** macro-F1 **0.781 ± 0.048**, accuracy **95.1%**, balanced accuracy 0.824.
Per category F1: Category_1 0.97, Category_2 0.78, Category_3 0.91, Category_4 0.44, Category_5 0.92,
Category_6 0.67. The ± is real: Categories 3–6 have 2–21 rows each, so a few rows swing macro-F1 by up to 0.1
between repeats. Category_4 is the weakest category.

**Vendors the model has never seen.** 98% of hold-out rows come from a vendor that is also in training, so the
scores above describe known vendors. With each vendor kept entirely on one side of the split, macro-F1 drops to
**0.36** and accuracy to 91.2%, barely above always guessing Category_1 (89.2%). Most rare categories come from
only 1–8 vendors, so this is a limit of the data, not of tuning. The prediction pipeline marks rows from unknown
vendors so they can be reviewed by hand.

**Hold-out check** (522 rows): accuracy 97.3%, macro-F1 0.905, balanced accuracy 0.970. Category_5 has no
rows here and the other rare categories have 1–2, so use the cross-validated results as the main estimate.

LightGBM was not used: it needs the system `libomp` library, a fragile dependency for a laptop demo;
scikit-learn's `HistGradientBoostingClassifier` is the same algorithm family.

**Explainer** — on demand only. The OpenAI agent chooses among three tools: occlusion attributions (neutralise
each column / word token and measure the change in predicted-class probability — model-agnostic), nearest
labelled training records, and class profiles. Its tool calls are shown in the UI for transparency.
