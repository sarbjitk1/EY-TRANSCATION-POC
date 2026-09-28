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
| `src/pipeline.py` | Label cleaning, schema coercion, feature engineering, sklearn preprocessor (shared by training and inference); `DROPPED_COLS` lists the columns the model ignores |
| `train.py` | Dedupe → split → 5-fold CV comparison of 4 models → grid search → hold-out evaluation → artifacts |
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
level (0.947 → 0.946), so we left them out. Uploads still need all seven columns, and both columns still
appear in the dashboard's Data exploration charts. The dashboard shows a note saying they are ignored.

**Features** — everything inside one sklearn `Pipeline`, so the upload path runs identical code
(6,217 model features in total):
- Col1/Col4/Col6 (anonymised word tokens): TF-IDF, whitespace tokens, uni+bigrams (6,206 features; Col4 alone is 4,371).
- Col3: parsed float → signed log1p, negative & round-number flags. Outliers kept (they are real).
- Col7: one-hot (values seen only once are grouped). Unknown categories ignored at inference.
- Word counts for Col1/Col4/Col6 and a Col4-missing flag. Numeric features are median-filled and standard-scaled.

**Split & validation**
- Stratified 90/10 hold-out (seed 42), used exactly once at the end.
- 5-fold stratified CV inside the 90% for both model comparison and tuning — a fixed validation set would
  contain only 1–2 examples of the rare classes.
- Selection on **macro-F1** (always predicting Category_1 already gives 89% accuracy).

**Results** (5-fold CV on train → then hold-out)

| Model | CV accuracy | CV macro-F1 |
|---|---|---|
| **Logistic Regression (balanced, C=10)** | 0.946 | **0.784** |
| Random Forest | 0.950 | 0.665 |
| Hist Gradient Boosting | 0.935 | 0.603 |
| Soft-voting ensemble | 0.949 | 0.646 |

Hold-out (522 rows): **accuracy 96.4%, macro-F1 0.898, balanced accuracy 0.981**. Hold-out rare-class
support is tiny (1–2 rows), so its macro-F1 is noisy — the CV macro-F1 (0.78) is the more reliable estimate.
Category_5 has only 2 rows in total, both in training, so it cannot be evaluated.

LightGBM was not used: it needs the system `libomp` library, a fragile dependency for a laptop demo;
scikit-learn's `HistGradientBoostingClassifier` is the same algorithm family.

**Explainer** — on demand only. The OpenAI agent chooses among three tools: occlusion attributions (neutralise
each column / word token and measure the change in predicted-class probability — model-agnostic), nearest
labelled training records, and class profiles. Its tool calls are shown in the UI for transparency.
