"""EY classification challenge - interactive dashboard.

Run:  .venv/bin/streamlit run app.py
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.explain import fallback_explanation, feature_contributions, predict_frame, run_agent
from src.pipeline import (COLUMN_DESCRIPTIONS, INPUT_COLS, LABEL_COL, NUMERIC_FEATURES, coerce_schema,
                          engineer_features, id_shape, normalize_label, parse_amount)

ROOT = Path(__file__).parent
st.set_page_config(page_title="Transaction Classifier", page_icon="📊", layout="wide")

# Validated categorical order (one fixed hue per class, never cycled) + recessive chart chrome.
CLASS_COLORS = {"Category_1": "#2a78d6", "Category_2": "#eb6834", "Category_3": "#1baf7a",
                "Category_4": "#eda100", "Category_5": "#e87ba4", "Category_6": "#008300"}
CLASS_ORDER = list(CLASS_COLORS)
SEQ = "Blues"


def style(fig, height=340):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(size=13), legend_title_text="", hoverlabel=dict(font_size=12))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.07)")
    return fig


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_bundle():
    return joblib.load(ROOT / "artifacts" / "model.joblib")


@st.cache_data
def load_metrics():
    return json.loads((ROOT / "artifacts" / "metrics.json").read_text())


@st.cache_data
def load_raw():
    df = pd.read_csv(ROOT / "Challenge_Data.csv", dtype=str)
    df["label"] = df[LABEL_COL].map(normalize_label)
    eng = engineer_features(df[INPUT_COLS])
    df["amount"] = parse_amount(df["Col3"])
    df["date"] = pd.to_datetime(df["Col5"], errors="coerce", format="mixed")
    df["col2_shape"] = eng["col2_shape"]
    return df, eng


if not (ROOT / "artifacts" / "model.joblib").exists():
    st.error("No trained model found. Run `.venv/bin/python train.py` first.")
    st.stop()

bundle, M = load_bundle(), load_metrics()
raw, eng_all = load_raw()

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Model")
    st.metric("Selected model", M["best_family"])
    st.metric("Hold-out accuracy", f"{M['holdout']['accuracy']:.1%}")
    st.metric("Hold-out macro-F1", f"{M['holdout']['f1_macro']:.3f}")
    st.caption(f"Tuned params: {M['best_params']}")
    st.divider()
    st.header("AI explainer")
    default_key = os.environ.get("OPENAI_API_KEY", "")
    try:
        default_key = default_key or st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        pass
    api_key = st.text_input("OpenAI API key", value=default_key, type="password",
                            help="Read from OPENAI_API_KEY or .streamlit/secrets.toml if set. "
                                 "Without a key a rule-based explanation is shown.")
    llm_model = st.text_input("OpenAI model", value=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    st.caption("🟢 Agent enabled" if api_key else "⚪ No key - rule-based fallback")

st.title("Transaction Classification - EY Challenge")
tab1, tab2 = st.tabs(["① Exploration, Preprocessing & Model Results", "② Hold-Out Prediction Pipeline"])

# =========================================================================== #
# TAB 1
# =========================================================================== #
with tab1:
    d = M["data"]
    k = st.columns(5)
    k[0].metric("Raw rows", f"{d['raw_rows']:,}")
    k[1].metric("Exact duplicates removed", f"{d['duplicates_removed']:,}")
    k[2].metric("Classes (after cleaning)", raw["label"].nunique(), delta=f"from {raw[LABEL_COL].nunique()} spellings",
                delta_color="off")
    k[3].metric("Missing Col4", int(raw["Col4"].isna().sum()))
    k[4].metric("Majority-class baseline", f"{M['holdout']['majority_baseline_accuracy']:.1%}",
                help="Accuracy from always predicting Category_1 - the bar to beat.")

    # ---------------------------------------------------------------- EDA
    st.header("1 · Data exploration")
    c1, c2 = st.columns(2)
    with c1:
        vc = raw["label"].value_counts().reindex(CLASS_ORDER).fillna(0).reset_index()
        vc.columns = ["label", "rows"]
        log_y = st.toggle("Log scale", value=True, key="logcls")
        fig = px.bar(vc, x="label", y="rows", color="label", color_discrete_map=CLASS_COLORS, text="rows",
                     log_y=log_y, title="Class distribution - severe imbalance")
        fig.update_traces(textposition="outside", marker_cornerradius=4)
        st.plotly_chart(style(fig).update_layout(showlegend=False), width="stretch")
    with c2:
        lm = pd.DataFrame(d["label_map"]).rename(columns={LABEL_COL: "raw label", "label": "cleaned"})
        st.markdown("**Label cleaning** - 11 spellings collapse to 6 classes (regex on the digit)")
        st.dataframe(lm, hide_index=True, width="stretch", height=300)

    st.subheader("Explore a column against the target")
    col_pick = st.selectbox("Column", INPUT_COLS + ["Col2 shape (engineered)"],
                            format_func=lambda c: f"{c} - {COLUMN_DESCRIPTIONS.get(c, 'reference-ID format')}")
    c1, c2 = st.columns([3, 2])
    with c1:
        if col_pick == "Col3":
            show = raw.dropna(subset=["amount"]).assign(abs_amount=lambda x: x["amount"].abs().clip(lower=0.01))
            fig = px.box(show, x="label", y="abs_amount", color="label", log_y=True, points="outliers",
                         color_discrete_map=CLASS_COLORS, category_orders={"label": CLASS_ORDER},
                         title="|Amount| by class (log scale)")
            st.plotly_chart(style(fig, 380).update_layout(showlegend=False), width="stretch")
        elif col_pick == "Col5":
            ct = pd.crosstab(raw["date"].dt.to_period("M").astype(str), raw["label"], normalize="index")
            fig = px.imshow(ct.T.reindex(CLASS_ORDER).fillna(0), color_continuous_scale=SEQ, aspect="auto",
                            text_auto=".0%", title="Class mix by posting month (share of month)")
            st.plotly_chart(style(fig, 380), width="stretch")
        else:
            src = "col2_shape" if col_pick.startswith("Col2 shape") else col_pick
            topn = st.slider("Top N values", 5, 25, 12)
            top = raw[src].fillna("<missing>").value_counts().head(topn).index
            sub = raw[raw[src].fillna("<missing>").isin(top)].assign(v=lambda x: x[src].fillna("<missing>"))
            ct = sub.groupby(["v", "label"]).size().reset_index(name="rows")
            fig = px.bar(ct, y="v", x="rows", color="label", orientation="h", color_discrete_map=CLASS_COLORS,
                         category_orders={"label": CLASS_ORDER, "v": list(top)},
                         title=f"Top {topn} values of {src} - stacked by class")
            fig.update_traces(marker_line_color="white", marker_line_width=1)
            st.plotly_chart(style(fig, 420).update_yaxes(title=""), width="stretch")
    with c2:
        src = "col2_shape" if col_pick.startswith("Col2 shape") else col_pick
        purity = (raw.groupby(src)["label"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
                  .rename("purity").to_frame().join(raw[src].value_counts().rename("n")))
        w_purity = float((purity["purity"] * purity["n"]).sum() / purity["n"].sum())
        st.metric("Distinct values", f"{raw[src].nunique():,}")
        st.metric("Missing", int(raw[src].isna().sum()))
        st.metric("Weighted label purity", f"{w_purity:.1%}",
                  help="If you knew only this column's value, how often would its majority label be right?")
        if col_pick == "Col2":
            st.info("Excel mangled some IDs into scientific notation (`4.80Z+11`). Raw IDs are near-unique, "
                    "so the model uses their **shape** (`KBNZ072618` → `A9`) - which identifies a vendor's "
                    "numbering format without memorising IDs.")
        if col_pick == "Col3":
            q1, q3 = np.log1p(raw["amount"].abs()).quantile([.25, .75])
            out = ((np.log1p(raw["amount"].abs()) > q3 + 1.5 * (q3 - q1)) |
                   (np.log1p(raw["amount"].abs()) < q1 - 1.5 * (q3 - q1))).sum()
            st.metric("Negative amounts (credits)", int((raw["amount"] < 0).sum()))
            st.metric("IQR outliers on log|amount|", int(out))
            st.caption("Outliers are kept: large amounts are genuine (e.g. 8.8M payroll-style rows) and are "
                       "tamed by a signed log transform instead of being dropped.")

    c1, c2 = st.columns(2)
    with c1:
        miss = raw[INPUT_COLS].isna().sum().reset_index()
        miss.columns = ["column", "missing"]
        fig = px.bar(miss, x="column", y="missing", text="missing", title="Missing values per column")
        fig.update_traces(marker_color=CLASS_COLORS["Category_1"], marker_cornerradius=4)
        st.plotly_chart(style(fig), width="stretch")
    with c2:
        corr = eng_all[NUMERIC_FEATURES].assign(
            **{f"is_{c}": (raw["label"] == c).astype(float) for c in ["Category_1", "Category_2"]}).corr()
        fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                        title="Correlation - engineered numeric features vs top classes")
        st.plotly_chart(style(fig, 420), width="stretch")

    # ---------------------------------------------------------------- Preprocessing
    st.header("2 · Cleaning & preprocessing")
    steps = pd.DataFrame([
        ["Labels", "Regex-normalise 11 spellings → Category_1..6", "Typos would otherwise create fake classes"],
        ["Duplicates", f"Drop {d['duplicates_removed']} exact duplicate rows BEFORE splitting",
         "Prevents identical rows landing in both train and test (leakage)"],
        ["Schema", "Coerce any upload to Col1..Col7 (Col#, A–G, or positional); missing cols → blank",
         "Robust live pipeline"],
        ["Col1 / Col4 / Col6", "TF-IDF on whitespace tokens, uni+bigrams, sublinear tf", "Token IDs carry vendor/account semantics"],
        ["Col2", "Character-class shape → one-hot (rare shapes pooled); length, digit share, separator, sci-notation flags",
         "IDs are unique; their format is informative"],
        ["Col3", "Strip commas/$ → float; signed log1p; negative & round-number flags", "Heavy right skew, credits"],
        ["Col5", "Parse date → month, day, month-start / month-end flags", "Posting-period patterns"],
        ["Col7", "One-hot (unknown values ignored)", "Low-cardinality categorical"],
        ["Missing values", "Text → empty string; numeric → median imputation (fit on train only)", "No row is ever dropped at inference"],
        ["Scaling", "StandardScaler on numeric features (fit on train only)", "Comparable scale for the linear model"],
    ], columns=["Step", "Transformation", "Why"])
    st.dataframe(steps, hide_index=True, width="stretch")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Before → after** (first rows of the training split)")
        view = st.radio("View", ["Raw input", "Engineered features"], horizontal=True, label_visibility="collapsed")
        if view == "Raw input":
            st.dataframe(pd.read_csv(ROOT / "data" / "train.csv", dtype=str).head(8), hide_index=True,
                         width="stretch")
        else:
            st.dataframe(pd.DataFrame(M["engineered_sample"]), hide_index=True, width="stretch")
    with c2:
        dims = pd.Series(M["feature_dims"]).rename_axis("block").reset_index(name="features")
        fig = px.bar(dims, x="features", y="block", orientation="h", text="features",
                     title=f"Model input width: {dims['features'].sum():,} features")
        fig.update_traces(marker_color=CLASS_COLORS["Category_1"], marker_cornerradius=4)
        st.plotly_chart(style(fig, 300).update_yaxes(title=""), width="stretch")

    # ---------------------------------------------------------------- Split
    st.header("3 · Split strategy")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(f"""
- **Deduplicate first** ({d['raw_rows']:,} → {d['dedup_rows']:,} rows), then split - otherwise copies of a
  test row sit in training and inflate every score.
- **Stratified 90 / 10 split** ({d['train_rows']:,} train / {d['test_rows']:,} hold-out, seed 42) so rare
  classes appear in both parts in proportion. The hold-out is saved as `data/holdout_test.csv` and is
  **touched once**, after model selection.
- **Validation = 5-fold stratified CV inside the 90%**: used for model comparison *and* hyper-parameter tuning.
  With only 12–23 rows in some classes, a single fixed validation set would hold 1–2 examples per rare class;
  CV uses every training row for validation once and gives a variance estimate.
- **Selection metric = macro-F1** (every class weighs equally). Accuracy alone rewards ignoring rare classes -
  predicting Category_1 always already scores {M['holdout']['majority_baseline_accuracy']:.0%}.
- Category_5 has only **2 rows in the whole dataset**; both fell in training, so it cannot be evaluated on the hold-out.
""")
    with c2:
        sc = pd.DataFrame(d["split_counts"])
        st.dataframe(sc, hide_index=True, width="stretch")

    # ---------------------------------------------------------------- Models
    st.header("4 · Model training & evaluation")
    cv = pd.DataFrame(M["cv_comparison"])
    tbl = cv[["model", "accuracy_mean", "accuracy_std", "f1_macro_mean", "f1_macro_std",
              "balanced_accuracy_mean", "f1_weighted_mean", "fit_seconds"]].rename(columns={
        "accuracy_mean": "Accuracy", "accuracy_std": "± acc", "f1_macro_mean": "Macro-F1",
        "f1_macro_std": "± F1", "balanced_accuracy_mean": "Balanced acc", "f1_weighted_mean": "Weighted F1",
        "fit_seconds": "5-fold time (s)"})
    st.dataframe(tbl.style.format({c: "{:.3f}" for c in tbl.columns[1:-1]})
                 .highlight_max(subset=["Macro-F1"], color="#dbe9fa"), hide_index=True, width="stretch")

    c1, c2 = st.columns(2)
    metric = c1.radio("CV metric per fold", ["f1_macro", "accuracy", "balanced_accuracy"], horizontal=True)
    folds = pd.DataFrame([{"model": r["model"], "fold": i + 1, "score": s}
                          for r in M["cv_comparison"] for i, s in enumerate(r[f"{metric}_folds"])])
    fig = px.strip(folds, x="model", y="score", hover_data=["fold"], title=f"5-fold CV {metric} - each dot is a fold")
    fig.update_traces(marker=dict(size=11, color=CLASS_COLORS["Category_1"], line=dict(width=2, color="white")))
    means = folds.groupby("model")["score"].mean()
    fig.add_trace(go.Scatter(x=means.index, y=means.values, mode="markers", name="mean",
                             marker=dict(symbol="line-ew-open", size=40, color="#0b0b0b", line_width=2)))
    c1.plotly_chart(style(fig, 360), width="stretch")
    with c2:
        st.markdown(f"**Why {M['best_family']}?**")
        st.markdown(
            "- Best macro-F1: the TF-IDF token space is high-dimensional and sparse - linear models with "
            "`class_weight='balanced'` separate rare classes well where trees need more examples per leaf.\n"
            "- Tree models / ensemble edge ahead on raw accuracy but lose the rare classes.\n"
            "- Fast (seconds to train), stable across folds, and its probabilities are well-behaved for explanation.")
        st.markdown("**Hyper-parameter tuning** (GridSearchCV, same 5 folds, refit on macro-F1)")
        st.dataframe(pd.DataFrame([{**t["params"], "Macro-F1": round(t["f1_macro"], 4),
                                    "Accuracy": round(t["accuracy"], 4)} for t in M["tuning"]]),
                     hide_index=True, width="stretch")

    st.subheader("Hold-out results (10%, never seen during training or tuning)")
    H = M["holdout"]
    k = st.columns(4)
    k[0].metric("Accuracy", f"{H['accuracy']:.2%}")
    k[1].metric("Macro-F1 (classes present)", f"{H['f1_macro']:.3f}")
    k[2].metric("Balanced accuracy", f"{H['balanced_accuracy']:.3f}")
    k[3].metric("Weighted F1", f"{H['f1_weighted']:.3f}")

    c1, c2 = st.columns(2)
    with c1:
        norm = st.toggle("Row-normalise (recall per class)", value=False)
        cm = np.array(H["confusion_matrix"], dtype=float)
        present = [i for i, l in enumerate(H["labels"]) if cm[i].sum() > 0 or cm[:, i].sum() > 0]
        cm = cm[np.ix_(present, present)]
        labs = [H["labels"][i] for i in present]
        if norm:
            cm = cm / cm.sum(1, keepdims=True).clip(min=1)
        fig = px.imshow(cm, x=labs, y=labs, color_continuous_scale=SEQ, text_auto=".2f" if norm else "d",
                        labels=dict(x="Predicted", y="Actual", color="share" if norm else "rows"),
                        title="Confusion matrix")
        st.plotly_chart(style(fig, 420), width="stretch")
    with c2:
        rep = pd.DataFrame(H["report"]).T
        rep = rep[rep["support"] > 0].drop(index=["accuracy"], errors="ignore")
        st.markdown("**Classification report** (classes with hold-out support)")
        st.dataframe(rep.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}",
                                       "support": "{:.0f}"}), width="stretch")
        imp = pd.DataFrame(M["column_importance"]).T.reset_index(names="column").sort_values("mean")
        fig = px.bar(imp, x="mean", y="column", error_x="std", orientation="h",
                     title="Which raw columns matter? (permutation: drop in hold-out macro-F1)")
        fig.update_traces(marker_color=CLASS_COLORS["Category_1"], marker_cornerradius=4)
        st.plotly_chart(style(fig, 320).update_xaxes(title="macro-F1 drop when shuffled"), width="stretch")

# =========================================================================== #
# TAB 2
# =========================================================================== #
with tab2:
    st.markdown("Upload the hold-out file (**Columns A–G**; a label column is optional). The full preprocessing "
                "and inference pipeline runs live on every upload.")
    c1, c2 = st.columns([3, 1])
    up = c1.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    with c2:
        st.caption("Sample files")
        for f in ["holdout_test.csv", "holdout_test_with_labels.csv"]:
            p = ROOT / "data" / f
            if p.exists():
                st.download_button(f, p.read_bytes(), file_name=f, width="stretch")

    if up is not None:
        content = up.getvalue()
        key = hashlib.md5(content).hexdigest()
        if st.session_state.get("file_key") != key:
            st.session_state.update(file_key=key, explanations={})
            log, t_all = [], time.time()
            with st.status("Running pipeline…", expanded=True) as status:
                try:
                    t = time.time()
                    if up.name.lower().endswith((".xlsx", ".xls")):
                        df_up = pd.read_excel(io.BytesIO(content), dtype=str)
                    else:
                        df_up = pd.read_csv(io.BytesIO(content), dtype=str, encoding_errors="replace",
                                            sep=None, engine="python")
                    st.write(f"✅ Read **{len(df_up):,} rows × {df_up.shape[1]} cols** ({time.time()-t:.2f}s)")
                    known = {c.lower() for c in INPUT_COLS} | set("abcdefg") | {LABEL_COL.lower()}
                    if not any(str(c).strip().lower().replace(" ", "") in known for c in df_up.columns):
                        # No recognisable header -> the first row is data, not column names.
                        reader = pd.read_excel if up.name.lower().endswith((".xlsx", ".xls")) else pd.read_csv
                        df_up = reader(io.BytesIO(content), dtype=str, header=None)
                        st.write("ℹ️ No header row detected - treating the first row as data.")
                    if df_up.empty:
                        raise ValueError("The file has no data rows.")
                    df_up = df_up.dropna(how="all")

                    t = time.time()
                    X_up, warns = coerce_schema(df_up)
                    for w in warns:
                        st.warning(w)
                    st.write(f"✅ Schema validated → Col1..Col7 ({time.time()-t:.2f}s)")

                    t = time.time()
                    eng = engineer_features(X_up)
                    bad_amt = int(eng["amount_signed_log"].isna().sum() - X_up["Col3"].isna().sum())
                    bad_dt = int(eng["month"].isna().sum() - X_up["Col5"].isna().sum())
                    st.write(f"✅ Cleaned & engineered {eng.shape[1]} features ({time.time()-t:.2f}s)"
                             + (f" - {bad_amt} unparseable amounts, {bad_dt} unparseable dates → imputed"
                                if bad_amt or bad_dt else ""))

                    t = time.time()
                    preds = predict_frame(bundle, X_up)
                    st.write(f"✅ Encoded (TF-IDF / one-hot / scaling) and scored with **{bundle['model_name']}** "
                             f"({time.time()-t:.2f}s)")
                    truth = None
                    lab_col = next((c for c in df_up.columns if c.strip().lower() in
                                    {LABEL_COL.lower(), "label", "h", "col8"}), None)
                    if lab_col:
                        truth = df_up[lab_col].map(normalize_label).values
                        preds.insert(2, "Actual", truth)
                    st.session_state.update(preds=preds.reset_index(drop=True), elapsed=time.time() - t_all)
                    status.update(label=f"Pipeline complete - {len(preds):,} rows in {time.time()-t_all:.2f}s",
                                  state="complete", expanded=False)
                except Exception as e:
                    st.session_state.pop("preds", None)
                    status.update(label="Pipeline failed", state="error")
                    st.error(f"Could not process this file: {e}")

    preds = st.session_state.get("preds")
    if up is not None and preds is not None:
        k = st.columns(4)
        k[0].metric("Rows scored", f"{len(preds):,}")
        k[1].metric("Pipeline time", f"{st.session_state['elapsed']:.2f}s")
        k[2].metric("Low-confidence rows (<70%)", int((preds["Confidence"] < 0.7).sum()))
        if "Actual" in preds and preds["Actual"].notna().any():
            m = preds["Actual"].notna()
            k[3].metric("Live accuracy vs provided labels", f"{(preds.loc[m, 'Actual'] == preds.loc[m, 'Predicted']).mean():.2%}")

        f1, f2 = st.columns([2, 1])
        cls_filter = f1.multiselect("Filter predicted class", CLASS_ORDER,
                                    default=[c for c in CLASS_ORDER if c in set(preds["Predicted"])])
        only_low = f2.toggle("Only low-confidence rows")
        view = preds[preds["Predicted"].isin(cls_filter)]
        if only_low:
            view = view[view["Confidence"] < 0.7]

        st.markdown("**Predictions** - click a row, then press **Explain** (≈25 rows visible, scroll for more)")
        show_cols = ["Predicted", "Confidence"] + (["Actual"] if "Actual" in view else []) + INPUT_COLS
        event = st.dataframe(
            view[show_cols], height=920, width="stretch", on_select="rerun",
            selection_mode="single-row", key="pred_table",
            column_config={"Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1,
                                                                         format="%.3f")})
        st.download_button("Download predictions CSV", preds.to_csv(index=False).encode(),
                           file_name="predictions.csv")

        sel = event.selection.rows if event and event.selection else []
        st.subheader("🤖 Prediction explainer")
        if not sel:
            st.info("Select a row in the table above to explain it.")
        else:
            idx = view.index[sel[0]]
            row = preds.loc[idx]
            c1, c2 = st.columns([2, 3])
            with c1:
                st.markdown(f"**Row {idx}** → **{row['Predicted']}** ({row['Confidence']:.1%})")
                st.dataframe(row[INPUT_COLS].rename("value").to_frame(), width="stretch")
                go_btn = st.button("✨ Explain this prediction", type="primary", width="stretch")
            with c2:
                cache = st.session_state.setdefault("explanations", {})
                if go_btn:
                    with st.spinner("Agent is gathering evidence…"):
                        try:
                            if api_key:
                                text, trace = run_agent(bundle, row, api_key, llm_model)
                            else:
                                text, trace = fallback_explanation(bundle, row), []
                        except Exception as e:
                            text = (f"⚠️ AI agent unavailable ({type(e).__name__}: {e}).\n\n"
                                    + fallback_explanation(bundle, row))
                            trace = []
                        cache[idx] = (text, trace, feature_contributions(bundle, row))
                if idx in cache:
                    text, trace, fc = cache[idx]
                    st.markdown(text)
                    cc = pd.DataFrame(fc["column_contributions"]).sort_values("contribution")
                    fig = px.bar(cc, x="contribution", y="column", orientation="h",
                                 hover_data=["value", "prediction_if_neutralised"],
                                 title=f"Contribution to P({fc['predicted_class']}) - drop when column is neutralised")
                    fig.update_traces(marker_color=np.where(cc["contribution"] >= 0, "#2a78d6", "#e34948"),
                                      marker_cornerradius=4)
                    st.plotly_chart(style(fig, 300), width="stretch")
                    if trace:
                        with st.expander(f"Agent tool calls ({len(trace)})"):
                            for t in trace:
                                st.markdown(f"**{t['tool']}** `{json.dumps(t['args'])}`")
                                st.json(t["result"], expanded=False)
                else:
                    st.caption("Explanations are generated on demand only.")
    elif up is None:
        st.session_state.pop("preds", None)
