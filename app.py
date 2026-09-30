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
from sklearn.metrics import classification_report, confusion_matrix

from src.explain import fallback_explanation, feature_contributions, predict_frame, run_agent
from src.pipeline import (COLUMN_DESCRIPTIONS, DROPPED_COLS, INPUT_COLS, LABEL_COL, MODEL_NUMERIC_FEATURES,
                          base_pipeline, coerce_schema, engineer_features, normalize_label, parse_amount)

ROOT = Path(__file__).parent
MODEL_COLS = [c for c in INPUT_COLS if c not in DROPPED_COLS]
SHORT_NAMES = {"Col1": "vendor name", "Col2": "reference ID", "Col3": "amount", "Col4": "line description",
               "Col5": "posting date", "Col6": "account description", "Col7": "document type"}
DROPPED_NOTE = ("**The model ignores " + " and ".join(f"{SHORT_NAMES[c]} ({c})" for c in DROPPED_COLS) + ".** "
                "Almost every reference ID is unique and the posting date only marks the monthly batch, so neither "
                "tells the model anything about a new transaction. It uses " + ", ".join(SHORT_NAMES[c] for c in MODEL_COLS[:-1])
                + " and " + SHORT_NAMES[MODEL_COLS[-1]] + ". "
                "The ignored columns still appear in the Data exploration charts.")
st.set_page_config(page_title="Transaction Classifier", page_icon="📊", layout="wide")

# Validated categorical order (one fixed hue per class, never cycled) + recessive chart chrome.
CLASS_COLORS = {"Category_1": "#2a78d6", "Category_2": "#eb6834", "Category_3": "#1baf7a",
                "Category_4": "#eda100", "Category_5": "#e87ba4", "Category_6": "#008300"}
CLASS_ORDER = list(CLASS_COLORS)
SEQ = "Blues"
PRIMARY, BEFORE = CLASS_COLORS["Category_1"], "#b8bec6"


def rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def get_encoder(model):
    """Fitted ColumnTransformer from the model (or the first member of an ensemble)."""
    return base_pipeline(model).named_steps["prep"].named_steps["encode"]


def style(fig, height=340):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(size=13), legend_title_text="", hoverlabel=dict(font_size=12))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.07)")
    return fig


def fmt_amount(x: float) -> str:
    for div, unit in [(1e6, "M"), (1e3, "K")]:
        if abs(x) >= div:
            return f"{x / div:,.1f}".rstrip("0").rstrip(".") + unit
    return f"{x:,.0f}"


def confusion_fig(cm, labels, norm, title):
    cm = np.array(cm, dtype=float)
    present = [i for i in range(len(labels)) if cm[i].sum() > 0 or cm[:, i].sum() > 0]
    cm, labs = cm[np.ix_(present, present)], [labels[i] for i in present]
    if norm:
        cm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig = px.imshow(cm, x=labs, y=labs, color_continuous_scale=SEQ, text_auto=".0%" if norm else ".0f",
                    labels=dict(x="Predicted", y="Actual", color="share" if norm else "rows"), title=title)
    return style(fig, 440)


def report_table(report):
    rep = pd.DataFrame(report).T
    rep = rep[rep["support"] > 0].drop(index=["accuracy"], errors="ignore")
    return rep.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}", "support": "{:.0f}"})


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
# The file's modification time is part of the cache key, so a retrain is picked up without restarting.
@st.cache_resource
def load_bundle(mtime: float):
    return joblib.load(ROOT / "artifacts" / "model.joblib")


@st.cache_data
def load_metrics(mtime: float):
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


@st.cache_data
def column_association(df: pd.DataFrame, shuffles: int = 5) -> pd.DataFrame:
    """Strength of relationship between each raw column and the category, plus the value the
    same measure gives when the categories are shuffled (chance level).

    Text / category columns: Cramér's V from the column x category table. Amount: correlation
    ratio (eta) on log |amount|, the equivalent 0-1 measure for a number vs a category.
    Dates are compared by posting month."""
    y = df["label"].to_numpy()
    rng = np.random.default_rng(0)

    def cramers_v(x, labels):
        ct = pd.crosstab(x, labels).to_numpy(dtype=float)
        n = ct.sum()
        expected = ct.sum(1, keepdims=True) * ct.sum(0, keepdims=True) / n
        chi2 = ((ct - expected) ** 2 / expected).sum()
        return float(np.sqrt(chi2 / n / max(min(ct.shape) - 1, 1)))

    def eta(x, labels):
        m = x.mean()
        between = sum((labels == c).sum() * (x[labels == c].mean() - m) ** 2 for c in np.unique(labels))
        return float(np.sqrt(between / ((x - m) ** 2).sum()))

    keys = {c: df[c].fillna("<missing>").to_numpy() for c in ["Col1", "Col2", "Col4", "Col6", "Col7"]}
    keys["Col5"] = df["date"].dt.strftime("%Y-%m").fillna("<missing>").to_numpy()
    amount = np.log10(df["amount"].abs().clip(lower=0.01))
    ok = amount.notna().to_numpy()
    rows = []
    for c in INPUT_COLS:
        if c == "Col3":
            x, lab_, f, measure = amount.to_numpy()[ok], y[ok], eta, "correlation ratio"
        else:
            x, lab_, f, measure = keys[c], y, cramers_v, "Cramér's V"
        strength = f(x, lab_)
        chance = float(np.mean([f(x, rng.permutation(lab_)) for _ in range(shuffles)]))
        rows.append({"column": c, "measure": measure, "strength": strength, "chance": chance,
                     "gap": strength - chance})
    return pd.DataFrame(rows)


if not (ROOT / "artifacts" / "model.joblib").exists():
    st.error("No trained model found. Run `.venv/bin/python train.py` first.")
    st.stop()

bundle = load_bundle((ROOT / "artifacts" / "model.joblib").stat().st_mtime)
M = load_metrics((ROOT / "artifacts" / "metrics.json").stat().st_mtime)
raw, eng_all = load_raw()

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Model")
    st.metric("Selected model", M["best_family"])
    st.metric("Cross-validated macro-F1", f"{M['cv_results']['with_cutoff']['f1_macro_mean']:.3f}")
    st.metric("Cross-validated accuracy", f"{M['cv_results']['with_cutoff']['accuracy_mean']:.1%}")
    st.caption(f"Tuned params: {M['best_params']}")

api_key = os.environ.get("OPENAI_API_KEY", "")
try:
    api_key = api_key or st.secrets.get("OPENAI_API_KEY", "")
except Exception:
    pass
llm_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

st.title("Transaction Classification - EY Challenge")
tab1, tab2 = st.tabs(["① Exploration, Preprocessing & Model Results", "② Hold-Out Prediction Pipeline"])

# =========================================================================== #
# TAB 1
# =========================================================================== #
with tab1:
    d = M["data"]
    dup_mask = raw.duplicated(subset=INPUT_COLS + ["label"])
    dedup = raw[~dup_mask]
    encoder = get_encoder(bundle["model"])

    k = st.columns(5)
    k[0].metric("Transactions", f"{d['raw_rows']:,}")
    k[1].metric("Input columns", len(INPUT_COLS))
    k[2].metric("Target classes", raw["label"].nunique())
    k[3].metric("Missing cells", f"{int(raw[INPUT_COLS].isna().sum().sum()):,}")
    k[4].metric("Duplicate rows", f"{int(dup_mask.sum()):,}")

    eda, prep, split, results = st.tabs(["📊 Data exploration", "🧹 Cleaning & preprocessing",
                                         "✂️ Split strategy", "🤖 Model training & evaluation"])

    # ------------------------------------------------------------------ EDA
    with eda:
        st.subheader("Dataset at a glance")
        c1, c2 = st.columns([2, 3])
        with c1:
            profile = pd.DataFrame({
                "Column": INPUT_COLS,
                "Content": [COLUMN_DESCRIPTIONS[c] for c in INPUT_COLS],
                "Distinct": [raw[c].nunique() for c in INPUT_COLS],
                "Missing": [int(raw[c].isna().sum()) for c in INPUT_COLS],
                "Example": [raw[c].dropna().iloc[0] for c in INPUT_COLS],
            })
            st.dataframe(profile, hide_index=True, width="stretch", height=300)
        with c2:
            st.dataframe(raw[INPUT_COLS + [LABEL_COL]], width="stretch", height=300)

        c1, c2 = st.columns(2)
        with c1:
            vc = raw["label"].value_counts().reindex(CLASS_ORDER).fillna(0).reset_index()
            vc.columns = ["label", "rows"]
            log_y = st.toggle("Log scale", value=True, key="logcls")
            fig = px.bar(vc, x="label", y="rows", color="label", color_discrete_map=CLASS_COLORS, text="rows",
                         log_y=log_y, title="Target class distribution")
            fig.update_traces(textposition="outside", marker_cornerradius=4)
            st.plotly_chart(style(fig).update_layout(showlegend=False).update_xaxes(title=""), width="stretch")
        with c2:
            miss = raw[INPUT_COLS].isna().sum().reset_index()
            miss.columns = ["column", "missing"]
            miss["share"] = miss["missing"] / len(raw)
            st.write("")
            st.write("")
            fig = px.bar(miss, x="column", y="missing", text="missing", hover_data={"share": ":.1%"},
                         title="Missing values per column")
            fig.update_traces(marker_color=PRIMARY, marker_cornerradius=4, textposition="outside")
            st.plotly_chart(style(fig).update_xaxes(title=""), width="stretch")

        st.subheader("Explore a column against the target")
        col_pick = st.selectbox("Column", INPUT_COLS + ["Col2 shape (engineered)"],
                                format_func=lambda c: f"{c} - {COLUMN_DESCRIPTIONS.get(c, 'reference-ID format')}")
        src = "col2_shape" if col_pick.startswith("Col2 shape") else col_pick
        c1, c2 = st.columns([4, 1])
        with c1:
            if col_pick == "Col3":
                show = raw.dropna(subset=["amount"]).assign(abs_amount=lambda x: x["amount"].abs().clip(lower=0.01))
                fig = px.box(show, x="label", y="abs_amount", color="label", log_y=True, points="outliers",
                             color_discrete_map=CLASS_COLORS, category_orders={"label": CLASS_ORDER},
                             title="Amount by class (absolute value, log scale)")
                st.plotly_chart(style(fig, 400).update_layout(showlegend=False).update_xaxes(title=""),
                                width="stretch")
            elif col_pick == "Col5":
                ct = pd.crosstab(raw["date"].dt.to_period("M").astype(str), raw["label"], normalize="index")
                fig = px.imshow(ct.T.reindex(CLASS_ORDER).fillna(0), color_continuous_scale=SEQ, aspect="auto",
                                text_auto=".0%", title="Class mix by posting month (share of month)")
                st.plotly_chart(style(fig, 400), width="stretch")
            else:
                topn = st.slider("Top N values", 5, 25, 12)
                vals = raw[src].fillna("<missing>")
                top = vals.value_counts().head(topn).index
                ct = raw.assign(v=vals)[vals.isin(top)].groupby(["v", "label"]).size().reset_index(name="rows")
                fig = px.bar(ct, y="v", x="rows", color="label", orientation="h", color_discrete_map=CLASS_COLORS,
                             category_orders={"label": CLASS_ORDER, "v": list(top)},
                             title=f"Top {topn} values of {src} - stacked by class")
                fig.update_traces(marker_line_color="white", marker_line_width=1)
                st.plotly_chart(style(fig, 440).update_yaxes(title=""), width="stretch")
        with c2:
            purity = (raw.groupby(src)["label"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
                      .rename("purity").to_frame().join(raw[src].value_counts().rename("n")))
            st.metric("Distinct values", f"{raw[src].nunique():,}")
            st.metric("Missing", int(raw[src].isna().sum()))
            st.metric("Label purity", f"{(purity['purity'] * purity['n']).sum() / purity['n'].sum():.1%}",
                      help="How often the most common class for a value is the right answer.")

        st.subheader("Outliers")
        la = np.log10(raw["amount"].abs().clip(lower=0.01)).dropna()
        q1, q3 = la.quantile([.25, .75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        fig = px.histogram(la, nbins=60, title="Amount distribution with IQR outlier fences")
        fig.update_traces(marker_color=PRIMARY)
        for x in (lo, hi):
            fig.add_vline(x=x, line_dash="dash", line_color="#e34948")
        fig.update_xaxes(title="log10 |amount|", tickvals=list(range(-2, 8)),
                         ticktext=["0.01", "0.1", "1", "10", "100", "1K", "10K", "100K", "1M", "10M"])
        st.plotly_chart(style(fig, 360).update_layout(showlegend=False).update_yaxes(title="rows"),
                        width="stretch")
        m = st.columns(3)
        m[0].metric("Outliers (IQR)", int(((la < lo) | (la > hi)).sum()))
        m[1].metric("Negative amounts", int((raw["amount"] < 0).sum()))
        m[2].metric("Largest amount", f"{raw['amount'].abs().max():,.0f}")

        st.subheader("How each column relates to the category")
        lab = dedup.dropna(subset=["label"])
        assoc = column_association(lab[INPUT_COLS + ["label", "amount", "date"]])
        assoc["name"] = assoc["column"] + " - " + assoc["column"].map(SHORT_NAMES) \
            + np.where(assoc["column"].isin(DROPPED_COLS), " (not used)", "")
        assoc = assoc.sort_values("gap")
        long = assoc.melt(["name", "measure"], ["strength", "chance"], var_name="kind", value_name="value")
        long["kind"] = long["kind"].map({"strength": "Relationship in the data",
                                         "chance": "Same measure with categories shuffled (pure chance)"})
        fig = px.bar(long, x="value", y="name", color="kind", barmode="group", orientation="h",
                     text=long["value"].round(2), hover_data={"measure": True, "name": False},
                     category_orders={"name": assoc["name"].tolist()[::-1]},
                     color_discrete_sequence=[PRIMARY, BEFORE],
                     title="Strength of relationship with the category (0 = none, 1 = the column fixes the category)")
        fig.update_traces(marker_cornerradius=4, textposition="outside")
        st.plotly_chart(style(fig, 480).update_xaxes(title="Cramér's V",
                                                     range=[0, 1.1])
                        .update_yaxes(title="").update_layout(legend=dict(orientation="h", y=-0.14)),
                        width="stretch")

    # ------------------------------------------------------------------ Cleaning & preprocessing
    with prep:
        st.info(DROPPED_NOTE, icon="ℹ️")
        STEPS = ["Labels", "Duplicates", "Amount", "Missing values", "Scaling"]
        step = st.segmented_control("Pipeline step", [f"{i + 1}. {s}" for i, s in enumerate(STEPS)],
                                    default="1. Labels", key="prep_step") or "1. Labels"
        step = step.split(". ", 1)[1]
        st.divider()

        if step == "Labels":
            lm = pd.DataFrame(d["label_map"])
            k = st.columns(2)
            k[0].metric("Before: label spellings", lm[LABEL_COL].nunique())
            k[1].metric("After: classes", lm["label"].nunique())
            equal = st.toggle("Equal-width links (show rare spellings clearly)", value=True)
            tgt = [c for c in CLASS_ORDER if c in set(lm["label"])]
            src_n = [f"{r[LABEL_COL]!r} ({r['rows']:,})" for _, r in lm.iterrows()]
            tgt_n = [f"{c} ({lm.loc[lm['label'] == c, 'rows'].sum():,})" for c in tgt]
            fig = go.Figure(go.Sankey(
                node=dict(label=src_n + tgt_n, pad=10, thickness=16,
                          color=["#b8bec6"] * len(src_n) + [CLASS_COLORS[c] for c in tgt]),
                link=dict(source=list(range(len(lm))), target=[len(lm) + tgt.index(l) for l in lm["label"]],
                          value=[1] * len(lm) if equal else lm["rows"].tolist(), customdata=lm["rows"],
                          hovertemplate="%{source.label} → %{target.label}<br>%{customdata:,} rows<extra></extra>",
                          color=[rgba(CLASS_COLORS[l], .35) for l in lm["label"]])))
            st.plotly_chart(style(fig, 460).update_layout(title="Raw label → cleaned class"), width="stretch")

        elif step == "Duplicates":
            k = st.columns(3)
            k[0].metric("Before", f"{len(raw):,} rows")
            k[1].metric("After", f"{len(dedup):,} rows", delta=f"-{int(dup_mask.sum()):,}", delta_color="off")
            k[2].metric("Duplicates removed", f"{dup_mask.mean():.1%}")
            c1, c2 = st.columns(2)
            with c1:
                ba = pd.DataFrame({"Before": raw["label"].value_counts(), "After": dedup["label"].value_counts()}) \
                    .reindex(CLASS_ORDER).fillna(0).reset_index(names="label").melt("label", var_name="stage",
                                                                                     value_name="rows")
                fig = px.bar(ba, x="label", y="rows", color="stage", barmode="group", log_y=True, text="rows",
                             color_discrete_map={"Before": BEFORE, "After": PRIMARY},
                             title="Rows per class, before vs after")
                fig.update_traces(marker_cornerradius=4, textposition="outside")
                st.plotly_chart(style(fig, 380).update_xaxes(title=""), width="stretch")
            with c2:
                st.markdown("**Duplicate groups** (identical on every column)")
                dups = raw[raw.duplicated(subset=INPUT_COLS + ["label"], keep=False)]
                st.dataframe(dups.sort_values(INPUT_COLS)[INPUT_COLS + ["label"]], hide_index=True,
                             width="stretch", height=340)

        elif step == "Amount":
            ex = pd.concat([raw[raw["Col3"].str.contains(",", na=False)].head(3),
                            raw[raw["amount"] < 0].head(2),
                            raw[~raw["Col3"].str.contains(",", na=True)].head(2)])
            st.dataframe(pd.DataFrame({"Before: raw text": ex["Col3"], "Parsed number": ex["amount"],
                                       "After: signed log": eng_all.loc[ex.index, "amount_signed_log"].round(3),
                                       "Is negative": eng_all.loc[ex.index, "amount_is_negative"],
                                       "Is round": eng_all.loc[ex.index, "amount_is_round"]}),
                         hide_index=True, width="stretch")
            c1, c2 = st.columns(2)
            fig = px.histogram(raw["amount"].dropna(), nbins=80, title="Before: raw amount")
            c1.plotly_chart(style(fig.update_traces(marker_color=BEFORE)).update_layout(showlegend=False)
                            .update_xaxes(title="amount").update_yaxes(title="rows"), width="stretch")
            fig = px.histogram(eng_all["amount_signed_log"].dropna(), nbins=80, title="After: signed log(1 + |amount|)")
            c2.plotly_chart(style(fig.update_traces(marker_color=PRIMARY)).update_layout(showlegend=False)
                            .update_xaxes(title="signed log amount").update_yaxes(title="rows"), width="stretch")

        elif step == "Missing values":
            c1, c2 = st.columns([3, 2])
            with c1:
                mv = pd.DataFrame({"column": MODEL_COLS, "Before": raw[MODEL_COLS].isna().sum().values,
                                   "After": 0}).melt("column", var_name="stage", value_name="missing")
                fig = px.bar(mv, x="column", y="missing", color="stage", barmode="group", text="missing",
                             color_discrete_map={"Before": BEFORE, "After": PRIMARY},
                             title="Missing values, before vs after")
                fig.update_traces(marker_cornerradius=4, textposition="outside")
                st.plotly_chart(style(fig, 360).update_xaxes(title=""), width="stretch")
            with c2:
                st.dataframe(pd.DataFrame([
                    ["Col1, Col4, Col6", "Empty text"],
                    ["Col3", "Median (from training data)"],
                    ["Col7", "'<missing>' category"],
                    ["Col4", "+ flag: col4_missing"],
                ], columns=["Column", "Filled with"]), hide_index=True, width="stretch")

        else:  # Scaling
            num = encoder.named_transformers_["numeric"]
            feat = st.selectbox("Numeric feature", MODEL_NUMERIC_FEATURES)
            imputed = num.named_steps["impute"].transform(eng_all[MODEL_NUMERIC_FEATURES])
            scaled = pd.DataFrame(num.named_steps["scale"].transform(imputed), columns=MODEL_NUMERIC_FEATURES)
            before = pd.Series(imputed[:, MODEL_NUMERIC_FEATURES.index(feat)])
            c1, c2 = st.columns(2)
            for col, s, name, color in [(c1, before, "Before", BEFORE), (c2, scaled[feat], "After", PRIMARY)]:
                fig = px.histogram(s, nbins=50, title=f"{name}: {feat}")
                fig.update_traces(marker_color=color)
                col.plotly_chart(style(fig, 320).update_layout(showlegend=False).update_xaxes(title="")
                                 .update_yaxes(title="rows"), width="stretch")
                m = col.columns(2)
                m[0].metric("Mean", f"{s.mean():.2f}")
                m[1].metric("Std", f"{s.std(ddof=0):.2f}")


    # ------------------------------------------------------------------ Split
    with split:
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = go.Figure(go.Sankey(
                node=dict(label=[f"All rows ({d['raw_rows']:,})", f"Duplicates removed ({d['duplicates_removed']:,})",
                                 f"Unique rows ({d['dedup_rows']:,})", f"Train 90% ({d['train_rows']:,})",
                                 f"Hold-out test 10% ({d['test_rows']:,})"],
                          color=[BEFORE, "#e34948", PRIMARY, PRIMARY, CLASS_COLORS["Category_2"]],
                          pad=18, thickness=18),
                link=dict(source=[0, 0, 2, 2], target=[1, 2, 3, 4],
                          value=[d["duplicates_removed"], d["dedup_rows"], d["train_rows"], d["test_rows"]],
                          color="rgba(42,120,214,0.18)")))
            st.plotly_chart(style(fig, 340).update_layout(title="How the data was partitioned"), width="stretch")
        with c2:
            sc = pd.DataFrame(d["split_counts"]).set_index("label")
            sh = (sc / sc.sum()).reset_index().melt("label", var_name="split", value_name="share")
            sh["split"] = sh["split"].map({"train": "Train", "holdout": "Hold-out"})
            fig = px.bar(sh, x="split", y="share", color="label", color_discrete_map=CLASS_COLORS,
                         category_orders={"label": CLASS_ORDER}, title="Class mix is identical in both parts",
                         hover_data={"share": ":.2%"})
            fig.update_yaxes(tickformat=".0%", title="")
            st.plotly_chart(style(fig, 340).update_xaxes(title=""), width="stretch")

        c1, c2 = st.columns([3, 2])
        with c1:
            z = np.eye(5)
            fig = go.Figure(go.Heatmap(z=z, x=[f"Part {i}" for i in range(1, 6)], y=[f"Fold {i}" for i in range(1, 6)],
                                       colorscale=[[0, "#dbe9fa"], [1, CLASS_COLORS["Category_4"]]], showscale=False,
                                       text=np.where(z == 1, "Validate", "Train"), texttemplate="%{text}",
                                       xgap=3, ygap=3, hoverinfo="skip"))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(style(fig, 300).update_layout(title="5-fold stratified cross-validation inside the 90%"),
                            width="stretch")
        with c2:
            t = sc.assign(total=sc.sum(axis=1), **{"hold-out %": (sc["holdout"] / sc.sum(axis=1)).map("{:.0%}".format)})
            st.dataframe(t, width="stretch")
            st.markdown(
                "- Duplicates removed **before** splitting\n"
                "- **Stratified** 90 / 10 split (seed 42)\n"
                "- 5-fold CV on the 90% for model choice & tuning\n"
                "- Hold-out scored **once**, at the end")

    # ------------------------------------------------------------------ Model results
    with results:
        CV = M["cv_results"]
        n_rep = CV["n_repeats"]
        METRICS = {"f1_macro": "Macro-F1", "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy"}
        metric = st.segmented_control("Metric", list(METRICS), format_func=METRICS.get, default="f1_macro",
                                      key="cv_metric") or "f1_macro"
        c1, c2 = st.columns(2)
        with c1:
            cv = pd.DataFrame(M["cv_comparison"])
            cv["selected"] = np.where(cv["model"] == M["best_family"], "Selected", "Other")
            fig = px.bar(cv, x="model", y=f"{metric}_mean", error_y=f"{metric}_std", color="selected",
                         text=cv[f"{metric}_mean"].round(3), hover_data={"fit_seconds": True},
                         color_discrete_map={"Selected": PRIMARY, "Other": BEFORE},
                         title=f"Algorithms compared - {METRICS[metric]}, average of {n_rep} repeats")
            fig.update_traces(marker_cornerradius=4, textposition="inside")
            st.plotly_chart(style(fig, 380).update_layout(showlegend=False).update_xaxes(title="")
                            .update_yaxes(title=METRICS[metric]), width="stretch")
        with c2:
            folds = pd.DataFrame([{"model": r["model"], "repeat": i + 1, "score": s}
                                  for r in M["cv_comparison"] for i, s in enumerate(r[f"{metric}_repeats"])])
            fig = px.strip(folds, x="model", y="score", hover_data=["repeat"], title="Score on each repeat")
            fig.update_traces(marker=dict(size=9, color=PRIMARY, line=dict(width=1.5, color="white")))
            means = folds.groupby("model")["score"].mean()
            fig.add_trace(go.Scatter(x=means.index, y=means.values, mode="markers", name="mean",
                                     marker=dict(symbol="line-ew-open", size=40, color="#0b0b0b", line_width=2)))
            st.plotly_chart(style(fig, 380).update_xaxes(title="").update_yaxes(title=METRICS[metric]),
                            width="stretch")
        others = [r for r in M["cv_comparison"] if r["model"] != M["best_family"]]
        tune = pd.DataFrame([{**t["params"], "Macro-F1": t["f1_macro"], "Accuracy": t["accuracy"]}
                             for t in M["tuning"]])
        pcols = [c for c in tune.columns if c not in ("Macro-F1", "Accuracy")]
        c1, c2 = st.columns([3, 2])
        with c1:
            if len(pcols) == 1 and pd.api.types.is_numeric_dtype(tune[pcols[0]]):
                p = pcols[0]
                tl = tune.melt(p, ["Macro-F1", "Accuracy"], var_name="metric", value_name="score")
                fig = px.line(tl, x=p, y="score", color="metric", markers=True, log_x=True,
                              color_discrete_map={"Macro-F1": PRIMARY, "Accuracy": CLASS_COLORS["Category_2"]},
                              title=f"Hyper-parameter tuning - {M['best_family']} ({p.split('__')[-1]})")
                fig.add_vline(x=float(M["best_params"][p]), line_dash="dash", line_color="#6b7280",
                              annotation_text="selected")
                st.plotly_chart(style(fig, 320).update_xaxes(title=p.split("__")[-1]), width="stretch")
            else:
                st.markdown(f"**Hyper-parameter tuning - {M['best_family']}**")
                st.dataframe(tune, hide_index=True, width="stretch")
        with c2:
            st.metric("Selected model", M["best_family"])
            st.metric("Best parameters", ", ".join(f"{k.split('__')[-1]} = {v}" for k, v in M["best_params"].items()))
            st.metric("Tuning search", f"{len(tune)} settings × {n_rep} repeats")

        # ---------------------------------------------------------------- main results
        st.subheader("Results on all training rows")
        W = CV["with_cutoff"]
        k = st.columns(3)
        k[0].metric("Macro-F1", f"{W['f1_macro_mean']:.3f} ± {W['f1_macro_std']:.3f}")
        k[1].metric("Accuracy", f"{W['accuracy_mean']:.2%}")
        k[2].metric("Balanced accuracy", f"{W['balanced_accuracy']:.3f}")
        c1, c2 = st.columns(2)
        with c1:
            norm = st.toggle("Show as % of actual class", value=False, key="cv_norm")
            st.plotly_chart(confusion_fig(W["confusion_matrix"], W["labels"], norm,
                                          "Confusion matrix (average per repeat)"), width="stretch")
        with c2:
            st.markdown("**Per-category results**")
            st.dataframe(report_table(W["report"]), width="stretch")
            st.caption("Every training row counts here, so each rare category is judged on all its rows "
                       "(e.g. 14 for Category_4) instead of the 1-2 it has in the 10% test file (Tab 2).")

        imp = (pd.DataFrame(M["column_importance"]["current"]["columns"]).T.reindex(MODEL_COLS)
               .reset_index(names="column").sort_values("mean"))
        imp["name"] = imp["column"] + " - " + imp["column"].map(SHORT_NAMES)
        fig = px.bar(imp, x="mean", y="name", error_x="std", orientation="h", text=imp["mean"].round(3),
                     title="Column importance (drop in macro-F1 when shuffled)")
        fig.update_traces(marker_color=PRIMARY, marker_cornerradius=4, textposition="outside")
        st.plotly_chart(style(fig, 320).update_xaxes(title="macro-F1 drop").update_yaxes(title=""),
                        width="stretch")
        st.caption(f"Each column's values are shuffled in turn inside cross-validation; the bar is the average "
                   f"drop in macro-F1 over {n_rep} repeats, the whisker its spread. A bigger drop means the model "
                   "leans on that column more.")


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
                    st.write(f"✅ Encoded features and scored with **{bundle['model_name']}** "
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
        k = st.columns(5)
        k[0].metric("Rows scored", f"{len(preds):,}")
        k[1].metric("Pipeline time", f"{st.session_state['elapsed']:.2f}s")
        k[2].metric("Low-confidence rows (<70%)", int((preds["Confidence"] < 0.7).sum()))
        k[3].metric("Rows from unknown vendors", int((~preds["Known vendor"]).sum()),
                    help="Vendor (Col1) never seen in training. The model is much less reliable on these - "
                         "review them by hand.")
        if "Actual" in preds and preds["Actual"].notna().any():
            m = preds["Actual"].notna()
            k[4].metric("Live accuracy vs provided labels", f"{(preds.loc[m, 'Actual'] == preds.loc[m, 'Predicted']).mean():.2%}")
            y_true, y_pred = preds.loc[m, "Actual"], preds.loc[m, "Predicted"]
            labels = CLASS_ORDER + sorted(set(y_true) - set(CLASS_ORDER))
            c1, c2 = st.columns(2)
            with c1:
                norm = st.toggle("Show as % of actual class", value=False, key="live_norm")
                st.plotly_chart(confusion_fig(confusion_matrix(y_true, y_pred, labels=labels), labels, norm,
                                              f"Confusion matrix on uploaded labels ({int(m.sum()):,} rows)"),
                                width="stretch")
            with c2:
                st.markdown("**Per-class report**")
                st.dataframe(report_table(classification_report(y_true, y_pred, labels=sorted(set(y_true)),
                                                                output_dict=True, zero_division=0)),
                             width="stretch")

        f1, f2, f3 = st.columns([2, 1, 1])
        cls_filter = f1.multiselect("Filter predicted class", CLASS_ORDER,
                                    default=[c for c in CLASS_ORDER if c in set(preds["Predicted"])])
        only_low = f2.toggle("Only low-confidence rows")
        only_new = f3.toggle("Only unknown vendors")
        view = preds[preds["Predicted"].isin(cls_filter)]
        if only_low:
            view = view[view["Confidence"] < 0.7]
        if only_new:
            view = view[~view["Known vendor"]]

        st.markdown("**Predictions** - click a row, then press **Explain** (≈25 rows visible, scroll for more)")
        show_cols = ["Predicted", "Confidence", "Known vendor"] + (["Actual"] if "Actual" in view else []) + INPUT_COLS
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
