"""Train, compare, tune and evaluate models. Writes everything the dashboard needs.

Usage:  .venv/bin/python train.py
Outputs:
  data/train.csv                      90% training split (deduplicated, cleaned labels)
  data/holdout_test.csv               10% hold-out, Columns A-G only  -> upload in Tab 2
  data/holdout_test_with_labels.csv   same rows + true label (upload to see live accuracy)
  artifacts/model.joblib              fitted end-to-end pipeline + explainer context
  artifacts/metrics.json              CV comparison, tuning, cross-validated results, hold-out check
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import (ParameterGrid, StratifiedGroupKFold, StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.pipeline import Pipeline

from src.pipeline import (DROPPED_COLS, INPUT_COLS, LABEL_COL, ThresholdedClassifier, base_pipeline,
                          build_preprocessor, densify, engineer_features, macro_f1, normalize_label,
                          thresholded_predict)

warnings.filterwarnings("ignore")  # tiny classes trigger 'least populated class' noise in CV
SEED = 42
N_REPEATS = 3              # 5-fold CV repeated with 3 different shuffles
CUTOFF_CLASS = "Category_2"  # the model over-predicts it (Category_1 rows labelled Category_2)
CUTOFF_GRID = np.round(np.arange(0.30, 0.951, 0.05), 2)
ROOT = Path(__file__).parent
DATA, ART = ROOT / "data", ROOT / "artifacts"
DATA.mkdir(exist_ok=True)
ART.mkdir(exist_ok=True)


def make_models(all_columns: bool = False) -> dict[str, Pipeline]:
    """all_columns=True builds the same models with Col2 / Col5 included (for comparison only)."""
    logreg = Pipeline([("prep", build_preprocessor(ngram_max=2, all_columns=all_columns)),
                       ("clf", LogisticRegression(C=10, class_weight="balanced", max_iter=5000))])
    rf = Pipeline([("prep", build_preprocessor(ngram_max=1, max_features=500, all_columns=all_columns)),
                   ("clf", RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                                  min_samples_leaf=1, n_jobs=1, random_state=SEED))])
    hgb = Pipeline([("prep", build_preprocessor(ngram_max=1, max_features=250, all_columns=all_columns)),
                    ("dense", densify()),
                    ("clf", HistGradientBoostingClassifier(learning_rate=0.1, max_iter=120,
                                                           class_weight="balanced",
                                                           random_state=SEED))])
    ensemble = VotingClassifier([("logreg", logreg), ("rf", rf), ("hgb", hgb)], voting="soft")
    return {"Logistic Regression": logreg, "Random Forest": rf,
            "Hist Gradient Boosting": hgb, "Soft-Voting Ensemble": ensemble}


PARAM_GRIDS = {
    "Logistic Regression": {"clf__C": [1, 3, 10, 30, 100]},
    "Random Forest": {"clf__n_estimators": [300, 600], "clf__max_features": ["sqrt", 0.1]},
    "Hist Gradient Boosting": {"clf__learning_rate": [0.05, 0.1], "clf__max_leaf_nodes": [15, 31]},
    "Soft-Voting Ensemble": {"weights": [[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2]]},
}


def column_permutation_importance(model, X, y, min_proba, cols):
    """Model-agnostic, on the training data: in each CV fold, fit on 4/5, then shuffle one raw
    input column at a time in the held-out 1/5 and measure the drop in macro-F1 (pooled over
    all rows of a repeat). Explains the *raw* columns, not 6k TF-IDF terms. Also returns the
    full cross-validated scores of the unshuffled predictions, so models can be compared."""
    rng = np.random.default_rng(SEED)
    drops, scores, bases = {c: [] for c in cols}, [], []
    for r in range(N_REPEATS):
        base = np.empty(len(y), dtype=object)
        shuffled = {c: np.empty(len(y), dtype=object) for c in cols}
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(X, y):
            m = ThresholdedClassifier(clone(model).fit(X.iloc[tr], y.iloc[tr]), min_proba)
            X_te = X.iloc[te]
            base[te] = m.predict(X_te)
            for c in cols:
                Xp = X_te.copy()
                Xp[c] = rng.permutation(Xp[c].values)
                shuffled[c][te] = m.predict(Xp)
        scores.append(macro_f1(y, base))
        bases.append(base)
        for c in cols:
            drops[c].append(scores[-1] - macro_f1(y, shuffled[c]))
    return {"f1_macro": float(np.mean(scores)), "cv": summarise(y, bases, sorted(y.unique())),
            "columns": {c: {"mean": float(np.mean(d)), "std": float(np.std(d))} for c, d in drops.items()}}


def oof_scores(model, X, y) -> dict:
    """Out-of-fold predictions for every training row, once per repeat; each metric is
    computed over all rows of a repeat. Repeat r uses the same splits for every model."""
    out = {}
    for r in range(N_REPEATS):
        pred = cross_val_predict(clone(model), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED + r),
                                 n_jobs=-1)
        for k, v in {"accuracy": accuracy_score(y, pred), "f1_macro": macro_f1(y, pred),
                     "balanced_accuracy": balanced_accuracy_score(y, pred)}.items():
            out.setdefault(f"{k}_repeats", []).append(float(v))
    for k in list(out):
        m = k.removesuffix("_repeats")
        out[f"{m}_mean"], out[f"{m}_std"] = float(np.mean(out[k])), float(np.std(out[k]))
    return out


def pick_cutoff(P, y, classes) -> float:
    """Cut-off for CUTOFF_CLASS that maximises macro-F1; ties go to the lower cut-off."""
    scores = [macro_f1(y, thresholded_predict(P, classes, {CUTOFF_CLASS: t})) for t in CUTOFF_GRID]
    return float(CUTOFF_GRID[int(np.argmax(scores))])


def cross_validated_results(model, X, y):
    """Out-of-fold probabilities for every training row, N_REPEATS times.

    The cut-off is chosen *inside* each outer fold (from an inner CV on that fold's
    training part) and then applied to the held-out part, so the thresholded scores
    are not tuned on the rows they are scored on."""
    classes = sorted(y.unique())
    runs = []
    for r in range(N_REPEATS):
        outer = StratifiedKFold(5, shuffle=True, random_state=SEED + r)
        P = np.zeros((len(y), len(classes)))
        pred_cut = np.empty(len(y), dtype=object)
        cutoffs = []
        for tr, te in outer.split(X, y):
            inner = StratifiedKFold(5, shuffle=True, random_state=SEED + r)
            P_in = cross_val_predict(clone(model), X.iloc[tr], y.iloc[tr], cv=inner, method="predict_proba",
                                     n_jobs=-1)
            t = pick_cutoff(P_in, y.iloc[tr], classes)
            P[te] = clone(model).fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])
            pred_cut[te] = thresholded_predict(P[te], classes, {CUTOFF_CLASS: t})
            cutoffs.append(t)
        runs.append({"P": P, "pred_plain": np.array(classes)[P.argmax(1)], "pred_cut": pred_cut,
                     "cutoffs": cutoffs})
        print(f"  repeat {r + 1}: macroF1 plain {macro_f1(y, runs[-1]['pred_plain']):.4f}  "
              f"with cut-off {macro_f1(y, pred_cut):.4f}  cut-offs {cutoffs}", flush=True)
    return classes, runs


def summarise(y, preds, classes):
    """Scores, per-class report and confusion matrix pooled over repeated OOF predictions."""
    y_all = np.tile(np.asarray(y), len(preds))
    p_all = np.concatenate(preds)
    labels = sorted(set(y))
    per_run_f1 = [macro_f1(y, p) for p in preds]
    per_run_acc = [accuracy_score(y, p) for p in preds]
    return {"f1_macro_mean": float(np.mean(per_run_f1)), "f1_macro_std": float(np.std(per_run_f1)),
            "accuracy_mean": float(np.mean(per_run_acc)),
            "balanced_accuracy": balanced_accuracy_score(y_all, p_all),
            "labels": classes,
            # average count per repeat, so the matrix adds up to the number of training rows
            "confusion_matrix": (confusion_matrix(y_all, p_all, labels=classes) / len(preds)).round(1).tolist(),
            "report": classification_report(y_all, p_all, labels=labels, output_dict=True, zero_division=0)}


def cutoff_curve(y, runs, classes):
    """Errors between Category_1 and Category_2 at each cut-off, averaged over repeats."""
    rows = []
    for t in CUTOFF_GRID:
        preds = [thresholded_predict(r["P"], classes, {CUTOFF_CLASS: t}) for r in runs]
        rows.append({"cutoff": float(t),
                     "cat1_as_cat2": float(np.mean([((y == "Category_1") & (p == "Category_2")).sum() for p in preds])),
                     "cat2_as_cat1": float(np.mean([((y == "Category_2") & (p == "Category_1")).sum() for p in preds])),
                     "f1_macro": float(np.mean([macro_f1(y, p) for p in preds]))})
    return rows


def main():
    t0 = time.time()
    raw = pd.read_csv(ROOT / "Challenge_Data.csv", dtype=str)
    raw["label"] = raw[LABEL_COL].map(normalize_label)
    unreadable = raw["label"].isna()
    if unreadable.any():  # report and set aside rather than guess a class
        print(f"dropped {unreadable.sum()} rows with unreadable labels: "
              f"{sorted(raw.loc[unreadable, LABEL_COL].astype(str).unique())}")
        raw = raw[~unreadable].reset_index(drop=True)
    label_map = (raw.groupby([LABEL_COL, "label"]).size().reset_index(name="rows")
                 .sort_values("label").to_dict("records"))

    n_raw = len(raw)
    dedup = raw.drop_duplicates(subset=INPUT_COLS + ["label"]).reset_index(drop=True)
    print(f"rows {n_raw} -> {len(dedup)} after removing exact duplicates")

    # A stratified split needs at least 2 rows per class; warn and set aside instead of crashing.
    counts = dedup["label"].value_counts()
    too_rare = counts[counts < 2].index.tolist()
    if too_rare:
        print(f"WARNING: classes with <2 rows cannot be split or evaluated, set aside: {too_rare}")
        dedup = dedup[~dedup["label"].isin(too_rare)].reset_index(drop=True)

    X, y = dedup[INPUT_COLS], dedup["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.10, stratify=y, random_state=SEED)

    pd.concat([X_tr, y_tr], axis=1).to_csv(DATA / "train.csv", index=False)
    X_te.to_csv(DATA / "holdout_test.csv", index=False)
    X_te.assign(ClassificationLabel=y_te.values).to_csv(DATA / "holdout_test_with_labels.csv", index=False)

    # ---- 1. compare model families on the 90% train split
    # Scored once per repeat on all out-of-fold predictions, not averaged over the 15 folds:
    # a fold holds 0-1 Category_5 rows, so per-fold macro-F1 swings by ~0.1 on a single row.
    # Every model sees the same splits, so the repeats can be compared one to one.
    comparison = []
    for name, model in make_models().items():
        s = time.time()
        row = {"model": name, **oof_scores(model, X_tr, y_tr)}
        row["fit_seconds"] = round(time.time() - s, 1)
        comparison.append(row)
        print(f"{name:24s} acc {row['accuracy_mean']:.4f}  macroF1 {row['f1_macro_mean']:.4f}  "
              f"({row['fit_seconds']}s)", flush=True)

    # Select on macro-F1 (every class counts equally); accuracy breaks ties.
    best = max(comparison, key=lambda r: (round(r["f1_macro_mean"], 3), r["accuracy_mean"]))
    best_name = best["model"]
    print("best family:", best_name)
    for row in comparison:
        diff = np.array(row["f1_macro_repeats"]) - np.array(best["f1_macro_repeats"])
        row["f1_macro_gap_to_best"] = float(diff.mean())
        row["repeats_beating_best"] = int((diff > 0).sum())

    # ---- 2. tune the winner (same out-of-fold scoring)
    tuning = []
    for params in ParameterGrid(PARAM_GRIDS[best_name]):
        sc = oof_scores(make_models()[best_name].set_params(**params), X_tr, y_tr)
        tuning.append({"params": {k: (str(v) if isinstance(v, list) else v) for k, v in params.items()},
                       "raw_params": params, "f1_macro": sc["f1_macro_mean"], "accuracy": sc["accuracy_mean"]})
        print(f"  {params}: macroF1 {sc['f1_macro_mean']:.4f}  acc {sc['accuracy_mean']:.4f}", flush=True)
    best_params = max(tuning, key=lambda t: (round(t["f1_macro"], 3), t["accuracy"]))["raw_params"]
    for t in tuning:
        del t["raw_params"]
    tuned = make_models()[best_name].set_params(**best_params).fit(X_tr, y_tr)  # refit on the full 90% split
    print("best params:", best_params)

    # ---- 3. cross-validated results on all training rows, with the Category_2 cut-off chosen in-fold
    print("cross-validated results (nested cut-off):")
    classes, runs = cross_validated_results(make_models()[best_name].set_params(**best_params), X_tr, y_tr)
    cv_plain = summarise(y_tr, [r["pred_plain"] for r in runs], classes)
    cv_cut = summarise(y_tr, [r["pred_cut"] for r in runs], classes)
    # Final cut-off: chosen on all out-of-fold probabilities (every training row, every repeat).
    cutoff = pick_cutoff(np.vstack([r["P"] for r in runs]), np.tile(y_tr.values, N_REPEATS), classes)
    final = ThresholdedClassifier(tuned, {CUTOFF_CLASS: cutoff})
    print(f"final {CUTOFF_CLASS} cut-off: {cutoff}")

    # ---- 4. vendors the model has never seen: keep each Col1 value on one side of the split
    grouped = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    P_g = cross_val_predict(make_models()[best_name].set_params(**best_params), X_tr, y_tr, cv=grouped,
                            groups=X_tr["Col1"].fillna(""), method="predict_proba", n_jobs=-1)
    pred_g = thresholded_predict(P_g, classes, {CUTOFF_CLASS: cutoff})
    unseen_vendor = {"f1_macro": macro_f1(y_tr, pred_g), "accuracy": accuracy_score(y_tr, pred_g),
                     "majority_baseline_accuracy": float((y_tr == y_tr.mode()[0]).mean()),
                     "report": classification_report(y_tr, pred_g, labels=classes, output_dict=True,
                                                     zero_division=0),
                     "holdout_rows_with_known_vendor": float(X_te["Col1"].isin(X_tr["Col1"]).mean())}
    print(f"unseen vendors: macroF1 {unseen_vendor['f1_macro']:.4f}  acc {unseen_vendor['accuracy']:.4f}")

    # ---- 5. final check on the untouched 10% hold-out
    pred = final.predict(X_te)
    present = sorted(set(y_te))
    holdout = {
        "accuracy": accuracy_score(y_te, pred),
        "f1_macro": macro_f1(y_te, pred),
        "balanced_accuracy": balanced_accuracy_score(y_te, pred),
        "labels": classes,
        "confusion_matrix": confusion_matrix(y_te, pred, labels=classes).tolist(),
        "report": classification_report(y_te, pred, labels=present, output_dict=True, zero_division=0),
        "majority_baseline_accuracy": float((y_te == y_tr.mode()[0]).mean()),
    }
    print(f"HOLD-OUT  acc {holdout['accuracy']:.4f}  macroF1 {holdout['f1_macro']:.4f}")

    # Same model and settings, with and without Col2 / Col5, to show what those columns add.
    model_cols = [c for c in INPUT_COLS if c not in DROPPED_COLS]
    importance = {
        "current": column_permutation_importance(make_models()[best_name].set_params(**best_params), X_tr, y_tr,
                                                 {CUTOFF_CLASS: cutoff}, model_cols),
        "all_columns": column_permutation_importance(
            make_models(all_columns=True)[best_name].set_params(**best_params), X_tr, y_tr,
            {CUTOFF_CLASS: cutoff}, INPUT_COLS),
    }
    print("column importance (macro-F1 drop): current",
          {c: round(v["mean"], 3) for c, v in importance["current"]["columns"].items()},
          "| all 7 columns", {c: round(v["mean"], 3) for c, v in importance["all_columns"]["columns"].items()},
          f"| macroF1 {importance['current']['f1_macro']:.4f} vs {importance['all_columns']['f1_macro']:.4f}")

    # Neutral values used by the per-row explainer (ablate one column at a time).
    amounts = pd.to_numeric(X_tr["Col3"].str.replace(",", ""), errors="coerce")
    baselines = {"Col1": "", "Col2": "", "Col3": f"{amounts.median():.2f}", "Col4": "",
                 "Col5": X_tr["Col5"].mode()[0], "Col6": "", "Col7": X_tr["Col7"].mode()[0]}

    joblib.dump({"model": final, "model_name": best_name, "best_params": best_params,
                 "classes": classes, "baselines": baselines,
                 "train": pd.concat([X_tr, y_tr], axis=1).reset_index(drop=True)},
                ART / "model.joblib", compress=3)

    eng = engineer_features(X_tr.head(8))
    metrics = {
        "data": {"raw_rows": n_raw, "duplicates_removed": n_raw - len(dedup), "dedup_rows": len(dedup),
                 "train_rows": len(X_tr), "test_rows": len(X_te), "label_map": label_map,
                 "split_counts": pd.DataFrame({"train": y_tr.value_counts(), "holdout": y_te.value_counts()})
                 .fillna(0).astype(int).reset_index(names="label").to_dict("records")},
        "cv_comparison": comparison, "best_family": best_name,
        "tuning": tuning, "best_params": {k: str(v) for k, v in best_params.items()},
        "cv_results": {"with_cutoff": cv_cut, "without_cutoff": cv_plain, "n_repeats": N_REPEATS,
                       "in_fold_cutoffs": [t for r in runs for t in r["cutoffs"]]},
        "cutoff": {"class": CUTOFF_CLASS, "value": cutoff, "curve": cutoff_curve(y_tr, runs, classes)},
        "unseen_vendor": unseen_vendor,
        "holdout": holdout, "column_importance": importance,
        "feature_dims": {name: int(final_dim) for name, final_dim in _block_dims(final).items()},
        "engineered_sample": eng.astype(str).to_dict("records"),
        "train_seconds": round(time.time() - t0, 1),
    }
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"done in {metrics['train_seconds']}s")


def _block_dims(model):
    """Width of each encoded feature block (for the preprocessing summary)."""
    prep = base_pipeline(model).named_steps["prep"].named_steps["encode"]
    names = prep.get_feature_names_out()
    dims = {}
    for n in names:
        block = n.split("__")[0]
        dims[block] = dims.get(block, 0) + 1
    return dims


if __name__ == "__main__":
    main()
