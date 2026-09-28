"""Train, compare, tune and evaluate models. Writes everything the dashboard needs.

Usage:  .venv/bin/python train.py
Outputs:
  data/train.csv                      90% training split (deduplicated, cleaned labels)
  data/holdout_test.csv               10% hold-out, Columns A-G only  -> upload in Tab 2
  data/holdout_test_with_labels.csv   same rows + true label (upload to see live accuracy)
  artifacts/model.joblib              fitted end-to-end pipeline + explainer context
  artifacts/metrics.json              CV comparison, tuning, hold-out results
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
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src.pipeline import (INPUT_COLS, LABEL_COL, build_preprocessor, densify, engineer_features,
                          normalize_label)

warnings.filterwarnings("ignore")  # tiny classes trigger 'least populated class' noise in CV
SEED = 42
ROOT = Path(__file__).parent
DATA, ART = ROOT / "data", ROOT / "artifacts"
DATA.mkdir(exist_ok=True)
ART.mkdir(exist_ok=True)


def make_models() -> dict[str, Pipeline]:
    logreg = Pipeline([("prep", build_preprocessor(ngram_max=2)),
                       ("clf", LogisticRegression(C=10, class_weight="balanced", max_iter=5000))])
    rf = Pipeline([("prep", build_preprocessor(ngram_max=1, max_features=500)),
                   ("clf", RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                                  min_samples_leaf=1, n_jobs=1, random_state=SEED))])
    hgb = Pipeline([("prep", build_preprocessor(ngram_max=1, max_features=250)),
                    ("dense", densify()),
                    ("clf", HistGradientBoostingClassifier(learning_rate=0.1, max_iter=120,
                                                           class_weight="balanced",
                                                           random_state=SEED))])
    ensemble = VotingClassifier([("logreg", logreg), ("rf", rf), ("hgb", hgb)], voting="soft")
    return {"Logistic Regression": logreg, "Random Forest": rf,
            "Hist Gradient Boosting": hgb, "Soft-Voting Ensemble": ensemble}


PARAM_GRIDS = {
    "Logistic Regression": {"clf__C": [1, 10, 50]},
    "Random Forest": {"clf__n_estimators": [300, 600], "clf__max_features": ["sqrt", 0.1]},
    "Hist Gradient Boosting": {"clf__learning_rate": [0.05, 0.1], "clf__max_leaf_nodes": [15, 31]},
    "Soft-Voting Ensemble": {"weights": [[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2]]},
}


def column_permutation_importance(model, X, y, n_repeats=5):
    """Model-agnostic: shuffle one raw input column at a time and measure the
    drop in hold-out macro-F1. Explains the *raw* columns, not 10k TF-IDF terms."""
    rng = np.random.default_rng(SEED)
    base = f1_score(y, model.predict(X), average="macro")
    out = {}
    for c in INPUT_COLS:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[c] = rng.permutation(Xp[c].values)
            drops.append(base - f1_score(y, model.predict(Xp), average="macro"))
        out[c] = {"mean": float(np.mean(drops)), "std": float(np.std(drops))}
    return out


def main():
    t0 = time.time()
    raw = pd.read_csv(ROOT / "Challenge_Data.csv", dtype=str)
    raw["label"] = raw[LABEL_COL].map(normalize_label)
    label_map = (raw.groupby([LABEL_COL, "label"]).size().reset_index(name="rows")
                 .sort_values("label").to_dict("records"))

    n_raw = len(raw)
    dedup = raw.drop_duplicates(subset=INPUT_COLS + ["label"]).reset_index(drop=True)
    print(f"rows {n_raw} -> {len(dedup)} after removing exact duplicates")

    X, y = dedup[INPUT_COLS], dedup["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.10, stratify=y, random_state=SEED)

    pd.concat([X_tr, y_tr], axis=1).to_csv(DATA / "train.csv", index=False)
    X_te.to_csv(DATA / "holdout_test.csv", index=False)
    X_te.assign(ClassificationLabel=y_te.values).to_csv(DATA / "holdout_test_with_labels.csv", index=False)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro", "balanced_accuracy": "balanced_accuracy",
               "f1_weighted": "f1_weighted"}

    # ---- 1. compare model families with 5-fold stratified CV on the 90% train split
    comparison = []
    for name, model in make_models().items():
        s = time.time()
        res = cross_validate(model, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=-1)
        row = {"model": name, "fit_seconds": round(time.time() - s, 1)}
        for k in scoring:
            row[f"{k}_mean"] = float(res[f"test_{k}"].mean())
            row[f"{k}_std"] = float(res[f"test_{k}"].std())
            row[f"{k}_folds"] = res[f"test_{k}"].tolist()
        comparison.append(row)
        print(f"{name:24s} acc {row['accuracy_mean']:.4f}  macroF1 {row['f1_macro_mean']:.4f}  "
              f"({row['fit_seconds']}s)", flush=True)

    # Select on macro-F1 (every class counts equally); accuracy breaks ties.
    best = max(comparison, key=lambda r: (round(r["f1_macro_mean"], 3), r["accuracy_mean"]))
    best_name = best["model"]
    print("best family:", best_name)

    # ---- 2. tune the winner
    gs = GridSearchCV(make_models()[best_name], PARAM_GRIDS[best_name], cv=cv, scoring=scoring,
                      refit="f1_macro", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    tuning = [{"params": {k: (str(v) if isinstance(v, list) else v) for k, v in p.items()},
               "f1_macro": float(f), "accuracy": float(a)}
              for p, f, a in zip(gs.cv_results_["params"], gs.cv_results_["mean_test_f1_macro"],
                                 gs.cv_results_["mean_test_accuracy"])]
    final = gs.best_estimator_  # refit on the full 90% train split
    print("best params:", gs.best_params_)

    # ---- 3. evaluate once on the untouched 10% hold-out
    pred = final.predict(X_te)
    classes = list(final.classes_)
    holdout = {
        "accuracy": accuracy_score(y_te, pred),
        "f1_macro": f1_score(y_te, pred, average="macro"),
        "f1_weighted": f1_score(y_te, pred, average="weighted"),
        "balanced_accuracy": balanced_accuracy_score(y_te, pred),
        "labels": classes,
        "confusion_matrix": confusion_matrix(y_te, pred, labels=classes).tolist(),
        "report": classification_report(y_te, pred, labels=classes, output_dict=True, zero_division=0),
        "majority_baseline_accuracy": float((y_te == y_tr.mode()[0]).mean()),
    }
    print(f"HOLD-OUT  acc {holdout['accuracy']:.4f}  macroF1 {holdout['f1_macro']:.4f}")

    importance = column_permutation_importance(final, X_te.reset_index(drop=True), y_te.reset_index(drop=True))

    # Neutral values used by the per-row explainer (ablate one column at a time).
    amounts = pd.to_numeric(X_tr["Col3"].str.replace(",", ""), errors="coerce")
    baselines = {"Col1": "", "Col2": "", "Col3": f"{amounts.median():.2f}", "Col4": "",
                 "Col5": X_tr["Col5"].mode()[0], "Col6": "", "Col7": X_tr["Col7"].mode()[0]}

    joblib.dump({"model": final, "model_name": best_name, "best_params": gs.best_params_,
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
        "tuning": tuning, "best_params": {k: str(v) for k, v in gs.best_params_.items()},
        "holdout": holdout, "column_importance": importance,
        "feature_dims": {name: int(final_dim) for name, final_dim in _block_dims(final).items()},
        "engineered_sample": eng.astype(str).to_dict("records"),
        "train_seconds": round(time.time() - t0, 1),
    }
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"done in {metrics['train_seconds']}s")


def _block_dims(model):
    """Width of each encoded feature block (for the preprocessing summary)."""
    pipes = [model] if isinstance(model, Pipeline) else [e for _, e in model.estimators]
    prep = pipes[0].named_steps["prep"].named_steps["encode"]
    names = prep.get_feature_names_out()
    dims = {}
    for n in names:
        block = n.split("__")[0]
        dims[block] = dims.get(block, 0) + 1
    return dims


if __name__ == "__main__":
    main()
