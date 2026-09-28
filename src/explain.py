"""Per-row explanation tools + the on-demand LLM agent that uses them.

Attribution is model-agnostic 'occlusion': replace one input column (or one
word token) with a neutral value, re-score the row, and record how much the
predicted class's probability drops. Positive delta = that input pushed the
model TOWARD the prediction. All perturbed variants are scored in one batch.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

from .pipeline import COLUMN_DESCRIPTIONS, INPUT_COLS, TEXT_COLS, coerce_schema


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def predict_frame(bundle: dict, X_raw: pd.DataFrame) -> pd.DataFrame:
    X, _ = coerce_schema(X_raw)
    proba = bundle["model"].predict_proba(X)
    classes = np.array(bundle["model"].classes_)
    top = proba.argmax(1)
    out = X.copy()
    out.insert(0, "Predicted", classes[top])
    out.insert(1, "Confidence", proba.max(1).round(4))
    for i, c in enumerate(classes):
        out[f"P({c})"] = proba[:, i].round(4)
    return out


# --------------------------------------------------------------------------- #
# Evidence tools
# --------------------------------------------------------------------------- #
def feature_contributions(bundle: dict, row: pd.Series) -> dict:
    model, base = bundle["model"], bundle["baselines"]
    classes = list(model.classes_)
    row = row[INPUT_COLS]

    variants, meta = [row.copy()], [("original", None, None)]
    for c in INPUT_COLS:
        r = row.copy()
        r[c] = base[c] or None
        variants.append(r)
        meta.append(("column", c, None))
    for c in TEXT_COLS:
        toks = str(row[c] or "").split()
        for i, t in enumerate(toks):
            r = row.copy()
            r[c] = " ".join(toks[:i] + toks[i + 1:]) or None
            variants.append(r)
            meta.append(("token", c, t))

    P = model.predict_proba(pd.DataFrame(variants).reset_index(drop=True))
    k = int(P[0].argmax())
    pred, p0 = classes[k], float(P[0, k])

    def effect(delta: float) -> str:
        if abs(delta) < 0.005:
            return "negligible"
        return f"supports {pred}" if delta > 0 else f"works against {pred}"

    columns, tokens = [], []
    for (kind, col, tok), p in zip(meta[1:], P[1:]):
        delta = round(p0 - float(p[k]), 4)
        if kind == "column":
            columns.append({"column": col, "meaning": COLUMN_DESCRIPTIONS[col], "value": row[col],
                            "neutral_value": base[col] or "<blank>",
                            "contribution": delta, "effect": effect(delta),
                            f"P({pred})_if_neutralised": round(float(p[k]), 4),
                            "prediction_if_neutralised": classes[int(p.argmax())]})
        else:
            tokens.append({"column": col, "token": tok, "contribution": delta, "effect": effect(delta)})
    columns.sort(key=lambda d: -abs(d["contribution"]))
    tokens.sort(key=lambda d: -abs(d["contribution"]))
    order = P[0].argsort()[::-1]
    return {"predicted_class": pred, "confidence": round(p0, 4),
            "runner_up_class": classes[int(order[1])], "runner_up_probability": round(float(P[0, order[1]]), 4),
            "probabilities": {c: round(float(v), 4) for c, v in zip(classes, P[0])},
            "how_to_read": f"contribution = P({pred}) with the real value minus P({pred}) with the input "
                           f"neutralised. Positive = the input pushes TOWARD {pred}; negative = it pushes "
                           f"AWAY from {pred} (i.e. toward the runner-up).",
            "column_contributions": columns, "top_token_contributions": tokens[:8]}


def _embedder(bundle: dict):
    """Fitted preprocessing from the final model, used as the similarity space."""
    m = bundle["model"]
    pipe = m if isinstance(m, Pipeline) else m.estimators_[0]
    return pipe.named_steps["prep"]


def similar_training_records(bundle: dict, row: pd.Series, k: int = 5) -> dict:
    if "_nn" not in bundle:
        prep = _embedder(bundle)
        train = bundle["train"]
        bundle["_nn"] = NearestNeighbors(metric="cosine").fit(prep.transform(train[INPUT_COLS]))
    prep, train = _embedder(bundle), bundle["train"]
    k = int(max(1, min(k, 15)))
    dist, idx = bundle["_nn"].kneighbors(prep.transform(pd.DataFrame([row[INPUT_COLS]])), n_neighbors=k)
    recs = train.iloc[idx[0]].copy()
    recs.insert(0, "similarity", (1 - dist[0]).round(3))
    return {"neighbours": recs.to_dict("records"),
            "label_votes": recs["label"].value_counts().to_dict()}


def class_profile(bundle: dict, class_name: str) -> dict:
    train = bundle["train"]
    sub = train[train["label"] == class_name]
    if sub.empty:
        return {"error": f"Unknown class {class_name}. Known: {sorted(train['label'].unique())}"}
    amt = pd.to_numeric(sub["Col3"].str.replace(",", ""), errors="coerce")
    return {"class": class_name, "training_rows": len(sub),
            "share_of_training": round(len(sub) / len(train), 4),
            "top_vendors_Col1": sub["Col1"].value_counts().head(5).to_dict(),
            "top_account_codes_Col6": sub["Col6"].value_counts().head(5).to_dict(),
            "doc_type_mix_Col7": sub["Col7"].value_counts(normalize=True).round(3).to_dict(),
            "amount_median": float(amt.median()), "amount_p10_p90": [float(amt.quantile(.1)), float(amt.quantile(.9))]}


# --------------------------------------------------------------------------- #
# Offline fallback (no API key): deterministic narrative from the same evidence
# --------------------------------------------------------------------------- #
def fallback_explanation(bundle: dict, row: pd.Series) -> str:
    fc = feature_contributions(bundle, row)
    sim = similar_training_records(bundle, row, 5)
    pos = [c for c in fc["column_contributions"] if c["contribution"] > 0.005][:3]
    lines = [f"**Prediction: {fc['predicted_class']}** (confidence {fc['confidence']:.1%}).", ""]
    if pos:
        lines.append("Inputs that pushed the model toward this class:")
        for c in pos:
            lines.append(f"- **{c['column']}** ({c['meaning']}) = `{c['value']}` - neutralising it drops "
                         f"confidence by {c['contribution']:.1%}.")
    else:
        lines.append("No single column dominates - the prediction comes from the combination of inputs.")
    toks = [t for t in fc["top_token_contributions"] if t["contribution"] > 0.005][:3]
    if toks:
        lines.append("Most influential words: " + ", ".join(f"`{t['token']}` ({t['column']})" for t in toks) + ".")
    votes = ", ".join(f"{k}: {v}" for k, v in sim["label_votes"].items())
    lines.append(f"The 5 most similar training records are labelled {votes}.")
    lines.append("\n_(Rule-based fallback - add an OpenAI API key for the AI agent's narrative.)_")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
TOOLS = [
    {"type": "function", "function": {
        "name": "get_feature_contributions",
        "description": "Occlusion attribution for this row: how much each input column and each word "
                       "token raised or lowered the probability of the predicted class. Also returns "
                       "the full class probability distribution.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "find_similar_training_records",
        "description": "Nearest labelled training records to this row (cosine similarity in the model's "
                       "feature space) and their label vote. Use to check whether the prediction is "
                       "consistent with precedent.",
        "parameters": {"type": "object", "properties": {"k": {"type": "integer", "minimum": 1, "maximum": 15}}}}},
    {"type": "function", "function": {
        "name": "get_class_profile",
        "description": "Training-set profile of a class: size, typical vendors, account codes, doc types, "
                       "amount range. Use to explain what characterises a class or a runner-up class.",
        "parameters": {"type": "object", "properties": {"class_name": {"type": "string"}},
                       "required": ["class_name"]}}},
]

SYSTEM_PROMPT = """You are a model-explanation analyst for an accounting-transaction classifier.
Input columns (values are anonymised tokens like 'Word563'): {cols}
The model is a {model_name} trained on {n_train} labelled rows. Classes are imbalanced
(Category_1 is ~88% of data; Categories 3-6 have fewer than 25 examples each).

Use the tools to gather evidence BEFORE answering. Always call get_feature_contributions.
Call other tools when they add value (e.g. low confidence, a rare class, or a close runner-up).
Then write a concise explanation (max ~180 words) in markdown:
1. One-line verdict: predicted class and confidence.
2. The 2-4 inputs that contributed most and in which direction, quoting the numbers.
3. Whether similar historical records agree.
4. A reliability note (confidence level, runner-up class, rare-class caveat) if relevant.
Rules:
- Use each contribution's 'effect' field for direction. Every contribution is about the PREDICTED class
  only; never say an input pushes toward some other class unless 'prediction_if_neutralised' shows it.
- The runner-up is 'runner_up_class' (second-highest model probability). Neighbour label votes are
  separate historical evidence - report them as "k of n similar records were labelled X".
- Do not invent meanings for anonymised tokens; refer to them by their token id and column role."""


def run_agent(bundle: dict, row: pd.Series, api_key: str, model: str = "gpt-4o-mini",
              max_steps: int = 6) -> tuple[str, list[dict]]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    tools = {
        "get_feature_contributions": lambda **_: feature_contributions(bundle, row),
        "find_similar_training_records": lambda k=5, **_: similar_training_records(bundle, row, k),
        "get_class_profile": lambda class_name, **_: class_profile(bundle, class_name),
    }
    cols = "; ".join(f"{c} = {d}" for c, d in COLUMN_DESCRIPTIONS.items())
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(cols=cols, model_name=bundle["model_name"],
                                                           n_train=len(bundle["train"]))},
        {"role": "user", "content": "Explain the model's prediction for this row:\n"
                                    + json.dumps({c: row[c] for c in INPUT_COLS}, default=str)},
    ]
    trace = []
    for _ in range(max_steps):
        resp = client.chat.completions.create(model=model, messages=messages, tools=TOOLS, temperature=0.2)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content or "", trace
        messages.append(msg.model_dump(exclude_none=True))
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            try:
                result = tools[call.function.name](**args)
            except Exception as e:  # surface tool errors to the model instead of crashing
                result = {"error": str(e)}
            trace.append({"tool": call.function.name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result, default=str)})
    return "The agent did not finish within the step limit.", trace
