"""Preprocessing and feature engineering shared by training and live inference.

Everything that touches raw columns Col1..Col7 lives here so the exact same
transformations run in train.py and in the dashboard's upload pipeline.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

INPUT_COLS = ["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7"]
LABEL_COL = "ClassificationLabel"
TEXT_COLS = ["Col1", "Col4", "Col6"]

COLUMN_DESCRIPTIONS = {
    "Col1": "Vendor / payee name (anonymised word tokens)",
    "Col2": "Document or invoice reference ID",
    "Col3": "Transaction amount",
    "Col4": "Line description (anonymised word tokens)",
    "Col5": "Posting date",
    "Col6": "Account / GL code description (anonymised word tokens)",
    "Col7": "Supporting document type",
}

NUMERIC_FEATURES = [
    "amount_signed_log", "amount_is_negative", "amount_is_round",
    "month", "day", "is_month_start", "is_month_end",
    "col2_len", "col2_digit_frac", "col2_has_sep", "col2_sci_notation",
    "col1_ntok", "col4_ntok", "col6_ntok", "col4_missing",
]

# Col2 (reference ID) and Col5 (posting date) are still engineered above for the
# EDA charts, but the model ignores them: dropping them left CV scores unchanged.
DROPPED_COLS = ["Col2", "Col5"]
MODEL_CATEGORICAL = ["col7"]
MODEL_NUMERIC_FEATURES = [f for f in NUMERIC_FEATURES
                          if not f.startswith("col2_") and f not in ("month", "day", "is_month_start", "is_month_end")]


# --------------------------------------------------------------------------- #
# Label cleaning
# --------------------------------------------------------------------------- #
WORD_NUMS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9}


def normalize_label(raw) -> str | float:
    """Map the 11 inconsistent spellings ('Category 3', 'Categry_6',
    'category_1', 'Category4', 'Category _3' ...) onto 'Category_<n>'.
    Also handles 'Category Three', 'Category_07' and 'v2 Category_4'
    (takes the number after 'cat...', not the first number). Unreadable -> NaN."""
    if pd.isna(raw):
        return np.nan
    s = str(raw).lower()
    for word, n in WORD_NUMS.items():                     # "three" -> "3"
        s = re.sub(rf"\b{word}\b", str(n), s)
    m = (re.search(r"cat[a-z]*\D*?(\d+)", s)              # number right after "cat..."
         or re.search(r"(\d+)", s))                       # fallback: any number ("3")
    return f"Category_{int(m.group(1))}" if m else np.nan  # int() drops leading zeros


# --------------------------------------------------------------------------- #
# Schema coercion: make any uploaded file look like Col1..Col7
# --------------------------------------------------------------------------- #
_ALIASES = {c.lower(): c for c in INPUT_COLS}
_ALIASES.update({l: f"Col{i}" for i, l in enumerate("abcdefg", start=1)})


def coerce_schema(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return a frame with exactly Col1..Col7 (as strings) plus warnings.

    Accepts headers Col1..Col7 (any case/whitespace) or A..G. If none match,
    the first seven columns are taken positionally. Missing columns are
    created empty so the pipeline still runs.
    """
    warnings: list[str] = []
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {c: _ALIASES[c.lower().replace(" ", "")] for c in df.columns
              if c.lower().replace(" ", "") in _ALIASES}
    df = df.rename(columns=rename)

    if not any(c in df.columns for c in INPUT_COLS):
        feature_like = [c for c in df.columns if c != LABEL_COL][:7]
        warnings.append("Headers not recognised - mapped the first 7 columns positionally.")
        df = df.rename(columns=dict(zip(feature_like, INPUT_COLS)))

    for c in INPUT_COLS:
        if c not in df.columns:
            warnings.append(f"Column {c} missing - filled with blanks.")
            df[c] = np.nan

    out = df[INPUT_COLS].astype(object)
    out = out.where(out.notna(), None)
    out = out.apply(lambda s: s.map(lambda v: None if v is None else str(v).strip()))
    out = out.replace({"": None, "nan": None, "NaN": None, "None": None})
    return out, warnings


# --------------------------------------------------------------------------- #
# Raw-value parsers
# --------------------------------------------------------------------------- #
def parse_amount(s: pd.Series) -> pd.Series:
    """'1,092.50' -> 1092.5 ; '(45.00)' -> -45 ; garbage -> NaN."""
    txt = s.astype("string").str.replace(",", "", regex=False).str.replace("$", "", regex=False).str.strip()
    neg_paren = txt.str.match(r"^\(.*\)$", na=False)
    txt = txt.str.replace(r"[()]", "", regex=True)
    val = pd.to_numeric(txt, errors="coerce")
    return val.where(~neg_paren, -val.abs())


def parse_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", format="mixed")


def id_shape(v) -> str:
    """Collapse an ID into its character-class shape so the model learns the
    *format* of a vendor's reference numbers, not the specific number.
    'KBNZGNNWMA072618' -> 'A9', '4.80Z+11' -> '9.9A+9', '108644-1359' -> '9-9'."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "<missing>"
    out, prev = [], None
    for ch in str(v):
        t = "9" if ch.isdigit() else "A" if ch.isupper() else "a" if ch.islower() else ch
        if t != prev:
            out.append(t)
        prev = t
    return "".join(out)


def _ntok(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.split().str.len()


# --------------------------------------------------------------------------- #
# Feature engineering (stateless -> safe inside FunctionTransformer)
# --------------------------------------------------------------------------- #
def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    X, _ = coerce_schema(X)
    amt = parse_amount(X["Col3"])
    dt = parse_date(X["Col5"])
    col2 = X["Col2"].fillna("").astype(str)

    f = pd.DataFrame(index=X.index)
    for c in TEXT_COLS:
        f[f"{c}_text"] = X[c].fillna("").astype(str)
    f["col2_shape"] = X["Col2"].map(id_shape)
    f["col7"] = X["Col7"].fillna("<missing>").astype(str)

    f["amount_signed_log"] = np.sign(amt) * np.log1p(amt.abs())
    f["amount_is_negative"] = (amt < 0).astype(float).where(amt.notna())
    f["amount_is_round"] = (amt % 100 == 0).astype(float).where(amt.notna())
    f["month"] = dt.dt.month
    f["day"] = dt.dt.day
    f["is_month_start"] = (dt.dt.day == 1).astype(float).where(dt.notna())
    f["is_month_end"] = dt.dt.is_month_end.astype(float).where(dt.notna())
    f["col2_len"] = col2.str.len()
    f["col2_digit_frac"] = col2.str.count(r"\d") / col2.str.len().replace(0, np.nan)
    f["col2_has_sep"] = col2.str.contains(r"[-/_ ]").astype(float)
    f["col2_sci_notation"] = col2.str.contains(r"^\d\.\d+[A-Z]?\+\d+$").astype(float)
    f["col1_ntok"] = _ntok(X["Col1"])
    f["col4_ntok"] = _ntok(X["Col4"])
    f["col6_ntok"] = _ntok(X["Col6"])
    f["col4_missing"] = X["Col4"].isna().astype(float)
    f[NUMERIC_FEATURES] = f[NUMERIC_FEATURES].astype(float)
    return f


def _tfidf(ngram_max: int, max_features: int | None) -> TfidfVectorizer:
    # Tokens are opaque 'WordNNN' ids: split on whitespace, no lowercasing.
    return TfidfVectorizer(token_pattern=r"\S+", lowercase=False, ngram_range=(1, ngram_max),
                           max_features=max_features, sublinear_tf=True)


def build_preprocessor(ngram_max: int = 2, max_features: int | None = None) -> Pipeline:
    """Linear models get uni+bigrams over the full vocabulary; tree models get a
    compact unigram vocabulary (trees split one feature at a time, so bigrams add
    little and a dense 10k-wide matrix would make boosting slow)."""
    encode = ColumnTransformer(
        transformers=[
            ("col1_tfidf", _tfidf(ngram_max, max_features), "Col1_text"),
            ("col4_tfidf", _tfidf(ngram_max, max_features), "Col4_text"),
            ("col6_tfidf", _tfidf(ngram_max, max_features), "Col6_text"),
            ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=2),
             MODEL_CATEGORICAL),
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                                  ("scale", StandardScaler())]), MODEL_NUMERIC_FEATURES),
        ],
        sparse_threshold=1.0,
    )
    return Pipeline([
        ("engineer", FunctionTransformer(engineer_features, validate=False)),
        ("encode", encode),
    ])


def to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else X


def densify() -> FunctionTransformer:
    return FunctionTransformer(to_dense, accept_sparse=True, validate=False)
