"""Kat-içi (fold-local) ön işleme.

Tüm öğrenilen parametreler (median, kategori sözlüğü, scaler istatistikleri, clipping eşikleri)
yalnızca `fit` çağrılan alt kümeden öğrenilir; diğer alt kümelere yalnızca `transform` uygulanır.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

from .data import CATEGORICAL_COLS, NUMERIC_COLS

SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
PDAYS_NEVER = 999


class PdaysEncoder(BaseEstimator, TransformerMixin):
    """Durumsuz (öğrenmeyen) kural: pdays=999 'daha önce temas yok' anlamına gelir.
    - `prev_contacted` ikili göstergesi eklenir (999 -> 0, aksi 1),
    - pdays içindeki 999 değeri NaN yapılır; sonra kat-içi median imputation uygulanır.
    Böylece 999 değeri aykırı gözlem gibi silinmez ve ölçekleyiciyi bozmaz."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        never = X["pdays"] == PDAYS_NEVER
        X["prev_contacted"] = (~never).astype(float)
        X.loc[never, "pdays"] = np.nan
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(input_features) + ["prev_contacted"], dtype=object)


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Eğitim alt kümesinde öğrenilen yüzdelik eşiklerle kırpma (winsorizing)."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99, enabled: bool = True):
        self.lower = lower
        self.upper = upper
        self.enabled = enabled

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lo_, self.hi_) if self.enabled else X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class UnknownToNaN(BaseEstimator, TransformerMixin):
    """'unknown' -> NaN (yalnızca 'unknown'u imputation ile ele alma alternatifinde)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X).replace("unknown", np.nan)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_preprocessor(scaler: str = "standard", clipping: bool = False,
                       unknown: str = "category") -> Pipeline:
    """scaler: standard|minmax|robust; clipping: kat-içi %1-%99 kırpma;
    unknown: 'category' (ayrı kategori, varsayılan) | 'most_frequent' (kat-içi imputation)."""
    num_cols = NUMERIC_COLS  # pdays dahil (999 -> NaN)
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clip", QuantileClipper(enabled=clipping)),
        ("scale", SCALERS[scaler]()),
    ])
    if unknown == "category":
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),  # gerçek NaN yoksa etkisiz
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
    elif unknown == "most_frequent":
        cat_pipe = Pipeline([
            ("unk", UnknownToNaN()),
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
    else:
        raise ValueError(unknown)
    ct = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("flag", "passthrough", ["prev_contacted"]),
        ("cat", cat_pipe, CATEGORICAL_COLS),
    ], remainder="drop", verbose_feature_names_out=False)
    return Pipeline([("pdays", PdaysEncoder()), ("ct", ct)])


def fitted_state(pre: Pipeline) -> dict:
    """Öğrenilmiş dönüşüm istatistiklerini JSON'a yazılabilir biçimde döndürür."""
    ct = pre.named_steps["ct"]
    num = ct.named_transformers_["num"]
    cat = ct.named_transformers_["cat"]
    scaler = num.named_steps["scale"]
    state = {
        "numeric_columns": list(NUMERIC_COLS),
        "numeric_medians": num.named_steps["impute"].statistics_.tolist(),
        "clipping_enabled": bool(num.named_steps["clip"].enabled),
        "clip_low": num.named_steps["clip"].lo_.tolist(),
        "clip_high": num.named_steps["clip"].hi_.tolist(),
        "scaler": type(scaler).__name__,
        "categorical_columns": list(CATEGORICAL_COLS),
        "categories": [list(map(str, c)) for c in cat.named_steps["ohe"].categories_],
        "feature_names_out": list(ct.get_feature_names_out()),
    }
    for attr in ("mean_", "scale_", "data_min_", "data_max_", "center_"):
        if hasattr(scaler, attr) and getattr(scaler, attr) is not None:
            state[f"scaler_{attr.rstrip('_')}"] = np.asarray(getattr(scaler, attr)).tolist()
    return state
