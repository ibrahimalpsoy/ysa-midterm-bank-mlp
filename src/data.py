"""Veri indirme, okuma, doğrulama ve %80/%20 stratified bölünme."""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C

TARGET = "y"
DROP_COLS = ["duration"]  # görüşme öncesi bilinmez -> performans görülmeden önce çıkarılır

NUMERIC_COLS = [
    "age", "campaign", "pdays", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
]
CATEGORICAL_COLS = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]
EXPECTED_N_ROWS = 4119
EXPECTED_N_INPUTS = 20


def download_uci(dest: Path = C.RAW_CSV) -> Path:
    """Resmî UCI zip'inden yalnızca bank-additional/bank-additional.csv dosyasını çıkarır."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(C.UCI_ZIP_URL, timeout=60) as r:
        outer = zipfile.ZipFile(io.BytesIO(r.read()))
    inner_name = next(n for n in outer.namelist() if n.endswith("bank-additional.zip"))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))
    csv_name = next(n for n in inner.namelist()
                    if n.endswith("bank-additional/bank-additional.csv"))
    dest.write_bytes(inner.read(csv_name))
    return dest


def load_raw(path: Path = C.RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    return df


def validate_raw(df: pd.DataFrame) -> dict:
    """Değişken adları, tipler ve hedef etiketlerinin doğru okunduğunu kontrol eder."""
    assert df.shape == (EXPECTED_N_ROWS, EXPECTED_N_INPUTS + 1), df.shape
    for c in NUMERIC_COLS + CATEGORICAL_COLS + DROP_COLS + [TARGET]:
        assert c in df.columns, f"eksik sütun: {c}"
    for c in NUMERIC_COLS:
        assert pd.api.types.is_numeric_dtype(df[c]), f"{c} sayısal değil"
    assert set(df[TARGET].unique()) == {"yes", "no"}, df[TARGET].unique()
    return {
        "n_rows": int(df.shape[0]),
        "n_inputs": int(df.shape[1] - 1),
        "n_numeric_after_drop": len(NUMERIC_COLS),
        "n_categorical": len(CATEGORICAL_COLS),
        "target_counts": df[TARGET].value_counts().to_dict(),
    }


def to_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """'duration' çıkarılır; y: yes=1, no=0."""
    X = df.drop(columns=DROP_COLS + [TARGET]).copy()
    y = (df[TARGET] == "yes").astype(int).to_numpy()
    return X, y


def duplicate_report(df: pd.DataFrame) -> dict:
    """Tam kopya kontrolü: (a) ham 21 sütun, (b) duration çıkarıldıktan sonra özellik+hedef,
    (c) yalnızca özellikler (çelişkili etiket kontrolü)."""
    X, y = to_xy(df)
    feat_hash = pd.util.hash_pandas_object(X, index=False)
    full = X.assign(_y=y)
    out = {
        "raw_exact_duplicates": int(df.duplicated().sum()),
        "model_input_exact_duplicates_with_target": int(full.duplicated().sum()),
        "model_input_feature_duplicates": int(X.duplicated().sum()),
    }
    grp = pd.DataFrame({"h": feat_hash, "y": y}).groupby("h")["y"].nunique()
    out["feature_groups_with_conflicting_labels"] = int((grp > 1).sum())
    out["n_feature_groups_size_gt1"] = int((feat_hash.value_counts() > 1).sum())
    return out


def stratified_split(df: pd.DataFrame, seed: int = C.SEEDS["split"]):
    """%80 development / %20 bağımsız test (stratified). Orijinal satır indeksleri korunur."""
    _, y = to_xy(df)
    idx = np.arange(len(df))
    dev_idx, test_idx = train_test_split(
        idx, test_size=C.TEST_SIZE, stratify=y, random_state=seed, shuffle=True
    )
    return np.sort(dev_idx), np.sort(test_idx)


def split_table(df: pd.DataFrame, dev_idx, test_idx) -> pd.DataFrame:
    _, y = to_xy(df)
    rows = [("development", i, y[i]) for i in dev_idx] + [("test", i, y[i]) for i in test_idx]
    t = pd.DataFrame(rows, columns=["subset", "original_row_index", "y"])
    t = t.sort_values("original_row_index").reset_index(drop=True)
    return t


def split_summary(df: pd.DataFrame, dev_idx, test_idx) -> pd.DataFrame:
    _, y = to_xy(df)
    recs = []
    for name, ix in (("development", dev_idx), ("test", test_idx), ("all", np.arange(len(y)))):
        recs.append({
            "subset": name, "n": len(ix), "n_pos": int(y[ix].sum()),
            "n_neg": int((1 - y[ix]).sum()), "pos_rate": float(y[ix].mean()),
        })
    return pd.DataFrame(recs)
