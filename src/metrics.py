"""Sınıflandırma metrikleri ve bootstrap güven aralıkları."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)

from . import config as C

METRIC_ORDER = ["AP", "ROC_AUC", "Accuracy", "Precision", "Sensitivity", "Specificity",
                "F1", "Balanced_Accuracy"]


def compute_metrics(y_true, proba, threshold: float = C.THRESHOLD) -> dict:
    """AP = average_precision_score (adım fonksiyonu); trapez PR-AUC ayrı sütunda raporlanır."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    prec, rec, _ = precision_recall_curve(y_true, proba)
    return {
        "AP": average_precision_score(y_true, proba),
        "ROC_AUC": roc_auc_score(y_true, proba),
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Sensitivity": recall_score(y_true, pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "F1": f1_score(y_true, pred, zero_division=0),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, pred),
        "PR_AUC_trapezoid": auc(rec, prec),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


def mean_sd(df: pd.DataFrame, group: str, cols: list[str]) -> pd.DataFrame:
    """Kat sonuçlarının ortalama ± örnek standart sapması (ddof=1)."""
    g = df.groupby(group)[cols]
    m, s = g.mean(), g.std(ddof=1)
    out = pd.DataFrame(index=m.index)
    for c in cols:
        out[c] = [f"{a:.4f} ± {b:.4f}" for a, b in zip(m[c], s[c])]
    return out


def bootstrap_ci(y_true, proba, n_boot: int = C.N_BOOTSTRAP, seed: int = C.SEEDS["bootstrap"],
                 threshold: float = C.THRESHOLD, alpha: float = 0.05) -> pd.DataFrame:
    """Yüzdelik bootstrap %95 CI (ROC-AUC, AP, F1). İki sınıfı içermeyen örneklem yeniden çekilir."""
    rng = np.random.default_rng(seed)
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    n = len(y_true)
    stats, redraws = {"ROC_AUC": [], "AP": [], "F1": []}, 0
    while len(stats["AP"]) < n_boot:
        ix = rng.integers(0, n, n)
        yt = y_true[ix]
        if yt.min() == yt.max():  # iki sınıf kontrolü
            redraws += 1
            continue
        p = proba[ix]
        stats["ROC_AUC"].append(roc_auc_score(yt, p))
        stats["AP"].append(average_precision_score(yt, p))
        stats["F1"].append(f1_score(yt, (p >= threshold).astype(int), zero_division=0))
    point = {"ROC_AUC": roc_auc_score(y_true, proba),
             "AP": average_precision_score(y_true, proba),
             "F1": f1_score(y_true, (proba >= threshold).astype(int), zero_division=0)}
    rows = []
    for k, v in stats.items():
        lo, hi = np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        rows.append({"metric": k, "point": point[k], "ci_low": lo, "ci_high": hi,
                     "n_boot": n_boot, "single_class_redraws": redraws})
    return pd.DataFrame(rows)
