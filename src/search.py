"""Manuel / otomatik (random) arama, nested CV, referans modeller ve nihai seçim."""
from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from sklearn.pipeline import Pipeline

from . import config as C
from .metrics import compute_metrics
from .model import MLPPipeline
from .preprocessing import build_preprocessor

# ----------------------------------------------------------------------------- adaylar
MANUAL_DEFAULTS = {
    "optimizer": "adamw", "hidden": [32], "scaler": "standard", "lr": 1e-3,
    "weight_decay": 1e-4, "scheduler": True, "batch_size": 64, "dropout": 0.0,
    "activation": "relu", "clipping": False, "unknown": "category",
}
MANUAL_SPEC = [
    ("M1", {}, "Temel manuel konfigürasyon"),
    ("M2", {"optimizer": "adam"}, "Optimizer etkisi (AdamW -> Adam)"),
    ("M3", {"optimizer": "sgd"}, "Optimizer etkisi (AdamW -> SGD+Momentum 0.9)"),
    ("M4", {"scaler": "minmax"}, "Ölçekleme etkisi (Standard -> MinMax)"),
    ("M5", {"hidden": [64, 32]}, "Mimari etkisi ((32,) -> (64,32))"),
    ("M6", {"scheduler": False}, "Learning-rate-drop etkisi (scheduler kapalı)"),
]

SEARCH_SPACE = {
    "optimizer": ["sgd", "adam", "adamw"],
    "hidden": [[16], [32], [64], [64, 32], [128, 64]],
    "activation": ["relu", "tanh"],
    "lr": {"adam": [1e-4, 1e-2], "adamw": [1e-4, 1e-2], "sgd": [1e-3, 1e-1], "scale": "log"},
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
    "batch_size": [32, 64, 128],
    "scaler": ["standard", "minmax", "robust"],
    "dropout": [0.0, 0.2],
    "clipping": [False, True],
    "candidates_per_optimizer": 2,
    "fixed": {"scheduler": True, "unknown": "category", "momentum_sgd": 0.9},
}


def manual_candidates() -> list[dict]:
    out = []
    for code, diff, purpose in MANUAL_SPEC:
        cfg = {**MANUAL_DEFAULTS, **diff}
        out.append({"code": code, "purpose": purpose, **cfg})
    return out


def random_candidates(seed: int = C.SEEDS["random_search"]) -> list[dict]:
    """Her optimizer için tam 2 aday; scikit-learn ParameterSampler, log-uniform LR."""
    rs = np.random.RandomState(seed)
    out, k = [], 1
    for opt in SEARCH_SPACE["optimizer"]:
        lo, hi = SEARCH_SPACE["lr"][opt]
        dist = {
            "hidden": SEARCH_SPACE["hidden"], "activation": SEARCH_SPACE["activation"],
            "lr": loguniform(lo, hi), "weight_decay": SEARCH_SPACE["weight_decay"],
            "batch_size": SEARCH_SPACE["batch_size"], "scaler": SEARCH_SPACE["scaler"],
            "dropout": SEARCH_SPACE["dropout"], "clipping": SEARCH_SPACE["clipping"],
        }
        for p in ParameterSampler(dist, n_iter=SEARCH_SPACE["candidates_per_optimizer"],
                                  random_state=rs):
            cfg = {"code": f"R{k}", "purpose": f"Random search ({opt})", "optimizer": opt,
                   "scheduler": True, "unknown": "category", **p}
            cfg["lr"] = float(cfg["lr"])
            cfg["hidden"] = list(cfg["hidden"])
            cfg["batch_size"] = int(cfg["batch_size"])
            cfg["weight_decay"] = float(cfg["weight_decay"])
            cfg["dropout"] = float(cfg["dropout"])
            cfg["clipping"] = bool(cfg["clipping"])
            out.append(cfg)
            k += 1
    return out


def load_candidates() -> dict[str, list[dict]]:
    return {
        "manual": json.loads((C.CONFIG_DIR / "manual_configs.json").read_text())["candidates"],
        "random": json.loads((C.CONFIG_DIR / "random_configs.json").read_text())["candidates"],
    }


def cfg_short(c: dict) -> str:
    return (f"{c['optimizer']}|{tuple(c['hidden'])}|{c['activation']}|lr={c['lr']:.2e}|"
            f"wd={c['weight_decay']:g}|bs={c['batch_size']}|{c['scaler']}|do={c['dropout']}|"
            f"clip={c['clipping']}|sched={c['scheduler']}")


# ----------------------------------------------------------------------------- iç CV
def inner_cv(X, y, candidates, stage: str, method: str, seed=C.SEEDS["inner_cv"],
             log_sink: list | None = None):
    """Adayları aynı iç kat bölünmeleriyle değerlendirir; AP ile karşılaştırır."""
    skf = StratifiedKFold(n_splits=C.N_INNER, shuffle=True, random_state=seed)
    folds = list(skf.split(X, y))
    rows = []
    for cand in candidates:
        for f, (tr, va) in enumerate(folds, 1):
            t0 = time.perf_counter()
            m = MLPPipeline(cand).fit(X.iloc[tr], y[tr])
            p = m.predict_proba(X.iloc[va])
            r = m.result_
            rows.append({
                "stage": stage, "method": method, "code": cand["code"], "inner_fold": f,
                "AP": average_precision_score(y[va], p), "best_epoch": r.best_epoch,
                "stopped_epoch": r.stopped_epoch, "train_time_s": r.train_time_s,
                "total_time_s": time.perf_counter() - t0, "n_params": r.n_params,
                "n_fit": m.n_fit_, "n_monitor": m.n_monitor_, "n_val": len(va),
                "warnings": "; ".join(r.warnings),
            })
            if log_sink is not None:
                h = r.history.copy()
                h.insert(0, "inner_fold", f)
                h.insert(0, "code", cand["code"])
                h.insert(0, "stage", stage)
                log_sink.append(h)
    long = pd.DataFrame(rows)
    return long, summarize_inner(long, candidates)


def summarize_inner(long: pd.DataFrame, candidates) -> pd.DataFrame:
    g = long.groupby(["stage", "method", "code"], sort=False)
    s = g.agg(AP_mean=("AP", "mean"), AP_sd=("AP", lambda v: v.std(ddof=1)),
              best_epoch_mean=("best_epoch", "mean"), stopped_epoch_mean=("stopped_epoch", "mean"),
              train_time_s_total=("train_time_s", "sum"), n_params=("n_params", "first"),
              n_warnings=("warnings", lambda v: int((v != "").sum()))).reset_index()
    cmap = {c["code"]: c for c in candidates}
    s["config"] = s["code"].map(lambda k: cfg_short(cmap[k]))
    return s


def select_best(summary: pd.DataFrame) -> str:
    """Önceden tanımlı kural: en yüksek iç CV mean AP; eşitlikte düşük SD, sonra düşük kod sırası."""
    s = summary.sort_values(["AP_mean", "AP_sd"], ascending=[False, True], kind="mergesort")
    return s.iloc[0]["code"]


# ----------------------------------------------------------------------------- referanslar
def make_logreg() -> Pipeline:
    """Sabit ön işleme kuralı: Standard, clipping yok, 'unknown' ayrı kategori."""
    return Pipeline([("pre", build_preprocessor("standard", False, "category")),
                     ("clf", LogisticRegression(max_iter=5000))])


def make_dummy() -> DummyClassifier:
    return DummyClassifier(strategy="prior")  # sabit sınıf önceliği olasılığı; 0.5 eşikte hep 0


def fit_reference(model, X, y):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        t0 = time.perf_counter()
        model.fit(X, y)
        el = time.perf_counter() - t0
    return model, el, [str(x.message) for x in w if issubclass(x.category, ConvergenceWarning)]


# ----------------------------------------------------------------------------- nested CV
def nested_cv(X, y, cands: dict[str, list[dict]], log_sink: list | None = None):
    outer = StratifiedKFold(n_splits=C.N_OUTER, shuffle=True, random_state=C.SEEDS["outer_cv"])
    inner_long, inner_summ, outer_rows, refit_logs = [], [], [], []
    for k, (tr, ev) in enumerate(outer.split(X, y), 1):
        Xtr, ytr, Xev, yev = X.iloc[tr], y[tr], X.iloc[ev], y[ev]
        for method in ("manual", "random"):
            long, summ = inner_cv(Xtr, ytr, cands[method], f"outer{k}", method,
                                  log_sink=log_sink)
            inner_long.append(long)
            inner_summ.append(summ)
            best = select_best(summ)
            cfg = next(c for c in cands[method] if c["code"] == best)
            t0 = time.perf_counter()
            m = MLPPipeline(cfg).fit(Xtr, ytr)  # outer-train üzerinde yeniden eğitim
            el = time.perf_counter() - t0
            p = m.predict_proba(Xev)           # outer evaluation: yalnızca ölçüm
            h = m.result_.history.copy()
            h.insert(0, "code", best)
            h.insert(0, "stage", f"outer{k}_refit_{method}")
            refit_logs.append(h)
            outer_rows.append({"outer_fold": k, "model": f"MLP-{method}", "selected": best,
                               "inner_AP_mean": float(summ.set_index("code").loc[best, "AP_mean"]),
                               "best_epoch": m.result_.best_epoch, "fit_time_s": el,
                               "n_outer_train": len(tr), "n_outer_eval": len(ev),
                               "warnings": "; ".join(m.result_.warnings),
                               **compute_metrics(yev, p)})
        for name, mk in (("LogisticRegression", make_logreg), ("Dummy(prior)", make_dummy)):
            mdl, el, w = fit_reference(mk(), Xtr, ytr)
            p = mdl.predict_proba(Xev)[:, 1]
            outer_rows.append({"outer_fold": k, "model": name, "selected": "-",
                               "inner_AP_mean": np.nan, "best_epoch": np.nan, "fit_time_s": el,
                               "n_outer_train": len(tr), "n_outer_eval": len(ev),
                               "warnings": "; ".join(w), **compute_metrics(yev, p)})
    return (pd.concat(inner_long, ignore_index=True), pd.concat(inner_summ, ignore_index=True),
            pd.DataFrame(outer_rows), pd.concat(refit_logs, ignore_index=True))
