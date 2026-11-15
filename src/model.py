"""PyTorch MLP ve eğitim döngüsü (early stopping + ReduceLROnPlateau)."""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn

from . import config as C
from .preprocessing import build_preprocessor

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh}


class MLP(nn.Module):
    """Girdi -> [Linear -> Aktivasyon -> (Dropout)] x L -> Linear(1) (tek logit)."""

    def __init__(self, n_in: int, hidden: tuple[int, ...], activation: str = "relu",
                 dropout: float = 0.0):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), ACTIVATIONS[activation]()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logit


def count_params(n_in: int, hidden: tuple[int, ...]) -> int:
    """Öğrenilebilir parametre sayısı: sum((girdi+1) * çıktı) (ağırlık + bias)."""
    dims = [n_in, *hidden, 1]
    return int(sum((dims[i] + 1) * dims[i + 1] for i in range(len(dims) - 1)))


def make_optimizer(name: str, params, lr: float, weight_decay: float):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(name)


@dataclass
class TrainResult:
    model: MLP
    history: pd.DataFrame
    best_epoch: int
    stopped_epoch: int
    train_time_s: float
    n_params: int
    warnings: list = field(default_factory=list)


def train_mlp(X_fit: np.ndarray, y_fit: np.ndarray, X_mon: np.ndarray, y_mon: np.ndarray,
              cfg: dict, torch_seed: int) -> TrainResult:
    """X_fit: ağırlık güncellemesi; X_mon: early-stop/scheduler izleme (BCE, mode=min)."""
    torch.manual_seed(torch_seed)
    g = torch.Generator().manual_seed(torch_seed)
    hidden = tuple(cfg["hidden"])
    model = MLP(X_fit.shape[1], hidden, cfg.get("activation", "relu"), cfg.get("dropout", 0.0))
    opt = make_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"], cfg["weight_decay"])
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=C.LR_FACTOR, patience=C.LR_PATIENCE, min_lr=C.MIN_LR)
        if cfg.get("scheduler", True) else None)
    loss_fn = nn.BCEWithLogitsLoss()

    Xf = torch.tensor(X_fit, dtype=torch.float32)
    yf = torch.tensor(y_fit, dtype=torch.float32)
    Xm = torch.tensor(X_mon, dtype=torch.float32)
    ym = torch.tensor(y_mon, dtype=torch.float32)
    n, bs = len(Xf), int(cfg["batch_size"])

    best_loss, best_state, best_epoch, wait = math.inf, None, 0, 0
    rows, warns = [], []
    t0 = time.perf_counter()
    epoch = 0
    for epoch in range(1, C.MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xf[idx]), yf[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        train_loss = tot / n
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xm), ym).item()
        lr_now = opt.param_groups[0]["lr"]
        rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr_now})
        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            warns.append(f"non-finite loss at epoch {epoch}")
            break
        if sched is not None:
            sched.step(val_loss)
        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch, wait = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= C.ES_PATIENCE:
                break
    elapsed = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)  # best weights restore
    else:
        warns.append("no valid epoch")
    if epoch == C.MAX_EPOCHS and wait < C.ES_PATIENCE:
        warns.append("max_epochs reached without early stop (possible non-convergence)")
    if best_epoch <= 2:
        warns.append(f"best epoch very early ({best_epoch})")
    return TrainResult(model, pd.DataFrame(rows), best_epoch, epoch, elapsed,
                       count_params(X_fit.shape[1], hidden), warns)


class MLPPipeline:
    """Ham DataFrame -> (fit/early-stop ayrımı) -> fold-local ön işleme -> MLP.

    fit(X, y):
      1) X stratified olarak %85 'fit' / %15 'early-stop monitor' alt kümelerine ayrılır,
      2) ön işleyici yalnızca 'fit' alt kümesinde fit edilir; monitor'a yalnızca transform,
      3) MLP 'fit' ile eğitilir, early stopping + scheduler monitor BCE kaybını izler.
    """

    def __init__(self, cfg: dict, torch_seed: int = C.SEEDS["torch"],
                 es_seed: int = C.SEEDS["early_stop_split"]):
        self.cfg = cfg
        self.torch_seed = torch_seed
        self.es_seed = es_seed

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        idx = np.arange(len(y))
        fit_idx, mon_idx = train_test_split(idx, test_size=C.EARLY_STOP_FRACTION, stratify=y,
                                            random_state=self.es_seed, shuffle=True)
        self.pre_ = build_preprocessor(self.cfg["scaler"], self.cfg.get("clipping", False),
                                       self.cfg.get("unknown", "category"))
        Xf = self.pre_.fit_transform(X.iloc[fit_idx])
        Xm = self.pre_.transform(X.iloc[mon_idx])
        self.result_ = train_mlp(Xf, y[fit_idx], Xm, y[mon_idx], self.cfg, self.torch_seed)
        self.n_fit_, self.n_monitor_ = len(fit_idx), len(mon_idx)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xt = torch.tensor(self.pre_.transform(X), dtype=torch.float32)
        m = self.result_.model
        m.eval()
        with torch.no_grad():
            return torch.sigmoid(m(Xt)).numpy()
