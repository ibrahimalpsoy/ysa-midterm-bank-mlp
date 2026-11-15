"""Sabitler, seed politikası ve dosya yolları.

Temel seed = öğrenci numarasının son beş hanesi + 2026 = 29040 + 2026 = 31066.
Tüm alt seed'ler temel seed'e sabit ofsetler eklenerek deterministik biçimde türetilir.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

STUDENT_LAST5 = 29040
BASE_SEED = STUDENT_LAST5 + 2026  # 31066

SEEDS = {
    "python": BASE_SEED + 0,          # random modülü
    "numpy": BASE_SEED + 1,           # NumPy global RNG
    "torch": BASE_SEED + 2,           # ağırlık başlangıcı + DataLoader karıştırma (birincil)
    "split": BASE_SEED + 3,           # %80 development / %20 test
    "outer_cv": BASE_SEED + 4,        # dış StratifiedKFold
    "inner_cv": BASE_SEED + 5,        # iç StratifiedKFold
    "random_search": BASE_SEED + 6,   # ParameterSampler
    "early_stop_split": BASE_SEED + 7,  # inner-train -> fit / %15 early-stop monitor
    "bootstrap": BASE_SEED + 8,       # test bootstrap CI
}
# Seed kararlılığı: birincil seed dahil önceden ilan edilmiş üç ağırlık-başlangıç seed'i
FINAL_TORCH_SEEDS = [SEEDS["torch"], BASE_SEED + 100, BASE_SEED + 200]
PRIMARY_TORCH_SEED = FINAL_TORCH_SEEDS[0]

# Deney sabitleri
TEST_SIZE = 0.20
N_OUTER = 3
N_INNER = 3
EARLY_STOP_FRACTION = 0.15
MAX_EPOCHS = 150
ES_PATIENCE = 15
LR_FACTOR = 0.5
LR_PATIENCE = 5
MIN_LR = 1e-5
THRESHOLD = 0.50
N_BOOTSTRAP = 1000

# Yollar
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "bank-additional.csv"
CONFIG_DIR = ROOT / "configs"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
MODEL_DIR = RESULTS_DIR / "models"
LOG_DIR = RESULTS_DIR / "training_logs"

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"


def set_global_seeds(torch_seed: int | None = None) -> None:
    """Python, NumPy ve PyTorch RNG'lerini sabitler."""
    random.seed(SEEDS["python"])
    np.random.seed(SEEDS["numpy"])
    os.environ["PYTHONHASHSEED"] = str(SEEDS["python"])
    try:
        import torch

        torch.manual_seed(SEEDS["torch"] if torch_seed is None else torch_seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)  # CPU'da bit-düzeyi tekrarlanabilirlik için
    except ImportError:
        pass


def ensure_dirs() -> None:
    for d in (DATA_DIR, CONFIG_DIR, RESULTS_DIR, FIG_DIR, MODEL_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
