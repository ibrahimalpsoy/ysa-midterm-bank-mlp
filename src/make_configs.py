"""Nested CV başlamadan ÖNCE çalıştırılır: arama uzayı, 6 manuel ve 6 otomatik aday,
eğitim ayarları configs/ altına yazılır ve ayrı bir commit ile sabitlenir.

Kullanım:  python -m src.make_configs
"""
from __future__ import annotations

import json

from . import config as C
from .search import MANUAL_DEFAULTS, SEARCH_SPACE, manual_candidates, random_candidates


def main() -> None:
    C.ensure_dirs()
    training = {
        "base_seed": C.BASE_SEED, "seed_formula": "son 5 hane (29040) + 2026",
        "derived_seeds": C.SEEDS, "final_torch_seeds": C.FINAL_TORCH_SEEDS,
        "primary_torch_seed": C.PRIMARY_TORCH_SEED,
        "test_size": C.TEST_SIZE, "outer_folds": C.N_OUTER, "inner_folds": C.N_INNER,
        "early_stop_monitor_fraction": C.EARLY_STOP_FRACTION, "max_epochs": C.MAX_EPOCHS,
        "early_stopping": {"monitor": "internal validation BCE loss", "mode": "min",
                           "patience": C.ES_PATIENCE, "restore_best_weights": True,
                           "min_delta": 1e-6},
        "lr_scheduler": {"type": "ReduceLROnPlateau", "mode": "min", "factor": C.LR_FACTOR,
                         "patience": C.LR_PATIENCE, "min_lr": C.MIN_LR,
                         "step": "her epoch sonunda early-stop alt kümesi BCE kaybı ile"},
        "loss": "BCEWithLogitsLoss (tek logit çıkış, olasılık = sigmoid)",
        "hp_selection_metric": "inner CV average_precision_score (AP)",
        "selection_rule": "en yüksek iç CV mean AP; eşitlikte düşük SD; sonra düşük kod sırası",
        "final_method_rule": "tam development iç CV'de en yüksek mean AP veren yöntemin (manuel/otomatik) seçtiği aday",
        "threshold": C.THRESHOLD, "threshold_search": False,
        "sgd_momentum": 0.9,
        "preprocessing_fixed_rules": {
            "drop": ["duration"],
            "pdays": "999 -> prev_contacted=0 göstergesi + pdays NaN -> kat-içi median",
            "categorical": "OneHotEncoder(handle_unknown='ignore'); 'unknown' ayrı kategori",
            "numeric_imputation": "median (kat-içi)",
            "clipping": "kapalı veya kat-içi %1-%99 yüzdelik kırpma",
        },
        "bootstrap": {"n": C.N_BOOTSTRAP, "seed": C.SEEDS["bootstrap"], "ci": "percentile 95%"},
    }
    (C.CONFIG_DIR / "training.json").write_text(json.dumps(training, indent=2, ensure_ascii=False))
    (C.CONFIG_DIR / "search_space.json").write_text(
        json.dumps({"manual_defaults": MANUAL_DEFAULTS, "random_search_space": SEARCH_SPACE},
                   indent=2, ensure_ascii=False))
    (C.CONFIG_DIR / "manual_configs.json").write_text(json.dumps(
        {"note": "Dış CV sonuçları görülmeden önce sabitlendi. Tek-faktör kontrollü.",
         "candidates": manual_candidates()}, indent=2, ensure_ascii=False))
    (C.CONFIG_DIR / "random_configs.json").write_text(json.dumps(
        {"note": "ParameterSampler, seed=%d; her optimizer için 2 aday; nested CV öncesi sabitlendi."
                 % C.SEEDS["random_search"],
         "candidates": random_candidates()}, indent=2, ensure_ascii=False))
    print("configs yazıldı:", sorted(p.name for p in C.CONFIG_DIR.glob("*.json")))


if __name__ == "__main__":
    main()
