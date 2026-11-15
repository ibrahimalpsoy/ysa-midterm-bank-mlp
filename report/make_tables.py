"""results/ altındaki CSV/JSON dosyalarından rapor tablolarını (LaTeX) üretir.
Rapordaki tüm sayılar bu betikle doğrudan sonuç dosyalarından alınır.  Kullanım: python report/make_tables.py
"""
import json
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
OUT = Path(__file__).resolve().parent / "tables"
OUT.mkdir(exist_ok=True)


def f(x, d=4):
    return f"{x:.{d}f}"


def pm(m, s, d=4):
    return f"{m:.{d}f} $\\pm$ {s:.{d}f}"


def write(name, body):
    (OUT / f"{name}.tex").write_text(body, encoding="utf-8")


def esc(s):
    return str(s).replace("_", "\\_").replace("%", "\\%")


# ---------------------------------------------------------------- Tablo 1a: veri seti / sınıf / özel kodlar
ov = pd.read_csv(R / "eda_feature_overview.csv")
ss = pd.read_csv(R / "split_summary.csv").set_index("subset")
dup = json.loads((R / "duplicate_report.json").read_text())
unk = ov.query("n_unknown>0")
rows = [
    ("Gözlem / girdi özelliği", "4.119 / 20 (\\texttt{duration} çıkarıldıktan sonra 19: 9 sayısal, 10 kategorik)"),
    ("Development (\\%80)", f"n={int(ss.loc['development','n'])}; pozitif {int(ss.loc['development','n_pos'])} (\\%{100*ss.loc['development','pos_rate']:.2f})"),
    ("Bağımsız test (\\%20)", f"n={int(ss.loc['test','n'])}; pozitif {int(ss.loc['test','n_pos'])} (\\%{100*ss.loc['test','pos_rate']:.2f})"),
    ("Gerçek NaN", "yok (tüm sütunlarda 0)"),
    ("\\texttt{unknown} (development)", "; ".join(f"{r.feature}: {r.n_unknown} (\\%{r.pct_unknown:.1f})" for r in unk.itertuples())),
    ("\\texttt{pdays}=999 (development)", f"{int(ov.set_index('feature').loc['pdays','n_pdays_999'])} / {int(ss.loc['development','n'])} (daha önce temas yok)"),
    ("Kopya kayıtlar (tüm veri)", f"ham 21 sütunda {dup['raw_exact_duplicates']}; \\texttt{{duration}} çıkarıldıktan sonra özellik+hedef {dup['model_input_exact_duplicates_with_target']}, "
                                   f"yalnız özellik {dup['model_input_feature_duplicates']} (1 grupta çelişkili etiket); development içinde aynısı bulunan test satırı: {dup['test_rows_with_identical_feature_vector_in_dev']}"),
]
write("t1a_dataset", "\\begin{tabular}{p{4.1cm}p{11.4cm}}\\toprule\nÖzellik & Değer \\\\\\midrule\n" +
      "\n".join(f"{a} & {b} \\\\" for a, b in rows) + "\n\\bottomrule\\end{tabular}")

# ---------------------------------------------------------------- Tablo 1b: sayısal tanımlayıcı istatistikler
ds = pd.read_csv(R / "eda_numeric_descriptives.csv", index_col=0)
body = "\\begin{tabular}{lrrrrrrr}\\toprule\nÖzellik & n & Ortalama & Medyan & SD & IQR & Min & Maks \\\\\\midrule\n"
for k, r in ds.iterrows():
    body += f"{esc(k)} & {int(r.n)} & {r['mean']:.3f} & {r['median']:.3f} & {r.sd:.3f} & {r.IQR:.3f} & {r['min']:.3f} & {r['max']:.3f} \\\\\n"
write("t1b_numeric", body + "\\bottomrule\\end{tabular}")

# ---------------------------------------------------------------- Tablo 2/3: manuel ve otomatik adaylar
cand = pd.read_csv(R / "candidate_table.csv").set_index("code")
for method, fn, name in (("manual", "manual_search_results.csv", "t2_manual"), ("random", "random_search_results.csv", "t3_random")):
    s = pd.read_csv(R / fn)
    body = ("\\begin{tabular}{l l r r r r r r r}\\toprule\n"
            "Kod & Konfigürasyon & Param. & \\multicolumn{4}{c}{İç CV AP (mean $\\pm$ SD, 3 iç kat)} & En iyi ep. & Süre (s) \\\\\n"
            "\\cmidrule(lr){4-7} & & & Dış kat 1 & Dış kat 2 & Dış kat 3 & Tam dev. & (tam dev.) & (tam dev.) \\\\\\midrule\n")
    for code in s.code.unique():
        c = cand.loc[code]
        conf = (f"{c.optimizer}, {c.hidden}, {c.activation}, lr={c.lr:.2g}, wd={c.weight_decay:g}, bs={c.batch_size}, "
                f"{c.scaler}, do={c.dropout:g}, clip={'var' if c.clipping else 'yok'}, sch={'açık' if c.scheduler else 'kapalı'}")
        cells = []
        for st in ("outer1", "outer2", "outer3", "full_dev"):
            r = s.query("stage==@st and code==@code").iloc[0]
            cells.append(pm(r.AP_mean, r.AP_sd, 3))
        fd = s.query("stage=='full_dev' and code==@code").iloc[0]
        body += f"{code} & {{\\scriptsize {conf}}} & {int(c.n_params)} & " + " & ".join(cells) + f" & {fd.best_epoch_mean:.1f} & {fd.train_time_s_total:.2f} \\\\\n"
    write(name, body + "\\bottomrule\\end{tabular}")

# ---------------------------------------------------------------- Tablo 4: dış kat ham sonuçlar
oc = pd.read_csv(R / "outer_cv_results.csv")
body = ("\\begin{tabular}{c l c r r r r r r r r}\\toprule\nKat & Model & Seçilen & İç AP & AP & ROC-AUC & F1 & Sens. & Spec. & Bal.Acc. & PR-AUC$_{\\mathrm{trapez}}$ \\\\\\midrule\n")
for r in oc.itertuples():
    ia = "--" if pd.isna(r.inner_AP_mean) else f"{r.inner_AP_mean:.3f}"
    body += f"{r.outer_fold} & {esc(r.model)} & {r.selected} & {ia} & {r.AP:.3f} & {r.ROC_AUC:.3f} & {r.F1:.3f} & {r.Sensitivity:.3f} & {r.Specificity:.3f} & {r.Balanced_Accuracy:.3f} & {r.PR_AUC_trapezoid:.3f} \\\\\n"
    if r.model.startswith("Dummy") and r.outer_fold < 3:
        body += "\\midrule\n"
write("t4_outer_folds", body + "\\bottomrule\\end{tabular}")

# ---------------------------------------------------------------- Tablo 5: dış CV Mean ± SD
mets = ["AP", "ROC_AUC", "F1", "Sensitivity", "Specificity", "Balanced_Accuracy", "Accuracy", "Precision"]
lab = ["AP", "ROC-AUC", "F1", "Sensitivity", "Specificity", "Bal. Acc.", "Accuracy", "Precision"]
order = ["MLP-manual", "MLP-random", "LogisticRegression", "Dummy(prior)"]
g = oc.groupby("model")
body = "\\begin{tabular}{l" + "c" * len(order) + "}\\toprule\nMetrik & " + " & ".join(esc(o) for o in order) + " \\\\\\midrule\n"
for m, l in zip(mets + ["fit_time_s"], lab + ["Eğitim süresi (s)"]):
    body += l + " & " + " & ".join(pm(g[m].mean()[o], g[m].std(ddof=1)[o], 3) for o in order) + " \\\\\n"
write("t5_outer_meansd", body + "\\bottomrule\\end{tabular}")

# ---------------------------------------------------------------- Tablo 6: test metrikleri + bootstrap CI
tm = pd.read_csv(R / "test_metrics.csv").set_index("model")
ci = pd.read_csv(R / "test_bootstrap_ci.csv")
cols = ["AP", "ROC_AUC", "Accuracy", "Precision", "Sensitivity", "Specificity", "F1", "Balanced_Accuracy", "Brier"]
clab = ["AP", "ROC-AUC", "Acc.", "Prec.", "Sens.", "Spec.", "F1", "Bal.Acc.", "Brier"]
body = "\\begin{tabular}{l" + "r" * len(cols) + "r}\\toprule\nModel & " + " & ".join(clab) + " & TP/FP/TN/FN \\\\\\midrule\n"
for k, r in tm.iterrows():
    body += esc(k) + " & " + " & ".join(f"{r[c]:.3f}" for c in cols) + f" & {int(r.TP)}/{int(r.FP)}/{int(r.TN)}/{int(r.FN)} \\\\\n"
body += "\\midrule\n\\multicolumn{11}{l}{\\textit{Bootstrap \\%95 yüzdelik güven aralıkları (1.000 tekrar)}} \\\\\n"
for mdl in ci.model.unique():
    q = ci.query("model==@mdl").set_index("metric")
    body += (f"\\multicolumn{{11}}{{l}}{{{esc(mdl)}: AP {q.loc['AP','point']:.3f} [{q.loc['AP','ci_low']:.3f}, {q.loc['AP','ci_high']:.3f}];\\ "
             f"ROC-AUC {q.loc['ROC_AUC','point']:.3f} [{q.loc['ROC_AUC','ci_low']:.3f}, {q.loc['ROC_AUC','ci_high']:.3f}];\\ "
             f"F1 {q.loc['F1','point']:.3f} [{q.loc['F1','ci_low']:.3f}, {q.loc['F1','ci_high']:.3f}]}} \\\\\n")
dd = pd.read_csv(R / "test_bootstrap_diff_mlp_vs_lr.csv")
body += ("\\multicolumn{11}{l}{Eşleştirilmiş bootstrap fark (MLP $-$ LR): " +
         ";\\ ".join(f"{esc(r.metric.split(' ')[0])} {r.point:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]" for r in dd.itertuples()) + "} \\\\\n")
write("t6_test", body + "\\bottomrule\\end{tabular}")

# ---------------------------------------------------------------- Tablo 7: seed kararlılığı
sd = pd.read_csv(R / "seed_stability_results.csv", index_col=0)
c7 = ["AP", "ROC_AUC", "F1", "Sensitivity", "Specificity", "Balanced_Accuracy", "Precision"]
body = "\\begin{tabular}{l" + "r" * len(c7) + "r}\\toprule\nSeed & AP & ROC-AUC & F1 & Sens. & Spec. & Bal.Acc. & Prec. & En iyi ep. \\\\\\midrule\n"
for k, r in sd.iterrows():
    if k in ("mean", "sd"):
        continue
    star = " (birincil)" if str(r.primary) == "True" else ""
    body += f"{k}{star} & " + " & ".join(f"{float(r[c]):.4f}" for c in c7) + f" & {int(float(r.best_epoch))} \\\\\n"
body += "\\midrule\nOrtalama $\\pm$ SD & " + " & ".join(pm(float(sd.loc['mean', c]), float(sd.loc['sd', c]), 4) for c in c7) + " & -- \\\\\n"
write("t7_seeds", body.replace("\\pm$ ", "\\pm$\\,") + "\\bottomrule\\end{tabular}")

# ---------------------------------------------------------------- Ek: hata analizi ve üç düzey
ea = pd.read_csv(R / "test_error_analysis.csv", index_col=0)
body = ("\\begin{tabular}{lrrrrrrr}\\toprule\nGrup & n & Medyan olas. & euribor3m (med.) & Önceden temas & poutcome=success & cellular & Yaş (med.) \\\\\\midrule\n")
for k, r in ea.iterrows():
    body += (f"{k} & {int(r.n)} & {r.proba_median:.3f} & {r.euribor3m_median:.3f} & \\%{100*r.prev_contacted_rate:.1f} & "
             f"\\%{100*r.poutcome_success_rate:.1f} & \\%{100*r.cellular_rate:.1f} & {r.age_median:.1f} \\\\\n")
write("t8_errors", body + "\\bottomrule\\end{tabular}")

lv = pd.read_csv(R / "summary_three_levels.csv")
body = "\\begin{tabular}{p{9.5cm}l}\\toprule\nDeğerlendirme düzeyi & AP \\\\\\midrule\n"
for r in lv.itertuples():
    body += f"{esc(r[1])} & {esc(r[2]).replace(chr(177), chr(36) + chr(92) + 'pm' + chr(36))} \\\\\n"
write("t9_levels", body + "\\bottomrule\\end{tabular}")

print("tablolar:", sorted(p.name for p in OUT.glob("*.tex")))
