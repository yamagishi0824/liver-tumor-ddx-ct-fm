"""Evaluate linear-probe OOF predictions in the MCT-LTDiag paper's benchmark format (Table 4):
4 classes (HCC, ICC, LM = BCLM+CRLM, HH); per-class specificity / sensitivity / F1 / AUC (one-vs-rest),
overall accuracy and Cohen's kappa; 95% CI by case-level bootstrap (2000 resamples).
5-class probabilities are collapsed by summing BCLM + CRLM. Prediction = argmax of the 4 collapsed probabilities
(repeat-averaged out-of-fold probabilities).
Usage: python eval_benchmark4.py  (from project root) -> results/benchmark4.csv
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score

CL4 = ["HCC", "ICC", "HH", "LM"]
RUNS = [("results/linear_probe_oof.npy", ("Merlin emb2048", "concat4"), "Merlin LP (4-phase)"),
        ("results/linear_probe_oof_radar.npy", ("RADAR liver query256", "concat4"), "RADAR LP (4-phase)")]
rng = np.random.default_rng(0)
B = 2000


def collapse(y5, P5, classes):
    c = list(classes)
    P = np.stack([P5[:, c.index("HCC")], P5[:, c.index("ICC")], P5[:, c.index("HH")],
                  P5[:, c.index("BCLM")] + P5[:, c.index("CRLM")]], 1)
    y = np.where(np.isin(y5, ["BCLM", "CRLM"]), "LM", y5)
    return y, P


def metrics(y, P):
    pred = np.array(CL4)[P.argmax(1)]
    out = {}
    for i, k in enumerate(CL4):
        t, p = y == k, pred == k
        tp, tn, fp, fn = (t & p).sum(), (~t & ~p).sum(), (~t & p).sum(), (t & ~p).sum()
        out[f"{k} Specificity"] = tn / (tn + fp)
        out[f"{k} Sensitivity"] = tp / (tp + fn)
        out[f"{k} F1"] = f1_score(t, p)
        out[f"{k} AUC"] = roc_auc_score(t, P[:, i])
    out["Overall Accuracy"] = accuracy_score(y, pred)
    out["Overall Kappa"] = cohen_kappa_score(y, pred)
    return out


rows = {}
for path, key, name in RUNS:
    o = np.load(path, allow_pickle=True).item()
    y, P = collapse(np.array(o["y"]), o["preds"][key].mean(0), o["classes"])
    point = metrics(y, P)
    boots = []
    n = len(y)
    while len(boots) < B:
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 4:
            continue
        boots.append(metrics(y[idx], P[idx]))
    bdf = pd.DataFrame(boots)
    rows[name] = {k: (f"{v:.4f} ({bdf[k].quantile(.025):.4f}-{bdf[k].quantile(.975):.4f})" if "Kappa" in k else
                      f"{100*v:.2f} ({100*bdf[k].quantile(.025):.2f}-{100*bdf[k].quantile(.975):.2f})")
                  for k, v in point.items()}

res = pd.DataFrame(rows)
res.to_csv("results/benchmark4.csv")
print(res.to_string())
