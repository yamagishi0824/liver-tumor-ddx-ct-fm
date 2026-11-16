"""Zero-shot evaluation.
RADAR: four-class differentiation (HCC, ICC, HH, LM = BCLM+CRLM) in the MCT-LTDiag benchmark format, using the
scores of four liver findings; prediction = finding with the highest score; 95% CI by case-level bootstrap.
Merlin: one-vs-rest AUC of phecode probabilities (HCC 155.1, ICC 155, LM 198.4); no phecode for HH.
Usage: python eval_zeroshot.py <labels_csv(id,type)> <radar_results_dir> <merlin_npz_dir> <phenotypes_csv> <out_dir>
  radar_results_dir: RADAR_infer_results_{nc,art,pvp,delay}.csv ; merlin_npz_dir: merlin_{phase}.npz
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score

CL4 = ["HCC", "ICC", "HH", "LM"]
FINDING = {"HCC": "Hepatocellular carcinoma", "ICC": "Intrahepatic cholangiocarcinoma", "HH": "Hemangioma", "LM": "Metastasis"}
PHECODE = {"HCC": ("155.1", ["HCC"]), "ICC": ("155", ["ICC"]), "LM": ("198.4", ["BCLM", "CRLM"])}
PHASES = ["nc", "art", "pvp", "delay"]


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


def boot_ci(y, P, fn, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    point, boots = fn(y, P), []
    while len(boots) < B:
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < len(np.unique(y)):
            continue
        boots.append(fn(y[idx], P[idx]))
    b = pd.DataFrame(boots)
    return {k: (v, b[k].quantile(.025), b[k].quantile(.975)) for k, v in point.items()}


def main():
    labels_csv, radar_dir, merlin_dir, pheno_csv, out_dir = sys.argv[1:6]
    os.makedirs(out_dir, exist_ok=True)
    lab = pd.read_csv(labels_csv, header=None, names=["id", "t"]).set_index("id")["t"]

    rows = {}
    for p in PHASES:
        d = pd.read_csv(os.path.join(radar_dir, f"RADAR_infer_results_{p}.csv"), encoding="utf-8-sig")
        ids = d.file_name.str.replace(".nii.gz", "", regex=False)
        P = np.stack([d[next(c for c in d.columns if c.endswith(f"(Liver_{FINDING[k]})"))].fillna(0).values for k in CL4], 1)
        y = ids.map(lab).values
        y = np.where(np.isin(y, ["BCLM", "CRLM"]), "LM", y)
        rows[f"RADAR zero-shot ({p})"] = {k: f"{v:.4f} ({lo:.4f}-{hi:.4f})" for k, (v, lo, hi) in boot_ci(y, P, metrics).items()}
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "radar_zeroshot_benchmark4.csv"))

    ph = pd.read_csv(pheno_csv, dtype={"phecode": str})
    out = []
    for p in PHASES:
        z = np.load(os.path.join(merlin_dir, f"merlin_{p}.npz"))
        y = pd.Series(z["ids"]).map(lab).values
        r = {"phase": p}
        for k, (code, pos) in PHECODE.items():
            s = z["pheno_prob"][:, ph.index[ph.phecode == code][0]]
            t = np.isin(y, pos).astype(int)
            ci = boot_ci(t, s[:, None], lambda yy, ss: {"AUC": roc_auc_score(yy, ss[:, 0])})["AUC"]
            r[f"{k} ({code})"] = f"{ci[0]:.4f} ({ci[1]:.4f}-{ci[2]:.4f})"
        out.append(r)
    pd.DataFrame(out).to_csv(os.path.join(out_dir, "merlin_zeroshot_auc.csv"), index=False)
    print("saved to", out_dir)


if __name__ == "__main__":
    main()
