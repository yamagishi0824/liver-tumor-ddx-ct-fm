"""Paired bootstrap of the change from zero-shot (PVP) to linear probing (four-phase concatenation).
RADAR: four classes (HCC, ICC, HH, LM), all 517 cases.
Merlin: matched to its zero-shot setting = three classes (HCC, ICC, LM) in the 421 cases without HH; the linear-probe
prediction is the largest of the HCC, ICC, and LM (= BCLM + CRLM) probabilities of the same five-class model.
Metrics: per-class sensitivity, specificity, F1, one-vs-rest AUC (x100), and overall accuracy (percentage points). The same case resample is used for both
conditions (paired); 2000 resamples; percentile 95% CI.
Usage: python eval_delta.py [concat4|pvp] (from project root) -> results/delta_zs_lp.csv or delta_zs_lp_pvp.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LPPH = sys.argv[1] if len(sys.argv) > 1 else "concat4"  # linear-probe input: concat4 (main) or pvp (supplement)
OUT = "results/delta_zs_lp.csv" if LPPH == "concat4" else f"results/delta_zs_lp_{LPPH}.csv"
lab = pd.read_csv("mctlt/labels_all.csv", header=None, names=["id", "t"]).set_index("id")["t"]
FIND = {"HCC": "Hepatocellular carcinoma", "ICC": "Intrahepatic cholangiocarcinoma", "HH": "Hemangioma", "LM": "Metastasis"}
PHE = {"HCC": "155.1", "ICC": "155", "LM": "198.4"}


def lp(path, key, classes_out):
    o = np.load(path, allow_pickle=True).item()
    P5, c = o["preds"][key].mean(0), list(o["classes"])
    col = {"HCC": P5[:, c.index("HCC")], "ICC": P5[:, c.index("ICC")], "HH": P5[:, c.index("HH")],
           "LM": P5[:, c.index("BCLM")] + P5[:, c.index("CRLM")]}
    return pd.DataFrame({k: col[k] for k in classes_out}, index=o["ids"])


def radar_zs():
    d = pd.read_csv(os.path.join(os.environ.get("RADAR_FEAT_DIR", "results_radar_feat"), "RADAR_infer_results_pvp.csv"), encoding="utf-8-sig")
    d.index = d.file_name.str.replace(".nii.gz", "", regex=False)
    return pd.DataFrame({k: d[next(c for c in d.columns if c.endswith(f"(Liver_{v})"))].fillna(0) for k, v in FIND.items()})


def merlin_zs():
    z = np.load(os.path.join(os.environ.get("MERLIN_DIR", "results_merlin"), "merlin_pvp.npz"))
    ph = pd.read_csv(os.environ.get("MERLIN_PHENOTYPES", "phenotypes.csv"), dtype={"phecode": str})
    return pd.DataFrame({k: z["pheno_prob"][:, ph.index[ph.phecode == v][0]] for k, v in PHE.items()}, index=z["ids"])


def stats(y, pred, P, cls):
    out = {}
    for j, k in enumerate(cls):
        t, p = y == k, pred == k
        tp, fp, fn = (t & p).sum(), (~t & p).sum(), (t & ~p).sum()
        out[f"{k} sensitivity"] = tp / t.sum() * 100
        out[f"{k} specificity"] = (~t & ~p).sum() / (~t).sum() * 100
        out[f"{k} F1"] = 2 * tp / (2 * tp + fp + fn) * 100
        out[f"{k} AUC"] = roc_auc_score(t, P[:, j]) * 100
    out["Overall accuracy"] = (pred == y).mean() * 100
    return out


rng = np.random.default_rng(0)
rows = []
for name, cls, zs, lpp in [
    ("RADAR", ["HCC", "ICC", "HH", "LM"], radar_zs(), lp("results/linear_probe_oof_radar.npy", ("RADAR liver query256", LPPH), ["HCC", "ICC", "HH", "LM"])),
    ("Merlin", ["HCC", "ICC", "LM"], merlin_zs(), lp("results/linear_probe_oof.npy", ("Merlin emb2048", LPPH), ["HCC", "ICC", "LM"])),
]:
    y = lab.reindex(lpp.index).replace({"BCLM": "LM", "CRLM": "LM"})
    keep = y.isin(cls)
    ids = y.index[keep]
    y = y[keep].values
    Pa, Pb = zs.loc[ids, cls].values, lpp.loc[ids, cls].values
    a, b = Pa.argmax(1), Pb.argmax(1)
    pa, pb = np.array(cls)[a], np.array(cls)[b]
    s0, s1 = stats(y, pa, Pa, cls), stats(y, pb, Pb, cls)
    boots = []
    while len(boots) < 2000:
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < len(cls):
            continue
        b0, b1 = stats(y[i], pa[i], Pa[i], cls), stats(y[i], pb[i], Pb[i], cls)
        boots.append({k: b1[k] - b0[k] for k in s0})
    B = pd.DataFrame(boots)
    for k in s0:
        rows.append({"model": name, "n": len(y), "metric": k, "zero_shot": s0[k], "linear_probe": s1[k],
                     "delta": s1[k] - s0[k], "lo": B[k].quantile(.025), "hi": B[k].quantile(.975)})
res = pd.DataFrame(rows)
res.to_csv(OUT, index=False)
print(res.round(1).to_string(index=False))
