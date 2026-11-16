"""Linear probing on MCT-LTDiag (5-class: HCC/ICC/BCLM/CRLM/HH).
Repeated stratified 5-fold CV; StandardScaler + L2 LogisticRegression with inner CV for C.
Features: Merlin emb2048 / emb512 / pheno_prob, RADAR 146 finding scores; per phase and 4-phase concat.
Usage: python linear_probe.py  (run from project root)
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
PHASES = ["nc", "art", "pvp", "delay"]
lab = pd.read_csv("mctlt/labels_all.csv", header=None, names=["id", "t"]).set_index("id")["t"]


def load_merlin(key):
    out = {}
    for p in PHASES:
        z = np.load(os.path.join(os.environ.get("MERLIN_DIR", "results_merlin"), f"merlin_{p}.npz"))
        out[p] = pd.DataFrame(z[key], index=z["ids"])
    return out


def load_radar():
    out = {}
    for p in PHASES:
        d = pd.read_csv(os.path.join(os.environ.get("RADAR_FEAT_DIR", "results_radar_feat"), f"RADAR_infer_results_{p}.csv"), encoding="utf-8-sig")
        d.index = d.pop("file_name").str.replace(".nii.gz", "", regex=False)
        out[p] = d.fillna(0.0)
    return out


def load_radar_feat(key, organ="肝"):
    import torch
    out = {}
    for p in PHASES:
        d = torch.load(os.path.join(os.environ.get("RADAR_FEAT_DIR", "results_radar_feat"), f"RADAR_organ_feats_{p}.pt"), weights_only=False)
        out[p] = pd.DataFrame({k.replace(".nii.gz", ""): np.asarray(v[organ][key]) for k, v in d.items()}).T
    return out


import sys

GROUP = sys.argv[1] if len(sys.argv) > 1 else "merlin"
if GROUP == "merlin":
    FEATS = {
        "Merlin emb2048": load_merlin("emb2048"),
        "Merlin emb512": load_merlin("emb512"),
        "Merlin phecode1692": load_merlin("pheno_prob"),
    }
elif GROUP == "radar":
    FEATS = {
        "RADAR liver proj256": load_radar_feat("proj"),
        "RADAR liver query256": load_radar_feat("query"),
    }
SUFFIX = "" if GROUP == "merlin" else f"_{GROUP}"

ids = sorted(lab.index)
y = lab.loc[ids].values
classes = np.unique(y)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
splits = list(cv.split(ids, y))

rows, preds = [], {}
for fname, per_phase in FEATS.items():
    sets = {p: per_phase[p].loc[ids].values for p in PHASES}
    sets["concat4"] = np.hstack([sets[p] for p in PHASES])
    for sname, X in sets.items():
        acc, bacc, auc = [], [], []
        oof = np.zeros((5, len(ids), len(classes)))  # per repeat
        for k, (tr, te) in enumerate(splits):
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegressionCV(Cs=np.logspace(-4, 1, 11), cv=5, max_iter=3000,
                                     scoring="balanced_accuracy", n_jobs=-1),
            )
            clf.fit(X[tr], y[tr])
            pr = clf.predict_proba(X[te])
            oof[k // 5, te] = pr
            yp = classes[pr.argmax(1)]
            acc.append(accuracy_score(y[te], yp))
            bacc.append(balanced_accuracy_score(y[te], yp))
            auc.append(roc_auc_score(y[te], pr, multi_class="ovr", average="macro"))
        preds[(fname, sname)] = oof
        rows.append(dict(features=fname, phase=sname,
                         acc=f"{np.mean(acc):.3f}±{np.std(acc):.3f}",
                         bal_acc=f"{np.mean(bacc):.3f}±{np.std(bacc):.3f}",
                         macroAUC=f"{np.mean(auc):.3f}±{np.std(auc):.3f}"))
        print(rows[-1], flush=True)

res = pd.DataFrame(rows)
res.to_csv(f"results/linear_probe_summary{SUFFIX}.csv", index=False)
np.save(f"results/linear_probe_oof{SUFFIX}.npy", {"ids": ids, "y": y, "classes": classes, "preds": preds}, allow_pickle=True)
print(res.to_string(index=False))
