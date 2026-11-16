"""Figure 3: RADAR outputs by true tumor type (all 517 cases).
Top: zero-shot (PVP) scores of the four liver tumor findings, normalized to sum to 1 in each case for display
(the original scores are computed independently for each finding and are not probabilities; pass "raw" to plot them as is).
Bottom: linear-probe probabilities (four-phase concatenation; BCLM + CRLM summed).
Inputs: mctlt/labels_all.csv, $RADAR_FEAT_DIR/RADAR_infer_results_pvp.csv, results/linear_probe_oof_radar.npy
Usage: python fig_score_dist.py [raw] (from WORK_DIR) -> results/figs/score_dist.{png,pdf}
"""
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e6e5e2"
C4 = {"HCC": "#2a78d6", "ICC": "#eb6834", "LM": "#1baf7a", "HH": "#eda100"}
lab = pd.read_csv("mctlt/labels_all.csv", header=None, names=["id", "t"]).set_index("id")["t"]
FIND = {"HCC": "Hepatocellular carcinoma", "ICC": "Intrahepatic cholangiocarcinoma", "LM": "Metastasis", "HH": "Hemangioma"}
CL = ["HCC", "ICC", "LM", "HH"]

# ---------------- (1) zero-shot score distributions ----------------
d = pd.read_csv(os.path.join(os.environ.get("RADAR_FEAT_DIR", "results_radar_feat"), "RADAR_infer_results_pvp.csv"), encoding="utf-8-sig")
d.index = d.file_name.str.replace(".nii.gz", "", regex=False)
S = pd.DataFrame({k: d[next(c for c in d.columns if c.endswith(f"(Liver_{v})"))].fillna(0) for k, v in FIND.items()})
t4 = lab.reindex(S.index).replace({"BCLM": "LM", "CRLM": "LM"})

o = np.load("results/linear_probe_oof_radar.npy", allow_pickle=True).item()
_P, _c = o["preds"][("RADAR liver query256", "concat4")].mean(0), list(o["classes"])
LPP = pd.DataFrame({"HCC": _P[:, _c.index("HCC")], "ICC": _P[:, _c.index("ICC")],
                    "LM": _P[:, _c.index("BCLM")] + _P[:, _c.index("CRLM")], "HH": _P[:, _c.index("HH")]}, index=o["ids"])
NORM = not (len(sys.argv) > 1 and sys.argv[1] == "raw")
S_ = S.div(S.sum(1).replace(0, np.nan), axis=0).fillna(0.25) if NORM else S
ROWS = [(S_, "RADAR zero-shot (PVP)\n" + ("normalized score (sum = 1)" if NORM else "finding score")), (LPP.reindex(S.index), "RADAR linear probe (four-phase)\nprobability")]

fig, axes = plt.subplots(2, 4, figsize=(13, 8.4), sharey=True)
for r, (DATA, ylab) in enumerate(ROWS):
    pred = DATA.idxmax(1)
    for ax, t in zip(axes[r], CL):
        sub = DATA[t4 == t]
        bp = ax.boxplot([sub[k] for k in CL], positions=range(4), widths=0.6, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, lw=1.5), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                        boxprops=dict(lw=0))
        for patch, k in zip(bp["boxes"], CL):
            patch.set_facecolor(C4[k])
        rng = np.random.default_rng(0)
        for i, k in enumerate(CL):
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(sub)), sub[k], s=5, color=INK, alpha=0.25, lw=0, zorder=3)
        ax.set_xticks(range(4), CL, color=INK)
        if r == 0:
            ax.set_title(f"True {t} (n={len(sub)})", fontsize=12, color=INK, pad=8)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(length=0, colors=INK)
    axes[r, 0].set_ylabel(ylab, color=INK)
    axes[r, 0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
for ax in axes[1]:
    ax.set_xlabel("Tumor type of the score", color=INK)
fig.tight_layout(rect=(0, 0, 1, 1), h_pad=2.5)
OUTN = "score_dist" if NORM else "score_dist_raw"
fig.savefig(f"results/figs/{OUTN}.png", dpi=200)
fig.savefig(f"results/figs/{OUTN}.pdf")
plt.close(fig)
print(f"saved results/figs/{OUTN}.{{png,pdf}}")
