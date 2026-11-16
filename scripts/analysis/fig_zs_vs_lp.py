"""Figure 1: per-class sensitivity and overall accuracy, zero-shot (PVP) vs linear probing (4-phase concatenation),
grouped bars with 95% bootstrap CIs, RADAR (left) | Merlin (right).
Inputs: results/benchmark4_with_zeroshot.csv, results/merlin_zeroshot_3class.csv
Colors: palette slots 1 (blue, zero-shot) and 2 (orange, linear probe), validated with validate_palette.js.
Usage: python fig_zs_vs_lp.py (from project root) -> results/figs/zs_vs_lp_bar.{png,pdf}
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a66", "#e6e5e2", "#ffffff"
C_ZS, C_LP = "#2a78d6", "#eb6834"
os.makedirs("results/figs", exist_ok=True)
BM = pd.read_csv("results/benchmark4_with_zeroshot.csv", index_col=0)
M3 = pd.read_csv("results/merlin_zeroshot_3class.csv", index_col=0)


def parse(v, scale):
    head, ci = str(v).split(" (")
    lo, hi = ci.rstrip(")").split("-")
    return float(head) * scale, float(lo) * scale, float(hi) * scale


ITEMS = [("HCC", "HCC Sensitivity"), ("ICC", "ICC Sensitivity"), ("HH", "HH Sensitivity"),
         ("LM", "LM Sensitivity"), ("Overall\naccuracy", "Overall Accuracy")]
DATA = {
    "RADAR": ([parse(BM.loc[k, "RADAR zero-shot (pvp)"], 1) for _, k in ITEMS],
              [parse(BM.loc[k, "RADAR LP (4-phase)"], 1) for _, k in ITEMS]),
    "Merlin": ([None if k.startswith("HH") else parse(M3.loc[k, "Merlin zero-shot 3class (pvp)"], 100) for _, k in ITEMS],
               [parse(BM.loc[k, "Merlin LP (4-phase)"], 1) for _, k in ITEMS]),
}

# ---------------- grouped bars ----------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
fig.patch.set_facecolor(SURFACE)
w, gap = 0.36, 0.02
for ax, (name, (zs, lp)) in zip(axes, DATA.items()):
    x = np.arange(len(ITEMS))
    for i in range(len(ITEMS)):
        if zs[i] is not None:
            v, lo, hi = zs[i]
            ax.bar(x[i] - w / 2 - gap / 2, v, w, color=C_ZS, zorder=2)
            ax.errorbar(x[i] - w / 2 - gap / 2, v, yerr=[[v - lo], [hi - v]], color=INK, lw=1, capsize=0, zorder=3)
            ax.text(x[i] - w / 2 - gap / 2, 2, f"{v:.0f}", ha="center", va="bottom", fontsize=10, color=SURFACE, zorder=4)
        else:
            ax.text(x[i] - w / 2 - gap / 2, 2, "n/a", ha="center", va="bottom", fontsize=9, color=MUTED)
        v, lo, hi = lp[i]
        ax.bar(x[i] + w / 2 + gap / 2, v, w, color=C_LP, zorder=2)
        ax.errorbar(x[i] + w / 2 + gap / 2, v, yerr=[[v - lo], [hi - v]], color=INK, lw=1, capsize=0, zorder=3)
        ax.text(x[i] + w / 2 + gap / 2, 2, f"{v:.0f}", ha="center", va="bottom", fontsize=10, color=SURFACE, zorder=4)
    ax.axvline(3.5, color=GRID, lw=1.5, ls="--", zorder=0)
    ax.set_xticks(x, [i[0] for i in ITEMS], color=INK)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(length=0, colors=INK)
    ax.set_title(name, fontsize=14, color=INK, pad=10)
axes[0].set_ylabel("Sensitivity or accuracy (%)", color=INK)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C_ZS, label="Zero-shot (portal venous phase)"),
                    plt.Rectangle((0, 0), 1, 1, color=C_LP, label="Linear probe (four-phase concatenation)")],
           loc="upper center", ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.0))
fig.text(0.01, 0.01, "Error bars, 95% CIs (case-level bootstrap). Merlin zero-shot: three classes in cases without HH "
         "(no phecode corresponds to HH). LM = liver metastasis.", color=MUTED, fontsize=9)
fig.tight_layout(rect=(0, 0.04, 1, 0.93))
fig.savefig("results/figs/zs_vs_lp_bar.png", dpi=200)
fig.savefig("results/figs/zs_vs_lp_bar.pdf")
plt.close(fig)

print("saved results/figs/zs_vs_lp_bar.{png,pdf}")
