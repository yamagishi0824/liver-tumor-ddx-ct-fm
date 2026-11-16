"""Figure: per-class sensitivity and overall accuracy, zero-shot (PVP) -> linear probing (4-phase concat),
RADAR (left) and Merlin (right), with 95% bootstrap CIs.
Inputs: results/benchmark4_with_zeroshot.csv (RADAR zero-shot + both linear probes, % strings),
        results/merlin_zeroshot_3class.csv (Merlin zero-shot, HH excluded, proportions).
Usage: python fig_zs_vs_lp.py (from project root) -> results/figs/zs_vs_lp.{png,pdf}
Colors: palette slots 1 (blue, zero-shot) and 2 (orange, linear probe), validated with validate_palette.js;
markers differ (open circle vs filled square) so identity is not color-only.
"""
import os

import matplotlib
import matplotlib.pyplot as plt
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

PANELS = [
    ("RADAR", lambda k: parse(BM.loc[k, "RADAR zero-shot (pvp)"], 1), lambda k: parse(BM.loc[k, "RADAR LP (4-phase)"], 1)),
    ("Merlin", lambda k: None if k.startswith("HH") else parse(M3.loc[k, "Merlin zero-shot 3class (pvp)"], 100),
     lambda k: parse(BM.loc[k, "Merlin LP (4-phase)"], 1)),
]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
fig.patch.set_facecolor(SURFACE)
dx = 0.17
for ax, (name, zs, lp) in zip(axes, PANELS):
    ax.set_facecolor(SURFACE)
    for x, (label, key) in enumerate(ITEMS):
        z, l = zs(key), lp(key)
        if z is not None:
            ax.annotate("", xy=(x + dx - 0.04, l[0]), xytext=(x - dx + 0.04, z[0]),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2, shrinkA=6, shrinkB=6))
            ax.errorbar(x - dx, z[0], yerr=[[z[0] - z[1]], [z[2] - z[0]]], fmt="o", ms=9, mfc=SURFACE, mec=C_ZS,
                        mew=2, ecolor=C_ZS, elinewidth=1.5, capsize=0, zorder=3)
            ax.text(x - dx - 0.09, z[0], f"{z[0]:.0f}", ha="right", va="center", fontsize=10, color=INK)
        else:
            ax.text(x - dx, 6, "n/a", ha="center", va="bottom", fontsize=10, color=MUTED)
        ax.errorbar(x + dx, l[0], yerr=[[l[0] - l[1]], [l[2] - l[0]]], fmt="s", ms=8.5, mfc=C_LP, mec=SURFACE,
                    mew=1.5, ecolor=C_LP, elinewidth=1.5, capsize=0, zorder=3)
        ax.text(x + dx + 0.09, l[0], f"{l[0]:.0f}", ha="left", va="center", fontsize=10, color=INK)
    ax.axvline(3.5, color=GRID, lw=1.5, ls="--", zorder=0)
    ax.set_xticks(range(len(ITEMS)), [i[0] for i in ITEMS], color=INK)
    ax.set_xlim(-0.6, len(ITEMS) - 0.4)
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(length=0, colors=INK)
    ax.set_title(name, fontsize=14, color=INK, pad=10)
axes[0].set_ylabel("Sensitivity or accuracy (%)", color=INK)

h_zs = plt.Line2D([], [], marker="o", ms=9, mfc=SURFACE, mec=C_ZS, mew=2, ls="none", label="Zero-shot (portal venous phase)")
h_lp = plt.Line2D([], [], marker="s", ms=8.5, mfc=C_LP, mec=SURFACE, ls="none", label="Linear probe (four-phase concatenation)")
fig.legend(handles=[h_zs, h_lp], loc="upper center", ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.0))
fig.text(0.01, 0.01, "Error bars, 95% CIs (case-level bootstrap). Merlin zero-shot: three classes in cases without HH "
         "(no phecode corresponds to HH). LM = liver metastasis.", color=MUTED, fontsize=9)
fig.tight_layout(rect=(0, 0.04, 1, 0.93))
fig.savefig("results/figs/zs_vs_lp.png", dpi=200)
fig.savefig("results/figs/zs_vs_lp.pdf")
print("saved results/figs/zs_vs_lp.{png,pdf}")
