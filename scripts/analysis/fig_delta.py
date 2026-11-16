"""Figure 1: change from zero-shot (PVP) to linear probing (four-phase concatenation) in per-class sensitivity, specificity, F1,
and one-vs-rest AUC (x100); point estimate and paired bootstrap 95% CI. RADAR (four classes, 517 cases) and
Merlin (three classes without HH, 421 cases) in the same panel.
Input: results/delta_zs_lp.csv (eval_delta.py). Output: results/figs/delta.{png,pdf}
Colors: palette slots 1 (blue) and 2 (orange), as in the previous Figure 1.
"""
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e6e5e2"
COL = {"RADAR": "#2a78d6", "Merlin": "#eb6834"}
OFF = {"RADAR": 0.17, "Merlin": -0.17}
d = pd.read_csv("results/delta_zs_lp.csv").set_index(["model", "metric"])
CLS = ["HCC", "ICC", "HH", "LM"]
PANELS = [("A", "Sensitivity", [f"{c} sensitivity" for c in CLS], CLS),
          ("B", "Specificity", [f"{c} specificity" for c in CLS], CLS),
          ("C", "F1 score", [f"{c} F1" for c in CLS], CLS),
          ("D", "AUC", [f"{c} AUC" for c in CLS], CLS)]

fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), sharex=True)
for ax, (tag, title, keys, labels) in zip(axes.flat, PANELS):
    n = len(keys)
    for i, k in enumerate(keys):
        y0 = n - 1 - i
        for m in ["RADAR", "Merlin"]:
            y = y0 + OFF[m]
            if (m, k) not in d.index:
                ax.text(1.5, y, "n/a", va="center", ha="left", fontsize=9, color=MUTED)
                continue
            r = d.loc[(m, k)]
            ax.plot([r.lo, r.hi], [y, y], color=COL[m], lw=1.8, zorder=2)
            ax.plot(r.delta, y, "o", ms=7, color=COL[m], mec="white", mew=1.2, zorder=3)
            ax.text(max(r.hi, 0) + 1.5, y, f"{r.delta:+.1f}".replace("-", "\u2212"), va="center", fontsize=9, color=INK)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(n), labels[::-1], color=INK)
    ax.set_ylim(-0.7, 3.7) if n > 1 else ax.set_ylim(-1.8, 1.8)
    ax.set_xlim(-20, 60)
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(length=0, colors=INK)
    ax.set_title(f"{tag}  {title}", loc="left", fontsize=13, color=INK)
for ax in axes[1]:
    ax.set_xlabel("Change, linear probe − zero-shot (percentage points)", color=INK)
    ax.tick_params(labelbottom=True)
fig.legend(handles=[plt.Line2D([], [], color=COL["RADAR"], marker="o", lw=1.8, label="RADAR (four classes, 517 cases)"),
                    plt.Line2D([], [], color=COL["Merlin"], marker="o", lw=1.8, label="Merlin (three classes without HH, 421 cases)")],
           loc="upper center", ncol=2, frameon=False, fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig("results/figs/delta.png", dpi=200)
fig.savefig("results/figs/delta.pdf")
print("saved results/figs/delta.{png,pdf}")
