"""Confusion matrices (rows = true, cols = predicted; shade = row-normalized %, text = count (row %)).
Top: Merlin linear probe (emb2048, 4-phase concat) | RADAR linear probe (liver query256, 4-phase concat)
Bottom: benign (HH) vs malignant for the same three.
Usage: python fig_confusion.py  (from project root) -> results/figs/confusion.{png,pdf}
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14})
INK, MUTED, SURFACE = "#1f1f1e", "#6b6a66", "#ffffff"
RAMP = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seqblue", RAMP)
os.makedirs("results/figs", exist_ok=True)

lab = pd.read_csv("mctlt/labels_all.csv", header=None, names=["id", "t"]).set_index("id")["t"]
ORDER5 = ["HCC", "ICC", "BCLM", "CRLM", "HH"]
BM = ["Malignant", "Benign"]


def lp_pred(path, key):
    """Returns y, 5-class argmax prediction, and HCC-vs-HH prediction (true HCC/HH cases only,
    argmax restricted to the HCC and HH probabilities of the same 5-class model)."""
    o = np.load(path, allow_pickle=True).item()
    y, C, P = np.array(o["y"]), np.array(list(o["classes"])), o["preds"][key].mean(0)
    m = np.isin(y, ["HCC", "HH"])
    ih, ic = list(C).index("HH"), list(C).index("HCC")
    pair = np.where(P[m, ic] >= P[m, ih], "HCC", "HH")
    return y, C[P.argmax(1)], (y[m], pair)


y_m, pred_m, pair_m = lp_pred("results/linear_probe_oof.npy", ("Merlin emb2048", "concat4"))
y_rl, pred_rl, pair_rl = lp_pred("results/linear_probe_oof_radar.npy", ("RADAR liver query256", "concat4"))


def bm(v):
    return np.where(np.asarray(v) == "HH", BM[1], BM[0])


TITLES = ["RADAR linear probe\n(liver features, 4-phase concat)", "Merlin linear probe\n(4-phase concat)"]
TOP = [(y_rl, pred_rl, ORDER5, ORDER5),
       (y_m, pred_m, ORDER5, ORDER5)]
BOTTOM = [(bm(y), bm(p), BM, BM) for y, p, _, _ in TOP]
PAIR = [(*pair_rl, ["HCC", "HH"], ["HCC", "HH"]), (*pair_m, ["HCC", "HH"], ["HCC", "HH"])]


def draw(ax, yt, yp, rows, cols, title=None, ylabel=False):
    yt, yp = np.asarray(yt), np.asarray(yp)
    cm = np.array([[np.sum((yt == r) & (yp == c)) for c in cols] for r in rows])
    pct = cm / cm.sum(1, keepdims=True)
    ax.imshow(pct, cmap=CMAP, vmin=0, vmax=1, aspect="equal")
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.0%})", ha="center", va="center", fontsize=14,
                    color=SURFACE if pct[i, j] > 0.5 else INK)
    ax.set_xticks(range(len(cols)), cols, color=INK)
    ax.set_yticks(range(len(rows)), [f"{r}\n(n={cm[i].sum()})" for i, r in enumerate(rows)], color=INK)
    ax.set_xlabel("Predicted", color=MUTED)
    if ylabel:
        ax.set_ylabel("True", color=MUTED)
    if title:
        ax.set_title(title, color=INK, fontsize=15)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-.5, len(rows)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)


from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(18, 11.5))
fig.patch.set_facecolor(SURFACE)
gs = GridSpec(2, 4, figure=fig, height_ratios=[3.6, 2.2], hspace=0.28, wspace=0.75)
for k in range(2):
    draw(fig.add_subplot(gs[0, 2 * k:2 * k + 2]), *TOP[k], title=TITLES[k], ylabel=(k == 0))
SMALL = [(BOTTOM[0], "RADAR\nbenign vs malignant"), (BOTTOM[1], "Merlin\nbenign vs malignant"),
         (PAIR[0], "RADAR\nHCC vs HH"), (PAIR[1], "Merlin\nHCC vs HH")]
for k, (args, t) in enumerate(SMALL):
    ax = draw(fig.add_subplot(gs[1, k]), *args, title=t, ylabel=(k == 0))
fig.subplots_adjust(left=0.09, right=0.99, top=0.95, bottom=0.04)
fig.savefig("results/figs/confusion.png", dpi=200)
fig.savefig("results/figs/confusion.pdf")
print("saved results/figs/confusion.{png,pdf}")
