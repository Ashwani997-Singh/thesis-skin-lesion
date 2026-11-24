"""Redraw fig14: declared vs tuned classifier settings, macro-F1 AND melanoma recall.

Run from D:\\thesis:
    python scripts\\fig14_tuning_v2.py

Reads results/run1_cloud/metrics/d29_tuning_table.csv only.
Saves results/figures/fig14_tuning_sensitivity.png (overwrites the old one-panel version).
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = Path("results/run1_cloud/metrics/d29_tuning_table.csv")
OUT = Path("results/figures/fig14_tuning_sensitivity.png")

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]

data = {}
with open(SRC, newline="") as f:
    for r in csv.DictReader(f):
        data[(r["condition"], r["classifier"], r["metric"])] = (float(r["mean"]), float(r["sd"]))

conds = ["A", "B", "C", "D"]
names = ["A: Baseline", "B: Classical", "C: WGAN-GP", "D: Diffusion"]
light = ["#c9c8c5", "#9bdac4", "#a0c2eb", "#f6bca5"]
dark = ["#8a8884", "#1baf7a", "#2a78d6", "#eb6834"]
metrics = [("macro_f1", "Macro-F1"), ("melanoma_recall", "Melanoma recall")]

print("condition  metric           declared            tuned")
for c in conds:
    for m, _ in metrics:
        d, t = data[(c, "declared", m)], data[(c, "tuned", m)]
        print(f"{c:9}  {m:15}  {d[0]:.3f} +/- {d[1]:.3f}   {t[0]:.3f} +/- {t[1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
x = np.arange(len(conds))
w = 0.36
for ax, (m, title) in zip(axes, metrics):
    for i, c in enumerate(conds):
        d, t = data[(c, "declared", m)], data[(c, "tuned", m)]
        ax.bar(x[i] - w / 2, d[0], w, color=light[i], yerr=d[1], capsize=3,
               error_kw={"linewidth": 1})
        ax.bar(x[i] + w / 2, t[0], w, color=dark[i], yerr=t[1], capsize=3,
               error_kw={"linewidth": 1})
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Test value (mean \u00b1 SD, three seeds)", fontsize=9)
    lo = 0.5 if m == "macro_f1" else 0.25
    ax.set_ylim(lo, 0.72)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
handles = [plt.Rectangle((0, 0), 1, 1, color="#c9c8c5"),
           plt.Rectangle((0, 0), 1, 1, color="#8a8884")]
fig.legend(handles, ["Declared classifier settings (lighter bar)",
                     "Tuned classifier settings (darker bar)"],
           loc="upper center", ncol=2, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.92))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=300)
print(f"Saved {OUT}")
