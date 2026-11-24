"""Redraw fig13 (pooled FID at each efficiency-curve checkpoint) with thesis styling.

Run from D:\\thesis:
    python scripts\\fig13_fid_curve_v2.py

Reads results/run1_cloud/metrics/fid_curve.csv only.
Saves results/figures/fig13_fid_curve.png (overwrites the old version; same data,
new axis label "lower = closer to the real images" and Times font).
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path("results/run1_cloud/metrics/fid_curve.csv")
OUT = Path("results/figures/fig13_fid_curve.png")

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]

pooled = {"gan": {}, "lora": {}}
with open(SRC, newline="") as f:
    for r in csv.DictReader(f):
        if r["class"] == "ALL":
            pooled[r["arm"]][float(r["checkpoint"].rstrip("h"))] = float(r["fid"])

print("Pooled FID (500 synthetic images per class):")
for arm in ("gan", "lora"):
    print(" ", arm, {k: round(v, 1) for k, v in sorted(pooled[arm].items())})

fig, ax = plt.subplots(figsize=(5.6, 3.4))
style = {"gan": ("Conditional WGAN-GP", "#2a78d6"),
         "lora": ("Stable Diffusion 1.5 + LoRA", "#eb6834")}
for arm, (label, colour) in style.items():
    xs = sorted(pooled[arm])
    ax.plot(xs, [pooled[arm][x] for x in xs], "-o", color=colour, label=label,
            linewidth=2, markersize=6, markeredgecolor="white")
ax.set_xticks([1, 2, 3, 4])
ax.set_xlabel("Generator training budget (GPU-hours)")
ax.set_ylabel("Pooled FID (lower = closer to the real images)")
ax.set_ylim(0, 230)
ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="lower left")
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=300)
print(f"Saved {OUT}")
