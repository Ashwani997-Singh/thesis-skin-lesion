"""Figure 5.5.x: confusion matrices for Conditions A-D (re-executed runs, D23).

Run from D:\\thesis:
    python scripts\\fig07_confusion_AtoD.py

Reads only the per-image prediction files (*_predictions.npz).
Sums the three seeds per condition, normalises each row to % of the true class,
prints the numbers (for checking against Table 5.2) and saves a 300 DPI figure to
    results/figures/fig07_confusion_matrices_AtoD.png
The old fig07_confusion_matrices_run1.png is NOT touched.
"""
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

M = Path("results/run1_cloud/metrics")
OUT = Path("results/figures/fig07_confusion_matrices_AtoD.png")
SEEDS = [42, 43, 44]
CONDITIONS = [
    ("A", "Baseline", "A_baseline_rerun_seed{}_predictions.npz"),
    ("B", "Classical augmentation", "B_classical_rerun_seed{}_predictions.npz"),
    ("C", "WGAN-GP synthetic images", "C_gan_seed{}_predictions.npz"),
    ("D", "Diffusion synthetic images", "D_diffusion_seed{}_predictions.npz"),
]

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]

classes = None
mats = {}
for key, _, pattern in CONDITIONS:
    total = None
    for s in SEEDS:
        p = M / pattern.format(s)
        assert p.exists(), f"missing {p}"
        d = np.load(p, allow_pickle=True)
        cls = [str(c) for c in d["classes"]]
        if classes is None:
            classes = cls
        assert cls == classes, f"class order differs in {p}"
        y, yp = d["y_true"], d["y_pred"]
        assert len(y) == 1502, f"{p}: {len(y)} test images, expected 1502"
        k = len(classes)
        cm = np.zeros((k, k), dtype=int)
        np.add.at(cm, (y, yp), 1)
        total = cm if total is None else total + cm
    mats[key] = total

labels = [c.upper() for c in classes]
k = len(classes)

# ---- print numbers for checking ------------------------------------------------
for key, name, _ in CONDITIONS:
    cm = mats[key]
    pct = 100 * cm / cm.sum(axis=1, keepdims=True)
    print(f"\n=== {key} {name}: counts summed over 3 seeds (rows = true class)")
    print("        " + " ".join(f"{l:>6}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l:>6}  " + " ".join(f"{v:6d}" for v in cm[i]))
    print("  recall (diagonal %): " +
          ", ".join(f"{labels[i]} {pct[i, i]:.1f}" for i in range(k)))
    mi, ni = classes.index("mel"), classes.index("nv")
    print(f"  MEL predicted as NV: {pct[mi, ni]:.1f}%")

# ---- figure -----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 9.2))
for ax, (key, name, _) in zip(axes.ravel(), CONDITIONS):
    cm = mats[key]
    pct = 100 * cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(k):
        for j in range(k):
            v = pct[i, j]
            if v >= 0.5:
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=8,
                        color="white" if v >= 50 else "black")
    ax.set_title(f"{key}: {name}", fontsize=11)
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted class", fontsize=10)
    ax.set_ylabel("True class", fontsize=10)
fig.subplots_adjust(left=0.08, right=0.88, top=0.95, bottom=0.08, wspace=0.35, hspace=0.38)
cax = fig.add_axes([0.91, 0.25, 0.02, 0.5])
cb = fig.colorbar(im, cax=cax)
cb.set_label("% of true class (three seeds pooled)", fontsize=10)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"\nSaved {OUT}")
