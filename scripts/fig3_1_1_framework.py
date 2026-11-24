"""Figure 3.1.1 Research framework of the study (marker point S7).

Pure diagram: every number is a fixed fact of the study design (Chapters 3-4);
no data files are read. 300 DPI, Times New Roman (Liberation Serif fallback).

Run from D:\\thesis:   python scripts\\fig3_1_1_framework.py
Output: results/figures/fig16_research_framework.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd()
OUT = ROOT / "results" / "figures" / "fig16_research_framework.png"

INK = "#222222"
EDGE = "#555555"
STAGE_FILL = "#f3f3f3"
BOX = "#ffffff"
GAN = "#dbe7f3"      # light blue  - WGAN-GP arm
DIF = "#f6e3d3"      # light orange - diffusion arm
KEY = "#e4efe0"      # light green - primary evaluation
SEC = "#eeeeee"      # grey - secondary analyses

fig, ax = plt.subplots(figsize=(6.5, 8.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 132)
ax.axis("off")


def box(x, y, w, h, title, body="", fill=BOX, fs=7.2, bold_title=True, lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                fc=fill, ec=EDGE, lw=lw))
    cx = x + w / 2
    if body:
        ax.text(cx, y + h - 1.6, title, ha="center", va="top", fontsize=fs + 0.6,
                fontweight="bold" if bold_title else "normal", color=INK)
        ax.text(cx, y + h - 5.0, body, ha="center", va="top", fontsize=fs,
                color=INK, linespacing=1.25)
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center", fontsize=fs + 0.6,
                fontweight="bold" if bold_title else "normal", color=INK)


def stage(y, h, label):
    ax.add_patch(FancyBboxPatch((0.5, y), 99, h, boxstyle="round,pad=0.2,rounding_size=1.5",
                                fc=STAGE_FILL, ec="none"))
    ax.text(2.2, y + h / 2, label, ha="center", va="center", rotation=90,
            fontsize=7.6, fontweight="bold", color="#444444")


def arrow(x1, y1, x2, y2, style="-|>", ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=EDGE, lw=0.9,
                                linestyle=ls, shrinkA=0, shrinkB=0,
                                mutation_scale=8))


# ---------------- Stage 1: data preparation ----------------
stage(108, 23.5, "1  Data")
box(6, 117.5, 27, 12, "HAM10000", "10,015 dermatoscopic\nimages, 7 classes")
box(36.5, 117.5, 27, 12, "Lesion-level split", "stratified by class;\nfrozen 23 Sep 2026")
box(67, 117.5, 27, 12, "Preprocessing", "resize to 128 px;\nImageNet normalisation")
arrow(33.4, 123.5, 36.1, 123.5)
arrow(63.9, 123.5, 66.6, 123.5)
box(6, 109.5, 27, 5.5, "Train  6,981", fs=6.8)
box(36.5, 109.5, 27, 5.5, "Validation  1,532", fs=6.8)
box(67, 109.5, 27, 5.5, "Test  1,502 (real only)", fs=6.8)
arrow(50, 117.1, 50, 115.4)

# ---------------- Stage 2: generator training ----------------
stage(79.5, 27, "2  Equal budget")
box(6, 97.5, 88, 7, "Generator source set: 1,526 training images of AKIEC, BCC, DF, MEL, VASC", fs=6.9)
arrow(19.5, 109.1, 19.5, 104.9)
ax.text(50, 94.6, "Identical budget: 4.0 GPU-hours per generator on one NVIDIA Tesla T4",
        ha="center", va="center", fontsize=7.0, style="italic", color=INK, zorder=6,
        bbox=dict(fc=STAGE_FILL, ec="none", pad=0.6))
box(6, 81, 42, 11, "Conditional WGAN-GP",
    "trained from scratch at 128 px\n3,943 optimiser steps", fill=GAN)
box(52, 81, 42, 11, "Stable Diffusion 1.5 + LoRA",
    "rank 8, 1.59 M trainable parameters\n512 px; 2,741 optimiser steps", fill=DIF)
arrow(27, 97.1, 27, 92.4)
arrow(73, 97.1, 73, 92.4)

# ---------------- Stage 3: synthesis and validation ----------------
stage(54.5, 23.5, "3  Synthesis")
box(6, 68.5, 88, 8.5, "Sampling from the final checkpoints",
    "1,000 images per class per arm (5,000 per arm), used at 128 px", fs=6.8)
arrow(27, 80.6, 27, 77.4)
arrow(73, 80.6, 73, 77.4)
box(6, 55.5, 42, 11.5, "Quality validation",
    "visual inspection of sample grids;\nFID per arm at equal sample counts", fs=6.6)
box(52, 55.5, 42, 11.5, "Memorization audit",
    "LPIPS nearest neighbour vs all 1,526 source\nimages; \u03c4 = 0.1333 (1st percentile of\nreal-to-real, same lesion excluded)", fs=6.3)
arrow(27, 68.1, 27, 67.4)
arrow(73, 68.1, 73, 67.4)

# ---------------- Stage 4: classification ----------------
stage(26, 25, "4  Classification")
cw, gap, x0 = 20.5, 2.0, 6
labels = [("A  Baseline", "real images only", BOX),
          ("B  Classical", "real + geometric and\nphotometric augmentation", BOX),
          ("C  GAN", "real + 5,000\nWGAN-GP images", GAN),
          ("D  Diffusion", "real + 5,000\ndiffusion images", DIF)]
for i, (t, b, f) in enumerate(labels):
    box(x0 + i * (cw + gap), 37.5, cw, 10, t, b, fill=f, fs=6.2)
box(6, 27.5, 88, 7, "ResNet-50 (ImageNet-pretrained), identical settings; seeds 42, 43, 44; "
    "model selection on validation macro-F1", fs=6.5, bold_title=False)
# synthetic images: right-hand channel from sampling to C and D only
cC = x0 + 2 * (cw + gap) + cw / 2
cD = x0 + 3 * (cw + gap) + cw / 2
yl = 49.6
ax.plot([94.4, 97.3, 97.3, cC], [72.75, 72.75, yl, yl], color=EDGE, lw=0.9, ls="--")
arrow(cC, yl, cC, 47.9)
arrow(cD, yl, cD, 47.9)
ax.text(cC - 1.5, yl, "synthetic images (C and D only)", ha="right", va="center",
        fontsize=6.0, style="italic", color="#444444",
        bbox=dict(fc=STAGE_FILL, ec="none", pad=0.5))
for i in range(4):
    cx = x0 + i * (cw + gap) + cw / 2
    arrow(cx, 37.1, cx, 34.9)

# ---------------- Stage 5: evaluation ----------------
stage(0.5, 23.5, "5  Evaluation")
box(6, 11, 46, 11, "Primary evaluation on the test set",
    "macro-F1 and melanoma recall (primary);\nper-class recall and specificity, balanced\n"
    "accuracy, macro AUROC and AUPRC", fill=KEY, fs=6.3)
box(56, 11, 38, 11, "Statistical comparison",
    "paired bootstrap, 1,000 resamples;\nall three seeds must agree;\nexact McNemar (descriptive)",
    fill=KEY, fs=6.3)
arrow(52.4, 16.5, 55.6, 16.5)
arrow(29, 27.1, 29, 22.4)
box(6, 2, 88, 6.5, "Secondary analyses: efficiency curve (1\u20134 GPU-hours), DF quantity ablation\n"
    "(100\u20132,000 images), classifier tuning and other sensitivity analyses",
    fill=SEC, fs=6.3, bold_title=False)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"written {OUT}")
