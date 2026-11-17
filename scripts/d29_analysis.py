"""
scripts/d29_analysis.py - analysis of the D29 session (notebooks/kaggle/k09_d29_tuning_fid.ipynb).

Reads only committed files in results/run1_cloud/metrics and writes:
  * paired_comparison_tuned.csv  - D23 paired analysis on the TUNED runs
                                   (same pairs, same 1,000 shared resamples,
                                   same all-three-seeds rule)
  * d29_nomel_comparison.csv     - exploratory: D without synthetic melanoma
                                   vs D trained in the same session with the
                                   same settings (TUNED_D used the declared
                                   setting, chosen on validation)
  * d29_tuning_table.csv         - declared vs tuned, mean +/- SD per condition
  * results/figures/fig13_fid_curve.png, fig14_tuning_sensitivity.png (300 DPI)

Run from the repository root:
    python scripts/d29_analysis.py | Tee-Object results/run1_cloud/logs/d29_analysis_output.txt
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "results" / "run1_cloud" / "metrics"
FIG = ROOT / "results" / "figures"
SEEDS = [42, 43, 44]

spec = importlib.util.spec_from_file_location("pc", ROOT / "scripts" / "paired_comparison.py")
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)


def run_pairs(names, pairs, idx, label):
    pc.NAMES = names
    rows = []
    for a, b in pairs:
        rs = [pc.compare(MET, a, b, s, idx) for s in SEEDS]
        for r in rs:
            r["status"] = label
            print(f"  {a}-{b} seed {r['seed']}: macro-F1 {r['macro_f1_diff']:+.3f} "
                  f"[{r['macro_f1_lo']:+.3f}, {r['macro_f1_hi']:+.3f}]   "
                  f"MEL recall {r['mel_recall_diff']:+.3f} "
                  f"[{r['mel_recall_lo']:+.3f}, {r['mel_recall_hi']:+.3f}]   "
                  f"McNemar p={r['mcnemar_p']:.3f}")
        print(f"  -> {a}-{b} macro-F1:   {pc.verdict(rs, 'macro_f1')}")
        print(f"  -> {a}-{b} MEL recall: {pc.verdict(rs, 'mel_recall')}\n")
        rows += rs
    return rows


# ---- 1. selection on validation -------------------------------------------
sel = pd.read_csv(MET / "tuning_selection.csv")
print("=== Step 1: setting chosen per condition on VALIDATION macro-F1 (seed 42) ===")
print(sel[sel.chosen][["condition", "lr", "loss", "val_macro_f1", "epochs_run"]]
      .round(4).to_string(index=False), "\n")

# ---- 2. declared vs tuned ---------------------------------------------------
DECL = {"A": "A_baseline_rerun", "B": "B_classical_rerun", "C": "C_gan", "D": "D_diffusion"}
rows = []
for c in "ABCD":
    for kind, stem in [("declared", DECL[c]), ("tuned", f"TUNED_{c}")]:
        rs = [json.loads((MET / f"{stem}_seed{s}.json").read_text()) for s in SEEDS]
        for m in ["macro_f1", "melanoma_recall", "balanced_accuracy", "macro_auroc", "macro_auprc"]:
            v = [r[m] for r in rs]
            rows.append({"condition": c, "classifier": kind, "metric": m,
                         "mean": np.mean(v), "sd": np.std(v, ddof=1)})
tab = pd.DataFrame(rows)
tab.round(4).to_csv(MET / "d29_tuning_table.csv", index=False)
wide = tab.assign(v=lambda d: d["mean"].map("{:.3f}".format) + " ± " + d["sd"].map("{:.3f}".format)) \
          .pivot_table(index=["metric", "classifier"], columns="condition", values="v", aggfunc="first")
print("=== Declared vs tuned classifier (mean ± SD, 3 seeds) ===")
print(wide.to_string(), "\n")

# ---- 3. D23 paired analysis on the tuned runs --------------------------------
TUNED = {c: f"TUNED_{c}" for c in "ABCD"}
pc.NAMES = TUNED
y0, _ = pc.load(MET, "A", 42)
rng = np.random.default_rng(pc.BOOT_SEED)
idx = [rng.integers(0, len(y0), len(y0)) for _ in range(pc.N_BOOT)]
print("=== D23 paired comparison on TUNED runs: declared pairs ===")
out = run_pairs(TUNED, pc.DECLARED, idx, "declared (tuned)")
print("=== supplementary (not declared in D23) ===")
out += run_pairs(TUNED, pc.SUPPLEMENTARY, idx, "supplementary (tuned)")
pd.DataFrame(out).round(4).to_csv(MET / "paired_comparison_tuned.csv", index=False)

# ---- 4. exploratory: D without synthetic melanoma ---------------------------
print("=== Exploratory: D without synthetic MEL (N) vs D, same session and settings (D) ===")
nomel = run_pairs({"N": "D_diffusion_noMEL", "D": "TUNED_D"}, [("N", "D")], idx, "exploratory")
pd.DataFrame(nomel).round(4).to_csv(MET / "d29_nomel_comparison.csv", index=False)

# ---- 5. FID per checkpoint --------------------------------------------------
fc = pd.read_csv(MET / "fid_curve.csv")
pooled = fc[fc["class"] == "ALL"].pivot(index="checkpoint", columns="arm", values="fid")
print("=== FID per checkpoint (500 per class; pooled) ===")
print(pooled.round(1).to_string())
print("\nper class:")
print(fc[fc["class"] != "ALL"].pivot_table(index=["arm", "class"], columns="checkpoint",
                                           values="fid").round(1).to_string(), "\n")

# ---- 6. figures ---------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})
COL = {"A": "#8a8985", "B": "#1baf7a", "C": "#2a78d6", "D": "#eb6834"}
FIG.mkdir(parents=True, exist_ok=True)

h = [1.0, 2.0, 3.0, 4.0]
fig, ax = plt.subplots(figsize=(5.5, 3.4))
for arm, lab, c in [("gan", "GAN (WGAN-GP)", COL["C"]), ("lora", "Diffusion (SD 1.5 + LoRA)", COL["D"])]:
    ax.plot(h, pooled.loc[[f"{x:.1f}h" for x in h], arm].values, "o-", color=c, label=lab, lw=1.8)
ax.set_xticks(h); ax.set_xlabel("Generator training budget (GPU-hours)")
ax.set_ylabel("Pooled FID (lower = more realistic)")
ax.legend(frameon=False); ax.set_ylim(0, None)
fig.tight_layout(); fig.savefig(FIG / "fig13_fid_curve.png", dpi=300); plt.close(fig)

f1 = tab[tab.metric == "macro_f1"]
fig, ax = plt.subplots(figsize=(5.5, 3.4))
x = np.arange(4); wdt = 0.36
for k, (kind, alpha) in enumerate([("declared", 0.45), ("tuned", 1.0)]):
    d = f1[f1.classifier == kind].set_index("condition").loc[list("ABCD")]
    ax.bar(x + (k - 0.5) * wdt, d["mean"], wdt, yerr=d["sd"], capsize=3,
           color=[COL[c] for c in "ABCD"], alpha=alpha, label=f"{kind} classifier settings")
ax.set_xticks(x); ax.set_xticklabels(["A Baseline", "B Classical", "C GAN", "D Diffusion"])
ax.set_ylabel("Test macro-F1 (mean ± SD, 3 seeds)"); ax.set_ylim(0.5, 0.72)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(FIG / "fig14_tuning_sensitivity.png", dpi=300); plt.close(fig)
print("saved fig13_fid_curve.png, fig14_tuning_sensitivity.png")
print("saved paired_comparison_tuned.csv, d29_nomel_comparison.csv, d29_tuning_table.csv")
