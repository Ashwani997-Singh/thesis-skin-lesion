"""
paired_comparison.py - the paired analysis fixed in advance by decision D23.

For each declared pair of conditions and each seed-matched pair of runs:
  * difference in macro-F1 and in melanoma recall on the SAME bootstrap
    resamples of the test set (1,000 resamples, 95% percentile interval);
  * McNemar's exact test on melanoma test images (correct vs incorrect).

A difference is called CONSISTENT only when all three seed-matched intervals
exclude zero in the same direction. p-values are descriptive only
(PRE_REGISTRATION.md section 8).

Declared pairs (D23): D vs C (RQ1); C vs B and D vs B (RQ3).
C vs A and D vs A are printed afterwards as SUPPLEMENTARY - not declared in D23.

Usage (from D:\\thesis):
    python paired_comparison.py results/run1_cloud/metrics
Writes paired_comparison.csv into the same folder.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, recall_score

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MEL = CLASSES.index("mel")
SEEDS = [42, 43, 44]
N_BOOT, CI, BOOT_SEED = 1000, 0.95, 42
NAMES = {"A": "A_baseline_rerun", "B": "B_classical_rerun",
         "C": "C_gan", "D": "D_diffusion"}
DECLARED = [("D", "C"), ("C", "B"), ("D", "B")]
SUPPLEMENTARY = [("C", "A"), ("D", "A")]


def load(folder, cond, seed):
    z = np.load(Path(folder) / f"{NAMES[cond]}_seed{seed}_predictions.npz")
    return z["y_true"], z["y_pred"]


def macro_f1(y, p):
    return f1_score(y, p, labels=range(len(CLASSES)), average="macro",
                    zero_division=0)


def mel_recall(y, p):
    m = y == MEL
    return float((p[m] == MEL).mean()) if m.any() else np.nan


def compare(folder, a, b, seed, rng_idx):
    y, pa = load(folder, a, seed)
    y2, pb = load(folder, b, seed)
    assert np.array_equal(y, y2), "test labels differ between runs"
    out = {"pair": f"{a}-{b}", "seed": seed}
    lo_q, hi_q = (1 - CI) / 2 * 100, (1 + CI) / 2 * 100
    for name, fn in [("macro_f1", macro_f1), ("mel_recall", mel_recall)]:
        diffs = np.array([fn(y[i], pa[i]) - fn(y[i], pb[i]) for i in rng_idx])
        diffs = diffs[~np.isnan(diffs)]
        out[f"{name}_diff"] = fn(y, pa) - fn(y, pb)
        out[f"{name}_lo"] = np.percentile(diffs, lo_q)
        out[f"{name}_hi"] = np.percentile(diffs, hi_q)
    # McNemar, exact, on melanoma images only
    m = y == MEL
    ca, cb = pa[m] == MEL, pb[m] == MEL
    only_a, only_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
    out["mel_only_first_correct"], out["mel_only_second_correct"] = only_a, only_b
    out["mcnemar_p"] = (binomtest(only_a, only_a + only_b, 0.5).pvalue
                        if only_a + only_b else 1.0)
    return out


def verdict(rows, metric):
    lo = np.array([r[f"{metric}_lo"] for r in rows])
    hi = np.array([r[f"{metric}_hi"] for r in rows])
    if (lo > 0).all():
        return "CONSISTENT, first higher"
    if (hi < 0).all():
        return "CONSISTENT, first lower"
    return "not consistent"


def main(folder):
    y0, _ = load(folder, "A", 42)
    rng = np.random.default_rng(BOOT_SEED)
    idx = [rng.integers(0, len(y0), len(y0)) for _ in range(N_BOOT)]
    print(f"test images {len(y0)}, melanoma {(y0 == MEL).sum()}, "
          f"{N_BOOT} shared bootstrap resamples\n")

    all_rows = []
    for label, pairs in [("DECLARED (D23)", DECLARED),
                         ("SUPPLEMENTARY (not declared in D23)", SUPPLEMENTARY)]:
        print(f"=== {label} ===")
        for a, b in pairs:
            rows = [compare(folder, a, b, s, idx) for s in SEEDS]
            for r in rows:
                r["status"] = "declared" if (a, b) in DECLARED else "supplementary"
                print(f"  {a}-{b} seed {r['seed']}: "
                      f"macro-F1 {r['macro_f1_diff']:+.3f} "
                      f"[{r['macro_f1_lo']:+.3f}, {r['macro_f1_hi']:+.3f}]   "
                      f"MEL recall {r['mel_recall_diff']:+.3f} "
                      f"[{r['mel_recall_lo']:+.3f}, {r['mel_recall_hi']:+.3f}]   "
                      f"McNemar MEL {r['mel_only_first_correct']}/"
                      f"{r['mel_only_second_correct']} p={r['mcnemar_p']:.3f}")
            print(f"  -> {a}-{b} macro-F1:   {verdict(rows, 'macro_f1')}")
            print(f"  -> {a}-{b} MEL recall: {verdict(rows, 'mel_recall')}\n")
            all_rows += rows

    out = Path(folder) / "paired_comparison.csv"
    pd.DataFrame(all_rows).round(4).to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/run1_cloud/metrics")
