"""Per-class specificity (and recall, for cross-check) for Conditions A-D.

Answers marker point S12: per-class specificity is defined in Section 3.12 and
promised in Objective 5, so it is reported in Chapter 5.

Reads the saved test predictions (no GPU, no retraining):
    {A_baseline_rerun, B_classical_rerun, C_gan, D_diffusion}_seed{42,43,44}_predictions.npz
Keys used: y_true, y_pred, classes.

Specificity_k = TN_k / (TN_k + FP_k)   (Section 3.12)
Mean +/- sample SD (n-1) over the three seeds, as in Tables 5.5.1 and 5.5.2.

Run from D:\\thesis:   python scripts\\ch5_specificity.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONDS = {
    "A": "A_baseline_rerun",
    "B": "B_classical_rerun",
    "C": "C_gan",
    "D": "D_diffusion",
}
SEEDS = [42, 43, 44]
EXPECTED_SUPPORT = {"AKIEC": 52, "BCC": 71, "BKL": 167, "DF": 20,
                    "MEL": 167, "NV": 1004, "VASC": 21}
OUT = ROOT / "results" / "tables" / "table_5_5_3_per_class_specificity.csv"


def find(stem):
    hits = [p for p in ROOT.joinpath("results").rglob(f"{stem}_predictions.npz")]
    if len(hits) != 1:
        raise SystemExit(f"Expected exactly one file for {stem}, found {len(hits)}: {hits}")
    return hits[0]


rows = []
for cond, prefix in CONDS.items():
    for seed in SEEDS:
        path = find(f"{prefix}_seed{seed}")
        d = np.load(path, allow_pickle=True)
        y, p = d["y_true"], d["y_pred"]
        classes = [str(c) for c in d["classes"]]
        assert len(y) == 1502, f"{path.name}: {len(y)} test images, expected 1502"
        for k, name in enumerate(classes):
            pos = y == k
            tp = int(np.sum(pos & (p == k)))
            fn = int(np.sum(pos & (p != k)))
            fp = int(np.sum(~pos & (p == k)))
            tn = int(np.sum(~pos & (p != k)))
            if name in EXPECTED_SUPPORT:
                assert tp + fn == EXPECTED_SUPPORT[name], f"{name} support {tp+fn}"
            rows.append(dict(condition=cond, seed=seed, cls=name,
                             specificity=tn / (tn + fp), recall=tp / (tp + fn),
                             fp=fp, tn=tn))
        print(f"read {path.relative_to(ROOT)}")

df = pd.DataFrame(rows)
summ = (df.groupby(["cls", "condition"])
          .agg(spec_mean=("specificity", "mean"),
               spec_sd=("specificity", lambda s: s.std(ddof=1)),
               recall_mean=("recall", "mean"),
               fp_total=("fp", "sum"))
          .reset_index())

wide = summ.pivot(index="cls", columns="condition",
                  values=["spec_mean", "spec_sd"])
table = pd.DataFrame(index=wide.index)
for c in CONDS:
    table[c] = [f"{m:.3f} ± {s:.3f}" for m, s in
                zip(wide[("spec_mean", c)], wide[("spec_sd", c)])]

OUT.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(OUT, encoding="utf-8-sig")
print("\nPer-class specificity, mean ± SD (n-1) over seeds 42/43/44")
print(table.to_string())
print("\nCross-check: mean recall should match Table 5.5.2; FP totals over 3 seeds")
print(summ.pivot(index="cls", columns="condition", values="recall_mean").round(3).to_string())
print(summ.pivot(index="cls", columns="condition", values="fp_total").to_string())
print(f"\nwritten {OUT.relative_to(ROOT)}")
