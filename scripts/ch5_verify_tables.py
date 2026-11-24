"""Read-only: recompute Table 5.1 / 5.2 from the per-seed metrics JSON files.

Run from D:\\thesis:
    python scripts\\ch5_verify_tables.py > results\\ch5_verify_output.txt

Prints, for each reported condition, every scalar metric found in the JSON
files as mean, SD with n-1 (sample) and SD with n (population), so the values
in results/tables/table_5_1_conditions.csv can be checked against both.
Nothing is written except to stdout.
"""
import json
import statistics as st
from pathlib import Path

M = Path("results/run1_cloud/metrics")

CONDITIONS = {
    "A baseline (rerun, D23)": "A_baseline_rerun_seed{}.json",
    "A baseline (original 2 Oct)": "A_baseline_seed{}.json",
    "B classical (rerun, D23)": "B_classical_rerun_seed{}.json",
    "B classical (original 2 Oct)": "B_classical_seed{}.json",
    "C GAN": "C_gan_seed{}.json",
    "D diffusion": "D_diffusion_seed{}.json",
}
SEEDS = [42, 43, 44]


def flat(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flat(v, f"{prefix}{k}."))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix[:-1]] = float(obj)
    return out


# 1. show the structure of one file so the key names are visible
first = M / CONDITIONS["C GAN"].format(42)
print("#" * 78)
print(f"# KEYS IN {first.name}")
print("#" * 78)
raw = json.loads(first.read_text(encoding="utf-8"))
for k, v in flat(raw).items():
    print(f"  {k} = {v:.4f}")

# 2. mean and both SDs per condition, for every scalar key
for name, pattern in CONDITIONS.items():
    rows = []
    for s in SEEDS:
        p = M / pattern.format(s)
        if not p.exists():
            print(f"\n!! missing {p}")
            continue
        rows.append(flat(json.loads(p.read_text(encoding="utf-8"))))
    if len(rows) < 2:
        continue
    print()
    print("=" * 78)
    print(f"{name}   (n = {len(rows)} seeds)")
    print("=" * 78)
    print(f"  {'metric':48} {'mean':>7} {'SD n-1':>7} {'SD n':>7}   per seed")
    keys = [k for k in rows[0] if all(k in r for r in rows)]
    for k in keys:
        vals = [r[k] for r in rows]
        mean = st.mean(vals)
        sd1 = st.stdev(vals)
        sd0 = st.pstdev(vals)
        seeds = ", ".join(f"{v:.3f}" for v in vals)
        print(f"  {k:48} {mean:7.3f} {sd1:7.3f} {sd0:7.3f}   [{seeds}]")
