"""Read-only: print compute / environment facts for Chapter 5.2.

Run from D:\\thesis:
    python scripts\\ch5_compute.py > results\\ch5_compute_output.txt

Nothing is written except to stdout. No data, image or checkpoint is opened.
"""
import json
from pathlib import Path

ROOT = Path("results/run1_cloud")
LOGS = ROOT / "logs"


def flat(obj, prefix="", depth=0, out=None):
    """Flatten nested JSON into key: value lines (short values only)."""
    if out is None:
        out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            flat(v, f"{prefix}{k}.", depth + 1, out)
    elif isinstance(obj, list):
        if len(obj) <= 6 and all(not isinstance(x, (dict, list)) for x in obj):
            out.append((prefix[:-1], obj))
        elif len(obj) <= 12 and depth < 4:
            for i, v in enumerate(obj):
                flat(v, f"{prefix}{i}.", depth + 1, out)
        else:
            out.append((prefix[:-1], f"<list of {len(obj)} items>"))
    else:
        s = str(obj)
        out.append((prefix[:-1], s if len(s) <= 120 else s[:117] + "..."))
    return out


def show(path):
    print("=" * 78)
    print(path.as_posix())
    print("-" * 78)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"  could not read: {e}")
        return
    for k, v in flat(data):
        print(f"  {k}: {v}")


print("#" * 78)
print("# 1. SESSION METADATA (run_meta_*_session.json)")
print("#" * 78)
for p in sorted(ROOT.glob("run_meta*.json")):
    show(p)

print()
print("#" * 78)
print("# 2. GENERATOR BUDGET STATES")
print("#" * 78)
for p in sorted(LOGS.glob("budget_state_*.json")):
    show(p)

print()
print("#" * 78)
print("# 3. SAMPLING LOGS")
print("#" * 78)
for p in sorted(LOGS.glob("sampling_*.json")):
    show(p)

print()
print("#" * 78)
print("# 4. CLASSIFIER RUN COUNTS (history files)")
print("#" * 78)
groups = {}
for p in LOGS.glob("*_history.json"):
    key = p.name.split("_")[0]
    groups[key] = groups.get(key, 0) + 1
for k in sorted(groups):
    print(f"  {k}: {groups[k]} runs")
print(f"  TOTAL classifier runs with a history file: {sum(groups.values())}")
