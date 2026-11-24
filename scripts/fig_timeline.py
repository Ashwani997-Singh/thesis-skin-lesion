"""Planned (proposal Table 9.1 / Figure 9.1) versus actual project timeline (marker point F8).

Planned weeks: Research Proposal, Table 9.1 (six phases, twelve weeks).
Actual dates: WORK_LOG.md Days 1-15 (13 Sep - 31 Oct 2026), the quantity ablation
(1-2 Nov, supervisor briefing), the D29 session (16 Nov, DECISION_LOG.md) and the
writing period (18 Nov - 2 Dec). Week 1 starts on Day 1, Sunday 13 September 2026.

Run from D:\\thesis:   python scripts\\fig_timeline.py
Output: results/figures/fig17_planned_vs_actual_timeline.png (300 DPI)
"""
from datetime import date
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd()
OUT = ROOT / "results" / "figures" / "fig17_planned_vs_actual_timeline.png"

START = date(2026, 9, 13)          # Day 1 of WORK_LOG = start of week 1


def wk(d, end=False):
    """Position in weeks from the start; end=True means the end of that day."""
    return ((d - START).days + (1 if end else 0)) / 7


def D(m, d):
    return date(2026, m, d)


# phase label, planned (first week, last week), actual segments [(start, end), ...]
PHASES = [
    ("1  Data, lesion-level split,\n    baseline (A)", (1, 2), [(D(9, 13), D(9, 25))]),
    ("2  Classical augmentation (B),\n    primary environment", (2, 3), [(D(9, 26), D(9, 28))]),
    ("3  Generator training\n    (4.0 GPU-h each), inspection", (3, 6), [(D(10, 4), D(10, 5))]),
    ("4  Synthesis; Conditions A\u2013D", (6, 8), [(D(10, 6), D(10, 6)), (D(10, 27), D(10, 27))]),
    ("5  Audit, FID, efficiency curve,\n    quantity ablation", (8, 10),
     [(D(10, 28), D(11, 2)), (D(11, 16), D(11, 16))]),
    ("6  Analysis and writing", (10, 12), [(D(11, 18), D(12, 2))]),
]

PLAN = "#c9d6e3"
ACT = "#2f5d8a"
PAUSE = "#ececec"

fig, ax = plt.subplots(figsize=(6.5, 3.9))
n = len(PHASES)
for i, (label, (a, b), segs) in enumerate(PHASES):
    y = n - 1 - i
    ax.barh(y + 0.17, b - (a - 1), left=a - 1, height=0.3, color=PLAN, edgecolor="#7d93aa", lw=0.6)
    for s, e in segs:
        ax.barh(y - 0.17, wk(e, True) - wk(s), left=wk(s), height=0.3, color=ACT)

# pause with no project work
p0, p1 = wk(D(10, 7)), wk(D(10, 26), True)
ax.axvspan(p0, p1, color=PAUSE, zorder=0)
ax.text(p0 + 0.08, n - 0.35, "no project work\n7\u201326 Oct", ha="left", va="top",
        fontsize=7, color="#555555", style="italic")

# milestones
for d, txt in [(D(10, 15), "proposal result\n15 Oct"), (D(12, 2), "submission\n2 Dec")]:
    x = wk(d, True) if d.month == 12 else wk(d) + 0.07
    ax.plot([x, x], [-0.7, n - 1.2], color="#999999", lw=0.7, ls="--", zorder=1)
    ax.text(x + 0.08, -0.62, txt, fontsize=6.5, color="#444444", va="bottom", ha="left")

ax.text(wk(D(11, 16)) + 0.25, 1 - 0.17, "sensitivity\nanalyses (D29)", fontsize=6.3,
        va="center", ha="left", color="#333333")

ax.set_yticks(range(n))
ax.set_yticklabels([p[0] for p in PHASES][::-1], fontsize=7.4)
ax.set_xlim(0, 12)
ax.set_ylim(-0.7, n - 0.3)
ax.set_xticks([k + 0.5 for k in range(12)])
ax.set_xticklabels([str(k + 1) for k in range(12)], fontsize=7.4)
ax.set_xticks(range(13), minor=True)
ax.tick_params(axis="x", which="major", length=0)
ax.tick_params(axis="x", which="minor", length=3)
ax.grid(axis="x", which="minor", color="#dddddd", lw=0.5)
ax.set_axisbelow(True)
ax.set_xlabel("Week (week 1 begins 13 September 2026)", fontsize=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(handles=[Patch(fc=PLAN, ec="#7d93aa", label="Planned (proposal Table 9.1)"),
                   Patch(fc=ACT, label="Actual (work log)")],
          loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=7.4)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"written {OUT}")
