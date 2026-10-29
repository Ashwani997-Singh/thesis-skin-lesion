"""
scripts/verification_checks.py - re-runs, in one place, every check that was
first done by hand in the terminal, and records the result.

Run from the repository root, on the local machine, with the output saved:

    python scripts/verification_checks.py | Tee-Object results/run1_cloud/logs/verification_checks_output.txt

Every check prints PASS or FAIL with the numbers behind it. A failing check does
not stop the others. Nothing is written to results/ except by Tee-Object.

Checks
  1  Repository and data protection (analysis plan commit, .gitignore, no raw data in git)
  2  Frozen split (counts, lesion-level disjointness, no synthetic leakage)
  3  Checkpoint integrity (manifest digest; files on disk against the manifest)
  4  Generator budget logs (4.0 GPU-h per arm, one device, stop reason)
  5  Sampling manifests (counts complete, seconds per image)
  6  Conditions A-D summary re-derived from per-run JSON files
  7  Paired comparison (D23) re-derived from per-image predictions
  8  Audit summary (D24) re-derived from per-image distances
  9  LPIPS embedding equivalence and same-lesion exclusion (D22) on this GPU
 10  Threshold tau recomputed locally and compared with the Kaggle value
"""

import hashlib
import importlib.util
import json
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
R1 = ROOT / "results" / "run1_cloud"
MET, LOGS = R1 / "metrics", R1 / "logs"
CKPT_DIR = Path(r"D:\thesis_checkpoints\run1_cloud")
IMAGES = Path(r"D:\thesis\data\raw\HAM10000\images")
EXPECTED_DIGEST = "81280427151b71c7eb1cc79d78415265d6924ec30c27210ee0bd0c708672dbf2"
SEEDS = [42, 43, 44]
CONDS = ["A_baseline_rerun", "B_classical_rerun", "C_gan", "D_diffusion"]
RESULTS = []


def check(name):
    def wrap(fn):
        print(f"\n=== {name} ===")
        try:
            ok, detail = fn()
        except Exception:
            ok, detail = False, traceback.format_exc(limit=2)
        print(detail)
        print("PASS" if ok else "FAIL")
        RESULTS.append((name, ok))
        return fn
    return wrap


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


@check("1  Repository and data protection")
def _():
    prereg = git("log", "--format=%h %ad %s", "--", "PRE_REGISTRATION.md")
    tracked = git("ls-files", "data").splitlines()
    raw_hist = git("log", "--all", "--oneline", "--", "data/raw", "data/synthetic")
    ign = git("check-ignore", "-v", "--non-matching",
              "results/run1_cloud/synthetic/x.png", "results/run2_local/checkpoints/x.pt",
              "results/run1_cloud/samples/x.png", "results/figures/x.png").splitlines()
    ignored = [l for l in ign if not l.startswith("::")]
    ok = (len(prereg.splitlines()) == 1 and prereg.startswith("02d7af7")
          and sorted(tracked) == ["data/splits/test.csv", "data/splits/train.csv", "data/splits/val.csv"]
          and raw_hist == "" and len(ignored) == 3)
    return ok, (f"analysis plan commits: {prereg}\n"
                f"tracked under data/: {tracked}\n"
                f"raw/synthetic ever committed: {'no' if raw_hist == '' else raw_hist}\n"
                f"ignored (expect synthetic, checkpoints, samples; figures tracked):\n  "
                + "\n  ".join(ign))


@check("2  Frozen split")
def _():
    from src.data import assert_no_synthetic_leakage
    sp = ROOT / "data" / "splits"
    tr, va, te = (pd.read_csv(sp / f"{n}.csv") for n in ["train", "val", "test"])
    a, b, c = set(tr.lesion_id), set(va.lesion_id), set(te.lesion_id)
    counts = (len(tr), len(va), len(te))
    assert_no_synthetic_leakage(sp / "val.csv", sp / "test.csv")
    ok = counts == (6981, 1532, 1502) and not (a & b or a & c or b & c) and sum(counts) == 10015
    return ok, f"train/val/test = {counts}; lesion overlap = {len(a & b) + len(a & c) + len(b & c)}"


@check("3  Checkpoint integrity")
def _():
    m = pd.read_csv(MET.parent / "checkpoint_manifest.csv")
    lines = sorted(f"{p}|{s}" for p, s in zip(m.path, m.sha256))
    digest = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    mism, missing = [], []
    for p, s in zip(m.path, m.sha256):
        f = CKPT_DIR / p
        if not f.exists():
            missing.append(p); continue
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for ch in iter(lambda: fh.read(1 << 20), b""):
                h.update(ch)
        if h.hexdigest() != s:
            mism.append(p)
    ok = digest == EXPECTED_DIGEST and len(m) == 38 and not mism and not missing
    return ok, (f"manifest files {len(m)}, digest {digest}\n"
                f"files on disk: {len(m) - len(missing)} found, {len(mism)} mismatched, "
                f"missing {missing or 'none'}")


@check("4  Generator budget logs")
def _():
    out, ok = [], True
    for arm in ["gan", "diffusion"]:
        b = json.loads((LOGS / f"budget_state_run1_cloud_{arm}.json").read_text())
        devs = {s["device"]["name"] for s in b["segments"]}
        reasons = {s["stop_reason"] for s in b["segments"]}
        ok &= (abs(b["cumulative_gpu_hours"] - 4.0) < 0.01 and devs == {"Tesla T4"}
               and b["budget_exhausted"] and reasons == {"gpu_hours_exhausted"})
        out.append(f"{arm}: {b['cumulative_gpu_hours']:.4f} GPU-h, {b['optimiser_steps']} steps, "
                   f"segments {len(b['segments'])} on {devs}, stop {reasons}, "
                   f"checkpoints {b['checkpoints_fired']}")
    return ok, "\n".join(out)


@check("5  Sampling manifests")
def _():
    out, ok = [], True
    for f in sorted(LOGS.glob("sampling_*.json")):
        if "smoke" in f.name:
            continue
        lt = json.loads(f.read_text())["latest"]
        ok &= lt["complete"]
        out.append(f"{f.name}: complete={lt['complete']}, counts={lt['counts_on_disk']}, "
                   f"s/image={lt['seconds_per_image']}, device={lt['device']}")
    return ok and bool(out), "\n".join(out)


@check("6  Conditions A-D summary re-derived")
def _():
    summ = pd.read_csv(MET / "ABCD_summary.csv")
    out, worst = [], 0.0
    for cond in CONDS:
        rs = [json.loads((MET / f"{cond}_seed{s}.json").read_text()) for s in SEEDS]
        for metric in ["macro_f1", "melanoma_recall", "balanced_accuracy", "macro_auroc", "macro_auprc"]:
            mean = float(np.mean([r[metric] for r in rs]))
            ref = summ[(summ.condition == cond) & (summ.metric == metric)]["mean"].iloc[0]
            worst = max(worst, abs(mean - ref))
            if metric in ("macro_f1", "melanoma_recall"):
                out.append(f"{cond:18s} {metric:16s} {mean:.4f} +/- "
                           f"{np.std([r[metric] for r in rs], ddof=1):.4f}")
    return worst < 1e-3, "\n".join(out) + f"\nmax |re-derived - saved summary| = {worst:.2e}"


@check("7  Paired comparison (D23) re-derived")
def _():
    spec = importlib.util.spec_from_file_location("pc", ROOT / "scripts" / "paired_comparison.py")
    pc = importlib.util.module_from_spec(spec); spec.loader.exec_module(pc)
    saved = pd.read_csv(MET / "paired_comparison.csv")
    y0, _ = pc.load(MET, "A", 42)
    rng = np.random.default_rng(pc.BOOT_SEED)
    idx = [rng.integers(0, len(y0), len(y0)) for _ in range(pc.N_BOOT)]
    worst, verdicts = 0.0, []
    for a, b in pc.DECLARED:
        rows = [pc.compare(MET, a, b, s, idx) for s in SEEDS]
        for r in rows:
            ref = saved[(saved.pair == f"{a}-{b}") & (saved.seed == r["seed"])].iloc[0]
            for k in ["macro_f1_diff", "macro_f1_lo", "macro_f1_hi", "mel_recall_diff", "mel_recall_lo", "mel_recall_hi"]:
                worst = max(worst, abs(r[k] - ref[k]))
        verdicts.append(f"{a}-{b}: macro-F1 {pc.verdict(rows, 'macro_f1')}; "
                        f"MEL recall {pc.verdict(rows, 'mel_recall')}")
    return worst < 1e-3, "\n".join(verdicts) + f"\nmax |re-derived - saved| = {worst:.2e}"


@check("8  Audit summary (D24) re-derived")
def _():
    h = pd.read_csv(MET / "audit_headline_per_image.csv")
    r2r = pd.read_csv(MET / "audit_real_to_real.csv")
    tau = float(np.percentile(r2r.nn_dist, 1.0))
    abl = pd.read_csv(MET / "audit_df_ablation.csv")
    nd = h.assign(nd=h.nn_dist < tau).groupby("arm").nd.sum()
    ok = abs(tau - 0.1333) < 5e-4 and int(nd.sum()) == 0 and int(abl.near_duplicates.sum()) == 0
    return ok, (f"tau from saved real-to-real distances = {tau:.4f}\n"
                f"near-duplicates per arm: {nd.to_dict()}\n"
                f"closest synthetic per arm: {h.groupby('arm').nn_dist.min().round(4).to_dict()}\n"
                f"DF ablation near-duplicates total: {int(abl.near_duplicates.sum())}")


@check("9 + 10  LPIPS equivalence, same-lesion exclusion, threshold recomputed locally")
def _():
    import torch
    from src.data import build_image_index, generator_source_frame
    from src.memorization import (LPIPSEmbedder, calibrate_threshold, embed_paths,
                                  load_images, real_to_real)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fr = generator_source_frame(ROOT / "data" / "splits" / "train.csv")
    idx = build_image_index(IMAGES)
    paths = [idx[i] for i in fr.image_id]
    e = LPIPSEmbedder(device=dev)
    x = load_images(paths[:32])
    err = e.verify_equivalence(x, x.flip(0))
    emb = embed_paths(e, paths)
    d, j = real_to_real(emb, fr.lesion_id.values)
    same = int((fr.lesion_id.values[j] == fr.lesion_id.values).sum())
    tau = calibrate_threshold(d, 1.0)
    kag = pd.read_csv(MET / "audit_real_to_real.csv").set_index("image_id")
    agree = float((kag.loc[fr.image_id, "nn_image_id"].values == fr.image_id.values[j]).mean())
    maxdiff = float(np.abs(kag.loc[fr.image_id, "nn_dist"].values - d).max())
    ok = err < 1e-3 and same == 0 and abs(tau - 0.1333) < 5e-4 and agree > 0.99
    return ok, (f"device {dev}: equivalence error {err:.2e}; same-lesion neighbours {same}\n"
                f"tau local {tau:.4f} vs Kaggle 0.1333; same nearest neighbour as Kaggle for "
                f"{agree:.1%} of 1,526 images; max |distance difference| {maxdiff:.2e}")


print("\n=== SUMMARY ===")
for name, ok in RESULTS:
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
print(f"{sum(ok for _, ok in RESULTS)}/{len(RESULTS)} checks passed")
