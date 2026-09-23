"""
src/run_registry.py - keeps the two executions separate and auditable.

PRE_REGISTRATION_2.md fixes Run 1 (Kaggle) as the primary execution and Run 2
(RTX 4060) as the secondary replication, with results reported separately and
never pooled. That separation has to be enforced by the filesystem, not by
remembering to type the right path: a single metrics JSON written to the wrong
directory would silently contaminate the comparison.

Every call to init_run appends a segment record with the device, so a run that
was interrupted and resumed carries a complete, timestamped hardware history.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

VALID_RUNS = {
    "run1_cloud": "PRIMARY - Kaggle free-tier GPU, matches proposal Table 8.2",
    "run2_local": "SECONDARY replication - local RTX 4060 (8 GB)",
}

SUBDIRS = ("metrics", "logs", "checkpoints", "synthetic", "figures")


def _utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def environment_fingerprint():
    """Everything needed to explain a result difference between segments."""
    info = {
        "timestamp": _utc(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update({
            "gpu": props.name,
            "vram_gb": round(props.total_memory / 1e9, 2),
            "capability": f"{props.major}.{props.minor}",
        })
        try:
            info["driver"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip().splitlines()[0]
        except Exception:
            info["driver"] = None
    else:
        info["gpu"] = "cpu"
    return info


def init_run(run_id, root="results", git_hash=None):
    """Create or extend an execution directory. Returns its path.

    Call at the top of EVERY notebook and script. The returned path is the only
    place that execution writes to.
    """
    if run_id not in VALID_RUNS:
        raise ValueError(
            f"Unknown run_id '{run_id}'. Valid: {list(VALID_RUNS)}. "
            f"Results from the two executions are never pooled, so the "
            f"identifier must be exact."
        )

    base = Path(root) / run_id
    for sub in SUBDIRS:
        (base / sub).mkdir(parents=True, exist_ok=True)

    meta_path = base / "run_meta.json"
    meta = (json.loads(meta_path.read_text()) if meta_path.exists()
            else {"run_id": run_id, "role": VALID_RUNS[run_id],
                  "created": _utc(), "segments": []})

    env = environment_fingerprint()
    if git_hash:
        env["git_hash"] = git_hash
    meta["segments"].append({"segment": len(meta["segments"]) + 1, **env})
    meta["last_seen"] = _utc()
    meta_path.write_text(json.dumps(meta, indent=2))

    seg = len(meta["segments"])
    print(f"[registry] {run_id} ({VALID_RUNS[run_id]})")
    print(f"[registry] segment {seg} on {env.get('gpu')} "
          f"| torch {env['torch']} | cuda {env['cuda_version']}")
    print(f"[registry] writing to {base}")

    if seg > 1:
        prior = {s.get("gpu") for s in meta["segments"][:-1]}
        if env.get("gpu") not in prior:
            print(f"[registry] WARNING: this segment is on {env.get('gpu')}, "
                  f"earlier segments used {prior}. Any generative arm resumed "
                  f"across this boundary is confounded - see "
                  f"BudgetTracker's device check.")
    return base


def assert_device_stable(run_id, root="results"):
    """Every segment of an execution used one GPU model."""
    meta = json.loads((Path(root) / run_id / "run_meta.json").read_text())
    gpus = {s.get("gpu") for s in meta["segments"]}
    if len(gpus) != 1:
        raise AssertionError(
            f"{run_id} spans multiple GPU models across sessions: {gpus}. "
            f"Identify which arms were trained on which, and rerun any arm "
            f"that straddles the boundary."
        )
    print(f"[registry] {run_id}: {len(meta['segments'])} segment(s), "
          f"all on {gpus.pop()}.")
    return True


def assert_run_isolation(root="results"):
    """No metrics file may sit outside a recognised execution directory.

    Catches the commonest contamination: a notebook run without init_run that
    wrote into results/metrics/ instead of results/run1_cloud/metrics/.
    """
    root = Path(root)
    stray = [p for p in root.glob("metrics/*.json")]
    stray += [p for p in root.glob("*.json") if p.name != "summary_both_runs.json"]
    if stray:
        raise AssertionError(
            f"Metrics found outside an execution directory: "
            f"{[str(p) for p in stray]}. Every notebook must call init_run() "
            f"first and write only under the path it returns."
        )
    for run_id in VALID_RUNS:
        d = root / run_id
        if d.exists():
            n = len(list((d / 'metrics').glob('*.json')))
            print(f"[registry] {run_id}: {n} metrics file(s)")
    return True


def run_provenance(run_id, root="results", state_dir=None):
    """Assemble the provenance record reported in the methodology chapter."""
    base = Path(root) / run_id
    meta = json.loads((base / "run_meta.json").read_text())
    state_dir = Path(state_dir) if state_dir else base / "logs"

    budgets = {}
    for arm in ("gan", "diffusion"):
        p = state_dir / f"budget_state_{run_id}_{arm}.json"
        if p.exists():
            s = json.loads(p.read_text())
            budgets[arm] = {
                "declared_gpu_hours": s["declared_gpu_hours"],
                "used_gpu_hours": s["cumulative_gpu_hours"],
                "optimiser_steps": s["optimiser_steps"],
                "segments": len(s["segments"]),
                "devices": sorted({sg["device"]["name"] for sg in s["segments"]}),
            }

    return {
        "run_id": run_id,
        "role": meta["role"],
        "sessions": len(meta["segments"]),
        "devices": sorted({s.get("gpu") for s in meta["segments"]}),
        "torch_versions": sorted({s["torch"] for s in meta["segments"]}),
        "budgets": budgets,
    }
