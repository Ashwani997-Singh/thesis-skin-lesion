"""
src/budget.py - declared training budget: enforcement, persistence, resumption.

DECLARED UNIT: GPU-hours (proposal Section 7.3, D1).
Optimiser steps are a supplementary hardware-invariant cross-check and a hard
cap. They do not redefine the declared unit.

WHY THIS VERSION EXISTS
-----------------------
A cloud session dies (Kaggle's 12-hour wall, a dropped connection, a browser
tab closing). The run resumes. A naive tracker restarts its clock at zero, so
an arm that had already consumed 2.5 GPU-hours receives another 4.0 and the
log reports 4.0. The equal-budget control - the entire contribution of this
thesis - is destroyed, and NOTHING RAISES AN ERROR.

This tracker persists cumulative seconds, cumulative optimiser steps and
already-fired D9 checkpoints to disk, and reloads them on resume. Budget is
enforced on the CUMULATIVE total across all segments.

It also refuses to resume onto a different GPU. Four GPU-hours on a P100 and
four on a T4 are not the same quantity of computation, so a segmented run
spanning two devices is confounded exactly as a two-device comparison would be.

Usage:

    with BudgetTracker("gan", run_id="run1_cloud", gpu_hours=4.0,
                       max_steps=20000, session_limit_hours=11.0) as bt:
        if bt.resumed:
            step = load_state(ckpt, models, optims)["step"]
        while not bt.exhausted():
            for batch in loader:
                ...
                bt.record_step()
                if bt.step_interval(1000):
                    save_state(resume_path, bt.steps, bt, models, optims)
                label = bt.checkpoint_due()
                if label:
                    save_state(f"gan_{label}.pt", bt.steps, bt, models, optims)
                if bt.exhausted() or bt.session_expiring():
                    break
            if bt.exhausted() or bt.session_expiring():
                break
        save_state(resume_path, bt.steps, bt, models, optims)
"""

from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch


class DeviceChanged(RuntimeError):
    """Raised when a resumed segment lands on a different GPU."""


def device_fingerprint():
    """Identity of the current accelerator, recorded in every segment."""
    if not torch.cuda.is_available():
        return {"name": "cpu", "vram_gb": None, "capability": None}
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "vram_gb": round(props.total_memory / 1e9, 2),
        "capability": f"{props.major}.{props.minor}",
    }


def _utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class BudgetTracker:
    """Cumulative, resumable, device-pinned GPU-hour budget.

    Parameters
    ----------
    arm : str                "gan" or "diffusion"
    run_id : str             "run1_cloud" or "run2_local"
    gpu_hours : float        the declared budget (config budget.gpu_hours_per_arm)
    max_steps : int          supplementary cap (config budget.max_optimiser_steps)
    checkpoint_at : seq      cumulative GPU-hours at which to dump the generator (D9)
    state_dir : str          where the persistent state file lives
    resume : bool            reload prior segments if a state file exists
    session_limit_hours : float or None
        Stop gracefully this long into the CURRENT segment, so a checkpoint is
        written before the platform kills the kernel. Kaggle's hard limit is
        12 hours; 11.0 leaves room to save and upload.
    allow_device_change : bool
        Off by default, and should stay off. Enabling it invalidates the
        equal-budget claim for that arm.
    """

    def __init__(self, arm, run_id, gpu_hours=4.0, max_steps=20000,
                 checkpoint_at=(1.0, 2.0, 3.0, 4.0), state_dir="logs",
                 resume=True, session_limit_hours=None,
                 allow_device_change=False, sync=True):
        self.arm = arm
        self.run_id = run_id
        self.budget_s = float(gpu_hours) * 3600.0
        self.max_steps = int(max_steps)
        self.sync = sync
        self.allow_device_change = allow_device_change
        self.session_limit_s = (session_limit_hours * 3600.0
                                if session_limit_hours else None)

        self.checkpoints = sorted({float(h) for h in checkpoint_at
                                   if 0 < float(h) <= float(gpu_hours)})

        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / f"budget_state_{run_id}_{arm}.json"

        self.device = device_fingerprint()
        self._base_s = 0.0          # seconds consumed in previous segments
        self.steps = 0
        self.checkpoint_seconds = 0.0   # D17: time spent writing checkpoints
        self._fired = set()
        self.segments = []
        self.history = []
        self.resumed = False
        self.stop_reason = None
        self._seg_start = None

        if resume and self.state_path.exists():
            self._load_state()

    # -- persistence --------------------------------------------------------

    def _load_state(self):
        state = json.loads(self.state_path.read_text())
        self._base_s = float(state["cumulative_seconds"])
        self.steps = int(state["optimiser_steps"])
        self.checkpoint_seconds = float(state.get("checkpoint_write_seconds", 0.0))
        self._fired = {float(h) for h in state.get("checkpoints_fired", [])}
        self.segments = state.get("segments", [])
        self.history = state.get("history", [])
        self.resumed = True

        prior = self.segments[-1]["device"] if self.segments else None
        if prior and prior["name"] != self.device["name"]:
            msg = (
                f"Resuming arm '{self.arm}' of {self.run_id} on "
                f"{self.device['name']}, but earlier segments ran on "
                f"{prior['name']}. Equal GPU-hours on unequal hardware is not "
                f"an equal budget. Either return to {prior['name']}, or delete "
                f"{self.state_path} and restart this arm from zero."
            )
            if not self.allow_device_change:
                raise DeviceChanged(msg)
            print(f"[budget] WARNING (device change permitted): {msg}")

        print(f"[budget] RESUMED {self.arm}: "
              f"{self._base_s/3600:.3f} GPU-h and {self.steps} steps already "
              f"consumed across {len(self.segments)} segment(s). "
              f"{self.remaining_hours():.3f} GPU-h remain.")

    def _write_state(self):
        state = {
            "arm": self.arm,
            "run_id": self.run_id,
            "declared_gpu_hours": round(self.budget_s / 3600, 4),
            "cumulative_seconds": round(self._cumulative_s(), 3),
            "cumulative_gpu_hours": round(self._cumulative_s() / 3600, 4),
            "optimiser_steps": self.steps,
            "max_optimiser_steps": self.max_steps,
            "checkpoint_write_seconds": round(self.checkpoint_seconds, 3),
            "checkpoint_write_share_of_budget": (
                round(self.checkpoint_seconds / self.budget_s, 5)
                if self.budget_s else None),
            "checkpoints_fired": sorted(self._fired),
            "checkpoints_pending": [c for c in self.checkpoints
                                    if c not in self._fired],
            "budget_exhausted": self._cumulative_s() >= self.budget_s,
            "segments": self.segments,
            "history": self.history,
            "updated": _utc(),
        }
        self.state_path.write_text(json.dumps(state, indent=2))
        return state

    # -- context management -------------------------------------------------

    def __enter__(self):
        self._maybe_sync()
        self._seg_start = time.perf_counter()
        self._seg_started_at = _utc()
        print(f"[budget] arm={self.arm} run={self.run_id} "
              f"device={self.device['name']} "
              f"declared={self.budget_s/3600:.2f} GPU-h "
              f"consumed={self._base_s/3600:.3f} "
              f"segment={len(self.segments)+1}")
        return self

    def __exit__(self, exc_type, exc, tb):
        self._maybe_sync()
        seg_s = time.perf_counter() - self._seg_start
        if self.stop_reason is None:
            self.stop_reason = "exception" if exc_type else "loop_ended"

        self.segments.append({
            "segment": len(self.segments) + 1,
            "device": self.device,
            "started": self._seg_started_at,
            "ended": _utc(),
            "seconds": round(seg_s, 3),
            "steps_at_end": self.steps,
            "stop_reason": self.stop_reason,
        })
        self._base_s += seg_s
        self._seg_start = None

        state = self._write_state()
        print(f"[budget] segment closed: {seg_s/3600:.3f} GPU-h. "
              f"cumulative {state['cumulative_gpu_hours']:.3f} / "
              f"{state['declared_gpu_hours']:.2f} GPU-h, "
              f"{self.steps} steps, reason={self.stop_reason}")
        if state["checkpoints_pending"] and state["budget_exhausted"]:
            print(f"[budget] WARNING: budget exhausted but checkpoints "
                  f"{state['checkpoints_pending']} never fired. The efficiency "
                  f"curve will have fewer points for this arm. Report it.")
        return False   # never suppress exceptions

    # -- clock --------------------------------------------------------------

    def _maybe_sync(self):
        if self.sync and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _segment_s(self):
        if self._seg_start is None:
            return 0.0
        self._maybe_sync()
        return time.perf_counter() - self._seg_start

    def _cumulative_s(self):
        return self._base_s + self._segment_s()

    def elapsed_hours(self):
        """Cumulative GPU-hours across ALL segments. This is the declared unit."""
        return self._cumulative_s() / 3600.0

    def remaining_hours(self):
        return max(0.0, (self.budget_s - self._cumulative_s()) / 3600.0)

    # -- step accounting ----------------------------------------------------

    def record_step(self, **extra):
        """Call once per OPTIMISER step, not per micro-batch.

        Under gradient accumulation, call only when opt.step() actually runs,
        or the step count will not mean what the dissertation says it means.
        """
        self.steps += 1
        if self.steps % 100 == 0:
            entry = {"step": self.steps,
                     "gpu_hours": round(self.elapsed_hours(), 4),
                     "segment": len(self.segments) + 1}
            entry.update(extra)
            self.history.append(entry)
        return self.steps

    def step_interval(self, every):
        """True every `every` optimiser steps. Use to trigger resume saves."""
        return self.steps > 0 and self.steps % every == 0

    def exhausted(self):
        if self._cumulative_s() >= self.budget_s:
            self.stop_reason = "gpu_hours_exhausted"
            return True
        if self.steps >= self.max_steps:
            self.stop_reason = "step_cap_reached"
            return True
        return False

    def session_expiring(self):
        """True when the CURRENT segment is near the platform's session wall.

        Not a budget condition - the run is unfinished and should be resumed.
        Distinguished from exhausted() so the two never get confused.
        """
        if self.session_limit_s is None:
            return False
        if self._segment_s() >= self.session_limit_s:
            self.stop_reason = "session_limit_reached"
            print("[budget] session limit reached - saving and stopping. "
                  "Resume in a new session; the budget clock carries over.")
            return True
        return False

    # -- D9 interval checkpointing -----------------------------------------

    def checkpoint_due(self):
        """Return "1.0h" etc. the first time a cumulative threshold is crossed.

        Fired thresholds persist across sessions, so a resumed run does not
        re-fire a checkpoint it already wrote.
        """
        now = self.elapsed_hours()
        for h in self.checkpoints:
            if h not in self._fired and now >= h:
                self._fired.add(h)
                self._write_state()          # persist immediately
                label = f"{h:.1f}h"
                print(f"[budget] checkpoint {label} at step {self.steps} "
                      f"(cumulative {now:.3f} GPU-h)")
                return label
        return None

    # -- D17: resume saves on a WALL-CLOCK interval -------------------------

    def save_due(self, every_minutes=10.0):
        """True once per `every_minutes` of CUMULATIVE budget.

        Scheduled in wall-clock time rather than optimiser steps, because
        wall-clock is the declared unit and the two arms run at very different
        step rates. A step-based interval would give the faster arm several
        times as many saves, so a policy meant to be symmetric would be
        asymmetric in the unit that is actually being controlled.

        Stateless across sessions: the next mark is recomputed from cumulative
        elapsed time, so a resumed run neither re-fires nor skips a save.
        """
        interval_h = every_minutes / 60.0
        if interval_h <= 0:
            return False
        now = self.elapsed_hours()
        mark = math.floor(now / interval_h)
        if mark > getattr(self, "_last_save_mark", -1):
            self._last_save_mark = mark
            return mark > 0        # do not fire at t=0
        return False

    def record_overhead(self, seconds):
        """Accumulate time spent writing checkpoints (D17).

        Charged to the budget, not excluded from it - but measured, so the
        dissertation reports the overhead rather than asserting it is small.
        """
        self.checkpoint_seconds += float(seconds)
        return self.checkpoint_seconds

    @contextmanager
    def timed_save(self):
        """Wrap any checkpoint write so its cost is measured.

            with bt.timed_save():
                save_state(path, bt.steps, bt, models, optims)
        """
        self._maybe_sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._maybe_sync()
            self.record_overhead(time.perf_counter() - t0)

    def snapshot(self):
        """Metadata stored with every checkpoint, carrying both budget units."""
        return {
            "arm": self.arm,
            "run_id": self.run_id,
            "device": self.device,
            "gpu_hours": round(self.elapsed_hours(), 4),
            "optimiser_steps": self.steps,
            "checkpoint_write_seconds": round(self.checkpoint_seconds, 3),
            "segment": len(self.segments) + 1,
            "resumed": self.resumed,
        }


# ---------------------------------------------------------------------------
# Cross-arm and cross-segment verification
# ---------------------------------------------------------------------------


def assert_same_device(run_id, state_dir="logs", arms=("gan", "diffusion")):
    """Both arms of one execution must have used one GPU model throughout.

    Checks every segment of every arm, not merely the final one - a resumed
    run that migrated between devices is confounded in exactly the way a
    two-device comparison would be.
    """
    seen = {}
    for arm in arms:
        path = Path(state_dir) / f"budget_state_{run_id}_{arm}.json"
        if not path.exists():
            raise FileNotFoundError(f"No budget state for '{arm}' in {run_id}: {path}")
        state = json.loads(path.read_text())
        for seg in state["segments"]:
            seen[f"{arm}/segment{seg['segment']}"] = seg["device"]["name"]

    names = set(seen.values())
    if len(names) != 1:
        raise AssertionError(
            f"{run_id} spans multiple GPU models: {seen}. Compute is "
            f"confounded with method. Rerun the affected arm from zero."
        )
    print(f"[budget] {run_id}: all segments of all arms on {names.pop()} "
          f"({len(seen)} segment(s)) - control holds.")
    return True


def assert_budget_respected(run_id, state_dir="logs", arms=("gan", "diffusion"),
                            tolerance_hours=0.05):
    """Both arms consumed the same declared budget, within tolerance.

    Catches the failure this module exists to prevent: an arm that silently
    received extra time by resuming with a reset clock.
    """
    used = {}
    for arm in arms:
        path = Path(state_dir) / f"budget_state_{run_id}_{arm}.json"
        state = json.loads(path.read_text())
        used[arm] = (state["cumulative_gpu_hours"], state["declared_gpu_hours"])

    for arm, (u, d) in used.items():
        if u > d + tolerance_hours:
            raise AssertionError(
                f"Arm '{arm}' consumed {u:.3f} GPU-h against a declared {d:.2f}. "
                f"Most likely a resumed segment restarted the clock."
            )
    declared = {d for _, d in used.values()}
    if len(declared) != 1:
        raise AssertionError(f"Arms declared different budgets: {used}")

    print(f"[budget] {run_id} budget respected: "
          + ", ".join(f"{a}={u:.3f}h" for a, (u, _) in used.items())
          + f" against {declared.pop():.2f}h declared.")
    return True


def budget_table(run_ids=("run1_cloud", "run2_local"), state_dir="logs",
                 arms=("gan", "diffusion")):
    """Declared vs consumed, per arm per execution - a dissertation table."""
    rows = []
    for run_id in run_ids:
        for arm in arms:
            path = Path(state_dir) / f"budget_state_{run_id}_{arm}.json"
            if not path.exists():
                continue
            s = json.loads(path.read_text())
            rows.append({
                "run": run_id,
                "arm": arm,
                "device": s["segments"][-1]["device"]["name"] if s["segments"] else "?",
                "declared_gpu_hours": s["declared_gpu_hours"],
                "used_gpu_hours": s["cumulative_gpu_hours"],
                "optimiser_steps": s["optimiser_steps"],
                "segments": len(s["segments"]),
            })
    return rows
