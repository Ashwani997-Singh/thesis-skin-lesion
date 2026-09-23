"""
src/checkpointing.py - crash-safe training state, and resumable sampling.

Two separate problems, both caused by sessions that end before the work does.

1. TRAINING. A killed session must resume with the model, the optimisers, the
   step count and the random number generators exactly where they were. Saving
   only model weights produces a run that is reproducible in name but not in
   fact: the optimiser's momentum buffers restart from zero and the data order
   changes.

2. SAMPLING. Generating 5,000 images at 512px takes hours and will be
   interrupted. Naive resumption reruns everything, or - worse - continues a
   sequential generator so the resumed images differ from those an
   uninterrupted run would have produced. Both are avoided here by deriving a
   deterministic per-image seed and skipping files that already exist.

Writes are atomic (temp file then rename), because a session killed midway
through a torch.save leaves a truncated file that fails to load, and the
resume point is lost.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# RNG state
# ---------------------------------------------------------------------------

def rng_state():
    """Capture every source of randomness set by src.utils.set_seed."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (torch.cuda.get_rng_state_all()
                 if torch.cuda.is_available() else None),
    }


def set_rng_state(state):
    if state is None:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------

def save_state(path, step, tracker=None, models=None, optimizers=None,
               extra=None, keep_rng=True):
    """Atomically write a resumable training checkpoint.

    models / optimizers are dicts, e.g. {"G": G, "C": critic}, so both arms
    use one function despite having different components.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "step": int(step),
        "models": {k: m.state_dict() for k, m in (models or {}).items()},
        "optimizers": {k: o.state_dict() for k, o in (optimizers or {}).items()},
        "budget": tracker.snapshot() if tracker is not None else None,
        "rng": rng_state() if keep_rng else None,
        "saved_at": time.time(),
        "extra": extra or {},
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)          # atomic on POSIX and on Windows (same volume)
    return path


def load_state(path, models=None, optimizers=None, restore_rng=True,
               map_location="cpu"):
    """Restore a checkpoint in place. Returns the payload minus the tensors.

    Returns None when the file does not exist, so a first run and a resumed
    run can share one code path.
    """
    path = Path(path)
    if not path.exists():
        return None

    payload = torch.load(path, map_location=map_location, weights_only=False)

    for key, module in (models or {}).items():
        if key not in payload["models"]:
            raise KeyError(f"checkpoint has no model '{key}' "
                           f"(has {list(payload['models'])})")
        module.load_state_dict(payload["models"][key])

    for key, opt in (optimizers or {}).items():
        if key in payload["optimizers"]:
            opt.load_state_dict(payload["optimizers"][key])

    if restore_rng:
        set_rng_state(payload.get("rng"))

    print(f"[ckpt] resumed from {path} at step {payload['step']}"
          + (f", {payload['budget']['gpu_hours']:.3f} GPU-h consumed"
             if payload.get("budget") else ""))
    return {"step": payload["step"], "budget": payload.get("budget"),
            "extra": payload.get("extra", {})}


# ---------------------------------------------------------------------------
# Deterministic, resumable sampling
# ---------------------------------------------------------------------------

def image_seed(base_seed, cls, idx):
    """A stable seed for one (class, index) image.

    Uses blake2b rather than Python's hash(), which is salted per process and
    would therefore produce different images on every session - silently
    breaking reproducibility across a resumed sampling run.
    """
    key = f"{base_seed}:{cls}:{idx}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big")


def resumable_generate(generate_one, classes, n_per_class, out_dir,
                       base_seed=42, ext="png", manifest_path=None,
                       tracker=None, session_limit_hours=None,
                       log_every=50):
    """Generate n_per_class images per class, skipping any already on disk.

    Parameters
    ----------
    generate_one : callable(cls, generator) -> PIL.Image
        Wraps the diffusion pipeline or the GAN. Must consume ONLY the supplied
        torch.Generator for randomness, so output depends on (class, index)
        alone and never on how many images preceded it in this session.
    session_limit_hours : float or None
        Stop cleanly this far into the current session. Already-written files
        make the next session resume where this one stopped.

    Sampling is EXCLUDED from the declared training budget (D2), but its cost
    is reported separately, so elapsed time is recorded in the manifest.
    """
    out_dir = Path(out_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start = time.perf_counter()
    limit_s = session_limit_hours * 3600 if session_limit_hours else None

    made, skipped = 0, 0
    for cls in classes:
        (out_dir / cls).mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            path = out_dir / cls / f"{cls}_{i:04d}.{ext}"
            if path.exists():
                skipped += 1
                continue

            gen = torch.Generator(device=device)
            gen.manual_seed(image_seed(base_seed, cls, i))
            img = generate_one(cls, gen)

            tmp = path.with_suffix(path.suffix + ".tmp")
            img.save(tmp)
            os.replace(tmp, path)
            made += 1

            if made % log_every == 0:
                rate = (time.perf_counter() - start) / made
                remaining = sum(
                    n_per_class - len(list((out_dir / c).glob(f"*.{ext}")))
                    for c in classes
                )
                print(f"[sample] {cls}: {made} made, {skipped} skipped, "
                      f"{rate:.2f}s/image, ~{remaining*rate/3600:.2f}h left")

            if limit_s and (time.perf_counter() - start) >= limit_s:
                print("[sample] session limit reached - stopping cleanly. "
                      "Rerun to resume; completed images are skipped.")
                _write_manifest(manifest_path, out_dir, classes, n_per_class,
                                base_seed, made, skipped, start, complete=False)
                return {"made": made, "skipped": skipped, "complete": False}

    result = _write_manifest(manifest_path, out_dir, classes, n_per_class,
                             base_seed, made, skipped, start, complete=True)
    return result


def _write_manifest(manifest_path, out_dir, classes, n_per_class, base_seed,
                    made, skipped, start, complete):
    elapsed = time.perf_counter() - start
    counts = {c: len(list((Path(out_dir) / c).glob("*"))) for c in classes}
    record = {
        "out_dir": str(out_dir),
        "classes": list(classes),
        "n_per_class_target": n_per_class,
        "counts_on_disk": counts,
        "base_seed": base_seed,
        "images_generated_this_session": made,
        "images_skipped_existing": skipped,
        "session_seconds": round(elapsed, 2),
        "seconds_per_image": round(elapsed / made, 3) if made else None,
        "complete": complete,
        "device": (torch.cuda.get_device_name(0)
                   if torch.cuda.is_available() else "cpu"),
    }
    if manifest_path:
        p = Path(manifest_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Accumulate sessions so total sampling cost can be reported (D2).
        prior = json.loads(p.read_text()) if p.exists() else {"sessions": []}
        prior["sessions"].append(record)
        prior["total_seconds"] = round(
            sum(s["session_seconds"] for s in prior["sessions"]), 2)
        prior["total_hours"] = round(prior["total_seconds"] / 3600, 3)
        prior["latest"] = record
        p.write_text(json.dumps(prior, indent=2))
    print(f"[sample] {json.dumps(record, indent=2)}")
    return record


def verify_synthetic_set(out_dir, classes, n_per_class, ext="png"):
    """Confirm every class reached its target before the set is used."""
    out_dir = Path(out_dir)
    missing = {}
    for cls in classes:
        have = len(list((out_dir / cls).glob(f"*.{ext}")))
        if have < n_per_class:
            missing[cls] = {"have": have, "want": n_per_class}
    if missing:
        raise AssertionError(
            f"Synthetic set incomplete in {out_dir}: {missing}. "
            f"Rerun resumable_generate to finish before training on it - an "
            f"incomplete set would change the synthetic-to-real ratio and "
            f"silently alter the experiment."
        )
    print(f"[sample] {out_dir}: all {len(classes)} classes at "
          f"{n_per_class} images.")
    return True
