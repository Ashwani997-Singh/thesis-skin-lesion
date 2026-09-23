"""
src/metrics.py - Equations 7.6 to 7.10, plus macro AUROC and bootstrap CIs.

Adds macro one-vs-rest AUROC as a SUPPLEMENTARY metric. AUROC appears in the
proposal's List of Abbreviations but nowhere in the body; computing it turns a
consistency gap into an extra reported measure. It never drives model
selection - that remains validation macro-F1 - and it is never a headline.

Bootstrap intervals and seed spread measure different things and are both
reported (D12): seeds capture training variance, the bootstrap captures the
small-test-set variance that dominates DF and VASC.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)

from src.data import CLASSES


@torch.no_grad()
def collect_predictions(model, loader, device="cuda"):
    """Return (y_true, y_pred, probs). Probabilities are needed for AUROC."""
    model.eval()
    ys, ps, probs = [], [], []
    for x, y in loader:
        logits = model(x.to(device))
        p = F.softmax(logits.float(), dim=1)
        probs.append(p.cpu().numpy())
        ps.append(logits.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    return (np.concatenate(ys), np.concatenate(ps), np.concatenate(probs))


def macro_auroc(y_true, probs, n_classes=len(CLASSES)):
    """Macro one-vs-rest AUROC.

    Guarded: roc_auc_score raises when a class is absent from y_true. That
    cannot happen on the full test set but will happen on bootstrap resamples
    of the rarest classes, so the guard is load-bearing, not defensive noise.
    """
    try:
        return float(roc_auc_score(
            y_true, probs, multi_class="ovr", average="macro",
            labels=list(range(n_classes)),
        ))
    except ValueError:
        return float("nan")


def compute_metrics(y_true, y_pred, probs=None):
    """All reported measures from a set of predictions."""
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(CLASSES)), zero_division=0
    )
    out = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float((y_true == y_pred).mean()),   # reported, never optimised
        "melanoma_recall": float(rec[CLASSES.index("mel")]),
        "per_class": {
            c: {"precision": float(prec[i]), "recall": float(rec[i]),
                "f1": float(f1[i]), "support": int(support[i])}
            for i, c in enumerate(CLASSES)
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=range(len(CLASSES))
        ).tolist(),
    }
    if probs is not None:
        out["macro_auroc"] = macro_auroc(y_true, probs)   # supplementary
    return out


def bootstrap_ci(y_true, y_pred, probs=None, n_iter=1000, ci=0.95, seed=42,
                 metrics=("macro_f1", "melanoma_recall", "balanced_accuracy")):
    """Percentile bootstrap over the TEST SET.

    Resamples test images with replacement. This is what quantifies the
    instability of DF and VASC recall, where roughly 17 test images mean a
    single reclassification moves recall by about six points.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    draws = {m: [] for m in metrics}

    for _ in range(n_iter):
        idx = rng.integers(0, n, n)
        m = compute_metrics(
            y_true[idx], y_pred[idx],
            probs[idx] if probs is not None else None,
        )
        for name in metrics:
            draws[name].append(m.get(name, float("nan")))

    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    out = {}
    for name, vals in draws.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            out[name] = {"lo": float("nan"), "hi": float("nan"), "n": 0}
            continue
        out[name] = {
            "lo": float(np.percentile(arr, lo_q)),
            "hi": float(np.percentile(arr, hi_q)),
            "n": int(arr.size),
        }
    return out


def per_class_bootstrap(y_true, y_pred, n_iter=1000, ci=0.95, seed=42):
    """Recall CI for every class. Drives the per-class recall figure."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    draws = {c: [] for c in CLASSES}

    for _ in range(n_iter):
        idx = rng.integers(0, n, n)
        _, rec, _, _ = precision_recall_fscore_support(
            y_true[idx], y_pred[idx], labels=range(len(CLASSES)),
            zero_division=0,
        )
        for i, c in enumerate(CLASSES):
            draws[c].append(rec[i])

    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return {c: {"lo": float(np.percentile(v, lo_q)),
                "hi": float(np.percentile(v, hi_q))}
            for c, v in draws.items()}


def evaluate(model, loader, device="cuda", bootstrap=True, n_iter=1000,
             ci=0.95, seed=42):
    """Full evaluation on the held-out partition."""
    y_true, y_pred, probs = collect_predictions(model, loader, device)
    out = compute_metrics(y_true, y_pred, probs)
    out["report"] = classification_report(
        y_true, y_pred, labels=range(len(CLASSES)), target_names=CLASSES,
        zero_division=0, digits=3,
    )
    if bootstrap:
        out["bootstrap"] = bootstrap_ci(y_true, y_pred, probs,
                                        n_iter=n_iter, ci=ci, seed=seed)
        out["bootstrap_per_class_recall"] = per_class_bootstrap(
            y_true, y_pred, n_iter=n_iter, ci=ci, seed=seed
        )
    return out


def save_metrics(m, name, root="results"):
    path = Path(root) / "metrics"
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{name}.json").write_text(json.dumps(m, indent=2))
    print(m["report"])
    print(f"macro-F1 {m['macro_f1']:.4f} | "
          f"balanced acc {m['balanced_accuracy']:.4f} | "
          f"MEL recall {m['melanoma_recall']:.4f} | "
          f"accuracy {m['accuracy']:.4f}"
          + (f" | macro-AUROC {m['macro_auroc']:.4f}"
             if "macro_auroc" in m else ""))
    return path / f"{name}.json"
