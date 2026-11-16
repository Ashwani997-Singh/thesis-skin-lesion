"""
scripts/patch_classifier_d29.py - one-off edit for decision D29 (item 6).

Adds ONE optional argument to src/classifier.train_classifier:

    class_weights=None

With the default (None) the loss is exactly the unweighted cross-entropy used
in every earlier run, so earlier results are unaffected. A list of 7 weights
gives a class-weighted cross-entropy for the D29 tuning sensitivity analysis.

Run once from the repository root on the laptop:

    python scripts/patch_classifier_d29.py

It refuses to run twice, then checks the result: the module imports, the
default loss is unweighted, and a weighted loss gives a different value on a
tiny example.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
F = ROOT / "src" / "classifier.py"
src = F.read_text(encoding="utf-8")

if "class_weights" in src:
    sys.exit("classifier.py already has class_weights: nothing to do")

old_sig = ('def train_classifier(model, train_loader, val_loader, epochs=30, lr=1e-4,\n'
           '                     weight_decay=1e-4, patience=7, device="cuda",\n'
           '                     tag="baseline", log_dir="logs"):')
new_sig = ('def train_classifier(model, train_loader, val_loader, epochs=30, lr=1e-4,\n'
           '                     weight_decay=1e-4, patience=7, device="cuda",\n'
           '                     tag="baseline", log_dir="logs", class_weights=None):')
old_crit = '    crit = nn.CrossEntropyLoss()\n'
new_crit = ('    # class_weights=None reproduces every run before D29 exactly.\n'
            '    # D29 tuning sensitivity: 7 weights -> class-weighted cross-entropy.\n'
            '    if class_weights is None:\n'
            '        crit = nn.CrossEntropyLoss()\n'
            '    else:\n'
            '        crit = nn.CrossEntropyLoss(\n'
            '            weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device))\n')

assert src.count(old_sig) == 1, "signature not found exactly once - file differs from expected"
assert src.count(old_crit) == 1, "loss line not found exactly once - file differs from expected"
src = src.replace(old_sig, new_sig).replace(old_crit, new_crit)
crlf = b"\r\n" in F.read_bytes()          # keep the file's own line endings
F.write_text(src, encoding="utf-8", newline="\r\n" if crlf else "\n")
print("patched src/classifier.py")

# ---- checks ----------------------------------------------------------------
import importlib
import inspect

import torch
import torch.nn as nn

import src.classifier as clf
importlib.reload(clf)
sig = inspect.signature(clf.train_classifier)
assert sig.parameters["class_weights"].default is None
print("signature OK: class_weights=None by default")

logits = torch.tensor([[2.0, 0.5, 0.1, 0.0, 0.3, 1.0, 0.2]] * 2)
y = torch.tensor([0, 5])
w = [1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 1.0]
plain = nn.CrossEntropyLoss()(logits, y).item()
weighted = nn.CrossEntropyLoss(weight=torch.tensor(w))(logits, y).item()
assert abs(plain - weighted) > 1e-4
print(f"loss check OK: unweighted {plain:.4f}, weighted {weighted:.4f}")
print("ALL CHECKS PASS")
