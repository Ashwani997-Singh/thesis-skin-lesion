"""
patch_metrics_d23.py - one-off edit of src/metrics.py for decision D23.

Adds, in response to the Second Marker's methodology feedback (S8, S12):
  1. Per-image predictions (true label, predicted label, class probabilities)
     saved next to every metrics JSON as <name>_predictions.npz. Required for
     paired comparisons between conditions on the same test images, and for
     AUPRC. Numbers only - no image data.
  2. Supplementary metrics: macro precision, per-class specificity, and
     macro one-vs-rest AUPRC (average precision).

Nothing that is already computed changes: macro-F1, recall, balanced
accuracy, AUROC, confusion matrix and bootstrap intervals are untouched, and
model selection is still on validation macro-F1.

Each replacement is asserted to match exactly once, so the script refuses to
run on a file that differs from the expected version. Run once, from D:\\thesis:
    python patch_metrics_d23.py
"""

from pathlib import Path

P = Path(r"D:\thesis\src\metrics.py")
src = P.read_text(encoding="utf-8")

EDITS = [
    # 1. import average_precision_score
    ("from sklearn.metrics import (balanced_accuracy_score, classification_report,\n"
     "                             confusion_matrix, f1_score,\n"
     "                             precision_recall_fscore_support, roc_auc_score)",
     "from sklearn.metrics import (average_precision_score,\n"
     "                             balanced_accuracy_score, classification_report,\n"
     "                             confusion_matrix, f1_score,\n"
     "                             precision_recall_fscore_support, roc_auc_score)"),

    # 2. macro AUPRC helper, placed after macro_auroc
    ("def compute_metrics(y_true, y_pred, probs=None):",
     'def macro_auprc(y_true, probs, n_classes=len(CLASSES)):\n'
     '    """Macro one-vs-rest average precision (AUPRC). Supplementary (D23).\n'
     '\n'
     '    Under heavy imbalance AUROC is flattered by the many true negatives;\n'
     '    precision-recall summarises minority-class behaviour more directly.\n'
     '    Classes absent from y_true (bootstrap resamples) are skipped.\n'
     '    """\n'
     '    aps = []\n'
     '    for c in range(n_classes):\n'
     '        pos = (y_true == c)\n'
     '        if pos.any():\n'
     '            aps.append(average_precision_score(pos.astype(int), probs[:, c]))\n'
     '    return float(np.mean(aps)) if aps else float("nan")\n'
     '\n'
     '\n'
     'def compute_metrics(y_true, y_pred, probs=None):'),

    # 3. macro precision and per-class specificity
    ('    out = {\n'
     '        "macro_f1": float(f1_score(y_true, y_pred, average="macro",\n'
     '                                   zero_division=0)),',
     '    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))\n'
     '    tn = cm.sum() - cm.sum(axis=0) - cm.sum(axis=1) + np.diag(cm)\n'
     '    fp = cm.sum(axis=0) - np.diag(cm)\n'
     '    spec = np.divide(tn, tn + fp, out=np.zeros(len(CLASSES)),\n'
     '                     where=(tn + fp) > 0)\n'
     '    out = {\n'
     '        "macro_f1": float(f1_score(y_true, y_pred, average="macro",\n'
     '                                   zero_division=0)),\n'
     '        "macro_precision": float(np.mean(prec)),          # supplementary (D23)'),

    ('            c: {"precision": float(prec[i]), "recall": float(rec[i]),\n'
     '                "f1": float(f1[i]), "support": int(support[i])}',
     '            c: {"precision": float(prec[i]), "recall": float(rec[i]),\n'
     '                "specificity": float(spec[i]),            # supplementary (D23)\n'
     '                "f1": float(f1[i]), "support": int(support[i])}'),

    ('        out["macro_auroc"] = macro_auroc(y_true, probs)   # supplementary',
     '        out["macro_auroc"] = macro_auroc(y_true, probs)   # supplementary\n'
     '        out["macro_auprc"] = macro_auprc(y_true, probs)   # supplementary (D23)'),

    # 4. keep the per-image predictions from evaluate()
    ('    out = compute_metrics(y_true, y_pred, probs)\n'
     '    out["report"] = classification_report(',
     '    out = compute_metrics(y_true, y_pred, probs)\n'
     '    # Per-image predictions for paired tests and AUPRC (D23). Removed from\n'
     '    # the JSON by save_metrics and written to <name>_predictions.npz.\n'
     '    out["_predictions"] = {"y_true": y_true, "y_pred": y_pred, "probs": probs}\n'
     '    out["report"] = classification_report('),

    # 5. save them next to the JSON
    ('    path = Path(root) / "metrics"\n'
     '    path.mkdir(parents=True, exist_ok=True)\n'
     '    (path / f"{name}.json").write_text(json.dumps(m, indent=2))',
     '    path = Path(root) / "metrics"\n'
     '    path.mkdir(parents=True, exist_ok=True)\n'
     '    preds = m.get("_predictions")\n'
     '    m = {k: v for k, v in m.items() if k != "_predictions"}\n'
     '    if preds is not None:\n'
     '        np.savez_compressed(path / f"{name}_predictions.npz",\n'
     '                            y_true=preds["y_true"], y_pred=preds["y_pred"],\n'
     '                            probs=preds["probs"], classes=np.array(CLASSES))\n'
     '    (path / f"{name}.json").write_text(json.dumps(m, indent=2))'),
]

for old, new in EDITS:
    n = src.count(old)
    assert n == 1, f"expected 1 match, found {n}:\n{old[:80]}..."
    src = src.replace(old, new)

P.write_text(src, encoding="utf-8")
print(f"patched {P}: {len(EDITS)} edits applied")
