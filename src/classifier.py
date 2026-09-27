import torch.nn as nn
from torchvision import models


def build_model(num_classes=7, arch="resnet50", pretrained=True):
    """ResNet-50 with ImageNet weights, final layer replaced for 7 classes."""
    weights = "IMAGENET1K_V2" if pretrained else None
    m = getattr(models, arch)(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m
import copy
import json
from pathlib import Path

import torch


def train_classifier(model, train_loader, val_loader, epochs=30, lr=1e-4,
                     weight_decay=1e-4, patience=7, device="cuda",
                     tag="baseline", log_dir="logs"):
    """Fine-tune, selecting the best epoch on validation macro-F1."""
    from src.metrics import collect_predictions, compute_metrics

    model.to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")

    best_f1, best_state, bad_epochs, history = -1.0, None, 0, []

    for ep in range(epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * x.size(0)
        sched.step()

        y_true, y_pred, _ = collect_predictions(model, val_loader, device)
        val = compute_metrics(y_true, y_pred)
        history.append({
            "epoch": ep,
            "train_loss": running / len(train_loader.dataset),
            "val_macro_f1": val["macro_f1"],
            "val_balanced_acc": val["balanced_accuracy"],
            "val_accuracy": val["accuracy"],
        })
        print(f"ep {ep:02d}  loss {history[-1]['train_loss']:.4f}  "
              f"val macroF1 {val['macro_f1']:.4f}  "
              f"val acc {val['accuracy']:.4f}")

        # SELECT ON MACRO-F1, NEVER ACCURACY.
        # Accuracy would pick the checkpoint most confident about NV.
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {ep} (best macro-F1 {best_f1:.4f})")
                break

    model.load_state_dict(best_state)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{log_dir}/{tag}_history.json").write_text(json.dumps(history, indent=2))
    return model, history