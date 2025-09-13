"""Evaluation loop."""
from __future__ import annotations
from typing import Dict, Any
import os, json, time
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from . import utils
from .data import get_dataloaders
from .models import build_mlp

def _to_device(batch, device: torch.device):
    x, y = batch
    x = x.view(x.size(0), -1).to(device)
    y = y.to(device)
    return x, y

def evaluate(cfg: Dict[str, Any], checkpoint: str, metrics_out: str, plots_out: str) -> None:
    os.makedirs(plots_out, exist_ok=True)
    device_str = utils.pick_device(cfg.get("device", "auto"))
    device = torch.device(device_str)

    # Data (only need test loader)
    _, _, test_loader = get_dataloaders(cfg)

    # Model
    model = build_mlp(cfg).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    running = 0.0
    total = 0
    correct = 0
    all_true = []
    all_pred = []
    with torch.no_grad():
        for batch in test_loader:
            x, y = _to_device(batch, device)
            logits = model(x)
            loss = criterion(logits, y)
            running += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            all_true.extend(y.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())

    test_loss = running / max(1, total)
    test_acc = correct / max(1, total)

    # Confusion matrix plot
    cm = confusion_matrix(all_true, all_pred, labels=list(range(10)))
    try:
        plt.figure()
        plt.imshow(cm, interpolation='nearest')
        plt.title('Confusion Matrix (counts)')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        cm_path = os.path.join(plots_out, 'confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()
    except Exception as e:
        cm_path = None
        with open(os.path.join(plots_out, 'plot_error.txt'), 'w', encoding='utf-8') as f:
            f.write(str(e))

    # Metrics JSON
    metrics = {
        "timestamp": int(time.time()),
        "checkpoint": checkpoint,
        "device": device_str,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "confusion_matrix_png": cm_path,
    }
    with open(metrics_out, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    # Markdown report
    md_path = os.path.join(os.path.dirname(metrics_out), 'evaluation.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Evaluation Report\n\n")
        f.write(f"- Checkpoint: `{checkpoint}`\n")
        f.write(f"- Device: `{device_str}`\n")
        f.write(f"- Test loss: **{test_loss:.4f}**\n")
        f.write(f"- Test accuracy: **{test_acc:.4f}**\n\n")
        if cm_path:
            f.write(f"![Confusion Matrix]({os.path.relpath(cm_path, os.path.dirname(md_path))})\n")
