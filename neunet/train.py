"""Training loop implementation."""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import os, json, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from . import utils
from .data import get_dataloaders
from .models import build_mlp

def _seed_everything(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _to_device(batch, device: torch.device):
    x, y = batch
    # Flatten 1x28x28 -> 784 for MLP
    x = x.view(x.size(0), -1).to(device)
    y = y.to(device)
    return x, y

def train(cfg: Dict[str, Any]) -> None:
    """Train the model on MNIST, save checkpoints/metrics/plots.
    
    - Loss: CrossEntropyLoss
    - Optimizer: Adam(lr)
    - Track train_loss (always) and val_loss/val_acc (if val split > 0)
    - Save checkpoints: models/last.pt (each epoch), models/best.pt (best metric)
    - Write runs/metrics.json and runs/loss_curve.png
    """
    seed = int(cfg.get("seed", 42))
    _seed_everything(seed)

    device_str = utils.pick_device(cfg.get("device", "auto"))
    device = torch.device(device_str)

    # Data
    train_loader, val_loader, _ = get_dataloaders(cfg)

    # Model/criterion/optim
    model = build_mlp(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]))

    epochs = int(cfg["train"]["epochs"])
    models_dir = cfg["artifacts"]["models_dir"]
    runs_dir = cfg["artifacts"]["runs_dir"]
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    history: List[dict] = []
    best_metric = float("inf")
    best_epoch = -1
    best_key = "val_loss" if val_loader is not None else "train_loss"

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)
        for batch in pbar:
            x, y = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            n += x.size(0)
            pbar.set_postfix(loss=loss.item())
        train_loss = running_loss / max(1, n)

        log = {"epoch": epoch, "train_loss": train_loss}

        # Validation (optional)
        if val_loader is not None:
            model.eval()
            val_running = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    x, y = _to_device(batch, device)
                    logits = model(x)
                    loss = criterion(logits, y)
                    val_running += loss.item() * x.size(0)
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == y).sum().item()
                    total += x.size(0)
            val_loss = val_running / max(1, total)
            val_acc = correct / max(1, total)
            log.update({"val_loss": val_loss, "val_acc": val_acc})

            metric = val_loss
        else:
            metric = train_loss

        # Save checkpoints
        last_path = os.path.join(models_dir, "last.pt")
        torch.save(model.state_dict(), last_path)

        if metric < best_metric - 1e-12:
            best_metric = metric
            best_epoch = epoch
            best_path = os.path.join(models_dir, "best.pt")
            torch.save(model.state_dict(), best_path)
            # Optional metadata for `neunet info`
            meta = {
                "saved_at": int(time.time()),
                "epoch": epoch,
                "best_key": best_key,
                "best_value": best_metric,
                "device": str(device),
                "model": "MLP",
                "layers": cfg["model"]["layers"],
                "activations": cfg["model"]["activations"],
            }
            with open(best_path + ".meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

        history.append(log)

        # Write metrics after each epoch
        with open(os.path.join(runs_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump({
                "history": history,
                "best": {"epoch": best_epoch, "key": best_key, "value": best_metric},
            }, f, indent=2)

    # Plot loss curves
    try:
        epochs_list = [h["epoch"] for h in history]
        tr = [h["train_loss"] for h in history]
        plt.figure()
        plt.plot(epochs_list, tr, label="train_loss")
        if any("val_loss" in h for h in history):
            vl = [h.get("val_loss") for h in history]
            plt.plot(epochs_list, vl, label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Loss curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(runs_dir, "loss_curve.png"))
        plt.close()
    except Exception as e:
        # Non-fatal
        with open(os.path.join(runs_dir, "plot_error.txt"), "w", encoding="utf-8") as f:
            f.write(str(e))
