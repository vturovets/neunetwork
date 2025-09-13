# neunet/train.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- dataloaders: accept either export name ---
try:
    from .data import build_dataloaders as _dl_factory
except Exception:
    try:
        from .data import get_dataloaders as _dl_factory  # type: ignore
    except Exception:
        _dl_factory = None  # type: ignore

# --- utils imports with fallbacks (for cross-branch compatibility) ---
try:
    from .utils import pick_device
except Exception:
    try:
        from .utils import get_device as pick_device  # type: ignore
    except Exception:
        from .utils import choose_device as pick_device  # type: ignore

try:
    from .utils import seed_all
except Exception:
    from .utils import set_seed as seed_all  # type: ignore

try:
    from .utils import ensure_dir
except Exception:
    from .utils import ensure_dirs as ensure_dir  # type: ignore

from .models import build_mlp_from_cfg

# --- optional training-report integration ---
try:
    from .report_train import generate_train_log_report as _gen_train_report
except Exception:
    _gen_train_report = None  # type: ignore


# ======================================================================
# Helpers
# ======================================================================

def _get_train_val_loaders(cfg: Dict) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Returns (train_loader, val_loader or None) using whichever data API exists.

    Expected factories:
      - build_dataloaders(cfg) -> (train, val[, test]) | (train, test) | dict
      - get_dataloaders(cfg)   -> same as above

    If no factory is available, we build MNIST train (and optional val) directly.
    """
    if _dl_factory is not None:
        out = _dl_factory(cfg)  # type: ignore
        if isinstance(out, (tuple, list)):
            if len(out) >= 2:
                # common shapes: (train, val, test) OR (train, test)
                train_loader = out[0]
                second = out[1]
                # Heuristic: if val_split==0 -> second is probably test → return None as val
                val_loader = second if float(cfg["data"].get("val_split", 0.0)) > 0.0 else None
                # If you have an explicit val regardless of val_split, keep it:
                if len(out) >= 3:  # (train, val, test)
                    val_loader = out[1]
                return train_loader, val_loader
            elif len(out) == 1:
                return out[0], None
        if isinstance(out, dict):
            train_loader = out.get("train") or out.get("train_loader")
            val_loader = (out.get("val") or out.get("valid") or
                          out.get("validation") or out.get("val_loader"))
            if train_loader is None:
                raise ValueError("Dataloader dict missing 'train'/'train_loader'.")
            return train_loader, val_loader  # val may be None

    # Fallback: build MNIST loaders directly (keeps behavior consistent)
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets, transforms

    norm = cfg["data"].get("normalization", {"mean": 0.1307, "std": 0.3081})
    t = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((norm["mean"],), (norm["std"],)),
    ])
    root = cfg["data"]["root"]
    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["data"]["num_workers"])
    val_split = float(cfg["data"].get("val_split", 0.0))

    full_train = datasets.MNIST(root, train=True, download=True, transform=t)
    if val_split and val_split > 0.0:
        n_total = len(full_train)
        n_val = int(round(n_total * val_split))
        n_train = n_total - n_val
        train_set, val_set = random_split(full_train, [n_train, n_val])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=False)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=False)
    else:
        train_loader = DataLoader(full_train, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=False)
        val_loader = None

    return train_loader, val_loader


def _write_metrics_json(path: Path, history: Dict[str, List[float]]) -> None:
    """
    Merge train/val loss into runs/metrics.json without clobbering
    test metrics that may have been written by eval.
    """
    existing: Dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    merged = dict(existing)
    merged["train_loss"] = list(history.get("train_loss", []))
    if history.get("val_loss"):
        merged["val_loss"] = list(history["val_loss"])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2), encoding="utf-8")


def _plot_losses(runs_dir: Path, history: Dict[str, List[float]]) -> None:
    """Plot loss curves using Matplotlib defaults (single plot)."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])

    if not train_loss:
        return

    fig = plt.figure()
    xs = list(range(1, len(train_loss) + 1))
    plt.plot(xs, train_loss, label="train_loss")
    if val_loss:
        plt.plot(xs, val_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss Curve")
    plt.legend()
    out = runs_dir / "loss_curve.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _save_checkpoint(
        path: Path,
        *,
        model: nn.Module,
        epoch: int,
        val_loss: Optional[float],
        cfg: Dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state": model.state_dict(),
            "val_loss": float(val_loss) if val_loss is not None else None,
            "config": cfg,
            "framework": {"torch": torch.__version__},
        },
        path,
    )


def _has_validation(val_loader: Optional[DataLoader]) -> bool:
    try:
        return val_loader is not None and len(val_loader) > 0
    except TypeError:
        return val_loader is not None


# ======================================================================
# Public API
# ======================================================================

def train(cfg: Dict) -> None:
    """
    Train an MLP on MNIST using Adam (with optional L2 weight decay and dropout).

    - Logs train loss (and val loss if a split is configured).
    - Saves checkpoints: models/last.pt every epoch; models/best.pt on val improvement.
      If no validation is used, best.pt is saved at the final epoch.
    - Writes/updates runs/metrics.json with {"train_loss":[...], "val_loss":[...]}.
    - Saves runs/loss_curve.png.
    - Generates runs/train_report.md (verdict/reason/actions) when report module is present.
    """
    # Device & seeding
    device = pick_device(cfg.get("device", "auto"))
    seed_all(int(cfg.get("seed", 42)))

    # Artifacts
    models_dir = Path(cfg["artifacts"]["models_dir"])
    runs_dir = Path(cfg["artifacts"]["runs_dir"])
    ensure_dir(models_dir)
    ensure_dir(runs_dir)

    # Data
    train_loader, val_loader = _get_train_val_loaders(cfg)
    use_val = _has_validation(val_loader)

    # Model
    model = build_mlp_from_cfg(cfg).to(device)

    # Optimizer (Adam) with L2
    lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"].get("weight_decay", 0.0))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Bookkeeping
    num_epochs = int(cfg["train"]["epochs"])
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    # Train
    for epoch in range(1, num_epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)  # model flattens internally
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = yb.size(0)
            running += float(loss.item()) * bs
            n_seen += bs

            pbar.set_postfix(train_loss=f"{running / max(n_seen,1):.4f}",
                             lr=f"{lr:g}", wd=f"{weight_decay:g}")

        epoch_train = running / max(n_seen, 1)
        history["train_loss"].append(epoch_train)

        # Validation
        if use_val:
            val_loss = _evaluate_loss(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                _save_checkpoint(
                    models_dir / "best.pt",
                    model=model,
                    epoch=epoch,
                    val_loss=val_loss,
                    cfg=cfg,
                    )

        # Always save "last"
        _save_checkpoint(
            models_dir / "last.pt",
            model=model,
            epoch=epoch,
            val_loss=(history["val_loss"][-1] if (use_val and history["val_loss"]) else None),
            cfg=cfg,
            )

        # Persist metrics + plot every epoch
        _write_metrics_json(runs_dir / "metrics.json", history)
        _plot_losses(runs_dir, history)

    # If no validation split, mark final as "best"
    if not use_val:
        _save_checkpoint(
            models_dir / "best.pt",
            model=model,
            epoch=num_epochs,
            val_loss=None,
            cfg=cfg,
            )

    # Optional: build/refresh the training report
    # Optional: build/refresh the training log
    try:
        if _gen_train_report is not None:
            _gen_train_report(
                metrics_path=str(runs_dir / "metrics.json"),
                eval_path=None,
                out_path=str(runs_dir / "train_log.md"),   # ← renamed filename
            )
    except Exception:
        pass


@torch.inference_mode()
def _evaluate_loss(
        model: nn.Module,
        loader: Optional[DataLoader],
        criterion: nn.Module,
        device: torch.device,
) -> float:
    if loader is None:
        return float("nan")
    model.eval()
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        bs = yb.size(0)
        total += float(loss.item()) * bs
        n += bs
    return total / max(n, 1)
