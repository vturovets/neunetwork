# neunet/eval.py
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.serialization as torch_serial
from pickle import UnpicklingError
from torch import nn
from torch.utils.data import DataLoader

from .config import resolve_config
from .models import build_mlp_from_cfg
from .utils import pick_device, ensure_dir

# --- accept multiple data APIs (any of these may exist) ---
try:
    from .data import get_dataloaders as _get_dataloaders  # type: ignore
except Exception:
    _get_dataloaders = None  # type: ignore

try:
    from .data import build_dataloaders as _build_dataloaders  # type: ignore
except Exception:
    _build_dataloaders = None  # type: ignore

try:
    from .data import get_test_loader as _get_test_loader  # type: ignore
except Exception:
    _get_test_loader = None  # type: ignore


# ---------------------------------------------------------------------------
# Checkpoint loading that works across PyTorch versions (2.6+ included)
# ---------------------------------------------------------------------------

def _load_checkpoint_any(ckpt_path: Path, device: torch.device):
    """
    Load checkpoints saved as:
      - OrderedDict state_dict
      - {"model_state": state_dict} or {"state_dict": state_dict}
      - whole nn.Module via torch.save(model, ...)
    Works with PyTorch 2.6+ (where torch.load defaults to weights_only=True).
    """
    # 1) Try safe (default) load first
    try:
        return torch.load(ckpt_path, map_location=device)
    except (UnpicklingError, RuntimeError, AttributeError):
        pass

    # 2) Allowlist TorchVersion for older checkpoints and retry safely
    try:
        # This adds a single harmless global type that older checkpoints may include
        torch_serial.add_safe_globals([torch.torch_version.TorchVersion])
        return torch.load(ckpt_path, map_location=device)
    except Exception:
        pass

    # 3) Final fallback: allow full pickle (only use on trusted files)
    return torch.load(ckpt_path, map_location=device, weights_only=False)


def _extract_state_dict(blob: Any) -> OrderedDict:
    if isinstance(blob, OrderedDict):
        return blob
    if isinstance(blob, nn.Module):
        return blob.state_dict()
    if isinstance(blob, dict):
        if "model_state" in blob and isinstance(blob["model_state"], (dict, OrderedDict)):
            return blob["model_state"]
        if "state_dict" in blob and isinstance(blob["state_dict"], (dict, OrderedDict)):
            return blob["state_dict"]
        if "model" in blob and isinstance(blob["model"], nn.Module):
            return blob["model"].state_dict()
    raise TypeError("Unsupported checkpoint format; expected a state_dict or a model container.")


# ---------------------------------------------------------------------------
# Data loader helpers
# ---------------------------------------------------------------------------

def _extract_test_loader(cfg: Dict) -> DataLoader:
    """Return a test DataLoader using whichever factory is available."""
    # 1) Dedicated helper, if provided
    if _get_test_loader is not None:
        tl = _get_test_loader(cfg)  # type: ignore
        if tl is not None:
            return tl

    # 2) Generic factory returning (train, val, test) OR dict
    factory = _get_dataloaders or _build_dataloaders
    if factory is not None:
        out = factory(cfg)  # type: ignore
        if isinstance(out, (list, tuple)):
            if len(out) >= 3:   # train, val, test
                return out[2]
            if len(out) == 2:   # train, test
                return out[1]
        if isinstance(out, dict):
            for k in ("test", "test_loader"):
                if k in out and out[k] is not None:
                    return out[k]

    # 3) Fallback: build MNIST test loader directly (keeps behavior consistent)
    from torchvision import datasets, transforms  # local import to keep optional
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
    test_set = datasets.MNIST(root, train=False, download=True, transform=t)
    return DataLoader(test_set, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=False)


# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device
              ) -> Tuple[float, float, List[int], List[int]]:
    """Return (test_loss, test_acc, y_true, y_pred)."""
    crit = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    total = 0
    correct = 0
    y_true: List[int] = []
    y_pred: List[int] = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)  # model flattens internally
        loss = crit(logits, yb)

        bs = yb.size(0)
        total_loss += float(loss.item()) * bs
        total += bs

        preds = logits.argmax(dim=-1)
        correct += int((preds == yb).sum().item())

        y_true.extend([int(t) for t in yb.detach().cpu().tolist()])
        y_pred.extend([int(p) for p in preds.detach().cpu().tolist()])

    test_loss = total_loss / max(total, 1)
    test_acc = correct / max(total, 1)
    return test_loss, test_acc, y_true, y_pred


def _save_confusion_matrix(y_true: List[int], y_pred: List[int], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "confusion_matrix.png"
    try:
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt

        cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(cm, aspect="auto")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_png)
        plt.close(fig)
    except Exception:
        # Fail-safe if plotting libraries are missing
        (out_dir / "confusion_matrix.json").write_text(
            json.dumps({"y_true": y_true, "y_pred": y_pred}), encoding="utf-8"
        )
    return out_png


def _update_metrics_json(path: Path, test_loss: float, test_acc: float) -> None:
    data: Dict[str, Any] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data["test_loss"] = float(test_loss)
    data["test_acc"] = float(test_acc)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(cfg: Dict, checkpoint: str = "models/best.pt",
             metrics_out: str = "runs/metrics.json", plots_out: str = "runs/") -> Dict[str, Any]:
    """
    Evaluate on the MNIST test set.

    - Loads checkpoint robustly across PyTorch versions.
    - Computes test loss/accuracy.
    - Updates runs/metrics.json.
    - Saves runs/confusion_matrix.png.
    - Writes runs/evaluation.md (if report module is available).
    """
    cfg = resolve_config(cfg)
    device = pick_device(cfg.get("device", "auto"))

    # dataloader
    test_loader = _extract_test_loader(cfg)

    # model + checkpoint
    model = build_mlp_from_cfg(cfg).to(device)
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    blob = _load_checkpoint_any(ckpt_path, device)
    state_dict = _extract_state_dict(blob)
    model.load_state_dict(state_dict, strict=False)

    # evaluate
    test_loss, test_acc, y_true, y_pred = _evaluate(model, test_loader, device)

    # artifacts
    plots_dir = Path(plots_out)
    metrics_path = Path(metrics_out)
    ensure_dir(plots_dir)

    _update_metrics_json(metrics_path, test_loss, test_acc)
    _save_confusion_matrix(y_true, y_pred, plots_dir)

    # markdown report (optional)
    try:
        from .report import generate_eval_report
        generate_eval_report(metrics_path, plots_dir / "evaluation.md")  # metrics_json -> MD
    except Exception:
        # If report module missing or errors, skip gracefully
        pass

    return {
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "metrics_json": str(metrics_path),
        "plots_dir": str(plots_dir),
        "checkpoint": str(ckpt_path),
    }
