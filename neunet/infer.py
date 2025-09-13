# neunet/infer.py
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.serialization as torch_serial
from pickle import UnpicklingError
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms

from .config import load_config, resolve_config
from .models import build_mlp_from_cfg
from .utils import pick_device, ensure_dir


# ---- checkpoint loading compatible with PyTorch 2.6+ ----
def _load_checkpoint_any(ckpt_path: Path, device: torch.device):
    # 1) try safe default
    try:
        return torch.load(ckpt_path, map_location=device)
    except (UnpicklingError, RuntimeError, AttributeError):
        pass
    # 2) allowlist TorchVersion and retry safely
    try:
        torch_serial.add_safe_globals([torch.torch_version.TorchVersion])
        return torch.load(ckpt_path, map_location=device)
    except Exception:
        pass
    # 3) final fallback (trusted files only)
    return torch.load(ckpt_path, map_location=device, weights_only=False)


def _extract_state_dict(blob: Any) -> OrderedDict:
    if isinstance(blob, OrderedDict):
        return blob
    if hasattr(blob, "state_dict"):
        return blob.state_dict()  # nn.Module
    if isinstance(blob, dict):
        if "model_state" in blob:
            return blob["model_state"]
        if "state_dict" in blob:
            return blob["state_dict"]
        if "model" in blob and hasattr(blob["model"], "state_dict"):
            return blob["model"].state_dict()
    raise TypeError("Unsupported checkpoint format; expected state_dict or model container.")


# ---- image discovery ----
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _gather_paths(path: str, recursive: bool) -> List[Path]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")
    if p.is_file():
        return [p]
    it = p.rglob("*") if recursive else p.glob("*")
    return [x for x in it if x.is_file() and x.suffix.lower() in _IMG_EXTS]


# ---- main API ----
def infer(images_path: str,
          checkpoint: str = "models/best.pt",
          out: str = "runs/infer.json",
          topk: int = 3,
          recursive: bool = False,
          config: str = "configs/default.yaml") -> List[Dict[str, Any]]:

    # config + device
    cfg = resolve_config(load_config(config))
    device = pick_device(cfg.get("device", "auto"))

    # model + checkpoint
    model = build_mlp_from_cfg(cfg).to(device)
    ckpt = Path(checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    blob = _load_checkpoint_any(ckpt, device)
    state_dict = _extract_state_dict(blob)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # transform (MNIST)
    norm = cfg["data"].get("normalization", {"mean": 0.1307, "std": 0.3081})
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((norm["mean"],), (norm["std"],)),
    ])

    # gather files
    files = _gather_paths(images_path, recursive=recursive)

    results: List[Dict[str, Any]] = []
    for fp in files:
        try:
            im = Image.open(fp).convert("L")
            x = tfm(im).unsqueeze(0).to(device)  # [1,1,28,28]
            logits = model(x)                   # model flattens internally
            probs = F.softmax(logits, dim=-1)[0]  # [10]
            p_max, pred_idx = torch.max(probs, dim=0)
            k = int(min(topk, probs.numel()))
            top_p, top_i = torch.topk(probs, k)
            results.append({
                "file": str(fp),
                "pred": int(pred_idx.item()),
                "pred_prob": float(p_max.item()),
                "topk": [{"label": int(i), "prob": float(p)} for p, i in zip(top_p.tolist(), top_i.tolist())],
            })
        except Exception as e:
            results.append({"file": str(fp), "error": str(e)})

    # write JSON
    out_path = Path(out)
    ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
