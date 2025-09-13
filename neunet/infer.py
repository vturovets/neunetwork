"""Inference pipeline."""
from __future__ import annotations
from typing import Dict, Any, List
import os, json, time
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from . import utils
from .models import build_mlp

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".ppm", ".pgm"}

def _collect_images(path: str, recursive: bool) -> List[str]:
    p = Path(path)
    files: List[Path] = []
    if p.is_file():
        files = [p]
    elif p.is_dir():
        if recursive:
            for dirpath, _, filenames in os.walk(p):
                for name in filenames:
                    if Path(name).suffix.lower() in IMG_EXTS:
                        files.append(Path(dirpath) / name)
        else:
            files = [c for c in p.iterdir() if c.is_file() and c.suffix.lower() in IMG_EXTS]
    return [str(f) for f in files]

def _preprocess(mean: float, std: float):
    return transforms.Compose([
        transforms.Grayscale(),      # ensure 1 channel
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ])

def infer(images_path: str, checkpoint: str, out_path: str, *, topk: int = 3, recursive: bool = False) -> None:
    mean = 0.1307
    std = 0.3081
    tfm = _preprocess(mean, std)

    device_str = utils.pick_device("auto")
    device = torch.device(device_str)

    model_cfg = {
        "model": {"input_size": 784, "layers": [128, 64], "activations": ["relu", "relu"], "output_size": 10, "dropout": 0.0}
    }
    # Build default MLP; real runs should use same config as training.
    from .config import default_config, deep_update
    cfg = default_config()
    deep_update(cfg, model_cfg)
    model = build_mlp(cfg).to(device)

    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    files = _collect_images(images_path, recursive=recursive)
    results = []
    for fp in files:
        try:
            img = Image.open(fp).convert("L")
            x = tfm(img).view(1, -1).to(device)
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1).cpu().view(-1)
                prob_vals, idxs = torch.topk(probs, k=min(topk, probs.numel()))
                top = [{"label": int(i.item()), "prob": float(p.item())} for p, i in zip(prob_vals, idxs)]
                pred = int(torch.argmax(probs).item())
                pred_prob = float(probs[pred].item())
            results.append({"file": fp, "topk": top, "pred": pred, "pred_prob": pred_prob})
        except Exception as e:
            results.append({"file": fp, "error": str(e)})

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
