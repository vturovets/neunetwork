# neunet/info.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import json
import torch

from .config import load_config, resolve_config
from .models import build_mlp_from_cfg, parameter_count
from .utils import pick_device

MD_TPL = """# Model Info

**Device:** {device}

## Architecture
- Input size: **{input_size}**
- Hidden layers: **{layers}**
- Activations: **{activations}**
- Dropout: **{dropout}**
- Output size: **{output_size}**
- Parameters (total): **{params:,}**

## Training setup
- Epochs: **{epochs}**
- Batch size: **{batch_size}**
- Learning rate (Adam): **{lr}**
- Weight decay (L2): **{weight_decay}**

{ckpt_block}
"""

def _ckpt_block(ckpt_path: str | None) -> str:
    if not ckpt_path:
        return "## Checkpoint\n_Not provided._"
    p = Path(ckpt_path)
    if not p.exists():
        return f"## Checkpoint\nPath: `{ckpt_path}`\n_Not found._"
    try:
        blob = torch.load(p, map_location="cpu")
        epoch = blob.get("epoch")
        val_loss = blob.get("val_loss")
        fw = blob.get("framework", {})
        cfg_snapshot = blob.get("config")
        return (
            "## Checkpoint\n"
            f"Path: `{ckpt_path}`\n\n"
            f"- Epoch: **{epoch}**\n"
            f"- Val loss: **{val_loss}**\n"
            f"- Framework: **{fw}**\n"
            f"- Saved with config keys: {list(cfg_snapshot.keys()) if isinstance(cfg_snapshot, dict) else 'n/a'}\n"
        )
    except Exception as e:
        return f"## Checkpoint\nPath: `{ckpt_path}`\n_Error reading checkpoint: {e}_"

def model_info(config_path: str = "configs/default.yaml",
               checkpoint: str | None = "models/best.pt",
               out_md: str | None = None) -> Dict[str, Any]:
    # config
    cfg = resolve_config(load_config(config_path))
    m = cfg["model"]
    t = cfg["train"]
    device = str(pick_device(cfg.get("device", "auto")))

    # model + params
    model = build_mlp_from_cfg(cfg)
    params = parameter_count(model)

    info: Dict[str, Any] = {
        "device": device,
        "input_size": int(m["input_size"]),
        "layers": [int(x) for x in m["layers"]],
        "activations": [str(a) for a in m["activations"]],
        "dropout": float(m.get("dropout", 0.0)),
        "output_size": int(m["output_size"]),
        "params": int(params),
        "epochs": int(t["epochs"]),
        "batch_size": int(t["batch_size"]),
        "lr": float(t["lr"]),
        "weight_decay": float(t.get("weight_decay", 0.0)),
        "checkpoint": checkpoint,
    }

    # optional markdown
    if out_md:
        ckpt_block = _ckpt_block(checkpoint)
        Path(out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(out_md).write_text(
            MD_TPL.format(
                device=info["device"],
                input_size=info["input_size"],
                layers=info["layers"],
                activations=info["activations"],
                dropout=info["dropout"],
                output_size=info["output_size"],
                params=info["params"],
                epochs=info["epochs"],
                batch_size=info["batch_size"],
                lr=info["lr"],
                weight_decay=info["weight_decay"],
                ckpt_block=ckpt_block,
            ),
            encoding="utf-8",
        )

    return info
