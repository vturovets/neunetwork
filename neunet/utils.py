from __future__ import annotations
import os, json
from typing import Iterable, Optional

def ensure_dirs(paths: Iterable[str]) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)

def path_exists(path: str) -> bool:
    return os.path.exists(path)

def pick_device(choice: str) -> str:
    """Return selected device string without importing torch at module import time."""
    if choice != "auto":
        return choice
    try:
        import torch  # defer import
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

def read_checkpoint_meta(path: str) -> Optional[dict]:
    """Attempt to read metadata stored alongside a checkpoint.

In the skeleton we don't save yet; return None if not present.
"""
    meta_path = f"{path}.meta.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None
