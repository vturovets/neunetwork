# neunet/utils.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import os
import random

import torch

try:
    import numpy as np
except Exception:  # numpy is listed in requirements, but be defensive
    np = None  # type: ignore


# ------------ filesystem helpers ------------
def ensure_dir(path: str | os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def ensure_dirs(paths: Iterable[str | os.PathLike]) -> None:
    for p in paths:
        ensure_dir(p)

def path_exists(path: str | os.PathLike) -> bool:
    return Path(path).exists()


# ------------ device & seeding ------------
def pick_device(requested: str = "auto") -> torch.device:
    """
    Choose a torch.device according to:
      - 'auto'  : cuda > mps > cpu
      - 'cuda'  : cuda if available else cpu
      - 'mps'   : mps  if available else cpu
      - 'cpu'   : cpu
    """
    r = (requested or "auto").lower()

    if r == "cpu":
        return torch.device("cpu")

    if r == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if r == "mps":
        mps_ok = getattr(torch.backends, "mps", None)
        if mps_ok is not None and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return torch.device("mps")
        return torch.device("cpu")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_ok = getattr(torch.backends, "mps", None)
    if mps_ok is not None and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if np is not None:
        np.random.seed(seed)  # type: ignore[call-arg]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic-ish; ok for classroom use
    try:
        torch.use_deterministic_algorithms(False)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass


# ------------ backward-compat aliases ------------
# If other modules expect these names, keep them pointing to the same functions.
get_device = pick_device
choose_device = pick_device
set_seed = seed_all

__all__ = [
    "ensure_dir", "ensure_dirs", "path_exists",
    "pick_device", "get_device", "choose_device",
    "seed_all", "set_seed",
]
