# neunet/config.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import yaml

# ---------- Defaults ----------
def default_config() -> Dict[str, Any]:
    return {
        "seed": 42,
        "device": "auto",
        "data": {
            "dataset": "MNIST",
            "root": "./data",
            "download": True,
            "num_workers": 2,
            "val_split": 0.0,
            "normalization": {"mean": 0.1307, "std": 0.3081},
        },
        "model": {
            "input_size": 784,
            "layers": [128, 64],
            "activations": ["relu", "relu"],
            "output_size": 10,
            "dropout": 0.0,
        },
        "train": {
            "epochs": 5,
            "batch_size": 64,
            "lr": 0.001,
            "weight_decay": 0.0,   # L2
            # (optimizer intentionally not present: Adam-only)
        },
        "artifacts": {
            "models_dir": "./models",
            "runs_dir": "./runs",
        },
    }

# ---------- IO ----------
def load_config(path: str | Path) -> Dict[str, Any]:
    """Read YAML config from disk. If the file does not exist, raise FileNotFoundError."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # allow very old configs to miss keys; merge shallowly with defaults
    cfg = default_config()
    for k, v in (data or {}).items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg

def save_config(cfg: Dict[str, Any], path: str | Path) -> None:
    """Write YAML config to disk (pretty)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

# ---------- Validation / Normalization ----------
def resolve_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize config in-place; return cfg."""
    # ensure required top-level sections
    for sect in ("data", "model", "train", "artifacts"):
        cfg.setdefault(sect, {})

    # hard-lock to Adam: drop legacy field if present
    cfg["train"].pop("optimizer", None)

    # model basics
    m = cfg["model"]
    m.setdefault("input_size", 784)
    m.setdefault("output_size", 10)
    if "layers" not in m or not m["layers"]:
        m["layers"] = [128, 64]
    if "activations" not in m or not m["activations"]:
        m["activations"] = ["relu"] * len(m["layers"])

    # lengths match
    if len(m["layers"]) != len(m["activations"]):
        raise ValueError("model.layers and model.activations must have the same length")

    # types / ranges
    m["layers"] = [int(x) for x in m["layers"]]
    m["activations"] = [str(a).lower() for a in m["activations"]]
    m["dropout"] = float(m.get("dropout", 0.0))
    if not (0.0 <= m["dropout"] < 1.0):
        raise ValueError("model.dropout must be in [0.0, 1.0)")

    t = cfg["train"]
    t["epochs"] = int(t.get("epochs", 5))
    t["batch_size"] = int(t.get("batch_size", 64))
    t["lr"] = float(t.get("lr", 1e-3))
    t["weight_decay"] = float(t.get("weight_decay", 0.0))
    if t["weight_decay"] < 0.0:
        raise ValueError("train.weight_decay must be ≥ 0.0")

    # data/artifacts defaults
    d = cfg["data"]
    d.setdefault("dataset", "MNIST")
    d.setdefault("root", "./data")
    d.setdefault("download", True)
    d.setdefault("num_workers", 2)
    d.setdefault("val_split", 0.0)
    d.setdefault("normalization", {"mean": 0.1307, "std": 0.3081})

    a = cfg["artifacts"]
    a.setdefault("models_dir", "./models")
    a.setdefault("runs_dir", "./runs")

    # seed & device
    cfg["seed"] = int(cfg.get("seed", 42))
    cfg["device"] = cfg.get("device", "auto")

    return cfg

# --- Backward-compat aliases (used by some branches/CLI code) ---
def load_yaml(path):
    return load_config(path)

def save_yaml(cfg, path):
    return save_config(cfg, path)

def resolve_yaml(cfg):
    return resolve_config(cfg)