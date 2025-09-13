from __future__ import annotations
from typing import Any, Dict
import yaml
import copy

DEFAULTS: Dict[str, Any] = {
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
        "optimizer": "adam",
        "lr": 1e-3,
        "early_stopping": {
            "enabled": False,
            "patience": 3,
            "min_delta": 0.0,
            "metric": "val_loss",
        },
    },
    "artifacts": {
        "models_dir": "./models",
        "runs_dir": "./runs",
    },
}

def default_config() -> Dict[str, Any]:
    return copy.deepcopy(DEFAULTS)

def save_yaml(cfg: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base

def load_and_resolve(path: str, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = default_config()
    try:
        file_cfg = load_yaml(path)
        deep_update(cfg, file_cfg or {})
    except FileNotFoundError:
        # OK if not present yet; caller may be running 'init'
        pass
    if overrides:
        deep_update(cfg, overrides)
    return cfg
