"""Data loading and transforms (MNIST)."""
from __future__ import annotations
from typing import Tuple, Optional, Any, Dict
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def _make_transform(mean: float, std: float):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ])

def get_dataloaders(cfg: Dict[str, Any]):
    """Return (train_loader, val_loader, test_loader) per config.
    
    - Dataset: MNIST (28x28 grayscale)
    - Normalization defaults: mean=0.1307, std=0.3081
    - If cfg['data']['val_split'] > 0, split train into train/val with fixed seed.
    """
    dcfg = cfg["data"]
    root = dcfg.get("root", "./data")
    mean = float(dcfg.get("normalization", {}).get("mean", 0.1307))
    std = float(dcfg.get("normalization", {}).get("std", 0.3081))
    num_workers = int(dcfg.get("num_workers", 2))
    download = bool(dcfg.get("download", True))
    val_split = float(dcfg.get("val_split", 0.0))
    batch_size = int(cfg["train"].get("batch_size", 64))

    tfm = _make_transform(mean, std)

    train_full = datasets.MNIST(root=root, train=True, download=download, transform=tfm)
    test_ds = datasets.MNIST(root=root, train=False, download=download, transform=tfm)

    val_loader = None
    if val_split and val_split > 0.0:
        n_total = len(train_full)
        n_val = int(n_total * val_split)
        n_train = n_total - n_val
        g = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
        train_ds, val_ds = random_split(train_full, [n_train, n_val], generator=g)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_full, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
