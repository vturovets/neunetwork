# tests/conftest.py
from __future__ import annotations
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import pytest

class TinyMNIST(Dataset):
    """
    Drop-in replacement for torchvision.datasets.MNIST used ONLY in tests.
    Generates tiny grayscale 28×28 images with labels 0..9.
    """
    def __init__(self, root, train: bool, download: bool, transform=None):
        rng = np.random.default_rng(123 if train else 456)
        n = 120 if train else 60  # small & fast
        self.images = (rng.integers(0, 256, size=(n, 28, 28), dtype=np.uint8))
        self.labels = rng.integers(0, 10, size=(n,), dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx], mode="L")
        y = int(self.labels[idx])
        if self.transform:
            x = self.transform(img)  # -> [1,28,28] tensor
        else:
            x = torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0) / 255.0
        return x, y

@pytest.fixture(autouse=True)
def patch_mnist(monkeypatch):
    """
    Automatically patch torchvision.datasets.MNIST for all tests.
    """
    import torchvision.datasets as dsets
    monkeypatch.setattr(dsets, "MNIST", TinyMNIST, raising=True)
    yield
