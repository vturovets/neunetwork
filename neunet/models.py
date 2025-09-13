# neunet/models.py
from __future__ import annotations

from typing import Iterable, List, Sequence
import torch
import torch.nn as nn


# ---------------- Activation registry ----------------
# Accept a few common aliases; map to torch.nn modules.
_ACTS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "linear": nn.Identity,
    "identity": nn.Identity,
    "none": nn.Identity,
}


def _normalize_activations(names: Iterable[str]) -> List[str]:
    out = []
    for a in names:
        key = str(a).strip().lower()
        if key not in _ACTS:
            raise ValueError(
                f"Unknown activation '{a}'. "
                f"Supported: {', '.join(sorted(set(_ACTS.keys())))}"
            )
        out.append(key)
    return out


# ---------------- MLP definition ----------------
class MLP(nn.Module):
    """
    Feed-forward multilayer perceptron with optional dropout after each hidden activation.
    Final layer outputs logits (no softmax).
    """

    def __init__(
            self,
            input_size: int,
            layers: Sequence[int],
            activations: Sequence[str],
            output_size: int,
            dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if len(layers) != len(activations):
            raise ValueError("layers and activations must have the same length")

        acts = _normalize_activations(activations)
        p = float(dropout)
        if not (0.0 <= p < 1.0):
            raise ValueError("dropout must be in [0.0, 1.0)")

        mods: List[nn.Module] = []
        in_dim = int(input_size)

        # Hidden blocks: Linear -> Activation -> (Dropout)
        for hid, act_name in zip(layers, acts):
            hid = int(hid)
            mods.append(nn.Linear(in_dim, hid))
            mods.append(_ACTS[act_name]())
            if p > 0.0:
                mods.append(nn.Dropout(p=p))
            in_dim = hid

        # Output layer: logits
        mods.append(nn.Linear(in_dim, int(output_size)))

        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept [B, 1, 28, 28] or already flattened [B, 784]
        if x.dim() > 2:
            x = x.view(x.size(0), -1)  # Flatten to [B, N]
        return self.net(x)


# ---------------- Factories / utils ----------------
def build_mlp_from_cfg(cfg: dict) -> MLP:
    """
    Create an MLP from the project's config dict.
    Expects:
        cfg["model"] = {
            "input_size": 784,
            "layers": [...],
            "activations": [...],
            "output_size": 10,
            "dropout": 0.0,
        }
    """
    m = cfg["model"]
    return MLP(
        input_size=int(m["input_size"]),
        layers=[int(h) for h in m["layers"]],
        activations=[str(a) for a in m["activations"]],
        output_size=int(m["output_size"]),
        dropout=float(m.get("dropout", 0.0)),
    )


def parameter_count(model: nn.Module) -> int:
    """Total number of parameters (useful for reports)."""
    return sum(p.numel() for p in model.parameters())


__all__ = ["MLP", "build_mlp_from_cfg", "parameter_count"]
