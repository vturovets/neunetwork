"""Model factory for MLP."""
from typing import List
import torch
import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "linear": nn.Identity,
}

class MLP(nn.Module):
    def __init__(self, input_size: int, layers: List[int], activations: List[str], output_size: int, dropout: float = 0.0):
        super().__init__()
        if len(layers) != len(activations):
            raise ValueError("layers and activations must have the same length")
        dims = [input_size] + layers + [output_size]
        mods = []
        for i in range(len(dims) - 2):  # hidden blocks
            mods.append(nn.Linear(dims[i], dims[i+1]))
            act_name = activations[i].lower()
            act_cls = ACTIVATIONS.get(act_name)
            if act_cls is None:
                raise ValueError(f"Unsupported activation: {activations[i]}")
            mods.append(act_cls())
            if dropout and dropout > 0:
                mods.append(nn.Dropout(dropout))
        # output layer (logits)
        mods.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)

def build_mlp(cfg) -> nn.Module:
    mcfg = cfg["model"]
    return MLP(
        input_size=mcfg["input_size"],
        layers=list(mcfg["layers"]),
        activations=[a.lower() for a in mcfg["activations"]],
        output_size=mcfg["output_size"],
        dropout=float(mcfg.get("dropout", 0.0))
    )
