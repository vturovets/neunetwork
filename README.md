# NeuNetwork

Minimal, extensible CLI to train, evaluate, and run inference for a feed‑forward NN on MNIST (PyTorch).

## Install

```bash
# inside repository root
pip install -e .
```

## Quick start

```bash
neunet init
# edit configs/default.yaml if desired
neunet train --layers 128,64 --activations relu,relu --epochs 5 --batch-size 64 --lr 1e-3
neunet eval --checkpoint models/best.pt
neunet infer --images ./some_folder --checkpoint models/best.pt --topk 3
neunet info --checkpoint models/best.pt
```

## Layout

```
neunet/           # package
configs/          # YAML configs (default.yaml)
models/           # checkpoints: best.pt, last.pt
runs/             # metrics.json, plots, evaluation.md
tests/            # pytest-based tests (stubs)
```

## Status

