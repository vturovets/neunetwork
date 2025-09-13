# NeuNetwork

A small, configurable **PyTorch** MNIST project with a clean **Typer** CLI. Train a feed‑forward MLP, evaluate on the test set, run inference on your images, and generate Markdown/plot reports.

> This README is aligned with **NN SRS V2** and **Solution Design V2**. It explicitly documents the `report.py`, `report_train.py`, and `info.py` modules.

---

## Features

- **Config‑driven MLP** (layers, activations, dropout, weight decay)
- **Train/Eval/Infer** commands with sensible defaults
- **Reports**
  - Training log (`training_log.md`) + loss curve via **report_train.py**
  - Evaluation summary (`evaluation.md`) + confusion matrix via **report.py**
- **Reproducibility**: seeds, saved resolved config, pinned deps
- **Transparency**: `neunet info` to inspect checkpoints (layers, params, device, config)
- **CPU‑friendly**, auto‑uses GPU/MPS if available

---

## Install

```bash
# 1) Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install
pip install -r requirements.txt
# (or) if packaged:
# pip install -e .
```

---

## Project Layout

```
neunet/
  __init__.py
  cli.py            # Typer app with subcommands
  config.py         # load/merge/validate YAML + flag overrides
  data.py           # MNIST datasets & transforms (+infer preprocessing)
  models.py         # MLP builder from YAML/flags
  train.py          # training loop, optimizer, checkpoints
  eval.py           # test-set evaluation entry
  infer.py          # image preprocessing & prediction (top-k)
  report.py         # evaluation reporting (evaluation.md, confusion matrix)
  report_train.py   # training reporting (loss curves, verdicts, recommended actions)
  info.py           # checkpoint inspection (layers, params, device, config)
  utils.py          # device pick, seeding, plotting, md helpers
configs/
  default.yaml
models/             # best.pt, last.pt
runs/               # metrics.json, plots, evaluation.md, training_log.md
tests/
README.md
requirements.txt
```

---

## Configuration

`configs/default.yaml` (example):
```yaml
seed: 42
device: auto

data:
  dataset: MNIST
  root: ./data
  download: true
  num_workers: 2
  val_split: 0.0
  normalization: {mean: 0.1307, std: 0.3081}

model:
  input_size: 784
  layers: [128, 64]
  activations: [relu, relu]
  output_size: 10
  dropout: 0.0

train:
  epochs: 5
  batch_size: 64
  lr: 0.001
  weight_decay: 0.0

artifacts:
  models_dir: ./models
  runs_dir: ./runs
```
> Flags override YAML at runtime. The **resolved config** is saved alongside artifacts for reproducibility.

---

## CLI Usage (Typer)

```bash
# Show help
python -m neunet.cli --help
```

### Init
```bash
python -m neunet.cli init
# Creates configs/default.yaml, models/, runs/, data/ (if missing)
```

### Train
```bash
python -m neunet.cli train   --layers 128,64   --activations relu,relu   --epochs 5   --batch-size 64   --lr 1e-3   --device auto   --config configs/default.yaml
# Artifacts: models/last.pt, models/best.pt, runs/loss_curve.png, runs/training_log.md
```

### Eval
```bash
python -m neunet.cli eval   --checkpoint models/best.pt
# Updates runs/metrics.json (test_loss, test_acc), writes runs/confusion_matrix.png, runs/evaluation.md
```

### Infer
```bash
python -m neunet.cli infer   --images ./my_imgs2   --checkpoint models/best.pt   --topk 3   --out runs/infer.json
# Robust to PNG/JPG/JPEG/BMP/TIFF; auto grayscale 28x28 + normalize
```

### Info
```bash
python -m neunet.cli info   --checkpoint models/best.pt
# Prints layers, activations, parameter count, device, and training config snapshot
```

---

## Outputs & Reports

- **Training (report_train.py)**
  - `runs/training_log.md` — epoch metrics, **verdict** (OK/Overfitting), **Recommended actions**
  - `runs/loss_curve.png`
- **Evaluation (report.py)**
  - `runs/evaluation.md` — dataset, resolved config, **test_loss**, **test_acc**, artifact links
  - `runs/confusion_matrix.png`
- **Inference JSON schema**
```json
[
  {
    "file": "path/to/img.png",
    "pred": 7,
    "pred_prob": 0.9912,
    "topk": [
      {"label": 7, "prob": 0.9912},
      {"label": 1, "prob": 0.0061},
      {"label": 9, "prob": 0.0010}
    ]
  }
]
```

---

## Tips & Troubleshooting

- **CUDA/MPS**: keep `--device auto` to use accelerators when available.
- **Determinism**: set `seed` and avoid nondeterministic ops for strict reproducibility.
- **Images for `infer`**: use high‑contrast digits on plain backgrounds for best results.
- **Checkpoints**: `best.pt` is selected by lowest validation loss (or final epoch when no val split). Use `info` to inspect.
- **Permissions**: on Unix, ensure the repo files are readable/executable as needed.

---

## Roadmap (Phase‑2)

- `neunet compare` for **P95 statistical comparison** (bootstrap CIs, p‑values)
- Advanced schedulers/regularization; more architectures (e.g., CNN baselines)
