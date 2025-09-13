# NeuNetwork

A minimal Python CLI to study how feed‑forward neural networks operate on MNIST.  
Built with **PyTorch** and **Typer**, it supports configurable layers/activations, training/evaluation, and inference on user images.  
Artifacts include metrics (JSON), plots (loss curve, confusion matrix), and checkpoints (`best.pt`, `last.pt`).

---

## Features

- **Dataset:** MNIST (28×28 grayscale)
- **Architecture:** `784 → [hidden_i] → 10`
  - Configurable hidden layers (`--layers`) and activations (`--activations`)
  - Supported activations: `relu`, `tanh`, `sigmoid`, `linear`
- **Regularization**
  - Dropout (`--dropout`, default 0.0)
  - L2 weight decay (`--weight-decay`, default 0.0)
- **Training**
  - Optimizer: Adam
  - Logs train loss per epoch; optional val split and val loss
  - Saves checkpoints (`last.pt`, `best.pt`)
  - Updates `runs/metrics.json` and plots `runs/loss_curve.png`
- **Evaluation**
  - Computes test loss and test accuracy
  - Generates `runs/confusion_matrix.png`
  - Produces `runs/evaluation.md`
- **Inference**
  - Accepts PNG/JPG/BMP/TIFF
  - Auto‑converts to grayscale 28×28, normalizes
  - Outputs JSON (`infer.json`) with predicted label, probability, and top‑K list
- **Device auto‑select:** uses CUDA/MPS if available, else CPU

---

## Project Structure

```
neunetwork/
  neunet/
    cli.py, config.py, data.py, models.py, train.py, eval.py, infer.py, utils.py
  configs/default.yaml
  models/               # best.pt, last.pt
  runs/                 # metrics.json, plots, evaluation.md
  tests/
  utilities/
    mcnemar_test.py     # NEW: McNemar matched-pairs comparator
  README.md
  requirements.txt
```
> The `utilities/mcnemar_test.py` script is provided as a standalone utility first; a native CLI
> command (`neunet compare`) can be added later if desired.

---

## Installation

```bash
git clone https://github.com/vturovets/neunetwork.git
cd neunetwork
pip install -e .
```

Requirements: Python 3.10+, PyTorch 2.x, torchvision, Matplotlib.

---

## CLI Usage

### Init
```bash
neunet init
```
Scaffolds config, models, runs folders.

### Train
```bash
# Baseline (Adam, no regularization)
neunet train --layers 128,64 --activations relu,relu --epochs 5 --batch-size 64 --lr 1e-3

# With Dropout
neunet train --layers 128,64 --activations relu,relu --dropout 0.2

# With L2 (weight decay)
neunet train --layers 256,128 --activations relu,relu --weight-decay 1e-4

# With both
neunet train --layers 256,128 --activations relu,relu --dropout 0.2 --weight-decay 1e-4
```
Produces:
- `models/last.pt`, `models/best.pt`
- `runs/metrics.json`, `runs/loss_curve.png`

### Eval
```bash
neunet eval --checkpoint models/best.pt
```
Outputs:
- Test loss, test accuracy (console + `metrics.json`)
- `runs/confusion_matrix.png`
- `runs/evaluation.md`

### Infer
```bash
neunet infer --images path/to/imgs --checkpoint models/best.pt --out runs/infer.json --topk 3
```

### Info
```bash
neunet info --checkpoint models/best.pt
```
Shows checkpoint metadata.

---

## Statistical Comparison (McNemar matched‑pairs)

Compare two model runs on the **same** image set using the McNemar test.

**Input assumptions**
- Each `infer.json` is a list of entries with at least:
  ```json
  {"file": "my_imgs/001_label3.png", "pred": 3, "pred_prob": 0.98, "topk": [...]}
  ```
- The ground‑truth label is embedded in the filename as `*_label<digit>.*` (e.g., `_label7.png`).

**Run**
```bash
# From repo root
python utilities/mcnemar_test.py runs/infer_modelA.json runs/infer_modelB.json

# Optional: dump per‑file agreement/discordance CSV
python utilities/mcnemar_test.py runs/infer_modelA.json runs/infer_modelB.json --dump runs/mcnemar_pairs.csv
```

**Output**
```
McNemar contingency (A vs B):
           B correct   B wrong
A correct         a           b
A wrong           c           d

Discordant pairs: b=..., c=..., n=b+c

McNemar exact (binomial, two-sided):
  p-value = ...
McNemar chi-square (with continuity correction):
  X^2 = ..., p-value = ...
```
Guidance:
- Prefer the **exact** p‑value (binomial) for small n (e.g., 20 images).
- If `b+c == 0` (no discordant pairs), the test returns p=1.0 (models indistinguishable on this set).

---

## Artifacts

- `models/best.pt` – lowest val loss
- `models/last.pt` – last epoch
- `runs/metrics.json` – `{"train_loss":[...], "val_loss":[...], "test_loss":..., "test_acc":...}`
- `runs/loss_curve.png` – train/val loss plot
- `runs/confusion_matrix.png` – test set confusion matrix
- `runs/evaluation.md` – Markdown summary for presentation
- `runs/mcnemar_pairs.csv` – (optional) per‑file (dis)agreement report

---

## Development & Testing

- Unit tests in `tests/` for model builder, config, preprocessing
- Integration: 1‑epoch CPU run produces artifacts
- Dependencies pinned in `requirements.txt`

---

## Roadmap (Phase 2)

- Promote `utilities/mcnemar_test.py` into `neunet compare` CLI (with `statsmodels` backend optional).
- Add bootstrap CIs for accuracy deltas and P95 latency comparison.

---

## License

MIT