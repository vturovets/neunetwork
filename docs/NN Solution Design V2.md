# NeuNetwork — Solution Design (V2, revised with info.py)

> **Status:** Updated to match the latest public repository state. This version preserves the style and structure of V1 while reflecting current CLI, config, artifacts, and flows.  
> Includes **report.py**, **report_train.py**, and **info.py** modules explicitly.

---

## 1) Architecture

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
  info.py           # checkpoint inspection (layers, params, device)
  utils.py          # device pick, seeding, plotting, md helpers
configs/
  default.yaml
models/             # best.pt, last.pt
runs/               # metrics.json, plots, evaluation.md, training_log.md
tests/
README.md
requirements.txt
```

**Notes**
- The codebase uses **Typer** for the CLI, **PyTorch** for NN, and **Matplotlib** for plots.
- Reporting responsibilities are separated: `report_train.py` (training phase), `report.py` (evaluation).  
- `info.py` centralizes checkpoint/model inspection.

---

## 2) CLI (Typer) — commands & key options

**Commands**

- `neunet init` — prepare default folders/configs.  
- `neunet train` — run training, log metrics, save checkpoints.  
- `neunet eval` — evaluate trained checkpoint, update metrics.json, generate evaluation report.  
- `neunet infer` — predict on files or folders, output JSON with top‑k probabilities.  
- `neunet info` — inspect checkpoint metadata (model layers, parameter count, device, training config).

---

## 3) Configuration (YAML)

*(unchanged from previous draft — dataset, model, train, artifacts sections.)*

---

## 4) Data & Transforms

*(same as V2 draft — MNIST, normalization, optional val split, flatten to 784 features.)*

---

## 5) Model

*(same as V2 draft — MLP with config‑driven hidden layers, activations, dropout.)*

---

## 6) Training

- **Driver**: `train.py`
- **Reports**: `report_train.py` generates:
  - `runs/loss_curve.png`
  - `runs/training_log.md` with epochs, losses, overfitting verdicts, and recommended actions
- **Loss**: CrossEntropyLoss  
- **Optimizer**: Adam (lr, weight_decay configurable)  
- **Checkpoints**: `models/last.pt`, `models/best.pt`  

---

## 7) Evaluation

- **Driver**: `eval.py`  
- **Reports**: `report.py` generates:
  - `runs/confusion_matrix.png`
  - `runs/evaluation.md` (dataset, config, metrics, artifact links)
- **Outputs**: updates metrics.json with `test_loss`, `test_acc`  

---

## 8) Inference

*(same as V2 draft — robust image loading, preprocessing, JSON schema with `file`, `pred`, `pred_prob`, `topk`.)*

---

## 9) Reporting Summary

- **report_train.py**
  - Epoch logs, train/val loss curves
  - Overfitting/OK verdicts
  - Recommended actions section
- **report.py**
  - Confusion matrix (Matplotlib)
  - Evaluation summary in Markdown

---

## 10) Info Command

- **Driver**: `info.py`  
- **Purpose**: Provide transparency into checkpoints and models.  
- **Outputs** (printed to console):
  - Layers and activations
  - Parameter count
  - Device (CPU/GPU/MPS)
  - Training config snapshot

---

## 11) Testing

*(same as previous — plus unit tests for `report.py`, `report_train.py`, and `info.py`.)*

---

## 12) Dependencies (pinned)

`torch`, `torchvision`, `pillow`, `numpy`, `typer`, `pyyaml`, `tqdm`, `matplotlib`, `scikit-learn`

---

## 13) Out of Scope / Roadmap (phase‑2)

- Advanced schedulers/regularization
- `neunet compare` for **P95 statistical comparison** (bootstrap CIs, p-values)

---

## 14) Change log (from V1 to V2 revised)

- Added **report.py**, **report_train.py**, **info.py** to architecture section.
- Clarified separation of responsibilities: training reporting vs. evaluation reporting vs. checkpoint inspection.
- Documented `neunet info` CLI command explicitly.
- Revised artifact list accordingly.
