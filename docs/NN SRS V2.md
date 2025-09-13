# NeuNetwork: SRS

# Software Requirements Specification (SRS): NeuNetwork – Python CLI Feed-Forward NN (Rev 4)

> Context: you’re building a Python CLI (PyTorch) that creates a simple feed-forward neural network with N≥1 hidden layers, trains on MNIST, evaluates accuracy (plus train/test loss), and performs inference on real images.  
> Revision 4: explicitly incorporates **report.py**, **report_train.py**, and **info.py** modules.

---

## Change log (since Rev 3)

- Added **report_train.py** (training reporting with verdicts, recommended actions).  
- Added **report.py** (evaluation reporting with confusion matrix, evaluation.md).  
- Added **info.py** (checkpoint/model inspection).  
- Updated acceptance criteria, CLI, reporting, and artifacts to reflect these modules.

---

## 1) Purpose & Business Value

Deliver a minimal, extensible CLI to (a) configure a dense NN, (b) train on MNIST, (c) report **accuracy + train/test loss** with clear training/evaluation reports, and (d) run inference on user images.  
Ensure reproducibility and transparency (via `info.py`) for stakeholders.

---

## 2) Confirmed Scope Decisions

1. **Dataset:** MNIST  
2. **Accuracy target:** none (report actuals)  
3. **Compute:** CPU by default; GPU/MPS auto-detected  
4. **Activations:** default ReLU; configurable {ReLU, Tanh, Sigmoid, Linear}  
5. **Reporting:** Markdown docs + metrics artifacts (JSON/PNGs/MD)  
6. **Metrics:** Training Loss, Test Loss, Accuracy  
7. **Checkpoints:** `best.pt` (lowest val loss or final if no val split)  
8. **P95 comparison:** Phase 2 (out of v1 scope)

---

## 3) Stakeholders, Roles & Responsibilities

- **Student Developer:** implement CLI + reporting modules; generate artifacts; ensure reproducibility.  
- **Instructor/Reviewer:** review code, artifacts, reports.  
- **Peer Users:** reproduce runs or try configs.

---

## 4) Assumptions & Constraints

- Python 3.10+; PyTorch 2.x; torchvision  
- Deterministic seeds; local MNIST download  
- Optional CUDA/MPS  
- Inference: local images (PNG/JPG/BMP/TIFF)

---

## 5) High-Level User Flows (v1)

- **Setup:** `neunet init` → scaffold configs/models/runs.  
- **Train:** `neunet train` → checkpoints, metrics, **training log (report_train.py)**.  
- **Evaluate:** `neunet eval` → test metrics, **evaluation report (report.py)**.  
- **Infer:** `neunet infer` → predictions with probabilities.  
- **Inspect:** `neunet info` → checkpoint/model metadata (info.py).

---

## 6) Functional Requirements (v1)

### 6.1 CLI Commands

- `neunet init` → scaffold (config.yaml, dirs).  
- `neunet train [--layers … --activations … --epochs 5 --batch-size 64 --lr 1e-3 --device auto]`  
- `neunet eval [--checkpoint models/best.pt]`  
- `neunet infer --images <file|dir> --checkpoint models/best.pt [--out runs/infer.json] [--topk 3]`  
- `neunet info [--checkpoint models/best.pt]`

### 6.2 Model Configuration

- `Input(784) → [Hidden_i × N] → Output(10)`  
- Activations per hidden layer; logits output; softmax only for probs

### 6.3 Training & Evaluation

- **train.py** runs loop; **report_train.py** generates:
  - `runs/loss_curve.png`  
  - `runs/training_log.md` (epochs, losses, overfitting verdicts, recommended actions)  
- **eval.py** runs evaluation; **report.py** generates:
  - `runs/confusion_matrix.png`  
  - `runs/evaluation.md` (config, metrics, artifact refs)  

### 6.4 Inference

- Accepts file/dir; converts to grayscale 28×28; normalizes; outputs JSON with `pred`, `pred_prob`, `topk`.

### 6.5 Info

- **info.py**: displays checkpoint details — layers, activations, parameter count, device, training config snapshot.

### 6.6 Non-Goals in v1

- P95 comparison (Phase 2)  
- Distributed training, advanced schedulers, CNN/Transformer models

---

## 7) Reporting (Markdown)

- **report_train.py → runs/training_log.md**  
  - Epoch metrics, loss curves, verdicts, recommendations  
- **report.py → runs/evaluation.md**  
  - Config summary, test loss/acc, confusion matrix, artifact links

---

## 8) Data, Files & Artifacts

### 8.1 Structure

```
neunet/
  cli.py, train.py, eval.py, infer.py, info.py
  report.py, report_train.py
  models/ (best.pt, last.pt)
  runs/ (metrics.json, training_log.md, evaluation.md, plots)
  configs/default.yaml
  tests/
```

### 8.2 Metrics & Plots

- `runs/metrics.json`: {train_loss, val_loss?, test_loss, test_acc}  
- `runs/loss_curve.png` (report_train.py)  
- `runs/confusion_matrix.png` (report.py)

### 8.3 Checkpoints

- `best.pt` = lowest val loss; metadata stored; printed by `neunet info`

---

## 9) Non-Functional Requirements

- Usability: clear CLI help, defaults  
- Reproducibility: fixed seeds, saved config  
- Performance: CPU-friendly, GPU auto  
- Reliability: resume from `last.pt`  
- Observability: progress bars, logs, reports

---

## 10) Acceptance Criteria (v1)

1. **Init** scaffolds config, models, runs.  
2. **Train** produces checkpoints + `training_log.md` (with verdicts & recommendations).  
3. **Eval** prints test metrics, writes metrics.json, confusion_matrix.png, evaluation.md.  
4. **Infer** robust to image formats, outputs JSON schema with topk.  
5. **Info** prints checkpoint details (layers, params, device, config).  
6. **Artifacts** reproducible, paths echoed in reports.  

---

## 11) Further Development – Phase 2: P95 Statistical Comparison

*(unchanged; future command `neunet compare`)*

---

## 12) Open/Edge Considerations

- Normalization constants configurable  
- Early stopping optional  
- MNIST balanced; no special handling

---

## 13) Next Steps

- Finalize v1 implementation with new reporting and info modules  
- After acceptance, schedule Phase 2 (P95 comparison)
