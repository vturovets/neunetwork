# NeuNetwork — System Requirements Specification (SRS) — V2

## 1. Purpose & Scope
NeuNetwork is a minimal CLI to study the behavior of feed‑forward neural networks on MNIST.
V2 extends scope with **model comparison** using McNemar’s matched‑pairs test over two `infer.json` files.

## 2. Definitions
- **Infer record:** JSON object containing `file`, `pred`, optional `pred_prob`, `topk`.
- **Truth in filename:** The ground truth digit is embedded as `*_label<digit>.*`.
- **Discordant pairs:** Samples where model decisions differ (one correct, one incorrect).

## 3. References
- Project README (installation, CLI)
- PyTorch 2.x documentation (informative)

## 4. Users & Environment
- Users: data scientists, students, and BAs exploring NN behavior.
- Runtime: Python 3.10+, CPU or CUDA/MPS if available.

## 5. High‑Level Use Cases
1. **Train model** → artifacts produced.
2. **Evaluate model** → test loss/accuracy, confusion matrix.
3. **Infer on folder** → `infer.json` per run.
4. **Compare two runs** (NEW) → McNemar test over `infer.json` pair.

## 6. Functional Requirements

### FR‑TRN (Training)
- FR‑TRN‑01: Train with configurable layers/activations; log train loss; save `best.pt` / `last.pt`.

### FR‑EVAL (Evaluation)
- FR‑EVAL‑01: Compute test loss/accuracy; save confusion matrix and `evaluation.md`.

### FR‑INFER (Inference)
- FR‑INFER‑01: Accept a folder of images; output `infer.json` with `file`, `pred`, `pred_prob`, `topk` (optional).

### FR‑CMP (Comparison — NEW)
- **FR‑CMP‑01:** Given two `infer.json` files from the **same image set**, the system shall compute a 2×2 contingency table:
  ```
             B correct   B wrong
  A correct       a          b
  A wrong         c          d
  ```
- **FR‑CMP‑02:** The system shall report:
  - two‑sided **exact binomial** p‑value on discordant pairs (b, c);
  - **chi‑square with continuity correction** statistic and p‑value.
- **FR‑CMP‑03:** If `b+c == 0`, p‑value shall be 1.0 (indistinguishable).
- **FR‑CMP‑04:** Optionally export per‑file agreement/discordance CSV.
- **FR‑CMP‑05:** Parser shall accept either a JSON list or an object with `results`/`items`/`predictions` list.

## 7. Data & Interfaces
- **Input:** `infer.json` schema (minimum): `{"file": "path/001_label3.png", "pred": 3}`.
- **Assumption:** truth parsed from filename pattern `_label<digit>` (0–9); case‑insensitive.
- **CLI:** `python utilities/mcnemar_test.py <inferA.json> <inferB.json> [--dump CSV] [--alpha 0.05]`.
- **Output:** Table (a,b,c,d), discordant counts (b,c,n), p‑values.

## 8. Constraints & Dependencies
- Python 3.10+; no GPU required.
- JSON files must refer to the **same filenames** (intersection is used; empty intersection is an error).

## 9. Quality Attributes
- **Performance:** O(N) over entries; handles ≥100k rows on a laptop.
- **Reliability:** Defensive parsing; informative errors for malformed inputs.
- **Usability:** Clear CLI, optional CSV report for presentation and debugging.
- **Reproducibility:** Results depend solely on provided JSON files.

## 10. Acceptance Criteria
- AC‑CMP‑01: For a synthetic pair with known discordants (`b=3`, `c=1`), the exact p‑value < 0.05.
- AC‑CMP‑02: With `b=c=0`, exact p‑value = 1.0.
- AC‑CMP‑03: `--dump` produces a CSV with `file,A_correct,B_correct,verdict`.

## 11. Test Plan (delta)
- Unit: filename parser, extractor, contingency builder, exact & chi‑square computations.
- Integration: run `utilities/mcnemar_test.py` on two real `infer.json` files (20 images) and capture outputs.
- Negative: mismatched file sets → error; missing `pred` → error.

## 12. Out of Scope (V2)
- Incorporating comparison into `neunet` CLI (`neunet compare`) — tracked for a next version.
- Bootstrap CIs for accuracy deltas; P95 latency comparisons.