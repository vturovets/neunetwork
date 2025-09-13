# neunet/report.py
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------
# Evaluation report (train/val/test summary)
# --------------------------------------------------------------------

_EVAL_TPL = """# Evaluation Report

| Metric        | Value |
|---------------|-------|
| Test Loss     | {test_loss} |
| Test Accuracy | {test_acc} |

**Verdict:** {verdict}
**Verdict reason:** {reason}

## Recommended actions
{actions}
"""

_ACTIONS_OK = "- Looks good. You can try deeper/wider layers or a small LR tune for marginal gains."
_ACTIONS_OVERFIT = """- Stop earlier (use the epoch with best `val_loss`) or enable early stopping.
- Increase dropout and/or weight decay.
- Reduce epochs or lower LR; consider gentle data augmentation.
"""
_ACTIONS_UNDERFIT = """- Train longer or reduce regularization (dropout/L2).
- Increase capacity (more/larger layers) or modestly increase LR.
- Check preprocessing; ensure normalization is correct.
"""


def _pct_change(a: float, b: float) -> float:
    if a == 0:
        return 0.0
    return (b - a) / abs(a)


def _trend(vals: List[float]) -> str:
    if len(vals) < 2:
        return "flat"
    ch = _pct_change(float(vals[0]), float(vals[-1]))
    if ch < -0.05:
        return "down"
    if ch > 0.05:
        return "up"
    return "flat"


def _verdict_from_history(history: Dict[str, Any]) -> Tuple[str, str, str]:
    tr = [float(x) for x in history.get("train_loss", []) or []]
    va = [float(x) for x in history.get("val_loss", []) or []]

    # If no history, default OK
    if not tr and not va:
        return "OK ✅", "No train/val history available; using test metrics only.", _ACTIONS_OK

    # Underfitting: train loss barely improves
    if tr and _pct_change(tr[0], tr[-1]) > -0.10:
        return "Underfitting 💤", "Training loss shows little improvement.", _ACTIONS_UNDERFIT

    # Overfitting: val worsens while train improves or gap grows
    if tr and va:
        best_val = min(va)
        last_val = va[-1]
        worse_than_best = (last_val - best_val) / best_val if best_val > 0 else 0.0
        gap = (last_val - tr[-1]) / last_val if last_val > 0 else 0.0
        if _trend(tr) == "down" and (worse_than_best > 0.05 or gap > 0.5):
            return "Overfitting ⚠️", "Validation loss worsened while training kept improving (or gap is large).", _ACTIONS_OVERFIT

    return "OK ✅", "Generalization looks acceptable for MNIST.", _ACTIONS_OK


def generate_eval_report(metrics_json: str | Path,
                         out_md: str | Path = "runs/evaluation.md") -> Path:
    """
    Build a concise evaluation Markdown from metrics.json.
    Expects (when available): train_loss[], val_loss[], test_loss, test_acc.
    """
    metrics_json = Path(metrics_json)
    out_md = Path(out_md)

    data: Dict[str, Any] = {}
    if metrics_json.exists():
        try:
            data = json.loads(metrics_json.read_text(encoding="utf-8"))
        except Exception:
            data = {}

    test_loss = data.get("test_loss", "n/a")
    test_acc = data.get("test_acc", data.get("test_accuracy", "n/a"))

    verdict, reason, actions = _verdict_from_history(data)
    md = _EVAL_TPL.format(
        test_loss=test_loss,
        test_acc=test_acc if isinstance(test_acc, str) else f"{float(test_acc):.4f}",
        verdict=verdict,
        reason=reason,
        actions=actions,
    )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    return out_md

# --------------------------------------------------------------------
# Inference report (kept from your existing behavior)
# --------------------------------------------------------------------

_DEFAULT_LABEL_REGEXES = [
    r"[_\-\.]label(?P<label>\d)\b",
    r"[\-_](?P<label>\d)\b",
    r"(?P<label>\d)\b",
]


def _extract_label_from_name(name: str, regexes: List[str]) -> Optional[int]:
    for pat in regexes:
        m = re.search(pat, name, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group("label"))
            except Exception:
                pass
    return None


def _extract_label(path: str, regex: Optional[str]) -> Optional[int]:
    p = Path(path)
    if regex:
        m = re.search(regex, p.name)
        if m:
            g = m.groupdict().get("label") or (m.group(1) if m.groups() else None)
            try:
                return int(g) if g is not None else None
            except Exception:
                return None
    lbl = _extract_label_from_name(p.stem, _DEFAULT_LABEL_REGEXES)
    if lbl is not None:
        return lbl
    try:
        parent = p.parent.name
        if parent.isdigit() and int(parent) in range(10):
            return int(parent)
    except Exception:
        pass
    return None


def build_report(infer_json_path: str, out_md_path: str, *, label_regex: Optional[str] = None) -> Dict[str, Any]:
    """
    Build a Markdown report for inference results (predictions on user images).
    """
    with open(infer_json_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    rows: List[Dict[str, Any]] = []
    n_total = 0
    n_with_label = 0
    n_correct = 0
    n_wrong = 0
    n_errors = 0
    no_label = 0

    for it in items:
        n_total += 1
        fp = it.get("file") or it.get("path") or ""
        err = it.get("error")
        if err:
            n_errors += 1
            rows.append({"file": fp, "status": "error", "error": err})
            continue

        pred = it.get("pred")
        pred_prob = it.get("pred_prob")
        topk = it.get("topk", []) or []

        true = _extract_label(fp, label_regex)
        if true is None:
            no_label += 1
            rows.append({"file": fp, "status": "no_label", "pred": pred, "pred_prob": pred_prob, "topk": topk})
            continue

        correct = (pred == true)
        status = "correct" if correct else "wrong"
        if correct:
            n_correct += 1
        else:
            n_wrong += 1
        n_with_label += 1

        rows.append({
            "file": fp,
            "status": status,
            "true": true,
            "pred": pred,
            "pred_prob": pred_prob,
            "topk": topk,
        })

    coverage = (n_with_label / n_total) if n_total else 0.0
    error_rate = (n_wrong / n_with_label) if n_with_label else 0.0
    accuracy = 1.0 - error_rate if n_with_label else 0.0

    summary = {
        "total": n_total,
        "with_label": n_with_label,
        "no_label": no_label,
        "errors": n_errors,
        "correct": n_correct,
        "wrong": n_wrong,
        "coverage": coverage,
        "error_rate": error_rate,
        "accuracy": accuracy,
        "generated_at": int(time.time()),
        "source": infer_json_path,
    }

    out_dir = os.path.dirname(out_md_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write("# Inference Report\n\n")
        f.write(f"- Source JSON: `{infer_json_path}`\n")
        f.write(f"- Images total: **{n_total}**\n")
        f.write(f"- With label: **{n_with_label}** (coverage: {coverage:.2%})\n")
        f.write(f"- Errors while reading: **{n_errors}**\n")
        f.write(f"- Correct: **{n_correct}**\n")
        f.write(f"- Wrong: **{n_wrong}**\n")
        f.write(f"- **Error rate** (wrong / with_label): **{error_rate:.2%}**\n")
        f.write(f"- Accuracy: **{accuracy:.2%}**\n\n")

        wrong_rows = [r for r in rows if r.get("status") == "wrong"]
        if wrong_rows:
            f.write("## Mismatches\n\n")
            f.write("| file | true | pred | pred_prob | top-3 |\n")
            f.write("|---|---:|---:|---:|---|\n")
            for r in wrong_rows[:200]:
                def _fmt(d):
                    try:
                        return f"{int(d.get('label'))}({float(d.get('prob')):.2f})"
                    except Exception:
                        return str(d)
                top = ", ".join([_fmt(d) for d in (r.get("topk") or [])[:3]])
                f.write(f"| {r['file']} | {r.get('true','')} | {r.get('pred','')} | {r.get('pred_prob','')} | {top} |\n")
            if len(wrong_rows) > 200:
                f.write(f"\n_+{len(wrong_rows)-200} more not shown_\n")

        nl = [r for r in rows if r.get("status") == "no_label"]
        if nl:
            f.write("\n## Files without detectable label\n\n")
            for r in nl[:200]:
                f.write(f"- {r['file']}\n")
            if len(nl) > 200:
                f.write(f"\n_+{len(nl)-200} more not shown_\n")

        errs = [r for r in rows if r.get("status") == "error"]
        if errs:
            f.write("\n## Files with errors\n\n")
            for r in errs[:200]:
                f.write(f"- {r['file']}: {r.get('error')}\n")
            if len(errs) > 200:
                f.write(f"\n_+{len(errs)-200} more not shown_\n")

    return {"summary": summary, "rows": rows}
