# neunet/report_train.py
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------- tiny helpers ----------
def _pct_change(a: float, b: float) -> float:
    if a == 0:
        return 0.0
    return (b - a) / abs(a)

def _safe_min(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return min(vals) if vals else None

def _trend(vals: List[float]) -> str:
    if len(vals) < 2:
        return "flat"
    start, end = float(vals[0]), float(vals[-1])
    ch = _pct_change(start, end)
    if ch < -0.05: return "down"
    if ch > 0.05:  return "up"
    return "flat"

def _md_table(headers: List[str], row: List[str]) -> str:
    sep = ["---"] * len(headers)
    return (
            "| " + " | ".join(headers) + " |\n" +
            "| " + " | ".join(sep)     + " |\n" +
            "| " + " | ".join(row)     + " |"
    )

# ---------- verdict + actions ----------
def _compute_verdict_and_actions_from_series(
        train_loss: List[float],
        val_loss: Optional[List[float]],
        test_acc: Optional[float]
) -> Tuple[str, str, List[str], Optional[int]]:
    has_val = bool(val_loss) and len(val_loss) > 0
    if not train_loss:
        return ("OK ✅", "No training loss found; defaulting to OK.", ["Run/inspect training first."], None)

    if not has_val:
        return (
            "No validation ℹ️",
            "No validation split was used, so generalization can't be assessed.",
            [
                "Re-run training with a validation split (e.g., set data.val_split to 0.1).",
                "Enable early stopping on `val_loss` once a val split is in place.",
            ],
            None,
        )

    tl_trend = _trend(train_loss)
    vl_trend = _trend(val_loss)
    last_tl  = train_loss[-1]
    last_vl  = val_loss[-1]
    min_vl   = _safe_min(val_loss)
    min_vl_epoch = None
    if min_vl is not None:
        try:
            min_vl_epoch = (val_loss.index(min_vl) + 1)
        except ValueError:
            pass

    gap = None
    if last_vl is not None and last_vl > 0:
        gap = (last_vl - last_tl) / last_vl  # relative gap at end

    # Underfitting
    if _pct_change(train_loss[0], last_tl) > -0.10:
        return (
            "Underfitting 💤",
            "Model hasn't learned enough: little train-loss improvement.",
            [
                "Train longer or reduce regularization (dropout/weight decay).",
                "Increase capacity (more/larger layers) or use a slightly higher learning rate.",
                "Check preprocessing; ensure inputs are normalized correctly.",
            ],
            min_vl_epoch,
        )

    # Overfitting
    if min_vl is not None:
        worse_than_min = (last_vl - min_vl) / min_vl if min_vl > 0 else 0.0
        if tl_trend == "down" and (worse_than_min > 0.05 or (gap is not None and gap > 0.5)):
            return (
                "Overfitting ⚠️",
                "Validation loss worsened while training loss kept improving (or end gap is large).",
                [
                    f"Stop earlier (best val loss at epoch {min_vl_epoch}). Enable early stopping.",
                    "Increase regularization: higher dropout and/or weight decay.",
                    "Reduce epochs or lower LR; consider gentle data augmentation.",
                ],
                min_vl_epoch,
            )

    # Still improving
    if min_vl is not None and abs(last_vl - min_vl) < 1e-8 and vl_trend == "down":
        return (
            "Still improving ⏳",
            "Validation loss is still decreasing; model may benefit from more epochs.",
            [
                "Continue training for a few more epochs.",
                "Optionally reduce LR (scheduler) to refine further.",
            ],
            min_vl_epoch,
        )

    # Good fit
    if gap is not None and gap <= 0.2 and (vl_trend in ("down", "flat")) and (test_acc is None or test_acc >= 0.80):
        return (
            "Good fit ✅",
            "Small train/val gap and no sign of worsening; generalization looks good.",
            [
                "Keep the current configuration; save and version the checkpoint.",
                "Optionally run multiple seeds to confirm stability.",
            ],
            min_vl_epoch,
        )

    # Fallback
    return (
        "OK ✅",
        "No strong signs of over/underfitting detected.",
        [
            "Monitor with a few more epochs or small hyperparameter tweaks.",
            "Evaluate on the test set to confirm.",
        ],
        min_vl_epoch,
    )

def _normalize_history(train_data: Dict[str, Any]) -> Tuple[List[float], Optional[List[float]]]:
    """
    Accept either:
      - {'history': [{'epoch':1, 'train_loss':..., 'val_loss':...}, ...]}
      - {'train_loss': [...], 'val_loss': [...]}   # v1 structure
    """
    if "history" in train_data and isinstance(train_data["history"], list):
        h = train_data["history"]
        tl = [float(e["train_loss"]) for e in h if "train_loss" in e]
        vl = [float(e["val_loss"]) for e in h if "val_loss" in e] if any("val_loss" in e for e in h) else None
        return tl, vl

    tl = [float(x) for x in train_data.get("train_loss", [])]
    vl_list = train_data.get("val_loss")
    vl = [float(x) for x in vl_list] if isinstance(vl_list, list) else None
    return tl, vl

# ---------- public API ----------
def generate_train_log_report(
        metrics_path: str,
        eval_path: str | None = None,
        out_path: str = "runs/train_log.md",   # ← default renamed
) -> Dict[str, Any]:
    """
    Build a compact training log (Markdown) with a verdict, reason, and actions.

    - reads train/val loss from runs/metrics.json
    - reads test metrics from either eval_path OR the same metrics.json
    """
    metrics_path = Path(metrics_path)
    train_data = json.loads(metrics_path.read_text(encoding="utf-8"))

    # test metrics (optional)
    test_loss = None
    test_acc  = None
    if eval_path and Path(eval_path).exists():
        try:
            ed = json.loads(Path(eval_path).read_text(encoding="utf-8"))
            test_loss = ed.get("test_loss")
            test_acc  = ed.get("test_accuracy") if "test_accuracy" in ed else ed.get("test_acc")
        except Exception:
            pass
    if test_loss is None:
        test_loss = train_data.get("test_loss")
    if test_acc is None:
        test_acc = train_data.get("test_acc") or train_data.get("test_accuracy")

    train_loss, val_loss = _normalize_history(train_data)
    if not train_loss:
        raise ValueError("No training loss found in metrics.json")

    verdict, reason, actions, best_epoch = _compute_verdict_and_actions_from_series(train_loss, val_loss, test_acc)

    # table row
    last_tl = f"{float(train_loss[-1]):.4f}"
    last_vl = f"{float(val_loss[-1]):.4f}" if val_loss else "—"
    last_va = "—"  # val acc not tracked in v1
    last_te = f"{float(test_loss):.4f}" if isinstance(test_loss, (int, float)) else "—"
    last_ta = f"{float(test_acc):.2%}"  if isinstance(test_acc, (int, float))  else "—"

    headers = ["Train Loss", "Val Loss", "Val Accuracy", "Test Loss", "Test Accuracy", "Verdict"]
    row     = [last_tl,      last_vl,      last_va,        last_te,     last_ta,        verdict]
    table_md = _md_table(headers, row)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# 📒 Training Log\n\n")                # ← heading renamed
        f.write(f"**Epochs:** {len(train_loss)}\n\n")
        if best_epoch is not None:
            f.write(f"**Best val-loss epoch:** {best_epoch}\n\n")
        f.write(table_md + "\n\n")
        f.write(f"**Verdict reason:** {reason}\n\n")
        f.write("## Recommended actions\n")
        for a in actions:
            f.write(f"- {a}\n")

    return {
        "epochs": len(train_loss),
        "best_epoch": best_epoch,
        "verdict": verdict,
        "reason": reason,
        "actions": actions,
        "summary": dict(zip(headers, row)),
        "path": out_path,
    }