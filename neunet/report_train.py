
from __future__ import annotations
import json, os
from typing import Any, Dict, List, Optional
from tabulate import tabulate

# ---------- small helpers ----------
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

# ---------- verdict + actions ----------
def _compute_verdict_and_actions(history: List[Dict[str, Any]], test_acc: Optional[float]):
    epochs = [e["epoch"] for e in history]
    train_loss = [float(e["train_loss"]) for e in history]
    has_val = any("val_loss" in e for e in history)
    val_loss = [float(e["val_loss"]) for e in history if "val_loss" in e] if has_val else []
    val_acc  = [float(e["val_acc"])  for e in history if "val_acc"  in e] if has_val else []

    if not has_val:
        verdict = "No validation ℹ️"
        reason = "No validation split was used, so generalization can't be assessed."
        actions = [
            "Re-run training with a validation split (e.g., `--val-split 0.1`).",
            "Use early stopping on `val_loss` once enabled."
        ]
        return verdict, reason, actions, None

    tl_trend = _trend(train_loss)
    vl_trend = _trend(val_loss) if val_loss else "flat"
    last_tl  = train_loss[-1]
    last_vl  = val_loss[-1] if val_loss else None
    min_vl   = _safe_min(val_loss)
    min_vl_epoch = None
    if min_vl is not None:
        for i, e in enumerate(history):
            if "val_loss" in e and float(e["val_loss"]) == min_vl:
                min_vl_epoch = e["epoch"]
                break

    gap = None
    if last_vl is not None and last_vl > 0:
        gap = (last_vl - last_tl) / last_vl  # relative gap at end

    # Underfitting
    if _pct_change(train_loss[0], last_tl) > -0.10 or (val_acc and val_acc[-1] < 0.60):
        verdict = "Underfitting 💤"
        reason = "Model hasn't learned enough: little train-loss improvement and/or low validation accuracy."
        actions = [
            "Train longer or use a warmup/lower regularization.",
            "Increase model capacity (more/larger layers) or try a slightly higher learning rate.",
            "Check data preprocessing; ensure inputs are normalized correctly."
        ]
        return verdict, reason, actions, min_vl_epoch

    # Overfitting
    if min_vl is not None:
        worse_than_min = (last_vl - min_vl) / min_vl if min_vl > 0 else 0.0
        if tl_trend == "down" and (worse_than_min > 0.05 or (gap is not None and gap > 0.5)):
            verdict = "Overfitting ⚠️"
            reason = "Validation loss worsened while training loss kept improving (or the train/val gap is large)."
            actions = [
                f"Stop earlier (best val loss at epoch {min_vl_epoch}). Enable early stopping on `val_loss`.",
                "Add regularization: increase dropout, add weight decay, or use data augmentation.",
                "Reduce epochs or lower the learning rate schedule to avoid late overfitting."
            ]
            return verdict, reason, actions, min_vl_epoch

    # Still improving
    if min_vl is not None and abs(last_vl - min_vl) < 1e-8 and vl_trend == "down":
        verdict = "Still improving ⏳"
        reason = "Validation loss is still decreasing; model may benefit from more epochs."
        actions = [
            "Continue training for a few more epochs.",
            "Optionally reduce the learning rate (scheduler) to refine further."
        ]
        return verdict, reason, actions, min_vl_epoch

    # Good fit
    if gap is not None and gap <= 0.2 and (vl_trend in ("down","flat")) and (test_acc is None or test_acc >= 0.80):
        verdict = "Good fit ✅"
        reason = "Small train/val gap and no sign of worsening; generalization looks good."
        actions = [
            "Keep the current configuration; save and version the checkpoint.",
            "Optionally run multiple seeds to confirm stability."
        ]
        return verdict, reason, actions, min_vl_epoch

    # Fallback
    verdict = "OK ✅"
    reason = "No strong signs of over/underfitting detected."
    actions = [
        "Monitor with a few more epochs or small hyperparameter tweaks.",
        "Evaluate on a held-out test set or your own images to confirm."
    ]
    return verdict, reason, actions, min_vl_epoch

# ---------- public API ----------
def generate_train_log_report(metrics_path: str, eval_path: str, out_path: str) -> Dict[str, Any]:
    with open(metrics_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    history: List[Dict[str, Any]] = train_data.get("history", [])
    if not history:
        raise ValueError("No training history found in metrics.json")

    epochs_count = len(history)
    last = history[-1]
    val_loss = last.get("val_loss")
    val_acc  = last.get("val_acc")
    test_loss = eval_data.get("test_loss")
    test_acc  = eval_data.get("test_accuracy")

    verdict, reason, actions, best_epoch = _compute_verdict_and_actions(history, test_acc)

    # Build markdown table
    headers = ["Train Loss", "Val Loss", "Val Accuracy", "Test Loss", "Test Accuracy", "Verdict"]
    row = [
        f"{float(last['train_loss']):.4f}",
        f"{float(val_loss):.4f}" if val_loss is not None else "—",
        f"{float(val_acc):.2%}" if val_acc is not None else "—",
        f"{float(test_loss):.4f}" if test_loss is not None else "—",
        f"{float(test_acc):.2%}" if test_acc is not None else "—",
        verdict,
    ]
    table_md = tabulate([row], headers, tablefmt="github")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# 📊 Final Training Report\n\n")
        f.write(f"**Epochs:** {epochs_count}\n\n")
        if best_epoch is not None:
            f.write(f"**Best val-loss epoch:** {best_epoch}\n\n")
        f.write("```example\n")
        f.write(table_md)
        f.write("\n```\n\n")
        f.write(f"**Verdict reason:** {reason}\n\n")
        f.write("## Recommended actions\n")
        for a in actions:
            f.write(f"- {a}\n")

    return {
        "epochs": epochs_count,
        "best_epoch": best_epoch,
        "verdict": verdict,
        "reason": reason,
        "actions": actions,
        "summary": dict(zip(headers, row)),
    }
