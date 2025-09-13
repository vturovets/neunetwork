from __future__ import annotations
from typing import List, Dict, Any, Optional
import os, json, re, time
from pathlib import Path

# Default filename/parent-folder label detectors (works for MNIST-style sets)
DEFAULT_REGEXES = [
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
        if m and ("label" in m.groupdict() or m.groups()):
            g = m.group("label") if "label" in m.groupdict() else m.group(1)
            try:
                return int(g)
            except Exception:
                return None
    lbl = _extract_label_from_name(p.stem, DEFAULT_REGEXES)
    if lbl is not None:
        return lbl
    # Parent folder fallback (e.g., ".../7/img.png")
    try:
        parent = p.parent.name
        if parent.isdigit() and int(parent) in range(10):
            return int(parent)
    except Exception:
        pass
    return None

def build_report(infer_json_path: str, out_md_path: str, *, label_regex: Optional[str] = None) -> Dict[str, Any]:
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
                # Defensive formatting for top-k
                def _fmt(d):
                    try:    return f"{int(d.get('label'))}({float(d.get('prob')):.2f})"
                    except: return str(d)
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
