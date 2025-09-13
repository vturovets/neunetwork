"""
mcnemar_test.py
----------------
Build a McNemar contingency table from two NeuNetwork `infer.json` files and run
McNemar's matched-pairs test (exact binomial + chi-square with continuity correction).

Usage
-----
python mcnemar_test.py path/to/inferA.json path/to/inferB.json \
    [--dump pairs.csv] [--alpha 0.05]

Outputs the 2x2 table:
           B correct   B wrong
A correct      a           b
A wrong        c           d

and p-values for:
  - Exact binomial test (two-sided on discordant pairs b and c)
  - Chi-square with continuity correction (Edwards' correction)

Assumptions
-----------
- Each infer.json entry is either:
    * a list of dicts with keys: "file", "pred" (others ignored), or
    * a dict containing a list under "results"/"items"/"predictions"
- Ground truth label is embedded in filename: "..._label<D>.<ext>"

If your schema differs, adjust `extract_items()` or `parse_truth_from_filename()`.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

LABEL_PAT = re.compile(r"_label(?P<label>\d+)\.", re.IGNORECASE)


@dataclass(frozen=True)
class ItemResult:
    file: str
    truth: str
    pred: str


# ---------- I/O & parsing ----------

def load_json(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_list(payload: dict | list) -> List[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("results", "items", "predictions"):
            if key in payload and isinstance(payload[key], list):
                return payload[key]
    raise ValueError("infer.json must be a list or a dict containing a list under 'results'/'items'/'predictions'.")


def parse_truth_from_filename(filename: str) -> str:
    m = LABEL_PAT.search(filename)
    if not m:
        raise ValueError(
            f"Cannot parse truth label from filename '{filename}'. "
            "Expected pattern like '_label3.'"
        )
    return m.group("label")


def extract_items(infer_json: dict | list) -> Dict[str, ItemResult]:
    out: Dict[str, ItemResult] = {}
    for entry in to_list(infer_json):
        file_name = str(entry.get("file") or entry.get("image") or "").strip()
        if not file_name:
            raise ValueError("Each entry must include a 'file' (or 'image') field.")
        pred_val = entry.get("pred")
        if pred_val is None:
            raise ValueError("Each entry must include a 'pred' field.")
        truth = parse_truth_from_filename(Path(file_name).name)
        out[file_name] = ItemResult(file=file_name, truth=str(truth), pred=str(pred_val))
    return out


# ---------- Contingency construction ----------

def build_contingency(a_map: Dict[str, ItemResult], b_map: Dict[str, ItemResult]) -> Tuple[List[List[int]], Dict[str, Tuple[bool, bool]]]:
    keys = sorted(set(a_map.keys()) & set(b_map.keys()))
    if not keys:
        raise ValueError("No overlapping files in the two infer.json files.")

    a = b = c = d = 0
    per_item: Dict[str, Tuple[bool, bool]] = {}

    for k in keys:
        a_res, b_res = a_map[k], b_map[k]
        a_ok = (a_res.pred == a_res.truth)
        b_ok = (b_res.pred == b_res.truth)
        per_item[k] = (a_ok, b_ok)
        if a_ok and b_ok:
            a += 1
        elif a_ok and not b_ok:
            b += 1
        elif (not a_ok) and b_ok:
            c += 1
        else:
            d += 1

    return [[a, b], [c, d]], per_item


# ---------- McNemar tests ----------

def mcnemar_exact(b: int, c: int) -> float:
    """
    Two-sided exact binomial test on discordant pairs (n=b+c, successes=min(b,c)).
    Returns p-value.
    """
    n = b + c
    k = min(b, c)
    if n == 0:
        return 1.0  # no discordant pairs: indistinguishable

    # Binomial(n, 0.5) probabilities
    def pmf(i: int) -> float:
        return math.comb(n, i) * (0.5 ** n)

    # two-sided: probability of outcomes at least as extreme as observed
    left = sum(pmf(i) for i in range(0, k + 1))
    right = sum(pmf(i) for i in range(n - k, n + 1))
    p = min(1.0, left + right)
    return p


def mcnemar_chi2_cc(b: int, c: int) -> Tuple[float, float]:
    """
    Chi-square with continuity correction (Edwards):
        X2 = (|b - c| - 1)^2 / (b + c)
    Returns (statistic, p_value) using 1 df.
    """
    n = b + c
    if n == 0:
        return 0.0, 1.0
    x2 = ((abs(b - c) - 1) ** 2) / n
    # For 1 df, p ≈ 2*(1 - Phi(sqrt(x2)))
    z = math.sqrt(x2)
    phi = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    p = 2 * (1 - phi)
    return x2, p


# ---------- Public API ----------

def run_mcnemar(infer_a: Path | str, infer_b: Path | str, alpha: float = 0.05) -> dict:
    a_map = extract_items(load_json(Path(infer_a)))
    b_map = extract_items(load_json(Path(infer_b)))
    table, _ = build_contingency(a_map, b_map)
    a, b, c, d = table[0][0], table[0][1], table[1][0], table[1][1]
    p_exact = mcnemar_exact(b, c)
    x2_cc, p_cc = mcnemar_chi2_cc(b, c)
    return {
        "table": {"a": a, "b": b, "c": c, "d": d},
        "discordant": {"b": b, "c": c, "n": b + c},
        "exact_p": p_exact,
        "chi2_cc": {"stat": x2_cc, "p": p_cc},
        "alpha": alpha,
        "significant_exact": p_exact < alpha,
        "significant_cc": p_cc < alpha,
    }


# ---------- CLI ----------

def main() -> None:
    ap = argparse.ArgumentParser(description="Run McNemar matched-pairs test on two NeuNetwork infer.json files.")
    ap.add_argument("infer_a", help="Path to first infer.json (Model A)")
    ap.add_argument("infer_b", help="Path to second infer.json (Model B)")
    ap.add_argument("--dump", metavar="CSV", help="Optional CSV to save per-file (A_correct,B_correct) and verdict")
    ap.add_argument("--alpha", type=float, default=0.05, help="Significance level for star marker (default: 0.05)")
    args = ap.parse_args()

    a_map = extract_items(load_json(Path(args.infer_a)))
    b_map = extract_items(load_json(Path(args.infer_b)))
    table, per_item = build_contingency(a_map, b_map)
    a, b, c, d = table[0][0], table[0][1], table[1][0], table[1][1]

    print("McNemar contingency (A vs B):")
    print("           B correct   B wrong")
    print(f"A correct     {a:>5}       {b:>5}")
    print(f"A wrong       {c:>5}       {d:>5}")
    print(f"\nDiscordant pairs: b={b}, c={c}, n={b+c}")

    p_exact = mcnemar_exact(b, c)
    x2_cc, p_cc = mcnemar_chi2_cc(b, c)

    star_exact = " *" if p_exact < args.alpha else ""
    star_cc = " *" if p_cc < args.alpha else ""

    print("\nMcNemar exact (binomial, two-sided):")
    print(f"  p-value = {p_exact:.6f}{star_exact}")
    print("McNemar chi-square (with continuity correction):")
    print(f"  X^2 = {x2_cc:.4f}, p-value = {p_cc:.6f}{star_cc}")
    print("\n* denotes significance at alpha =", args.alpha)

    if args.dump:
        out = Path(args.dump)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "A_correct", "B_correct", "verdict"])
            for file, (a_ok, b_ok) in sorted(per_item.items()):
                verdict = "Agree" if a_ok == b_ok else "Discordant"
                w.writerow([file, int(a_ok), int(b_ok), verdict])
        print(f"\nPer-file report saved to: {out}")

if __name__ == "__main__":
    main()
