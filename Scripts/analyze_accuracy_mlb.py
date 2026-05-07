"""
analyze_accuracy_mlb.py

Three-part accuracy analysis of graded MLB picks:
  1. Calibration curve — does fair_prob match actual win rate?
  2. Per-book ROI      — which bookmaker lines yield the best ROI?
  3. Direction edge    — OVER vs UNDER by stat, ranked by ROI

Outputs: terminal report + Data/mlb/accuracy_report.html

Run:
    python Scripts/analyze_accuracy_mlb.py
"""

from pathlib import Path
import pandas as pd
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent

GRADED_FILE    = _ROOT / "Data/mlb/history/graded_picks_mlb.csv"
OUTPUT_HTML    = _ROOT / "Data/mlb/accuracy_report.html"
BET_SIZE       = 100
HEAVY_JUICE    = -200
HITS_MAX_JUICE = -115
EXCLUDED_STATS = {"rbi", "runs", "hr"}
MIN_BUCKET     = 20
MIN_DIRECTION  = 10

CAL_BUCKETS = [
    (0.50, 0.55), (0.55, 0.60), (0.60, 0.65),
    (0.65, 0.70), (0.70, 0.75), (0.75, 0.80), (0.80, 1.01),
]


def american_to_profit(odds: float) -> float:
    odds = float(odds)
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def pnl(row) -> float:
    odds = float(row["bet_odds"])
    if row["hit_result"] == "WIN":
        return BET_SIZE * american_to_profit(odds)
    elif row["hit_result"] == "LOSS":
        return -BET_SIZE
    return 0.0


def load_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bet_odds"] = pd.to_numeric(df["bet_odds"], errors="coerce")
    df["stat"]     = df["stat"].str.lower().str.strip()
    df["side"]     = df["side"].str.upper().str.strip()

    df = df[df["hit_result"].isin(["WIN", "LOSS"])]
    df = df[~df["stat"].isin(EXCLUDED_STATS)]
    df = df[df["bet_odds"] >= HEAVY_JUICE]

    hits_mask = (df["stat"] == "hits") & (df["bet_odds"] < HITS_MAX_JUICE)
    df = df[~hits_mask]

    return df.reset_index(drop=True)


def calibration_curve(df: pd.DataFrame) -> list[dict]:
    rows = []
    for lo, hi in CAL_BUCKETS:
        bucket = df[(df["fair_prob"] >= lo) & (df["fair_prob"] < hi)]
        n = len(bucket)
        if n == 0:
            continue
        wins      = (bucket["hit_result"] == "WIN").sum()
        actual    = wins / n
        label_hi  = 1.00 if hi > 1.0 else hi
        predicted = (lo + label_hi) / 2
        rows.append({
            "bucket":    f"{lo:.2f}–{label_hi:.2f}",
            "n":         n,
            "predicted": predicted,
            "actual":    actual,
            "gap":       actual - predicted,
            "low_conf":  n < MIN_BUCKET,
        })
    return rows


def calibration_grade(cal_rows: list[dict]) -> str:
    valid = [r for r in cal_rows if not r["low_conf"]]
    if not valid:
        return "N/A"
    mean_gap = float(np.mean([abs(r["gap"]) for r in valid]))
    if mean_gap < 0.03:
        return "A"
    elif mean_gap < 0.06:
        return "B"
    elif mean_gap < 0.10:
        return "C"
    return "D"


def print_calibration(cal_rows: list[dict]) -> None:
    print(f"\n{'=' * 62}")
    print("  MODULE 1 — CALIBRATION CURVE")
    print(f"{'=' * 62}")
    print(f"  {'Bucket':<12} {'N':>5}  {'Predicted':>9}  {'Actual':>7}  {'Gap':>7}  Note")
    print(f"  {'-' * 58}")
    for r in cal_rows:
        note = "⚠ low-n" if r["low_conf"] else ""
        print(
            f"  {r['bucket']:<12} {r['n']:>5}  "
            f"{r['predicted'] * 100:>8.1f}%  {r['actual'] * 100:>6.1f}%  "
            f"{r['gap'] * 100:>+6.1f}%  {note}"
        )
    print(f"\n  Calibration grade: {calibration_grade(cal_rows)}")
    print("  (A=<3% avg gap  B=3–6%  C=6–10%  D=>10%)")


if __name__ == "__main__":
    if not GRADED_FILE.exists():
        print(f"Graded file not found: {GRADED_FILE}")
        raise SystemExit(1)

    raw = pd.read_csv(GRADED_FILE)
    df  = load_and_filter(raw)
    print(f"Loaded {len(raw)} rows → {len(df)} after filtering")
