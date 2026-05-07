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


if __name__ == "__main__":
    if not GRADED_FILE.exists():
        print(f"Graded file not found: {GRADED_FILE}")
        raise SystemExit(1)

    raw = pd.read_csv(GRADED_FILE)
    df  = load_and_filter(raw)
    print(f"Loaded {len(raw)} rows → {len(df)} after filtering")
