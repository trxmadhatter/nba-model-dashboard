"""
build_calibration_mlb.py

Analyzes graded EV picks to compute how well the model's fair_prob
matches actual win rates — then writes calibration_mlb.csv so
compute_ev_mlb.py can correct future probability estimates.

How it works:
  1. Load graded_picks_mlb.csv (WIN/LOSS only — NO ACTION excluded)
  2. For each (stat, side) pair with enough data, bucket picks by fair_prob
  3. Compute actual win rate per bucket
  4. Write calibration_mlb.csv: stat, side, prob_min, prob_max, actual_win_rate, n_picks
  5. Print a full report showing model accuracy vs actual

Calibration is side-specific (OVER vs UNDER) so directional biases like
TB OVER underperforming while TB UNDER outperforms are captured separately.

Only buckets with >= MIN_BUCKET picks are written (avoids overfitting on noise).
Stats with < MIN_STAT_PICKS total graded picks are excluded entirely.

Output: Data/mlb/processed/calibration_mlb.csv
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

GRADED_FILE      = ROOT / "Data/mlb/history/graded_picks_mlb.csv"
CALIBRATION_FILE = ROOT / "Data/mlb/processed/calibration_mlb.csv"
ISOTONIC_FILE    = ROOT / "Data/mlb/processed/calibration_isotonic_mlb.csv"

# Minimum graded picks for a stat to be calibrated at all
MIN_STAT_PICKS = 50

# Minimum picks per bucket to write a calibration entry
MIN_BUCKET_PICKS = 20

# Minimum picks in a calibration bucket before it's considered reliable enough
# to influence ELITE-tier decisions. Buckets below this are flagged elite_safe=False.
ELITE_MIN_BUCKET_PICKS = 75

# Bucket width: use narrow (5%) for stats with lots of data, wide (10%) otherwise
NARROW_THRESHOLD = 150   # if stat has >= this many graded picks, use 5% buckets
NARROW_STEP      = 0.05
WIDE_STEP        = 0.10

# Fair prob range to consider (ignore extreme tails with little data)
PROB_MIN = 0.45
PROB_MAX = 1.00


def make_buckets(step: float) -> list[tuple[float, float]]:
    buckets = []
    p = PROB_MIN
    while p < PROB_MAX - 1e-9:
        buckets.append((round(p, 4), round(min(p + step, PROB_MAX), 4)))
        p += step
    return buckets


def calibrate_stat(stat_df: pd.DataFrame, stat: str, side: str) -> list[dict]:
    n = len(stat_df)
    step = NARROW_STEP if n >= NARROW_THRESHOLD else WIDE_STEP
    buckets = make_buckets(step)
    rows = []

    for p_min, p_max in buckets:
        mask = (stat_df["fair_prob"] >= p_min) & (stat_df["fair_prob"] < p_max)
        bucket_df = stat_df[mask]
        n_b = len(bucket_df)
        if n_b < MIN_BUCKET_PICKS:
            continue

        wins        = (bucket_df["hit_result"] == "WIN").sum()
        actual_wr   = wins / n_b
        model_avg   = bucket_df["fair_prob"].mean()
        bias        = actual_wr - model_avg  # positive = model under-estimates
        rows.append({
            "stat":            stat,
            "side":            side,
            "prob_min":        p_min,
            "prob_max":        p_max,
            "model_avg_prob":  round(model_avg, 4),
            "actual_win_rate": round(actual_wr, 4),
            "bias":            round(bias, 4),
            "n_picks":         n_b,
            "elite_safe":      n_b >= ELITE_MIN_BUCKET_PICKS,
        })

    return rows


def print_report(all_rows: list[dict], graded: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  MLB MODEL CALIBRATION REPORT  (side-specific)")
    print("=" * 70)

    # Overall per stat+side summary
    print(f"\n{'Stat':<8} {'Side':<6} {'Graded':>7} {'Win%':>7} {'Model Avg':>10}  {'Bias':>7}  Status")
    print("-" * 60)
    for (stat, side), g in graded.groupby(["stat", "side"]):
        n     = len(g)
        wins  = (g["hit_result"] == "WIN").sum()
        wr    = wins / n
        mavg  = g["fair_prob"].mean()
        bias  = wr - mavg
        flag  = ""
        if abs(bias) >= 0.05:  flag = "*** RECALIBRATE"
        elif abs(bias) >= 0.03: flag = "* watch"
        print(f"{stat:<8} {side:<6} {n:>7} {wr:>7.1%} {mavg:>10.1%}  {bias:>+7.1%}  {flag}")

    # Per-bucket breakdown
    if all_rows:
        print(f"\n{'Stat':<8} {'Side':<6} {'Bucket':>14}  {'Model':>7} {'Actual':>8} {'Bias':>7} {'N':>5}")
        print("-" * 60)
        cur_key = None
        for r in sorted(all_rows, key=lambda x: (x["stat"], x["side"], x["prob_min"])):
            key = (r["stat"], r["side"])
            if key != cur_key:
                cur_key = key
                print()
            bias_str = f"{r['bias']:>+.1%}"
            flag = " ***" if abs(r["bias"]) >= 0.05 else (" *" if abs(r["bias"]) >= 0.03 else "")
            print(f"{r['stat']:<8} {r['side']:<6} {r['prob_min']:.2f}-{r['prob_max']:.2f}  "
                  f"{r['model_avg_prob']:>7.1%} {r['actual_win_rate']:>8.1%} "
                  f"{bias_str:>7} {r['n_picks']:>5}{flag}")

    print()


def fit_isotonic_curves(graded: pd.DataFrame) -> list[dict]:
    """
    Fit isotonic regression curves for (stat, side), (stat, BOTH), and _global_.
    Returns a list of row dicts: {stat, side, x, y}.
    Each (stat, side) entry has 100 rows — one per evenly-spaced x from 0.45 to 1.00.
    """
    rows: list[dict] = []
    x_eval = np.linspace(0.45, 1.00, 100)

    def _fit_and_append(subset: pd.DataFrame, stat_key: str, side_key: str) -> None:
        subset = subset.dropna(subset=["fair_prob"])
        if subset.empty:
            return
        X = subset["fair_prob"].values.reshape(-1, 1)
        y = (subset["hit_result"] == "WIN").astype(float).values
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(X.ravel(), y)
        y_pred = iso.predict(x_eval)
        for x_val, y_val in zip(x_eval, y_pred):
            rows.append({"stat": stat_key, "side": side_key, "x": round(float(x_val), 6), "y": round(float(y_val), 6)})

    # Per (stat, side) — only if enough picks
    for (stat, side), group in graded.groupby(["stat", "side"]):
        stat_str, side_str = str(stat), str(side)
        if len(group) >= MIN_STAT_PICKS:
            _fit_and_append(group, stat_str, side_str)

    # Per (stat, BOTH) — for stats whose total >= MIN_STAT_PICKS but some sides are thin
    stat_totals = graded.groupby("stat").size()
    for stat_str, total in stat_totals.items():
        if total < MIN_STAT_PICKS:
            continue
        stat_group = graded[graded["stat"] == stat_str]
        # Only write BOTH if at least one side is below threshold
        side_counts = stat_group.groupby("side").size()
        any_thin = any(n < MIN_STAT_PICKS for n in side_counts.values)
        if any_thin:
            _fit_and_append(stat_group, str(stat_str), "BOTH")

    # Global fallback
    _fit_and_append(graded, "_global_", "_global_")

    return rows


def main() -> None:
    if not GRADED_FILE.exists():
        print(f"Graded picks file not found: {GRADED_FILE}")
        return

    df = pd.read_csv(GRADED_FILE)
    df["hit_result"] = df["hit_result"].astype(str).str.upper().str.strip()

    # Use raw model probability to avoid double-calibration feedback loop.
    # fair_prob_raw is the pre-calibration Poisson output; fair_prob is post-calibration
    # for rows generated after calibration was active.
    if "fair_prob_raw" in df.columns:
        raw      = pd.to_numeric(df["fair_prob_raw"], errors="coerce")
        fallback = pd.to_numeric(df["fair_prob"],     errors="coerce")
        df["fair_prob"] = raw.where(raw.notna(), fallback)
        n_raw      = raw.notna().sum()
        n_fallback = raw.isna().sum()
        print(f"  fair_prob source: {n_raw} rows from fair_prob_raw, {n_fallback} rows fallback to fair_prob")
        # Warn if fair_prob_raw disagrees with side-specific raw probability columns
        if "raw_prob_over" in df.columns and "side" in df.columns:
            side_up = df["side"].str.strip().str.upper()
            expected = np.where(side_up == "OVER", df["raw_prob_over"], 1.0 - df["raw_prob_over"])
            expected = pd.to_numeric(pd.Series(expected, index=df.index), errors="coerce")
            mismatch = (raw.notna() & expected.notna() & (raw - expected).abs().gt(5e-4))
            n_mismatch = mismatch.sum()
            if n_mismatch:
                print(f"  WARNING: {n_mismatch} rows where fair_prob_raw disagrees with raw_prob_over by >0.05%")
    else:
        df["fair_prob"] = pd.to_numeric(df["fair_prob"], errors="coerce")
        print("  fair_prob source: fair_prob_raw column not found, using fair_prob for all rows")

    # Normalize stat/side to avoid casing splits in groupby
    df["stat"] = df["stat"].astype(str).str.strip().str.lower()
    df["side"] = df["side"].astype(str).str.strip().str.upper()

    # Only use WIN/LOSS rows with a valid fair_prob
    graded = df[df["hit_result"].isin(["WIN", "LOSS"]) & df["fair_prob"].notna()].copy()
    graded = graded[graded["fair_prob"].between(PROB_MIN, PROB_MAX)]

    print(f"Graded picks loaded: {len(graded)} (from {len(df)} total rows)")

    all_rows: list[dict] = []
    skipped: list[str]   = []

    for (stat, side), group_df in graded.groupby(["stat", "side"]):
        n = len(group_df)
        if n < MIN_STAT_PICKS:
            skipped.append(f"{stat}/{side} ({n} picks — need {MIN_STAT_PICKS})")
            continue
        rows = calibrate_stat(group_df.copy(), str(stat), str(side))
        all_rows.extend(rows)

    if skipped:
        print(f"\nSkipped (not enough data): {', '.join(skipped)}")

    print_report(all_rows, graded)

    if not all_rows:
        print("No calibration buckets met the minimum sample threshold. Run again later.")
        return

    # Write calibration file (only the columns compute_ev_mlb.py needs)
    out_df = pd.DataFrame(all_rows)[["stat", "side", "prob_min", "prob_max", "actual_win_rate", "n_picks", "elite_safe"]]
    out_df.to_csv(CALIBRATION_FILE, index=False)

    print(f"Calibration written: {len(out_df)} buckets -> {CALIBRATION_FILE}")

    # Isotonic regression pass — fits continuous curves for all-pick calibration
    iso_rows = fit_isotonic_curves(graded)
    if iso_rows:
        iso_df = pd.DataFrame(iso_rows)[["stat", "side", "x", "y"]]
        iso_df.to_csv(ISOTONIC_FILE, index=False)
        n_curves = iso_df.groupby(["stat", "side"]).ngroups
        print(f"Isotonic calibration written: {n_curves} curves -> {ISOTONIC_FILE}")
    else:
        print("Isotonic calibration: no data to fit.")
    print("\nStats calibrated:", sorted(out_df["stat"].unique().tolist()))
    print("\nRun compute_ev_mlb.py to apply calibration to today's picks.")
    print("DONE")


if __name__ == "__main__":
    main()
