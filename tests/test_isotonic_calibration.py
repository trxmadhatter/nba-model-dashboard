"""Tests for isotonic calibration fitting in build_calibration_mlb.py."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Scripts"))

from build_calibration_mlb import fit_isotonic_curves, MIN_STAT_PICKS


def _make_graded(stat, side, n, win_frac=0.55, prob_val=0.70):
    return pd.DataFrame({
        "stat":       [stat] * n,
        "side":       [side] * n,
        "fair_prob":  [prob_val] * n,
        "hit_result": ["WIN"] * int(n * win_frac) + ["LOSS"] * (n - int(n * win_frac)),
    })


def test_per_stat_side_written():
    df = _make_graded("hits", "OVER", 60)
    rows = fit_isotonic_curves(df)
    keys = {(r["stat"], r["side"]) for r in rows}
    assert ("hits", "OVER") in keys


def test_both_written_when_side_below_threshold():
    # hits OVER has 60, hits UNDER has 30 — UNDER below MIN_STAT_PICKS(50)
    # so (hits, BOTH) should appear but (hits, UNDER) should not
    df = pd.concat([
        _make_graded("hits", "OVER", 60),
        _make_graded("hits", "UNDER", 30),
    ])
    rows = fit_isotonic_curves(df)
    keys = {(r["stat"], r["side"]) for r in rows}
    assert ("hits", "BOTH") in keys
    assert ("hits", "UNDER") not in keys


def test_global_always_written():
    df = _make_graded("hits", "OVER", 60)
    rows = fit_isotonic_curves(df)
    keys = {(r["stat"], r["side"]) for r in rows}
    assert ("_global_", "_global_") in keys


def test_x_range_and_count():
    df = _make_graded("k", "UNDER", 60)
    rows = fit_isotonic_curves(df)
    stat_rows = [r for r in rows if r["stat"] == "k" and r["side"] == "UNDER"]
    assert len(stat_rows) == 100
    xs = [r["x"] for r in stat_rows]
    assert abs(min(xs) - 0.45) < 1e-6
    assert abs(max(xs) - 1.00) < 1e-6


def test_y_monotone():
    df = _make_graded("k", "OVER", 60)
    rows = fit_isotonic_curves(df)
    ys = [r["y"] for r in rows if r["stat"] == "k" and r["side"] == "OVER"]
    for i in range(len(ys) - 1):
        assert ys[i] <= ys[i + 1] + 1e-9, f"Non-monotone at index {i}: {ys[i]} > {ys[i+1]}"


def test_below_threshold_not_written():
    # Only 20 picks — below MIN_STAT_PICKS(50), but both sides < 50 so BOTH also < 50
    df = _make_graded("rbi", "OVER", 20)
    rows = fit_isotonic_curves(df)
    keys = {(r["stat"], r["side"]) for r in rows}
    assert ("rbi", "OVER") not in keys
    assert ("rbi", "BOTH") not in keys


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
