"""Tests for isotonic calibration fitting in build_calibration_mlb.py."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Scripts"))

from build_calibration_mlb import fit_isotonic_curves, MIN_STAT_PICKS
from compute_ev_mlb import load_isotonic_calibration, apply_calibration


def _make_graded(stat, side, n, win_frac=0.55, prob_val=0.70):
    return pd.DataFrame({
        "stat":       [stat] * n,
        "side":       [side] * n,
        "fair_prob":  [prob_val] * n,
        "hit_result": ["WIN"] * int(n * win_frac) + ["LOSS"] * (n - int(n * win_frac)),
    })


def _make_iso_csv(tmp_path, rows):
    """rows: list of (stat, side, x, y)"""
    df = pd.DataFrame(rows, columns=["stat", "side", "x", "y"])
    p = tmp_path / "calibration_isotonic_mlb.csv"
    df.to_csv(p, index=False)
    return p


# ── fit_isotonic_curves tests ─────────────────────────────────────────────────

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
    assert ("hits", "OVER") in keys   # fat side still gets its own curve


def test_global_always_written():
    df = _make_graded("hits", "OVER", 60)
    rows = fit_isotonic_curves(df)
    keys = {(r["stat"], r["side"]) for r in rows}
    assert ("_global_", "_GLOBAL_") in keys


def test_x_range_and_count():
    df = _make_graded("k", "UNDER", 60)
    rows = fit_isotonic_curves(df)
    stat_rows = [r for r in rows if r["stat"] == "k" and r["side"] == "UNDER"]
    assert len(stat_rows) == 100
    xs = [r["x"] for r in stat_rows]
    assert abs(min(xs) - 0.45) < 1e-6
    assert abs(max(xs) - 1.00) < 1e-6


def test_y_monotone():
    # Generate rows with varied fair_prob so the isotonic fit is non-trivial
    rng = np.random.default_rng(42)
    n = 80
    probs = np.linspace(0.50, 0.95, n)
    wins = (rng.random(n) < probs).astype(int)
    df = pd.DataFrame({
        "stat":       ["k"] * n,
        "side":       ["OVER"] * n,
        "fair_prob":  probs,
        "hit_result": np.where(wins, "WIN", "LOSS"),
    })
    rows = fit_isotonic_curves(df)
    ys = [r["y"] for r in rows if r["stat"] == "k" and r["side"] == "OVER"]
    assert len(ys) == 100
    for i in range(len(ys) - 1):
        assert ys[i] <= ys[i + 1] + 1e-9, f"Non-monotone at index {i}: {ys[i]} > {ys[i+1]}"


def test_below_threshold_not_written():
    # Only 20 picks — below MIN_STAT_PICKS(50), but both sides < 50 so BOTH also < 50
    df = _make_graded("rbi", "OVER", 20)
    rows = fit_isotonic_curves(df)
    keys = {(r["stat"], r["side"]) for r in rows}
    assert ("rbi", "OVER") not in keys
    assert ("rbi", "BOTH") not in keys


def test_both_not_written_when_both_sides_above_threshold():
    # Both OVER and UNDER have 60 picks — both above MIN_STAT_PICKS(50)
    # so BOTH should NOT appear
    df = pd.concat([
        _make_graded("hits", "OVER", 60),
        _make_graded("hits", "UNDER", 60),
    ])
    rows = fit_isotonic_curves(df)
    keys = {(r["stat"], r["side"]) for r in rows}
    assert ("hits", "BOTH") not in keys
    assert ("hits", "OVER") in keys
    assert ("hits", "UNDER") in keys


# ── load_isotonic_calibration / apply_calibration tests ──────────────────────

def test_load_isotonic_returns_dict(tmp_path):
    p = _make_iso_csv(tmp_path, [
        ("hits", "OVER", 0.60, 0.52),
        ("hits", "OVER", 0.70, 0.58),
    ])
    iso = load_isotonic_calibration(str(p))
    assert ("hits", "OVER") in iso
    xs, ys = iso[("hits", "OVER")]
    assert len(xs) == len(ys) == 2


def test_load_isotonic_missing_file_returns_empty():
    iso = load_isotonic_calibration("/nonexistent/path.csv")
    assert iso == {}


def test_load_isotonic_bad_columns_returns_empty(tmp_path):
    # File exists but missing required columns
    bad_df = pd.DataFrame([{"stat": "hits", "side": "OVER", "wrong_col": 0.5}])
    p = tmp_path / "bad.csv"
    bad_df.to_csv(p, index=False)
    iso = load_isotonic_calibration(str(p))
    assert iso == {}


def test_apply_calibration_uses_iso_first(tmp_path):
    # iso says hits OVER at 0.65 -> 0.50; bucket says 0.70 — iso should win
    p = _make_iso_csv(tmp_path, [
        ("hits", "OVER", 0.50, 0.45),
        ("hits", "OVER", 0.65, 0.50),
        ("hits", "OVER", 0.80, 0.56),
    ])
    iso = load_isotonic_calibration(str(p))
    cal_df = pd.DataFrame()  # empty — bucket logic unavailable
    result = apply_calibration(0.65, "hits", cal_df, side="OVER", iso_map=iso)
    assert abs(result - 0.50) < 0.01  # np.interp at 0.65 in [0.50, 0.65, 0.80]


def test_apply_calibration_falls_back_to_both(tmp_path):
    p = _make_iso_csv(tmp_path, [
        ("hits", "BOTH", 0.50, 0.44),
        ("hits", "BOTH", 0.80, 0.58),
    ])
    iso = load_isotonic_calibration(str(p))
    cal_df = pd.DataFrame()
    result = apply_calibration(0.65, "hits", cal_df, side="OVER", iso_map=iso)
    # Should use BOTH fallback — result between 0.44 and 0.58
    assert 0.44 <= result <= 0.58


def test_apply_calibration_falls_back_to_global(tmp_path):
    p = _make_iso_csv(tmp_path, [
        ("_global_", "_global_", 0.50, 0.43),
        ("_global_", "_global_", 0.80, 0.57),
    ])
    iso = load_isotonic_calibration(str(p))
    cal_df = pd.DataFrame()
    result = apply_calibration(0.65, "hits", cal_df, side="OVER", iso_map=iso)
    assert 0.43 <= result <= 0.57


def test_apply_calibration_falls_through_to_bucket_when_no_iso():
    # iso_map empty, bucket cal_df matches — should return bucket value
    cal_df = pd.DataFrame([{
        "stat": "hits", "side": "OVER",
        "prob_min": 0.60, "prob_max": 0.70,
        "actual_win_rate": 0.55, "n_picks": 80,
    }])
    result = apply_calibration(0.65, "hits", cal_df, side="OVER", iso_map={})
    assert result == 0.55


def test_apply_calibration_no_iso_no_bucket_returns_raw():
    result = apply_calibration(0.72, "tb", pd.DataFrame(), side="OVER", iso_map={})
    assert result == 0.72


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
