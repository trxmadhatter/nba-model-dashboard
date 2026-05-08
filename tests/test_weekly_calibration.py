"""Tests for build_weekly_calibration in generate_dashboard_mlb.py."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Scripts"))

from generate_dashboard_mlb import build_weekly_calibration


def _make_df(game_dates, hit_results, fair_prob_raws, bet_odds=None):
    n = len(game_dates)
    if bet_odds is None:
        bet_odds = [-110] * n
    return pd.DataFrame({
        "game_date":     game_dates,
        "hit_result":    hit_results,
        "fair_prob_raw": fair_prob_raws,
        "bet_odds":      bet_odds,
    })


def test_returns_html_string():
    df = _make_df(
        ["2026-04-07"] * 10,
        ["WIN"] * 6 + ["LOSS"] * 4,
        [0.70] * 10,
    )
    html = build_weekly_calibration(df)
    assert isinstance(html, str)
    assert "<table" in html


def test_week_label_present():
    # Apr 7 2026 is a Tuesday — week starts Mon Apr 6
    df = _make_df(
        ["2026-04-07"] * 10,
        ["WIN"] * 6 + ["LOSS"] * 4,
        [0.70] * 10,
    )
    html = build_weekly_calibration(df)
    assert "Apr 6" in html  # Monday of that week


def test_multiple_weeks_multiple_rows():
    df = _make_df(
        ["2026-04-07"] * 8 + ["2026-04-14"] * 8,
        ["WIN"] * 9 + ["LOSS"] * 7,
        [0.68] * 16,
    )
    html = build_weekly_calibration(df)
    assert "Apr 6" in html
    assert "Apr 13" in html


def test_thin_week_dimmed():
    # 3 picks — below MIN_WEEKLY_PICKS(5) — row should have opacity style
    df = _make_df(
        ["2026-04-07"] * 3,
        ["WIN", "WIN", "LOSS"],
        [0.65, 0.65, 0.65],
    )
    html = build_weekly_calibration(df)
    assert "opacity" in html


def test_gap_color_green():
    # win rate 60%, model 63% -> gap = -3 pp; |gap| = 3 pp < 5 pp -> green
    df = _make_df(
        ["2026-04-07"] * 10,
        ["WIN"] * 6 + ["LOSS"] * 4,
        [0.63] * 10,
    )
    html = build_weekly_calibration(df)
    assert "#00e5a0" in html


def test_gap_color_red():
    # win rate 40%, model 75% -> gap -35% -> |gap| > 10% -> red
    df = _make_df(
        ["2026-04-07"] * 10,
        ["WIN"] * 4 + ["LOSS"] * 6,
        [0.75] * 10,
    )
    html = build_weekly_calibration(df)
    assert "#f04e4e" in html


def test_empty_df_returns_empty_string():
    html = build_weekly_calibration(pd.DataFrame())
    assert html == ""


def test_missing_fair_prob_raw_rows_excluded():
    df = _make_df(
        ["2026-04-07"] * 10,
        ["WIN"] * 6 + ["LOSS"] * 4,
        [0.70] * 8 + [None, None],  # 2 rows with no fair_prob_raw
    )
    # Should not crash; rows without fair_prob_raw excluded from Model% calc
    html = build_weekly_calibration(df)
    assert "<table" in html


def test_missing_required_column_returns_empty():
    df = pd.DataFrame({"game_date": ["2026-04-07"], "hit_result": ["WIN"]})  # no bet_odds
    assert build_weekly_calibration(df) == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
