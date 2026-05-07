import os, sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "Scripts"))
from analyze_accuracy_mlb import (
    load_and_filter, american_to_profit, pnl,
    calibration_curve, calibration_grade,
    book_roi,
    direction_edge,
)


def make_fixture():
    return pd.DataFrame({
        "stat":           ["hits",  "tb",    "rbi",   "k",     "k"    ],
        "side":           ["OVER",  "OVER",  "UNDER", "OVER",  "UNDER"],
        "bet_odds":       [-120.0,  -130.0,  200.0,   -180.0,  -250.0 ],
        "hit_result":     ["WIN",   "LOSS",  "WIN",   "WIN",   "LOSS" ],
        "fair_prob":      [0.62,    0.57,    0.71,    0.64,    0.68   ],
        "pricing_source": ["bovada","bovada","betonline","bovada","bovada"],
        "game_date":      ["2026-04-01"] * 5,
    })


def test_excludes_rbi():
    result = load_and_filter(make_fixture())
    assert "rbi" not in result["stat"].values


def test_drops_heavy_juice():
    # k UNDER at -250 exceeds HEAVY_JUICE=-200 and should be dropped
    result = load_and_filter(make_fixture())
    assert not any((result["stat"] == "k") & (result["bet_odds"] == -250.0))


def test_drops_hits_juice():
    # hits at -120 is worse than HITS_MAX_JUICE=-115, should be dropped
    result = load_and_filter(make_fixture())
    assert not any(result["stat"] == "hits")


def test_keeps_k_at_valid_odds():
    # k OVER at -180 is within HEAVY_JUICE=-200, should be kept
    result = load_and_filter(make_fixture())
    assert any((result["stat"] == "k") & (result["bet_odds"] == -180.0))


def test_american_to_profit_negative():
    assert abs(american_to_profit(-200) - 0.5) < 0.001


def test_american_to_profit_positive():
    assert abs(american_to_profit(150) - 1.5) < 0.001


def test_pnl_win():
    row = pd.Series({"bet_odds": -110.0, "hit_result": "WIN"})
    assert abs(pnl(row) - (100 * 100 / 110)) < 0.01


def test_pnl_loss():
    row = pd.Series({"bet_odds": -110.0, "hit_result": "LOSS"})
    assert pnl(row) == -100


def test_pnl_no_action():
    row = pd.Series({"bet_odds": -110.0, "hit_result": "NO ACTION"})
    assert pnl(row) == 0.0


def test_calibration_curve_bucket_values():
    # 30 picks all in 0.60–0.65 bucket, 60% actual win rate
    df = pd.DataFrame({
        "stat":           ["k"] * 30,
        "side":           ["OVER"] * 30,
        "bet_odds":       [-110.0] * 30,
        "hit_result":     ["WIN"] * 18 + ["LOSS"] * 12,
        "fair_prob":      [0.62] * 30,
        "pricing_source": ["bovada"] * 30,
    })
    rows = calibration_curve(df)
    bucket = next(r for r in rows if r["bucket"] == "0.60–0.65")
    assert bucket["n"] == 30
    assert abs(bucket["actual"] - 0.60) < 0.001
    assert abs(bucket["predicted"] - 0.625) < 0.001
    assert not bucket["low_conf"]


def test_calibration_curve_low_conf_flag():
    # 10 picks in 0.70–0.75 bucket — below MIN_BUCKET=20
    df = pd.DataFrame({
        "stat":           ["k"] * 10,
        "side":           ["OVER"] * 10,
        "bet_odds":       [-110.0] * 10,
        "hit_result":     ["WIN"] * 6 + ["LOSS"] * 4,
        "fair_prob":      [0.72] * 10,
        "pricing_source": ["bovada"] * 10,
    })
    rows = calibration_curve(df)
    bucket = next(r for r in rows if "0.70" in r["bucket"])
    assert bucket["low_conf"] is True


def test_calibration_grade_A():
    rows = [{"gap": 0.02, "low_conf": False}, {"gap": -0.01, "low_conf": False}]
    assert calibration_grade(rows) == "A"


def test_calibration_grade_D():
    rows = [{"gap": 0.12, "low_conf": False}]
    assert calibration_grade(rows) == "D"


def test_calibration_grade_skips_low_conf():
    rows = [{"gap": 0.20, "low_conf": True}]  # only low-conf buckets
    assert calibration_grade(rows) == "N/A"


def test_book_roi_basic():
    df = pd.DataFrame({
        "stat":           ["k",      "k",      "k",         "k"       ],
        "side":           ["OVER",   "OVER",   "OVER",      "OVER"    ],
        "bet_odds":       [150.0,    150.0,    -110.0,      -110.0    ],
        "hit_result":     ["WIN",    "WIN",    "LOSS",      "LOSS"    ],
        "fair_prob":      [0.60] * 4,
        "pricing_source": ["bovada", "bovada", "betonline", "betonline"],
    })
    rows = book_roi(df)
    bovada    = next(r for r in rows if r["book"] == "bovada")
    betonline = next(r for r in rows if r["book"] == "betonline")
    assert bovada["n"] == 2
    assert abs(bovada["profit"] - 300.0) < 0.01   # 2 × $100 × 1.5
    assert abs(betonline["profit"] - (-200.0)) < 0.01
    # sorted by ROI desc — bovada first
    assert rows[0]["book"] == "bovada"


def test_book_roi_skips_nan_source():
    df = pd.DataFrame({
        "stat":           ["k"],
        "side":           ["OVER"],
        "bet_odds":       [-110.0],
        "hit_result":     ["WIN"],
        "fair_prob":      [0.60],
        "pricing_source": [np.nan],
    })
    rows = book_roi(df)
    assert len(rows) == 0


def test_book_roi_missing_column():
    df = pd.DataFrame({
        "stat":       ["k"],
        "side":       ["OVER"],
        "bet_odds":   [-110.0],
        "hit_result": ["WIN"],
        "fair_prob":  [0.60],
    })
    rows = book_roi(df)
    assert rows == []


def test_direction_edge_filters_min_picks():
    # Only 5 picks — below MIN_DIRECTION=10, should produce no rows
    df = pd.DataFrame({
        "stat":           ["k"] * 5,
        "side":           ["OVER"] * 5,
        "bet_odds":       [-110.0] * 5,
        "hit_result":     ["WIN"] * 3 + ["LOSS"] * 2,
        "fair_prob":      [0.60] * 5,
        "pricing_source": ["bovada"] * 5,
    })
    assert direction_edge(df) == []


def test_direction_edge_sorted_by_roi():
    # k OVER: 16 wins / 20 picks at +150 = positive ROI
    # hits UNDER: 8 wins / 20 picks at -110 = negative ROI
    df = pd.DataFrame({
        "stat":           ["k"]    * 20 + ["hits"] * 20,
        "side":           ["OVER"] * 20 + ["UNDER"] * 20,
        "bet_odds":       [150.0]  * 20 + [-110.0] * 20,
        "hit_result":     ["WIN"] * 16 + ["LOSS"] * 4 + ["WIN"] * 8 + ["LOSS"] * 12,
        "fair_prob":      [0.62] * 40,
        "pricing_source": ["bovada"] * 40,
    })
    rows = direction_edge(df)
    assert len(rows) == 2
    assert rows[0]["stat_side"] == "k OVER"
    assert rows[-1]["stat_side"] == "hits UNDER"
    assert rows[0]["roi"] > rows[-1]["roi"]


def test_direction_edge_win_rate_calculation():
    df = pd.DataFrame({
        "stat":           ["k"] * 20,
        "side":           ["OVER"] * 20,
        "bet_odds":       [-110.0] * 20,
        "hit_result":     ["WIN"] * 13 + ["LOSS"] * 7,
        "fair_prob":      [0.65] * 20,
        "pricing_source": ["bovada"] * 20,
    })
    rows = direction_edge(df)
    assert len(rows) == 1
    assert abs(rows[0]["win_rate"] - 0.65) < 0.001
    assert rows[0]["n"] == 20


if __name__ == "__main__":
    test_excludes_rbi()
    test_drops_heavy_juice()
    test_drops_hits_juice()
    test_keeps_k_at_valid_odds()
    test_american_to_profit_negative()
    test_american_to_profit_positive()
    test_pnl_win()
    test_pnl_loss()
    test_pnl_no_action()
    test_calibration_curve_bucket_values()
    test_calibration_curve_low_conf_flag()
    test_calibration_grade_A()
    test_calibration_grade_D()
    test_calibration_grade_skips_low_conf()
    test_book_roi_basic()
    test_book_roi_skips_nan_source()
    test_book_roi_missing_column()
    test_direction_edge_filters_min_picks()
    test_direction_edge_sorted_by_roi()
    test_direction_edge_win_rate_calculation()
    print("All tests passed (Task 1 + Task 2 + Task 3 + Task 4)")
