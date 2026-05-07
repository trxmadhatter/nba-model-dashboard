import os, sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "Scripts"))
from analyze_accuracy_mlb import load_and_filter, american_to_profit, pnl


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
    print("All Task 1 tests passed")
