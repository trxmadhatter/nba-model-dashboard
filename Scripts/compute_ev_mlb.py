"""
compute_ev_mlb.py

Full production EV engine for MLB player props and HR props.

Signals applied per pick:
  - Poisson distribution for discrete counting stats
  - Umpire K-rate / hit-rate multiplier
  - Bullpen fatigue boost: opponent bullpen >120% avg 48h usage → +5% hitting proj
  - Calibrated fair_prob from build_calibration_mlb.py (applied when file exists)
  - Sanity filters: proj/line ratio, prob range, plus-money suspicion gate
  - Juice caps: global -200 hard drop; hits capped at -115
  - Banned bets: TB OVER, K OVER (negative ROI by backtest)
  - Per-game limit: max 2 ELITE/STRONG picks per game
  - Per-player limit: max 1 ELITE/STRONG pick per player
"""

from __future__ import annotations

import math
import os
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from scipy.stats import nbinom as _nbinom

import numpy as np
import pandas as pd

from bet_rating import apply_ratings, RATING_DISPLAY
from config_mlb import (
    BOVADA_BOOKS, CONSENSUS_BOOKS,
    ALLOWED_MARKETS, EXCLUDED_MARKETS, BANNED_MARKET_SIDE_PAIRS,
    DEFAULT_MAX_JUICE, JUICE_CAPS_BY_MARKET,
    REQUIRE_CONSENSUS_MARKETS,
    MAX_BETS_PER_GAME, MAX_BETS_PER_PLAYER,
    FAIR_PROB_ELITE_MAX,
    ELITE_STAT_CAPS, DEFAULT_ELITE_STAT_CAP,
    MAX_ELITE_PICKS, MAX_STRONG_PICKS,
    MIN_EDGE, MIN_BOOK_COUNT, MIN_EV_TO_FLAG,
    SANITY_PROB_MIN, SANITY_PROB_MAX,
    SANITY_PROJ_LINE_MAX_RATIO, SANITY_PROJ_LINE_MIN_RATIO,
    SANITY_PLUS_MONEY_PROB_GATE,
    K_BIAS_CORRECTION,
    KELLY_FRACTION, MAX_BET_FRAC, BANKROLL, BETTABLE_RATINGS,
)

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROPS_FILE        = ROOT / "Data/mlb/lines/props_today_mlb.csv"
BATTER_PROJ_FILE  = ROOT / "Data/mlb/processed/mlb_batter_projections.csv"
PITCHER_PROJ_FILE = ROOT / "Data/mlb/processed/mlb_pitcher_projections.csv"
UMPIRES_FILE      = ROOT / "Data/mlb/lines/umpires_today_mlb.csv"
GAMELOGS_FILE     = ROOT / "Data/mlb/raw/mlb_pitcher_gamelogs_all.csv"
CALIBRATION_FILE  = ROOT / "Data/mlb/processed/calibration_mlb.csv"
ISOTONIC_FILE     = ROOT / "Data/mlb/processed/calibration_isotonic_mlb.csv"
AUTO_BANS_FILE    = ROOT / "Data/mlb/processed/auto_bans_mlb.csv"
OUT_FILE          = ROOT / "Data/mlb/processed/ev_props_today_mlb.csv"
HR_PROPS_FILE     = ROOT / "Data/mlb/processed/hr_props_mlb.csv"
DAILY_LOG_FILE    = ROOT / "Data/mlb/history/daily_picks_log_mlb.csv"
HR_HISTORY_FILE   = ROOT / "Data/mlb/history/hr_history_mlb.csv"
LINE_MOVE_FILE    = ROOT / "Data/mlb/lines/line_movement_mlb.csv"
IL_FILE           = ROOT / "Data/mlb/processed/il_players_mlb.csv"

ALLOWED_BOOKS = BOVADA_BOOKS | CONSENSUS_BOOKS

# hrb uses Negative Binomial (overdispersion ratio 2.31 from 159 graded picks;
# projection scale 0.858 corrects +0.385 upward bias in PROJ_HRB)
HRB_VAR_MEAN_RATIO = 2.31
HRB_PROJ_SCALE     = 0.858

POISSON_STATS = {"k", "hits", "sb", "h_allowed", "er", "bb_allowed"}
NEGBIN_STATS  = {"hrb"}
PITCHER_STATS = {"k", "h_allowed", "er", "bb_allowed", "outs"}

PROJ_COL: dict[str, str] = {
    "hits":       "PROJ_H",
    "hr":         "PROJ_HR",
    "tb":         "PROJ_TB",
    "hrb":        "PROJ_HRB",
    "sb":         "PROJ_SB",
    "runs":       "PROJ_R",
    "rbi":        "PROJ_RBI",
    "k":          "PROJ_K",
    "h_allowed":  "PROJ_H_ALLOWED",
    "er":         "PROJ_ER_ALLOWED",
    "bb_allowed": "PROJ_BB_ALLOWED",
    "outs":       "PROJ_IP",
}


# ── Utility ───────────────────────────────────────────────────────────────────

def normalize(text) -> str:
    if pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in text if not unicodedata.combining(ch)).lower().strip()


def american_to_implied(odds: float) -> float:
    odds = float(odds)
    return abs(odds) / (abs(odds) + 100) if odds < 0 else 100.0 / (odds + 100)


def implied_to_american(p: float) -> float:
    p = max(0.001, min(0.999, float(p)))
    return round(-p / (1 - p) * 100) if p > 0.5 else round((1 - p) / p * 100)


def american_to_profit(odds: float) -> float:
    """Profit per $1 wagered."""
    odds = float(odds)
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def compute_kelly(fair_prob: float, bet_odds: float) -> float:
    """Full Kelly fraction — caller applies KELLY_FRACTION and MAX_BET_FRAC."""
    p = max(0.0, min(1.0, float(fair_prob)))
    if pd.isna(bet_odds) or bet_odds == 0:
        return 0.0
    b = american_to_profit(float(bet_odds))
    if b <= 0:
        return 0.0
    return max(0.0, (b * p - (1 - p)) / b)


def poisson_over(lam: float, line: float) -> float:
    """P(X > line) using Poisson(lam). Line is typically X.5."""
    if pd.isna(lam) or lam <= 0:
        return 0.5
    k = int(math.floor(line))
    log_lam, log_fact, cdf = math.log(lam), 0.0, 0.0
    for i in range(k + 1):
        if i > 0:
            log_fact += math.log(i)
        cdf += math.exp(i * log_lam - lam - log_fact)
    return max(0.0, min(1.0, 1.0 - cdf))


def negbin_over(mu: float, line: float, var_mean_ratio: float) -> float:
    """P(X > line) using Negative Binomial with overdispersion.
    var_mean_ratio = Var/Mean from graded history (>1 = overdispersed vs Poisson)."""
    if pd.isna(mu):
        return 0.5
    if mu <= 0:
        return 0.0
    k = int(math.floor(line))
    var = mu * var_mean_ratio
    if var <= mu:
        return poisson_over(mu, line)
    r = mu ** 2 / (var - mu)       # dispersion parameter
    p = r / (r + mu)               # success probability in NegBin parameterization
    return float(_nbinom.sf(k, n=r, p=p))


def remove_vig(over_price: float, under_price: float) -> tuple[float, float]:
    """Return (fair_over_prob, fair_under_prob) after removing vig."""
    ov_imp = american_to_implied(over_price)
    un_imp = american_to_implied(under_price)
    total  = ov_imp + un_imp
    if total <= 0:
        return 0.5, 0.5
    return ov_imp / total, un_imp / total


def fmt_odds(v) -> str:
    try:
        v = float(v)
        return f"+{int(v)}" if v > 0 else str(int(v))
    except Exception:
        return "-"


# ── Calibration ───────────────────────────────────────────────────────────────

_STAT_ABBR = {
    "hits": "H", "hr": "HR", "hrb": "HRB", "k": "K",
    "tb": "TB", "runs": "R", "rbi": "RBI", "outs": "Outs",
}

def build_reason(row) -> str:
    parts: list[str] = []

    # Projection vs line
    abbr = _STAT_ABBR.get(str(row.get("stat", "")).lower(), str(row.get("stat", "")).upper())
    try:
        proj = float(row["projection"])
        line = float(row["line"])
        edge = float(row["edge"])
        parts.append(f"Proj {proj:.1f} {abbr} vs line {line:.1f} ({edge:+.1f})")
    except (TypeError, ValueError, KeyError):
        pass

    # Pitcher matchup (batter props only — skip when hand is blank)
    hand    = str(row.get("throw_hand_faced", "")).strip()
    starter = str(row.get("opp_starter", "")).strip()
    if hand in ("L", "R"):
        name = starter.split()[-1] if starter and starter not in ("nan", "") else "starter"
        parts.append(f"vs {hand}HP {name}")

    # Line movement
    lm    = str(row.get("line_move", "")).strip().upper()
    shift = row.get("line_move_shift")
    if lm in ("UP", "DOWN", "SHARP"):
        try:
            parts.append(f"Line {lm.lower()} {float(shift):+.2f}")
        except (TypeError, ValueError):
            parts.append(f"Line moved {lm.lower()}")

    # Book count
    try:
        books = int(row["book_count"])
        if books >= 3:
            parts.append(f"{books} books")
    except (TypeError, ValueError, KeyError):
        pass

    # Calibrated flag
    if row.get("calibrated"):
        parts.append("calibrated")

    return ". ".join(parts) + "." if parts else ""


def load_calibration() -> pd.DataFrame:
    """
    Load calibration table from build_calibration_mlb.py output.
    Returns empty DataFrame if file missing or malformed — caller falls back to raw fair_prob.
    """
    if not CALIBRATION_FILE.exists():
        return pd.DataFrame()
    try:
        cal = pd.read_csv(CALIBRATION_FILE)
        required = {"stat", "prob_min", "prob_max", "actual_win_rate"}
        if not required.issubset(cal.columns):
            print("  Calibration file missing required columns — using raw fair_prob")
            return pd.DataFrame()
        # Normalize so matching is case/whitespace insensitive
        cal["stat"] = cal["stat"].astype(str).str.strip().str.lower()
        if "side" in cal.columns:
            cal["side"] = cal["side"].astype(str).str.strip().str.upper()
        cal["prob_min"]        = pd.to_numeric(cal["prob_min"],        errors="coerce")
        cal["prob_max"]        = pd.to_numeric(cal["prob_max"],        errors="coerce")
        cal["actual_win_rate"] = pd.to_numeric(cal["actual_win_rate"], errors="coerce")
        cal = cal.dropna(subset=["prob_min", "prob_max", "actual_win_rate"])
        print(f"  Calibration loaded: {len(cal)} buckets ({cal['stat'].nunique()} stats)")
        return cal
    except Exception as e:
        print(f"  Calibration load failed: {e} — using raw fair_prob")
        return pd.DataFrame()


def load_isotonic_calibration(path: str | None = None) -> dict:
    """
    Load isotonic calibration curves from calibration_isotonic_mlb.csv.
    Returns dict: (stat, side) -> (x_array, y_array) for use with np.interp.
    Returns empty dict if file missing or malformed.
    """
    csv_path = Path(path) if path else ISOTONIC_FILE
    if not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path)
        if not {"stat", "side", "x", "y"}.issubset(df.columns):
            print("  Isotonic calibration file missing required columns — skipping")
            return {}
        df["stat"] = df["stat"].astype(str).str.strip().str.lower()
        df["side"] = df["side"].astype(str).str.strip().str.upper()
        df["x"]    = pd.to_numeric(df["x"], errors="coerce")
        df["y"]    = pd.to_numeric(df["y"], errors="coerce")
        df = df.dropna(subset=["x", "y"])
        iso_map: dict = {}
        for (stat, side), group in df.groupby(["stat", "side"]):
            g = group.sort_values("x")
            iso_map[(str(stat), str(side))] = (g["x"].values, g["y"].values)
        print(f"  Isotonic calibration loaded: {len(iso_map)} curves")
        return iso_map
    except Exception as e:
        print(f"  Isotonic calibration load failed: {e} — skipping")
        return {}


def apply_calibration(fair_prob: float, stat: str, cal_df: pd.DataFrame,
                      side: str = "", iso_map: dict | None = None) -> float:
    """
    Return calibrated win probability for (stat, side, fair_prob).

    Priority:
      1. Isotonic: iso_map[(stat, side)]     — np.interp
      2. Isotonic: iso_map[(stat, "BOTH")]   — np.interp
      3. Isotonic: iso_map[("_global_", "_GLOBAL_")] — np.interp
      4. Bucket: cal_df lookup (existing logic, unchanged)
    Falls back to raw fair_prob if nothing matches.
    """
    if iso_map is None:
        iso_map = {}

    stat_n = str(stat).strip().lower()
    side_n = str(side).strip().upper()

    # Isotonic lookup — three levels of fallback
    if iso_map:
        for key in [(stat_n, side_n), (stat_n, "BOTH"), ("_global_", "_GLOBAL_")]:
            if key in iso_map:
                xs, ys = iso_map[key]
                return float(np.interp(fair_prob, xs, ys))

    # Existing bucket logic (unchanged)
    if cal_df.empty:
        return fair_prob

    prob_mask = (cal_df["prob_min"] <= fair_prob) & (cal_df["prob_max"] > fair_prob)
    has_side_col = "side" in cal_df.columns

    # Stats that need thin-bucket protection: don't apply calibration from small samples
    ELITE_SENSITIVE_STATS = {"hits", "tb"}
    ELITE_MIN_BUCKET = 75

    if has_side_col:
        # Side-specific file: only match when side is known and found
        if side_n:
            match = cal_df[prob_mask & (cal_df["stat"] == stat_n) & (cal_df["side"] == side_n)]
            if not match.empty:
                row = match.iloc[0]
                # For unreliable stats, skip calibration buckets with too few picks
                if stat_n in ELITE_SENSITIVE_STATS:
                    n = int(row.get("n_picks", 0))
                    if n < ELITE_MIN_BUCKET:
                        return fair_prob
                return float(row["actual_win_rate"])
        # No match (or no side provided) — never cross-apply another side's calibration
        return fair_prob

    # Stat-only fallback (backward compat with pre-side calibration files)
    match = cal_df[prob_mask & (cal_df["stat"] == stat_n)]
    if not match.empty:
        return float(match.iloc[0]["actual_win_rate"])

    return fair_prob


def load_auto_bans() -> set[tuple[str, str]]:
    """
    Load auto-detected banned stat+side pairs from auto_ban_detection_mlb.py output.
    Merges with BANNED_MARKET_SIDE_PAIRS from config. Safe fallback if file missing.
    """
    bans = set(BANNED_MARKET_SIDE_PAIRS)
    if not AUTO_BANS_FILE.exists():
        return bans
    try:
        df = pd.read_csv(AUTO_BANS_FILE)
        auto = df[df["auto_banned"] == True]
        new_bans = set(zip(auto["stat"].str.lower(), auto["side"].str.upper()))
        added = new_bans - bans
        if added:
            print(f"  Auto-bans loaded: {sorted(added)}")
        bans |= new_bans
    except Exception as e:
        print(f"  Auto-ban load failed: {e} — using config bans only")
    return bans


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_umpire_map() -> dict[tuple[str, str], dict]:
    """Returns {(home_norm, away_norm): {k_mult, hit_mult}}."""
    if not UMPIRES_FILE.exists():
        return {}
    df = pd.read_csv(UMPIRES_FILE)
    df.columns = [c.strip().lower() for c in df.columns]
    result: dict[tuple[str, str], dict] = {}
    for _, r in df.iterrows():
        key = (normalize(r.get("home_team", "")), normalize(r.get("away_team", "")))
        result[key] = {
            "k_mult":   float(r.get("k_mult",  1.0)),
            "hit_mult": float(r.get("hit_mult", 1.0)),
        }
    return result


def load_bullpen_fatigue() -> dict[str, float]:
    """
    Per-team bullpen fatigue multiplier from last 48h pitcher gamelogs.
    Tired bullpens (>120% avg pitch count) give a 1.05 boost to opposing hitters.
    """
    if not GAMELOGS_FILE.exists():
        return {}
    try:
        logs = pd.read_csv(GAMELOGS_FILE, low_memory=False)
        logs.columns = [c.strip().upper() for c in logs.columns]
        logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"], errors="coerce")
        cutoff = pd.Timestamp.now() - timedelta(hours=48)
        recent = logs[
            (logs["GAME_DATE"] >= cutoff) &
            (logs["IS_STARTER"].astype(str).str.strip() == "0")
        ].copy()
        if recent.empty:
            return {}
        usage   = recent.groupby("TEAM")["PITCHES"].sum()
        avg_use = usage.mean()
        if avg_use <= 0:
            return {}
        return {
            normalize(str(team)): (1.05 if pitches > avg_use * 1.2 else 1.0)
            for team, pitches in usage.items()
        }
    except Exception as e:
        print(f"  Bullpen fatigue load failed: {e}")
        return {}


def load_line_movement() -> dict[tuple[str, str, float, str], tuple[str, float]]:
    """Load line movement signals keyed by (player_norm, stat, line, side)."""
    if not LINE_MOVE_FILE.exists():
        return {}
    try:
        df = pd.read_csv(LINE_MOVE_FILE)
        required = {"player_name", "stat", "line", "over_signal", "under_signal"}
        if not required.issubset(df.columns):
            print("  Line movement file missing required columns — skipping")
            return {}
        result: dict[tuple[str, str, float, str], tuple[str, float]] = {}
        for _, r in df.iterrows():
            player = normalize(str(r.get("player_name", "")))
            stat   = str(r.get("stat", "")).lower().strip()
            try:
                line = round(float(r.get("line", np.nan)), 1)
            except (ValueError, TypeError):
                continue
            if not player or not stat or pd.isna(line):
                continue
            over_sig    = str(r.get("over_signal",  "NEUTRAL"))
            under_sig   = str(r.get("under_signal", "NEUTRAL"))
            over_shift  = r.get("over_shift",  np.nan)
            under_shift = r.get("under_shift", np.nan)
            result[(player, stat, line, "OVER")]  = (over_sig,  float(over_shift)  if pd.notna(over_shift)  else np.nan)
            result[(player, stat, line, "UNDER")] = (under_sig, float(under_shift) if pd.notna(under_shift) else np.nan)
        print(f"  Line movement loaded: {len(df)} props tracked")
        return result
    except Exception as e:
        print(f"  Line movement load failed: {e} — using NO_DATA")
        return {}


def load_il_players() -> set[str]:
    """Return normalized names of all current IL players."""
    if not IL_FILE.exists():
        return set()
    try:
        df = pd.read_csv(IL_FILE)
        if "player_name_norm" not in df.columns:
            return set()
        if "il_type" in df.columns:
            df = df[df["il_type"].str.strip().str.lower() != "active"]
        return set(df["player_name_norm"].dropna().str.lower().str.strip())
    except Exception as e:
        print(f"  IL load failed: {e} — skipping IL filter")
        return set()


# ── Book aggregation ──────────────────────────────────────────────────────────

def aggregate_books(group: pd.DataFrame) -> dict:
    """
    Aggregate all book prices for one (player, stat, line) group.
    Returns bovada odds, consensus odds, best-book odds, book count,
    and bovada_book (which offshore book was actually used).
    """
    bk_col = "bookmaker_key" if "bookmaker_key" in group.columns else "bookmaker"

    book_prices: dict[str, tuple[float, float]] = {}
    for _, r in group.iterrows():
        bk = str(r.get(bk_col, "")).lower().strip()
        if bk not in ALLOWED_BOOKS:
            continue
        ov = r.get("over_price", np.nan)
        un = r.get("under_price", np.nan)
        try:
            ov = float(ov) if pd.notna(ov) else np.nan
            un = float(un) if pd.notna(un) else np.nan
        except (ValueError, TypeError):
            continue
        if pd.notna(ov) or pd.notna(un):
            book_prices[bk] = (ov, un)

    result: dict = {
        "bovada_over": np.nan, "bovada_under": np.nan, "bovada_book": "",
        "consensus_over": np.nan, "consensus_under": np.nan,
        "best_book_over": "", "best_book_price_over": np.nan,
        "best_book_under": "", "best_book_price_under": np.nan,
        "book_count": len(book_prices),
    }

    # Offshore: first matching book (bovada preferred, betonlineag fallback)
    for bk in BOVADA_BOOKS:
        if bk in book_prices:
            result["bovada_over"], result["bovada_under"] = book_prices[bk]
            result["bovada_book"] = bk  # track which offshore book was used
            break

    # Consensus: vig-removed average across sharp books
    sharp_ov_imps, sharp_un_imps = [], []
    for bk in CONSENSUS_BOOKS:
        if bk in book_prices:
            ov, un = book_prices[bk]
            if pd.notna(ov) and pd.notna(un):
                f_ov, f_un = remove_vig(ov, un)
                sharp_ov_imps.append(f_ov)
                sharp_un_imps.append(f_un)

    if sharp_ov_imps:
        avg_ov = float(np.mean(sharp_ov_imps))
        avg_un = float(np.mean(sharp_un_imps))
        result["consensus_over"]  = implied_to_american(avg_ov)
        result["consensus_under"] = implied_to_american(avg_un)

    # Best book: highest payout (most positive american odds) for each side
    best_ov_price, best_un_price = -np.inf, -np.inf
    best_ov_bk,   best_un_bk    = "", ""
    for bk, (ov, un) in book_prices.items():
        if pd.notna(ov) and float(ov) > best_ov_price:
            best_ov_price, best_ov_bk = float(ov), bk
        if pd.notna(un) and float(un) > best_un_price:
            best_un_price, best_un_bk = float(un), bk

    if best_ov_bk:
        result["best_book_over"]       = best_ov_bk
        result["best_book_price_over"] = best_ov_price
    if best_un_bk:
        result["best_book_under"]       = best_un_bk
        result["best_book_price_under"] = best_un_price

    return result


# ── HR props ──────────────────────────────────────────────────────────────────

def compute_hr_props(batter_proj: pd.DataFrame, props: pd.DataFrame) -> pd.DataFrame:
    """Compute HR prop EV for all batters with HR lines available today."""
    hr_props = props[props["stat"] == "hr"].copy()
    if hr_props.empty or batter_proj.empty:
        return pd.DataFrame()

    rows = []
    for (pname_norm, _line), grp in hr_props.groupby(["player_name_norm", "line"]):
        match = batter_proj[batter_proj["PLAYER_NAME_NORM"] == pname_norm]
        if match.empty:
            continue
        p       = match.iloc[0]
        proj_hr = float(p.get("PROJ_HR", np.nan))
        if pd.isna(proj_hr) or proj_hr <= 0:
            continue

        agg        = aggregate_books(grp)
        book_count = agg["book_count"]
        best_bk    = agg["best_book_over"]
        best_price = agg["best_book_price_over"]

        if pd.isna(best_price) or book_count == 0:
            continue

        prob_hr = poisson_over(proj_hr, float(_line))
        implied = american_to_implied(best_price)
        b       = american_to_profit(best_price)
        ev_pct  = round(((prob_hr * b) - (1 - prob_hr)) * 100, 2)

        rows.append({
            "player_name":    grp.iloc[0]["player_name"],
            "matchup":        f"{grp.iloc[0].get('away_team','?')} @ {grp.iloc[0].get('home_team','?')}",
            "team_name":      str(p.get("TEAM_NAME", "")),
            "commence_time":  grp.iloc[0].get("commence_time", ""),
            "proj_hr":        round(proj_hr, 3),
            "adj_lambda":     round(proj_hr, 3),
            "prob_hr":        round(prob_hr, 4),
            "prob_hr_pct":    round(prob_hr * 100, 2),
            "best_odds":      best_price,
            "best_book":      best_bk,
            "implied_prob":   round(implied, 4),
            "ev_pct":         ev_pct,
            "positive_ev":    ev_pct > 0 and book_count >= MIN_BOOK_COUNT,
            "book_count":     book_count,
            "park_component": float(p.get("PARK_COMPONENT", 1.0)),
            "form_5g":        float(p.get("HR_AVG_5", 0.0)),
            "opp_starter":    str(p.get("OPP_STARTER", "")),
            "throw_hand":     str(p.get("THROW_HAND_FACED", "")),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("prob_hr", ascending=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("COMPUTING MLB EV PROPS")
    print("=" * 55)

    if not PROPS_FILE.exists():
        print(f"  Props file missing: {PROPS_FILE}")
        return
    if not BATTER_PROJ_FILE.exists():
        print(f"  Batter projections missing: {BATTER_PROJ_FILE}")
        return

    # ── Load calibration (safe fallback if missing) ───────────────────────────
    calibration = load_calibration()
    iso_map   = load_isotonic_calibration()
    calibration_active = not calibration.empty

    # ── Load auto-detected bans (merges with hardcoded config bans) ───────────
    active_bans = load_auto_bans()

    # ── Load props ────────────────────────────────────────────────────────────
    props = pd.read_csv(PROPS_FILE)
    props.columns = [c.strip().lower() for c in props.columns]
    props["player_name_norm"] = props["player_name"].apply(normalize)
    bk_col = "bookmaker_key" if "bookmaker_key" in props.columns else "bookmaker"
    props_allowed = props[props[bk_col].str.lower().str.strip().isin(ALLOWED_BOOKS)].copy()
    print(f"  Props: {len(props)} rows | {props['player_name'].nunique()} players "
          f"| {props['stat'].unique().tolist()} | {props[bk_col].nunique()} books")

    # ── Load projections ──────────────────────────────────────────────────────
    batter_proj  = pd.read_csv(BATTER_PROJ_FILE)
    pitcher_proj = pd.read_csv(PITCHER_PROJ_FILE) if PITCHER_PROJ_FILE.exists() else pd.DataFrame()
    batter_proj["PLAYER_NAME_NORM"]  = batter_proj["PLAYER_NAME"].apply(normalize)
    if not pitcher_proj.empty:
        pitcher_proj["PLAYER_NAME_NORM"] = pitcher_proj["PLAYER_NAME"].apply(normalize)
    print(f"  Batter proj: {len(batter_proj)} | Pitcher proj: {len(pitcher_proj)}")

    # ── Situational signals ───────────────────────────────────────────────────
    umpire_map   = load_umpire_map()
    bullpen_mult = load_bullpen_fatigue()
    line_move    = load_line_movement()
    il_players   = load_il_players()
    print(f"  Umpire map: {len(umpire_map)} games")
    if il_players:
        print(f"  IL players filtered: {len(il_players)}")
    fatigued = [t for t, m in bullpen_mult.items() if m > 1.0]
    print(f"  Bullpen fatigue (>120% avg): {fatigued if fatigued else 'none'}")

    # ── Evaluate each (player, stat, line) group ──────────────────────────────
    ev_rows = []
    skipped_sanity = 0

    for (pname_norm, stat, line), grp in props_allowed.groupby(
            ["player_name_norm", "stat", "line"], sort=False):
        line = round(float(line), 1)
        stat = str(stat).lower().strip()

        if stat in EXCLUDED_MARKETS:
            continue
        if stat not in ALLOWED_MARKETS:
            continue
        if pname_norm in il_players:
            continue

        proj_col = PROJ_COL.get(stat)
        if proj_col is None:
            continue

        proj_df = pitcher_proj if stat in PITCHER_STATS else batter_proj
        if proj_df.empty:
            continue
        match = proj_df[proj_df["PLAYER_NAME_NORM"] == pname_norm]
        if match.empty:
            continue

        row_p      = match.iloc[0]
        projection = row_p.get(proj_col, np.nan)

        if stat == "outs" and pd.notna(projection):
            projection = float(projection) * 3.0  # IP → outs recorded

        if pd.isna(projection) or float(projection) <= 0:
            continue
        projection = float(projection)

        # Stat-specific adjustments
        if stat == "k":
            projection += K_BIAS_CORRECTION

        # Umpire multiplier
        home_norm = normalize(grp.iloc[0].get("home_team", ""))
        away_norm = normalize(grp.iloc[0].get("away_team", ""))
        ump       = umpire_map.get((home_norm, away_norm), {})
        if stat == "k":
            projection *= ump.get("k_mult", 1.0)
        elif stat in ("hits", "tb", "hrb", "sb", "h_allowed", "er", "bb_allowed"):
            projection *= ump.get("hit_mult", 1.0)

        # Bullpen fatigue boost for hitters
        opp_team = normalize(str(row_p.get("OPPONENT", "")))
        fatigue  = bullpen_mult.get(opp_team, 1.0)
        if stat in ("hits", "tb", "hrb", "rbi", "runs", "hr") and fatigue > 1.0:
            projection *= fatigue

        # Sanity check 1: projection vs line ratio
        # Catches stale projections or wrong line data before we waste time on book agg
        if line > 0:
            ratio = projection / line
            if ratio > SANITY_PROJ_LINE_MAX_RATIO or ratio < SANITY_PROJ_LINE_MIN_RATIO:
                skipped_sanity += 1
                continue

        # Book aggregation
        agg          = aggregate_books(grp)
        book_count   = agg["book_count"]
        bovada_over  = agg["bovada_over"]
        bovada_under = agg["bovada_under"]
        consensus_ov = agg["consensus_over"]
        consensus_un = agg["consensus_under"]

        # Require offshore or 2+ books to proceed
        if book_count < MIN_BOOK_COUNT and pd.isna(bovada_over) and pd.isna(bovada_under):
            continue

        # For NegBin stats, side decision uses the bias-corrected projection so the
        # edge sign and the probability agree (raw proj could straddle the line
        # differently than adj proj, causing contradictory OVER label + sub-0.5 prob).
        proj_for_edge = projection * HRB_PROJ_SCALE if stat in NEGBIN_STATS else projection
        edge = round(proj_for_edge - line, 3)
        if abs(edge) < MIN_EDGE:
            continue

        recommendation = "OVER" if edge > 0 else "UNDER"

        # Hard ban: config bans + auto-detected negative-ROI combos
        if (stat, recommendation) in active_bans:
            continue

        # Model probability — distribution chosen per stat family.
        # Consensus is a quality gate only. Raw fair_prob is kept for reference.
        if stat in NEGBIN_STATS:
            # Negative Binomial: corrects for overdispersion (Var/Mean=2.31) and
            # upward projection bias (+0.385 avg) observed in 159 graded hrb picks.
            proj_adj        = projection * HRB_PROJ_SCALE
            model_prob_over = negbin_over(proj_adj, line, HRB_VAR_MEAN_RATIO)
        elif stat in POISSON_STATS:
            model_prob_over = poisson_over(projection, line)
        else:
            xbh_rate = float(row_p.get("XBH_RATE", np.nan)) if pd.notna(row_p.get("XBH_RATE")) else np.nan
            if stat == "tb" and pd.notna(xbh_rate):
                std_mult = max(0.80, min(1.30, 1.0 + (xbh_rate - 0.40) * 1.0))
            else:
                std_mult = 1.0
            std             = max(0.5, projection * 0.35 * std_mult)
            z               = (projection - line) / std
            model_prob_over = min(0.97, max(0.03, 0.5 + 0.5 * math.erf(z / math.sqrt(2))))

        fair_prob_raw = model_prob_over if recommendation == "OVER" else (1.0 - model_prob_over)

        # Sanity check 2: model probability range
        if fair_prob_raw < SANITY_PROB_MIN or fair_prob_raw > SANITY_PROB_MAX:
            skipped_sanity += 1
            continue

        # Consensus gate — required for non-exempt stats
        if pd.notna(consensus_ov) and pd.notna(consensus_un):
            has_consensus = True
        else:
            has_consensus = False
            if stat in REQUIRE_CONSENSUS_MARKETS:
                continue  # stale/thin market — skip

        # Determine bet price: offshore first, best book as fallback
        if recommendation == "OVER":
            best_book       = agg["best_book_over"]
            best_book_price = agg["best_book_price_over"]
            bovada_bet      = bovada_over
            consensus_odds  = consensus_ov
        else:
            best_book       = agg["best_book_under"]
            best_book_price = agg["best_book_price_under"]
            bovada_bet      = bovada_under
            consensus_odds  = consensus_un

        # Bovada-only pricing: skip if Bovada doesn't have this line
        bet_odds = bovada_bet
        if pd.isna(bet_odds):
            continue

        bet_book_actual = agg["bovada_book"] or "bovada"
        pricing_source  = "offshore"

        # Sanity check 3: plus-money + high model prob without consensus
        # If book offers plus-money AND model is very confident AND no consensus exists,
        # the line is likely stale or wrong — skip it
        if float(bet_odds) >= 0 and fair_prob_raw >= SANITY_PLUS_MONEY_PROB_GATE and not has_consensus:
            skipped_sanity += 1
            continue

        # Per-market juice cap (e.g., hits capped at -115; config_mlb.py)
        juice_cap = JUICE_CAPS_BY_MARKET.get(stat, DEFAULT_MAX_JUICE)
        if float(bet_odds) < juice_cap:
            continue


        # Apply calibration — adjusts fair_prob to match historical actual win rates.
        # Side-specific (OVER/UNDER) when available; falls back to stat-only, then raw.
        # hrb is excluded: its calibration buckets were built on old Poisson/capped picks
        # and would double-correct against the new NegBin model. Re-enable once enough
        # NegBin-based graded picks accumulate (~300+).
        if stat == "hrb":
            fair_prob = fair_prob_raw
        else:
            fair_prob = apply_calibration(fair_prob_raw, stat, calibration, side=recommendation, iso_map=iso_map)

        # EV using calibrated probability
        b      = american_to_profit(float(bet_odds))
        ev_pct = round(((fair_prob * b) - (1 - fair_prob)) * 100, 2)

        lm_signal, lm_shift = line_move.get(
            (pname_norm, stat, line, recommendation),
            ("NO_DATA", np.nan),
        )

        kelly_raw = compute_kelly(fair_prob, float(bet_odds))
        k_frac    = min(kelly_raw * KELLY_FRACTION, MAX_BET_FRAC)

        ev_rows.append({
            # Identity
            "player_name":        grp.iloc[0]["player_name"],
            "stat":               stat,
            "line":               line,
            "side":               recommendation,
            # Projection
            "projection":         round(projection, 3),
            "edge":               edge,
            # Probability (keep both for comparison and debugging)
            "fair_prob":          round(fair_prob, 4),           # calibrated (used for EV/rating)
            "fair_prob_raw":      round(fair_prob_raw, 4),       # raw model output
            "calibrated":         calibration_active and (fair_prob != fair_prob_raw),
            # EV
            "ev_pct":             ev_pct,
            "positive_ev":        ev_pct >= MIN_EV_TO_FLAG and book_count >= MIN_BOOK_COUNT,
            # Kelly sizing (bet_size zeroed for non-bettable tiers after ratings assigned)
            "kelly_pct":          round(kelly_raw * 100, 2),
            "bet_size":           round(k_frac * BANKROLL, 2),
            # Pricing — bet_book tracks the actual source of bet_odds
            "bet_odds":           float(bet_odds),
            "bet_book":           bet_book_actual,   # actual book for bet_odds
            "pricing_source":     pricing_source,    # "offshore" or "best_book"
            "bovada_over":        bovada_over,
            "bovada_under":       bovada_under,
            "best_book":          best_book,          # best available price (display)
            "best_book_price":    best_book_price,
            "consensus_odds":     consensus_odds,
            "book_count":         book_count,
            # Context
            "opp_starter":        str(row_p.get("OPP_STARTER", "")),
            "throw_hand_faced":   str(row_p.get("THROW_HAND_FACED", "")),
            "confidence_tier":    str(row_p.get("CONFIDENCE", "")),
            "line_move":          lm_signal,
            "line_move_shift":    lm_shift,
            "matchup":            f"{grp.iloc[0].get('away_team','?')} @ {grp.iloc[0].get('home_team','?')}",
            "home_team":          grp.iloc[0].get("home_team", ""),
            "away_team":          grp.iloc[0].get("away_team", ""),
            "commence_time":      grp.iloc[0].get("commence_time", ""),
            "has_consensus":      has_consensus,
        })

    if skipped_sanity:
        print(f"  Sanity filters rejected: {skipped_sanity} picks")

    if not ev_rows:
        print("  No EV rows generated.")
        return

    ev_df = pd.DataFrame(ev_rows)
    ev_df = ev_df.drop_duplicates(subset=["player_name", "stat", "line", "side"])

    # ── Apply ratings ─────────────────────────────────────────────────────────
    # Ratings use the calibrated fair_prob column
    ev_df = apply_ratings(ev_df, ev_col="ev_pct", prob_col="fair_prob", book_col="book_count", stat_col="stat")

    if "BET_RATING" in ev_df.columns:
        # Degenerate line downgrade — prob above ceiling is likely a bad/stale book line
        degen_mask = (ev_df["fair_prob"] > FAIR_PROB_ELITE_MAX) & ev_df["BET_RATING"].isin(["ELITE", "STRONG"])
        ev_df.loc[degen_mask, "BET_RATING"] = "GOOD"
        if degen_mask.sum():
            print(f"  Degenerate prob downgrade (>{FAIR_PROB_ELITE_MAX}): {degen_mask.sum()} picks -> GOOD")

        # Hard drop: juice worse than DEFAULT_MAX_JUICE
        heavy_drop = ev_df["bet_odds"] < DEFAULT_MAX_JUICE
        if heavy_drop.sum():
            print(f"  Heavy juice drop (<{DEFAULT_MAX_JUICE}): {heavy_drop.sum()} picks removed")
        ev_df = ev_df[~heavy_drop].copy()

        # Model-only cap: picks without consensus → cap at AVERAGE (except proven stats)
        NO_CONSENSUS_ALLOWED = {"k", "hrb", "outs", "hits", "sb", "h_allowed", "er", "bb_allowed"}
        if "has_consensus" in ev_df.columns:
            drop_model_mask = (~ev_df["has_consensus"]) & ~ev_df["stat"].isin(NO_CONSENSUS_ALLOWED)
            ev_df = ev_df[~drop_model_mask].copy()
            if drop_model_mask.sum():
                print(f"  Model-only drop (unreliable stat): {drop_model_mask.sum()} removed")

            model_only_mask = (
                (~ev_df["has_consensus"]) &
                ev_df["BET_RATING"].isin(["ELITE", "STRONG", "GOOD"]) &
                ~ev_df["stat"].isin({"k", "hrb"})  # only K/HRB exempt (proven ROI)
            )
            ev_df.loc[model_only_mask, "BET_RATING"] = "AVERAGE"
            if model_only_mask.sum():
                print(f"  Model-only cap (no consensus): {model_only_mask.sum()} picks -> AVERAGE")

    ev_df = ev_df.sort_values("ev_pct", ascending=False)

    # ── hits / tb ELITE block (must run BEFORE per-stat ELITE cap) ──────────
    # hits: 43% win rate at ELITE (-4.56u). tb: 47% (-2.54u). Models too weak for ELITE.
    # tb OVER loses at every bettable tier (ELITE -10u, STRONG -6u, GOOD -7u) → cap at AVERAGE.
    if "BET_RATING" in ev_df.columns and "stat" in ev_df.columns and "side" in ev_df.columns:
        hits_tb_elite_mask = (
            ev_df["stat"].isin(["hits", "tb"]) &
            ev_df["BET_RATING"].isin(["ELITE"])
        )
        ev_df.loc[hits_tb_elite_mask, "BET_RATING"] = "STRONG"
        n_blocked = hits_tb_elite_mask.sum()
        if n_blocked:
            print(f"  hits/tb ELITE block: {n_blocked} -> STRONG (model unreliable at ELITE tier)")

        tb_over_mask = (
            (ev_df["stat"] == "tb") &
            (ev_df["side"] == "OVER") &
            ev_df["BET_RATING"].isin(["ELITE", "STRONG", "GOOD"])
        )
        ev_df.loc[tb_over_mask, "BET_RATING"] = "AVERAGE"
        n_tb_over = tb_over_mask.sum()
        if n_tb_over:
            print(f"  tb OVER cap: {n_tb_over} -> AVERAGE (ELITE -10u, STRONG -6u, GOOD -7u; all tiers losing)")

    # ── HRB cap at GOOD (any side; must run BEFORE per-stat ELITE cap) ───────
    # OVER STRONG: 132 bets, 47.7% win rate, -14.22u — model overestimates at STRONG.
    # UNDER STRONG: 20 bets, 60% — encouraging but too thin to validate STRONG sizing.
    # hrb is in ELITE_BLOCKED_STATS so bet_rating.py already demotes ELITE->STRONG;
    # this cap catches those STRONG picks and lands all hrb at GOOD.
    if "BET_RATING" in ev_df.columns and "stat" in ev_df.columns:
        hrb_cap_mask = (
            (ev_df["stat"] == "hrb") &
            ev_df["BET_RATING"].isin(["ELITE", "STRONG"])
        )
        n_hrb_capped = hrb_cap_mask.sum()
        ev_df.loc[hrb_cap_mask, "BET_RATING"] = "GOOD"
        if n_hrb_capped:
            print(f"  HRB cap: {n_hrb_capped} ELITE/STRONG -> GOOD (OVER STRONG -14.22u; UNDER sample too thin)")

    # ── Per-stat ELITE cap ────────────────────────────────────────────────────
    # Prevents correlated props (e.g., 12 HRB picks) from flooding ELITE
    if "BET_RATING" in ev_df.columns:
        stat_elite_counts: dict[str, int] = {}
        for idx in ev_df.index:
            if ev_df.at[idx, "BET_RATING"] != "ELITE":
                continue
            s   = str(ev_df.at[idx, "stat"])
            cap = ELITE_STAT_CAPS.get(s, DEFAULT_ELITE_STAT_CAP)
            stat_elite_counts[s] = stat_elite_counts.get(s, 0) + 1
            if stat_elite_counts[s] > cap:
                ev_df.at[idx, "BET_RATING"] = "STRONG"
        n_stat_capped = sum(
            1 for s, c in stat_elite_counts.items()
            if c > ELITE_STAT_CAPS.get(s, DEFAULT_ELITE_STAT_CAP)
        )
        if n_stat_capped:
            print(f"  Per-stat ELITE cap: {n_stat_capped} stats had excess -> STRONG")

    # ── Global ELITE / STRONG cap ─────────────────────────────────────────────
    if "BET_RATING" in ev_df.columns:
        elite_idx  = ev_df[ev_df["BET_RATING"] == "ELITE"].head(MAX_ELITE_PICKS).index
        strong_idx = ev_df[ev_df["BET_RATING"] == "STRONG"].head(MAX_STRONG_PICKS).index
        capped_mask = (
            ev_df["BET_RATING"].isin(["ELITE", "STRONG"]) &
            ~ev_df.index.isin(elite_idx.union(strong_idx))
        )
        ev_df.loc[capped_mask, "BET_RATING"] = "GOOD"
        if capped_mask.sum():
            print(f"  Global cap: {capped_mask.sum()} picks beyond {MAX_ELITE_PICKS}E/{MAX_STRONG_PICKS}S -> GOOD")

    # ── Per-game and per-player limits ────────────────────────────────────────
    # Cap ELITE/STRONG at 2 per game and 1 per player to avoid correlated exposure.
    # Excess picks drop to GOOD (still visible, just not bet-sized).
    if "BET_RATING" in ev_df.columns:
        game_counts:   dict[str, int] = {}
        player_counts: dict[str, int] = {}
        n_game_capped = 0
        n_plyr_capped = 0
        for idx in ev_df.index:
            if ev_df.at[idx, "BET_RATING"] not in ("ELITE", "STRONG"):
                continue
            game   = str(ev_df.at[idx, "matchup"])
            player = str(ev_df.at[idx, "player_name"])
            game_counts[game]     = game_counts.get(game, 0) + 1
            player_counts[player] = player_counts.get(player, 0) + 1
            if game_counts[game] > MAX_BETS_PER_GAME:
                ev_df.at[idx, "BET_RATING"] = "GOOD"
                n_game_capped += 1
            elif player_counts[player] > MAX_BETS_PER_PLAYER:
                ev_df.at[idx, "BET_RATING"] = "GOOD"
                n_plyr_capped += 1
        if n_game_capped or n_plyr_capped:
            print(f"  Position limits: {n_game_capped} game-capped, {n_plyr_capped} player-capped -> GOOD")

    # ── Zero bet_size for non-bettable tiers ─────────────────────────────────
    if "BET_RATING" in ev_df.columns and "bet_size" in ev_df.columns:
        non_bettable = ~ev_df["BET_RATING"].isin(BETTABLE_RATINGS)
        ev_df.loc[non_bettable, "bet_size"] = 0.0

    # ── Recompute display label after ALL rating caps are applied ─────────────
    ev_df["bet_rating_display"] = ev_df["BET_RATING"].map(RATING_DISPLAY) if "BET_RATING" in ev_df.columns else ""

    # ── Summary ───────────────────────────────────────────────────────────────
    positive = ev_df[ev_df["positive_ev"] == True]
    print(f"\n  Total props evaluated : {len(ev_df)}")
    print(f"  Positive EV picks     : {len(positive)}")

    if "BET_RATING" in ev_df.columns:
        es = ev_df[ev_df["BET_RATING"].isin(["ELITE", "STRONG"])]
        if not es.empty:
            print(f"\n  ELITE/STRONG ({len(es)} picks):")
            for _, r in es.head(20).iterrows():
                cal_flag = " [cal]" if r.get("calibrated") else ""
                print(f"    {r['BET_RATING']:6}  {r['player_name']:<22}  {r['stat']:5}  {r['side']:5}  "
                      f"EV={r['ev_pct']:+.1f}%  prob={r['fair_prob']:.2f}{cal_flag}  "
                      f"odds={fmt_odds(r['bet_odds'])}@{r['bet_book']}  books={r['book_count']}")

    # ── Generate reason comments ──────────────────────────────────────────────
    ev_df["reason"] = ev_df.apply(build_reason, axis=1)

    # ── Save props ────────────────────────────────────────────────────────────
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    ev_df.to_csv(OUT_FILE, index=False)
    print(f"\n  Saved -> {OUT_FILE}  ({len(ev_df)} rows)")

    # ── Daily picks log (ELITE/STRONG/GOOD) ───────────────────────────────────
    DAILY_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    today_date  = datetime.now().date().isoformat()
    rating_col  = "BET_RATING" if "BET_RATING" in ev_df.columns else "bet_rating"
    elite_today = ev_df[ev_df[rating_col].isin(["ELITE", "STRONG", "GOOD"])].copy()
    if not elite_today.empty:
        elite_today.insert(0, "log_date", today_date)
        elite_today["result"] = "PENDING"
        elite_today["profit"] = np.nan
        if DAILY_LOG_FILE.exists():
            log_hist = pd.read_csv(DAILY_LOG_FILE)
            if "log_date" in log_hist.columns:
                log_hist = log_hist[~(
                    (log_hist["log_date"] == today_date) & (log_hist["result"] == "PENDING")
                )]
            combined_log = pd.concat([log_hist, elite_today], ignore_index=True)
        else:
            combined_log = elite_today
        combined_log.to_csv(DAILY_LOG_FILE, index=False)
        print(f"  Auto-saved {len(elite_today)} ELITE/STRONG/GOOD picks to daily log")

    # ── HR props ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("Computing HR Props...")
    print("=" * 55)

    hr_df = compute_hr_props(batter_proj, props_allowed)
    if not hr_df.empty:
        HR_PROPS_FILE.parent.mkdir(parents=True, exist_ok=True)
        hr_df.to_csv(HR_PROPS_FILE, index=False)
        print(f"\n  {'Player':<25} {'Matchup':<22} {'P(HR)':>6} {'Odds':>8} {'EV%':>7}")
        print("  " + "-" * 72)
        for _, r in hr_df.head(15).iterrows():
            print(f"  {r['player_name']:<25} {r['matchup']:<22} "
                  f"{r['prob_hr_pct']:>5.1f}% {fmt_odds(r['best_odds']):>8} "
                  f"{r['ev_pct']:>+7.1f}%")
        pos_hr = hr_df[hr_df["positive_ev"] == True]
        print(f"\n  Positive EV HR picks: {len(pos_hr)}")
        print(f"  Saved -> {HR_PROPS_FILE}  ({len(hr_df)} players)")

        # Append to HR history
        HR_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        save_hr = hr_df[hr_df["prob_hr"] >= 0.10].head(30).copy()
        save_hr.insert(0, "date", today_date)
        save_hr["hit_hr"] = pd.NA
        save_hr["result"] = "PENDING"
        save_hr["profit"] = pd.NA
        if HR_HISTORY_FILE.exists():
            hr_hist = pd.read_csv(HR_HISTORY_FILE)
            if "date" in hr_hist.columns and today_date in hr_hist["date"].values:
                print(f"  HR history already has {today_date} — skipping append.")
            else:
                hr_hist = pd.concat([hr_hist, save_hr], ignore_index=True)
                hr_hist.to_csv(HR_HISTORY_FILE, index=False)
                print(f"  Appended {len(save_hr)} HR predictions to history.")
        else:
            save_hr.to_csv(HR_HISTORY_FILE, index=False)
            print(f"  Created HR history with {len(save_hr)} predictions.")
    else:
        print("  No HR props computed.")

    print("\nDONE")


if __name__ == "__main__":
    main()
