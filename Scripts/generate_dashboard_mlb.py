"""
generate_dashboard_mlb.py

Tabs:
  1. Today's Plays   — filters by stat, side, rating, EV%, matchup
  2. Game Lines      — spreads, totals, moneylines with EV
  3. Results History — collapsible by day, P&L strip
  4. Performance     — win rate by stat, side, probability bucket
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

os.chdir(Path(__file__).parent.parent)

EV_FILE        = Path("Data/mlb/processed/ev_props_today_mlb.csv")
LINES_FILE     = Path("Data/mlb/processed/ev_game_lines_mlb.csv")
HISTORY_FILE   = Path("Data/mlb/history/graded_picks_mlb.csv")
SERIES_FILE    = Path("Data/mlb/processed/4game_series_mlb.csv")
HR_FILE        = Path("Data/mlb/processed/hr_props_mlb.csv")
HR_HIST_FILE   = Path("Data/mlb/history/hr_history_mlb.csv")
BATTER_PROJ    = Path("Data/mlb/processed/mlb_batter_projections.csv")
PITCHER_PROJ   = Path("Data/mlb/processed/mlb_pitcher_projections.csv")
BATTER_LOG     = Path("Data/mlb/raw/mlb_batter_gamelogs_all.csv")
PITCHER_LOG    = Path("Data/mlb/raw/mlb_pitcher_gamelogs_all.csv")
OUTPUT_FILE    = Path("Data/mlb/dashboard_mlb.html")

UNIT = 10.0

STAT_LABELS = {
    "hits": "Hits", "runs": "Runs", "rbi": "RBI",
    "hr": "HR", "tb": "Total Bases", "k": "Strikeouts", "outs": "Outs Recorded",
}


def safe_read(path):
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


_STAT_ABBR = {
    "hits": "H", "hr": "HR", "hrb": "HRB", "k": "K",
    "tb": "TB", "runs": "R", "rbi": "RBI", "outs": "Outs",
}

_TEAM_ABBR = {
    "arizona diamondbacks": "ARI", "atlanta braves": "ATL", "baltimore orioles": "BAL",
    "boston red sox": "BOS", "chicago cubs": "CHC", "chicago white sox": "CWS",
    "cincinnati reds": "CIN", "cleveland guardians": "CLE", "colorado rockies": "COL",
    "detroit tigers": "DET", "houston astros": "HOU", "kansas city royals": "KC",
    "los angeles angels": "LAA", "los angeles dodgers": "LAD", "miami marlins": "MIA",
    "milwaukee brewers": "MIL", "minnesota twins": "MIN", "new york mets": "NYM",
    "new york yankees": "NYY", "athletics": "OAK", "oakland athletics": "OAK",
    "philadelphia phillies": "PHI", "pittsburgh pirates": "PIT", "san diego padres": "SD",
    "san francisco giants": "SF", "seattle mariners": "SEA", "st. louis cardinals": "STL",
    "tampa bay rays": "TB", "texas rangers": "TEX", "toronto blue jays": "TOR",
    "washington nationals": "WSH",
}

def abbrev_team(name: str) -> str:
    return _TEAM_ABBR.get(str(name).strip().lower(), str(name).strip()[:3].upper())

def abbrev_matchup(matchup: str) -> str:
    if " @ " not in str(matchup):
        return matchup
    away, home = matchup.split(" @ ", 1)
    return f"{abbrev_team(away)}@{abbrev_team(home)}"

def build_reason(row) -> str:
    parts: list[str] = []
    abbr = _STAT_ABBR.get(str(row.get("stat", "")).lower(), str(row.get("stat", "")).upper())
    try:
        proj = float(row["projection"])
        line = float(row["line"])
        edge = float(row["edge"])
        parts.append(f"Proj {proj:.1f} {abbr} vs line {line:.1f} ({edge:+.1f})")
    except (TypeError, ValueError, KeyError):
        pass
    hand    = str(row.get("throw_hand_faced", "")).strip()
    starter = str(row.get("opp_starter", "")).strip()
    if hand in ("L", "R"):
        name = starter.split()[-1] if starter and starter not in ("nan", "") else "starter"
        parts.append(f"vs {hand}HP {name}")
    lm    = str(row.get("line_move", "")).strip().upper()
    shift = row.get("line_move_shift")
    if lm in ("UP", "DOWN", "SHARP"):
        try:
            parts.append(f"Line {lm.lower()} {float(shift):+.2f}")
        except (TypeError, ValueError):
            parts.append(f"Line moved {lm.lower()}")
    try:
        books = int(row["book_count"])
        if books >= 3:
            parts.append(f"{books} books")
    except (TypeError, ValueError, KeyError):
        pass
    if row.get("calibrated"):
        parts.append("calibrated")
    return ". ".join(parts) + "." if parts else ""


def fmt_odds(v):
    try:
        v = float(v)
        return f"+{int(v)}" if v > 0 else str(int(v))
    except: return "-"


def fmt_pct(v, decimals=1):
    try: return f"{float(v):.{decimals}f}%"
    except: return "-"


def fmt_num(v, decimals=2):
    try: return f"{float(v):.{decimals}f}"
    except: return "-"


def parse_commence(val):
    try:
        dt = pd.to_datetime(val, utc=True).tz_convert("America/Los_Angeles")
        return dt.strftime("%I:%M %p PT")
    except: return "-"


def recommendation_badge(rec):
    rec = str(rec).upper()
    if rec == "OVER":  return '<span class="badge badge-over">OVER</span>'
    if rec == "UNDER": return '<span class="badge badge-under">UNDER</span>'
    return f'<span class="badge">{rec}</span>'


def result_badge(result):
    result = str(result).upper()
    if result == "WIN":  return '<span class="badge badge-win">WIN</span>'
    if result == "LOSS": return '<span class="badge badge-loss">LOSS</span>'
    if result == "PUSH": return '<span class="badge badge-push">PUSH</span>'
    return f'<span class="badge">{result}</span>'


def rating_badge(rating):
    rating = str(rating).strip().upper()
    if "ELITE"   in rating: return '<span class="bet-rating rating-elite">🔥 ELITE</span>'
    if "STRONG"  in rating: return '<span class="bet-rating rating-strong">✅ STRONG</span>'
    if "GOOD"    in rating: return '<span class="bet-rating rating-good">👍 GOOD</span>'
    if "AVERAGE" in rating: return '<span class="bet-rating rating-average">➖ AVERAGE</span>'
    if "POOR"    in rating: return '<span class="bet-rating rating-poor">⚠️ POOR</span>'
    if "BAD"     in rating: return '<span class="bet-rating rating-bad">❌ BAD</span>'
    return '<span class="bet-rating">-</span>'


def ev_badge(ev):
    try:
        e = float(ev)
        if e >= 5:  return f'<span style="color:#00e5a0;font-family:monospace;font-weight:600">+{e:.1f}%</span>'
        if e >= 2:  return f'<span style="color:#44d4a0;font-family:monospace;font-weight:600">+{e:.1f}%</span>'
        if e >= 0:  return f'<span style="color:#f5a623;font-family:monospace">+{e:.1f}%</span>'
        return f'<span style="color:#f04e4e;font-family:monospace">{e:.1f}%</span>'
    except: return "-"


def confidence_badge(conf):
    conf = str(conf).strip().upper()
    if conf == "HIGH":   return '<span style="font-size:0.65rem;background:rgba(0,229,160,0.15);color:#00e5a0;border-radius:3px;padding:1px 5px;font-family:monospace;letter-spacing:0.04em">HIGH</span>'
    if conf == "MEDIUM": return '<span style="font-size:0.65rem;background:rgba(245,166,35,0.15);color:#f5a623;border-radius:3px;padding:1px 5px;font-family:monospace;letter-spacing:0.04em">MED</span>'
    return '<span style="font-size:0.65rem;background:rgba(150,150,150,0.15);color:#888;border-radius:3px;padding:1px 5px;font-family:monospace;letter-spacing:0.04em">LOW</span>'


def hand_tag(hand, starter):
    if not hand or str(hand).strip() in ("", "nan"):
        return '<span style="font-size:0.65rem;color:var(--muted)">TBD</span>'
    color = "#f04e4e" if hand == "L" else "#3d8ef8"
    label = f"vs {'LHP' if hand == 'L' else 'RHP'}"
    tip   = str(starter).strip() if starter and str(starter).strip() not in ("", "nan") else ""
    title = f' title="{tip}"' if tip else ""
    return f'<span style="font-size:0.65rem;color:{color};font-family:monospace;font-weight:600"{title}>{label}</span>'


def line_move_badge(signal, shift=None):
    """Display line movement signal inline on a pick."""
    s = str(signal).strip().upper()
    if s == "STEAM_WITH":
        shift_str = f" {shift*100:+.1f}pp" if shift is not None and not pd.isna(shift) else ""
        return f'<span style="font-size:0.62rem;background:rgba(0,229,160,0.15);color:#00e5a0;border-radius:3px;padding:1px 5px;font-family:monospace" title="Sharp money confirms this side{shift_str}">STEAM</span>'
    if s == "STEAM_AGAINST":
        shift_str = f" {shift*100:+.1f}pp" if shift is not None and not pd.isna(shift) else ""
        return f'<span style="font-size:0.62rem;background:rgba(240,78,78,0.15);color:#f04e4e;border-radius:3px;padding:1px 5px;font-family:monospace" title="Sharp money fades this side{shift_str}">FADE</span>'
    if s == "NO_DATA":
        return ""   # No pre-game pull yet — show nothing
    return ""   # NEUTRAL — no badge needed


def is_top_play(r):
    """Returns True for ELITE and STRONG picks — all sides included."""
    rating = str(r.get("bet_rating_display", r.get("bet_rating", ""))).upper()
    return any(x in rating for x in ["ELITE", "STRONG"])

def is_pick6_eligible(r):
    """Kept for backward compat — now same as is_top_play."""
    return is_top_play(r)


def calc_pnl(rows):
    pnl = 0.0
    for _, r in rows.iterrows():
        result = str(r.get("hit_result", "")).upper()
        if result not in ("WIN", "LOSS"): continue
        try:
            odds = float(r.get("bet_odds", r.get("price", np.nan)))
            if pd.isna(odds): odds = -110.0
        except: odds = -110.0
        if result == "WIN":
            pnl += (odds / 100.0 * UNIT) if odds > 0 else (100.0 / abs(odds) * UNIT)
        else:
            pnl -= UNIT
    return pnl


def pnl_chip(label, pnl):
    color = "#00e5a0" if pnl >= 0 else "#f04e4e"
    sign  = "+" if pnl >= 0 else ""
    units = pnl / UNIT
    return f"""
    <div class="stat-chip" style="border-color:{color}40">
      <span class="chip-val" style="color:{color}">{sign}${pnl:.2f}</span>
      <span class="chip-lbl">{label} P&amp;L</span>
    </div>
    <div class="stat-chip" style="border-color:{color}40">
      <span class="chip-val" style="color:{color}">{sign}{units:.1f}u</span>
      <span class="chip-lbl">{label} Units</span>
    </div>"""


def build_picks_json(df):
    if df.empty: return "[]"
    picks = []
    src = df[df["positive_ev"] == True] if "positive_ev" in df.columns else df
    for _, r in src.iterrows():
        player     = str(r.get("player_name", "")).strip()
        stat       = str(r.get("stat", "")).strip()
        stat_label = STAT_LABELS.get(stat, stat.upper())
        line       = str(r.get("line", "")).strip()
        side       = str(r.get("side", r.get("recommendation", ""))).strip().upper()
        matchup    = str(r.get("matchup", f"{r.get('away_team','?')} @ {r.get('home_team','?')}")).strip()
        rating     = str(r.get("BET_RATING", r.get("bet_rating", ""))).strip()
        try:
            odds_raw = float(r.get("bet_odds", r.get("price", 0)))
            odds = f"+{int(odds_raw)}" if odds_raw > 0 else str(int(odds_raw))
        except: odds = ""
        if player:
            picks.append({
                "player": f"{player} ({stat_label})", "prop": stat_label,
                "line": line, "side": side, "odds": odds,
                "rating": rating, "matchup": matchup, "sport": "MLB",
            })
    return json.dumps(picks, ensure_ascii=False)


# Simulation support maps
_STAT_GAMELOG_COL = {
    "hits": "h", "hr": "hr", "tb": "tb", "runs": "r",
    "rbi": "rbi", "sb": "sb", "k": "k", "outs": "ip",
    "h_allowed": "h_allowed",
}
_PROJ_COL = {
    "hits": "proj_h", "hr": "proj_hr", "tb": "proj_tb",
    "runs": "proj_r", "rbi": "proj_rbi", "sb": "proj_sb",
    "k": "proj_k", "outs": "proj_ip",
    "h_allowed": "proj_h_allowed",
}
_STD_COL = {
    "hits": "h_std_5", "tb": "tb_std_5", "hr": "hr_std_5",
}
_SEASON_AVG_COL = {
    "hits": "h_season_avg", "hr": "hr_season_avg", "tb": "tb_season_avg",
    "runs": "r_season_avg",  "rbi": "rbi_season_avg", "sb": "sb_season_avg",
    "k": "k_season_avg",    "outs": "ip_season_avg",
    "h_allowed": "h_allowed_season_avg",
}
_POISSON_STATS = {"hits", "hr", "k", "outs", "rbi", "runs", "sb"}
_NORMAL_STATS  = {"tb"}


def _load_gamelog_index(log_df, stat_col):
    """Return {player_id_str: [last-30 float values, newest-first]} for one stat column."""
    idx = {}
    if log_df.empty:
        return idx
    cols = list(log_df.columns)
    pid_col  = next((c for c in ["player_id", "playerid"] if c in cols), None)
    date_col = next((c for c in ["game_date", "gamedate", "date"] if c in cols), None)
    if not pid_col or not date_col or stat_col not in cols:
        return idx
    sub = log_df[[pid_col, date_col, stat_col]].dropna(subset=[pid_col, stat_col])
    sub = sub.sort_values(date_col, ascending=False)
    for pid, grp in sub.groupby(pid_col):
        idx[str(pid).split(".")[0]] = [float(v) for v in grp[stat_col].head(30)]
    return idx


def build_sim_inputs(df):
    """Return JSON string of simulation inputs for ELITE picks only."""
    if df.empty:
        return "[]"

    rating_col = next((c for c in df.columns if c.upper() == "BET_RATING"), None)
    if rating_col is None:
        return "[]"

    elite = df[df[rating_col].str.upper().str.contains("ELITE", na=False)].copy()
    if elite.empty:
        return "[]"

    # Load all data sources once
    bat_raw = safe_read(BATTER_LOG)
    pit_raw = safe_read(PITCHER_LOG)
    bat_prj = safe_read(BATTER_PROJ)
    pit_prj = safe_read(PITCHER_PROJ)

    # Normalize columns to lowercase after safe_read already did it
    def name_lookup(proj_df):
        """Return {lowercase_player_name: row_dict} keyed on most-recent row."""
        out = {}
        if proj_df.empty or "player_name" not in proj_df.columns:
            return out
        date_col = next((c for c in ["game_date","gamedate"] if c in proj_df.columns), None)
        if date_col:
            proj_df = proj_df.sort_values(date_col, ascending=False)
        deduped = proj_df.drop_duplicates(subset="player_name", keep="first")
        for _, r in deduped.iterrows():
            out[str(r["player_name"]).strip().lower()] = r.to_dict()
        return out

    bat_idx = name_lookup(bat_prj)
    pit_idx = name_lookup(pit_prj)

    # Lazy gamelog index cache: stat_col -> {player_id -> [samples]}
    bat_log_cache = {}
    pit_log_cache = {}

    def bat_samples(stat_col, player_id):
        if stat_col not in bat_log_cache:
            bat_log_cache[stat_col] = _load_gamelog_index(bat_raw, stat_col)
        return bat_log_cache[stat_col].get(player_id, [])

    def pit_samples(stat_col, player_id):
        if stat_col not in pit_log_cache:
            pit_log_cache[stat_col] = _load_gamelog_index(pit_raw, stat_col)
        return pit_log_cache[stat_col].get(player_id, [])

    def safe_float(v, fallback=0.0):
        try:
            f = float(v)
            return fallback if pd.isna(f) else f
        except (TypeError, ValueError):
            return fallback

    results = []
    for _, row in elite.iterrows():
        player    = str(row.get("player_name", "")).strip()
        if not player:
            continue
        stat      = str(row.get("stat", "")).strip().lower()
        try:
            line = float(row.get("line", 0))
        except (TypeError, ValueError):
            continue
        if line <= 0:
            continue

        side       = str(row.get("side", "")).upper()
        odds_raw   = safe_float(row.get("bet_odds", row.get("price", -110)), -110.0)
        fair_prob  = safe_float(row.get("fair_prob", 0.55), 0.55)
        projection = safe_float(row.get("projection", line), line)
        matchup    = str(row.get("matchup", ""))
        stat_label = STAT_LABELS.get(stat, stat.upper())
        is_pitcher = stat in ("k", "outs", "h_allowed")

        proj_lookup = pit_idx if is_pitcher else bat_idx
        proj_row    = proj_lookup.get(player.lower(), {})
        player_id   = str(proj_row.get("player_id", "")).split(".")[0]
        if not player_id:
            print(f"  [sim] no player_id for {player!r} ({stat}) — bootstrap will use season-avg padding")

        # Projection mean from PROJ_* column, fall back to EV projection
        proj_col  = _PROJ_COL.get(stat)
        proj_mean = projection
        if proj_col and proj_col in proj_row:
            v = safe_float(proj_row[proj_col], 0.0)
            if v > 0:
                proj_mean = v * 3 if stat == "outs" else v

        # Projection std from STD_* column, fall back to 35% of mean
        std_col   = _STD_COL.get(stat)
        proj_std  = max(0.3, proj_mean * 0.35)
        if std_col and std_col in proj_row:
            v = safe_float(proj_row[std_col], 0.0)
            if v > 0:
                proj_std = max(0.1, v)

        # Game samples for bootstrap
        log_col = _STAT_GAMELOG_COL.get(stat, stat)
        raw_samples = pit_samples(log_col, player_id) if is_pitcher else bat_samples(log_col, player_id)
        samples = [s * 3 if stat == "outs" else s for s in raw_samples]

        # Pad to at least 10 samples using season average
        if len(samples) < 10:
            season_col = _SEASON_AVG_COL.get(stat)
            pad_val    = projection
            if season_col and season_col in proj_row:
                v = safe_float(proj_row[season_col], 0.0)
                if v > 0:
                    pad_val = v * 3 if stat == "outs" else v
            while len(samples) < 10:
                samples.append(pad_val)

        results.append({
            "player":          player,
            "stat":            stat,
            "stat_label":      stat_label,
            "line":            line,
            "side":            side,
            "odds":            odds_raw,
            "fair_prob":       round(fair_prob, 4),
            "proj_mean":       round(proj_mean, 4),
            "proj_std":        round(max(0.1, proj_std), 4),
            "samples":         [round(s, 2) for s in samples[:30]],
            "stat_type":       "normal" if stat in _NORMAL_STATS else "poisson",
            "matchup":         matchup,
            "has_real_samples": len(raw_samples) > 0,
        })

    return json.dumps(results, ensure_ascii=False)


def sanitize_js(text):
    """Remove characters that break JS string literals."""
    return str(text).replace("'", "").replace('"', '').replace('\n', ' ').replace('\r', '').strip()


def build_parlay(props_df, game_lines_df):
    """
    Build a Parlay of the Day targeting ~+200 odds.
    Prefers 2 legs, allows 3. Uses highest fair_prob picks from props + game lines.
    """
    import itertools

    def american_to_decimal(odds):
        if pd.isna(odds): return None
        o = float(odds)
        return (o/100 + 1) if o > 0 else (100/abs(o) + 1)

    def decimal_to_american(d):
        if d >= 2.0: return f"+{int((d-1)*100)}"
        return f"-{int(100/(d-1))}"

    candidates = []

    # Add prop picks — only ELITE/STRONG, exclude underperforming stats (tb=37% win rate)
    PARLAY_EXCLUDED_STATS = {"tb", "rbi", "runs"}
    if not props_df.empty:
        rating_col = "BET_RATING" if "BET_RATING" in props_df.columns else "bet_rating"
        pos = props_df[
            props_df.get(rating_col, pd.Series(dtype=str)).isin(["ELITE", "STRONG"]) &
            ~props_df["stat"].isin(PARLAY_EXCLUDED_STATS)
        ].copy() if rating_col in props_df.columns else pd.DataFrame()
        for _, r in pos.iterrows():
            side = str(r.get("side","")).upper()
            bov = r.get("bovada_over" if side=="OVER" else "bovada_under", np.nan)
            if pd.isna(bov): continue
            dec = american_to_decimal(bov)
            if dec is None or dec <= 1: continue
            candidates.append({
                "label": f"{r.get('player_name','')} {r.get('stat','').title()} {r.get('line','')} {side}",
                "fair_prob": float(r.get("fair_prob", 0)),
                "bovada_odds": int(bov),
                "decimal": dec,
                "matchup": str(r.get("matchup","")),
                "type": "prop"
            })

    # Add game line picks
    if not game_lines_df.empty and "ev_pct" in game_lines_df.columns:
        gl = game_lines_df[game_lines_df["ev_pct"] > 0].copy()
        for _, r in gl.iterrows():
            bov = r.get("best_odds", np.nan)
            if pd.isna(bov): continue
            dec = american_to_decimal(bov)
            if dec is None or dec <= 1: continue
            candidates.append({
                "label": f"{r.get('side','')} ({r.get('market','')}) — {r.get('matchup','')}",
                "fair_prob": float(r.get("fair_prob", 0)),
                "bovada_odds": int(bov),
                "decimal": dec,
                "matchup": str(r.get("matchup","")),
                "type": "game"
            })

    if len(candidates) < 2:
        return '<p class="empty-msg">Not enough picks to build a parlay today.</p>'

    # Sort by fair_prob descending
    candidates.sort(key=lambda x: x["fair_prob"], reverse=True)

    def build_parlay_html(legs, combined, label, color, icon, target_note):
        combined_american = decimal_to_american(combined)
        combined_prob = 1.0
        for leg in legs:
            combined_prob *= leg["fair_prob"]
        legs_html = ""
        for i, leg in enumerate(legs):
            odds_str = f'+{leg["bovada_odds"]}' if leg["bovada_odds"] > 0 else str(leg["bovada_odds"])
            legs_html += f"""
            <div style="display:flex;align-items:center;gap:0.75rem;padding:0.5rem 0;border-bottom:1px solid var(--border)">
              <span style="background:rgba({color},0.15);color:rgb({color});border-radius:50%;width:22px;height:22px;display:flex;align-items:center;justify-content:center;font-size:0.75rem;font-weight:700;flex-shrink:0">{i+1}</span>
              <div style="flex:1">
                <div style="font-weight:600;font-size:0.85rem">{leg["label"]}</div>
                <div style="font-size:0.72rem;color:var(--muted)">{leg["matchup"]} &nbsp;·&nbsp; Fair prob: {leg["fair_prob"]*100:.1f}%</div>
              </div>
              <span style="font-family:monospace;font-weight:700;color:rgb({color})">{odds_str}</span>
            </div>"""
        safe_label = label.replace("'","").replace('"','')
        parlay_id   = f"pu_{abs(hash(label)) % 99999}"
        legs_json   = "[" + ",".join("'" + lg['label'].replace("'","").replace('"','') + "'" for lg in legs) + "]"
        return f"""
        <div style="background:rgba({color},0.06);border:1px solid rgba({color},0.3);border-radius:10px;padding:1.2rem 1.4rem;margin-bottom:1rem">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.75rem">
            <div style="font-family:'DM Mono',monospace;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;color:rgb({color});font-weight:600">
              {icon} {label} &nbsp;·&nbsp; {len(legs)} Legs
            </div>
            <div style="display:flex;align-items:center;gap:0.75rem">
              <div style="font-family:monospace;font-size:1.1rem;font-weight:700;color:rgb({color})">{combined_american}</div>
              <select id="{parlay_id}" style="background:var(--surface2);border:1px solid var(--border);border-radius:4px;padding:0.15rem 0.3rem;color:var(--text);font-size:0.72rem;font-family:monospace">
                <option value="0.1">0.1u</option><option value="0.5">0.5u</option>
                <option value="1" selected>1u</option><option value="2">2u</option>
              </select>
              <button onclick="trackParlay('{safe_label}','{combined_american}',{legs_json},parseFloat(document.getElementById('{parlay_id}').value))"
                style="background:rgba(0,229,160,0.1);border:1px solid rgba(0,229,160,0.3);color:#00e5a0;border-radius:4px;padding:0.2rem 0.5rem;font-size:0.7rem;cursor:pointer;font-family:monospace">
                Track Parlay
              </button>
            </div>
          </div>
          {legs_html}
          <div style="margin-top:0.75rem;font-size:0.72rem;color:var(--muted)">
            Combined fair probability: {combined_prob*100:.1f}% &nbsp;·&nbsp; {target_note} &nbsp;·&nbsp; Verify odds on Bovada before placing
          </div>
        </div>"""

    # ── Parlay of the Day (~+200) ─────────────────────────────
    TARGET_MIN = 2.8
    TARGET_MAX = 4.5
    main_parlay = None
    best_diff = 999
    for n_legs in [2, 3]:
        top = candidates[:12]
        for combo in itertools.combinations(top, n_legs):
            matchups = [c["matchup"] for c in combo]
            if len(set(matchups)) < len(matchups): continue
            combined_decimal = 1.0
            for c in combo: combined_decimal *= c["decimal"]
            diff = abs(combined_decimal - 3.0)
            if TARGET_MIN <= combined_decimal <= TARGET_MAX and diff < best_diff:
                best_diff = diff
                main_parlay = (tuple(combo), combined_decimal)
        if main_parlay: break
    if not main_parlay:
        combo = tuple(candidates[:2])
        combined_decimal = combo[0]["decimal"] * combo[1]["decimal"]
        main_parlay = (combo, combined_decimal)

    # ── Shot in the Dark (4-7 legs, max EV) ─────────────────
    # Goal: highest probability of hitting × biggest payout = max parlay EV
    # EV = combined_fair_prob × combined_decimal (return per $1 wagered)
    # We iterate 4-7 leg combos from the top unique-matchup candidates
    # and pick whichever combo has the best EV — no arbitrary payout target.
    shot_parlay = None
    seen_matchups = set()
    shot_legs = []
    for c in candidates:
        if c["matchup"] not in seen_matchups:
            shot_legs.append(c)
            seen_matchups.add(c["matchup"])
        if len(shot_legs) == 10:  # wider pool → more combos to score
            break
    if len(shot_legs) >= 4:
        best_shot    = None
        best_shot_ev = -999.0
        for n in range(4, min(8, len(shot_legs)+1)):
            for combo in itertools.combinations(shot_legs, n):
                cd = 1.0
                cp = 1.0
                for c in combo:
                    cd *= c["decimal"]
                    cp *= c["fair_prob"]
                if cd < 6.0:   # require at least +500 payout
                    continue
                parlay_ev = cp * cd  # expected return per $1 — higher is better
                if parlay_ev > best_shot_ev:
                    best_shot_ev = parlay_ev
                    best_shot    = (tuple(combo), cd)
        if best_shot:
            shot_parlay = best_shot

    # Build HTML
    main_html = build_parlay_html(
        main_parlay[0], main_parlay[1],
        "Parlay of the Day", "245,166,35", "🎯", "Targeting ~+200 odds"
    )
    shot_html = ""
    if shot_parlay:
        shot_html = build_parlay_html(
            shot_parlay[0], shot_parlay[1],
            "Shot in the Dark", "180,100,255", "🚀", "Best prob × payout combo · Small unit only (0.1u max)"
        )

    return main_html + shot_html


def build_today(df, game_lines_df=None, player_team: dict | None = None):
    if df.empty:
        return '<p class="empty-msg">No plays found. Run compute_ev_mlb.py first.</p>', False

    rating_col = "BET_RATING" if "BET_RATING" in df.columns else "bet_rating"
    elite_strong_mask = df[rating_col].isin(["ELITE", "STRONG"]) if rating_col in df.columns else pd.Series(False, index=df.index)
    elite_strong = df[elite_strong_mask].copy()

    # Only show ELITE and STRONG — no fallback to GOOD
    # ELITE: 57.5% win rate +$77.74 profit | STRONG: 58.2% win rate +$41.76 profit
    positive = elite_strong
    fallback_mode = False

    if positive.empty:
        today_str = pd.Timestamp.now(tz="America/Los_Angeles").strftime("%B %d")
        return (f'<p class="empty-msg">No actionable picks for {today_str}. '
                f'The model found no edge on today\'s Bovada/BetOnline lines vs sharp consensus. '
                f'Check back after lines move or tomorrow\'s slate.</p>'), False

    # Sort by game time then matchup alphabetically, then by ev_pct within each game
    positive["_sort_time"] = pd.to_datetime(positive["commence_time"], utc=True, errors="coerce")
    positive["_sort_matchup"] = positive["matchup"].fillna("").str.strip()
    positive = positive.sort_values(["_sort_time", "_sort_matchup", "ev_pct"],
                                     ascending=[True, True, False])

    # Mark started games (commence_time in the past) — shown grayed-out with Track still enabled
    now_utc = pd.Timestamp.now(tz="UTC")
    positive["_game_started"] = positive["_sort_time"].apply(
        lambda t: pd.notna(t) and t < now_utc
    )

    pick6_html = ""  # Top box removed — single unified table below

    stat_opts    = "".join(f'<option value="{s}">{STAT_LABELS.get(s,s.upper())}</option>'
                           for s in sorted(positive["stat"].dropna().unique())) if "stat" in positive.columns else ""
    matchup_opts = "".join(f'<option value="{m}">{m}</option>'
                           for m in sorted(positive["matchup"].dropna().unique())) if "matchup" in positive.columns else ""

    filter_bar = f"""
    <div class="filter-bar">
      <select id="f-stat" class="filter-select" onchange="applyFilters()">
        <option value="">All Stats</option>{stat_opts}
      </select>
      <select id="f-side" class="filter-select" onchange="applyFilters()">
        <option value="">All Sides</option>
        <option value="OVER">OVER</option><option value="UNDER">UNDER</option>
      </select>
      <select id="f-rating" class="filter-select" onchange="applyFilters()">
        {'<option value="" selected>All Ratings</option><option value="ELITE,STRONG">Elite + Strong Only</option>' if fallback_mode else '<option value="ELITE,STRONG" selected>Elite + Strong Only</option>'}
        <option value="ELITE">ELITE Only</option>
        <option value="STRONG">STRONG Only</option>
      </select>
      <select id="f-ev" class="filter-select" onchange="applyFilters()">
        <option value="">Min EV%</option>
        <option value="2">2%+</option><option value="4">4%+</option>
        <option value="6">6%+</option><option value="8">8%+</option>
      </select>
      <select id="f-fairprob" class="filter-select" onchange="applyFilters()">
        <option value="">Min Fair%</option>
        <option value="55">55%+</option><option value="57">57%+</option>
        <option value="60">60%+</option><option value="63">63%+</option>
      </select>
      <select id="f-matchup" class="filter-select" onchange="applyFilters()">
        <option value="">All Games</option>{matchup_opts}
      </select>
      <button class="filter-reset" onclick="resetFilters()">Reset</button>
    </div>"""

    # Orange highlight = top 5 ELITE/STRONG picks by EV% (best plays of the day).
    # Exactly 5 picks highlighted — no more, no less.
    TOP_PLAYS_MAX = 5
    top5_keys  = set()   # the 5 highlighted picks
    capped_keys = set()  # ELITE/STRONG but outside top 5
    if not positive.empty:
        rating_col = "BET_RATING" if "BET_RATING" in positive.columns else "bet_rating"
        top_picks = positive[positive[rating_col].isin(["ELITE", "STRONG"])].copy() if rating_col in positive.columns else positive.iloc[:0].copy()
        if not top_picks.empty and "ev_pct" in top_picks.columns:
            top_picks = top_picks.sort_values("ev_pct", ascending=False)
        for i, (_, r5) in enumerate(top_picks.iterrows()):
            k = (sanitize_js(r5.get('player_name','')),
                 str(r5.get('stat','')),
                 str(r5.get('line','')),
                 str(r5.get('side','')).upper())
            if i < TOP_PLAYS_MAX:
                top5_keys.add(k)
            else:
                capped_keys.add(k)

    rows_html = ""
    for _, r in positive.iterrows():
        stat_label = STAT_LABELS.get(str(r.get("stat","")), str(r.get("stat","")).upper())
        game_time  = parse_commence(r.get("commence_time"))
        matchup    = str(r.get("matchup", f"{r.get('away_team','?')} @ {r.get('home_team','?')}"))
        p6         = is_pick6_eligible(r)
        side       = str(r.get("side", r.get("recommendation",""))).upper()
        rating_raw   = r.get("BET_RATING", r.get("bet_rating", ""))   # clean key for filter + badge
        game_started = bool(r.get("_game_started", False))
        ev_val       = r.get("ev_pct", 0)
        try: ev_f = float(ev_val)
        except: ev_f = 0
        try: fair_f = round(float(r.get("fair_prob", 0)) * 100, 1)
        except: fair_f = 0

        player_name  = sanitize_js(r.get('player_name','-'))
        odds_val     = fmt_odds(r.get('bet_odds', r.get('price','')))
        pick_key     = f"{player_name}|{stat_label}|{fmt_num(r.get('line',''),1)}|{side}"
        conf_tier    = str(r.get('confidence_tier', r.get('confidence', ''))).strip().upper()
        throw_hand   = str(r.get('throw_hand_faced', '')).strip()
        opp_starter  = str(r.get('opp_starter', '')).strip()
        lm_signal    = str(r.get('line_move', 'NO_DATA')).strip().upper()
        lm_shift     = r.get('line_move_shift', np.nan)

        # Bovada specific odds
        if side == "OVER":
            bov_price = r.get("bovada_over", np.nan)
        else:
            bov_price = r.get("bovada_under", np.nan)
        bov_str_display = fmt_odds(bov_price) if pd.notna(bov_price) else '<span style="color:var(--muted)">-</span>'
        bov_str = fmt_odds(bov_price) if pd.notna(bov_price) else '-'  # plain for onclick (no HTML quotes)
        has_bov = pd.notna(bov_price)
        bov_color = "#00e5a0" if has_bov and float(bov_price) < 0 else "#f5a623" if has_bov else "var(--muted)"

        # Best-book line shopping
        best_bk       = str(r.get("best_book", "") or "").strip()
        best_bk_price = r.get("best_book_price", np.nan)
        BOOK_SHORT = {
            "draftkings": "DK", "fanduel": "FD", "betmgm": "MGM",
            "caesars": "CZR", "fanatics": "FAN", "espnbet": "ESPN",
            "betrivers": "BR", "bovada": "BOV", "mybookieag": "MYB",
            "betonlineag": "BOL",
        }
        best_bk_label = BOOK_SHORT.get(best_bk, best_bk.upper()[:4]) if best_bk else "-"
        if pd.notna(best_bk_price) and best_bk:
            _best_color = "#00e5a0" if float(best_bk_price) >= 0 else "var(--text)"
            best_bk_html = f'<span style="color:{_best_color};font-weight:600;font-family:monospace">{fmt_odds(best_bk_price)}</span> <span style="color:var(--muted);font-size:11px">{best_bk_label}</span>'
        else:
            best_bk_html = '<span style="color:var(--muted)">-</span>'

        _row_key   = (player_name, str(r.get('stat','')), str(r.get('line','')), side)
        _is_top5   = _row_key in top5_keys
        if game_started:
            _row_style = " style='opacity:0.45;border-left:3px solid #555'"
        elif _is_top5:
            _row_style = " style='background:rgba(245,166,35,0.07);border-left:3px solid #f5a623'"
        else:
            _row_style = ""
        _corr_attr = ''
        _started_attr = ' data-started="1"' if game_started else ''
        game_time_display = f'<span style="color:#888;font-size:10px">LIVE</span>' if game_started else game_time
        rows_html += f"""<tr data-stat="{r.get('stat','')}" data-side="{side}" data-rating="{str(rating_raw).upper()}" data-ev="{ev_f}" data-fairprob="{fair_f}" data-matchup="{matchup}"{_corr_attr}{_started_attr}{_row_style}>
          <td>{rating_badge(rating_raw)}</td>
          <td>
            <div style="display:flex;align-items:center;gap:4px">
              <select id="u_{pick_key}" style="background:var(--surface2);border:1px solid var(--border);border-radius:4px;padding:0.15rem 0.3rem;color:var(--text);font-size:0.72rem;font-family:monospace">
                <option value="1">1u</option>
                <option value="2">2u</option>
                <option value="3">3u</option>
                <option value="4">4u</option>
                <option value="5">5u</option>
              </select>
              <button onclick="trackBet('{player_name}','{stat_label}','{fmt_num(r.get('line',''),1)}','{side}','{bov_str}','{matchup}','{game_time}','MLB')"
                id="tb_{pick_key}"
                style="background:rgba(0,229,160,0.1);border:1px solid rgba(0,229,160,0.3);color:#00e5a0;border-radius:4px;padding:0.2rem 0.5rem;font-size:0.7rem;cursor:pointer;font-family:monospace;white-space:nowrap">
                Track
              </button>
            </div>
          </td>
          <td><div><strong>{player_name}</strong></div><div style="margin-top:1px;font-size:0.68rem;color:var(--muted);font-family:monospace">{(player_team or {}).get(r.get('player_name','').strip().lower(), '')}</div><div style="margin-top:2px;display:flex;gap:4px;align-items:center">{confidence_badge(conf_tier)}{line_move_badge(lm_signal, lm_shift)}</div></td>
          <td><span class="stat-pill">{stat_label}</span></td>
          <td>{fmt_num(r.get('line',''),1)}</td>
          <td><div style="color:#3d8ef8;font-weight:600">{fmt_num(r.get('projection',''),2)}</div><div style="margin-top:2px">{hand_tag(throw_hand, opp_starter)}</div></td>
          <td style="color:#00e5a0;font-weight:600;font-family:monospace">+{fmt_num(r.get('edge',0),2)}</td>
          <td>{recommendation_badge(side)}</td>
          <td style="font-family:monospace;color:{bov_color};font-weight:600">{bov_str_display}</td>
          <td>{fmt_pct(float(r.get('fair_prob',0))*100)}</td>
          <td class="ev-cell">{fmt_pct(ev_val)}</td>
          <td style="font-size:0.72rem;color:var(--muted);min-width:220px;max-width:320px;white-space:normal">{r.get('reason','')}</td>
          <td class="muted" style="white-space:nowrap;font-family:monospace;font-size:0.78rem">{abbrev_matchup(matchup)}</td>
          <td class="muted">{game_time_display}</td>
        </tr>"""

    # Build parlay
    gl_df = game_lines_df if game_lines_df is not None else pd.DataFrame()
    parlay_html = build_parlay(positive, gl_df)

    total_picks = len(positive)
    fallback_banner = """
    <div style="background:rgba(245,166,35,0.1);border:1px solid rgba(245,166,35,0.35);border-radius:6px;padding:0.5rem 0.85rem;margin-bottom:0.75rem;font-size:0.75rem;color:#f5a623">
      No ELITE or STRONG picks today &mdash; showing model-based GOOD/AVERAGE plays. AVERAGE = model only (no sharp consensus). Smaller units or skip.
    </div>""" if fallback_mode else ""
    return parlay_html + fallback_banner + filter_bar + f"""
    <p style="font-size:0.75rem;color:var(--muted);margin-bottom:0.75rem">{total_picks} picks today &nbsp;·&nbsp; Sorted by game time &nbsp;·&nbsp; Check Bovada odds before betting</p>
    <div class="table-wrap">
      <table class="data-table" id="mlb-table"><thead><tr>
        <th class="tt" data-tip="Overall bet quality based on EV% and probability">Rating</th>
        <th>Track</th>
        <th>Player</th>
        <th class="tt" data-tip="The stat being bet on (Hits, Strikeouts, Outs, etc.)">Stat</th>
        <th class="tt" data-tip="The book's over/under line for this stat">Line</th>
        <th class="tt" data-tip="Model's projected stat value based on season averages">Projection</th>
        <th class="tt" data-tip="Projection minus line. Positive = model favors OVER, negative = model favors UNDER">Edge</th>
        <th class="tt" data-tip="Whether the model recommends OVER or UNDER">Pick</th>
        <th class="tt" data-tip="Bovada's actual odds — this is what you can bet">Bovada</th>
        <th class="tt" data-tip="Model's estimated probability of winning after removing vig">Fair%</th>
        <th class="tt" data-tip="Expected Value % — how much profit per $100 wagered if the model is correct. 5% EV means +$5 per $100 bet long term">EV%</th>
        <th class="tt" data-tip="Key reasons the model likes this pick">Why</th>
        <th>Matchup</th>
        <th>Time</th>
      </tr></thead><tbody>{rows_html}</tbody></table>
    </div>""", fallback_mode


def build_game_lines(df):
    if df.empty:
        return '<p class="empty-msg">No game lines data. Run compute_ev_game_lines.py first.</p>'

    positive = df[df["ev_pct"] > 0].copy() if "ev_pct" in df.columns else df.copy()
    if positive.empty:
        return '<p class="empty-msg">No positive EV game lines found today.</p>'

    # Hide heavy-juice lines (-155 or worse) from straight-bet display; parlay maker uses full set
    if "best_odds" in positive.columns:
        odds_numeric = pd.to_numeric(positive["best_odds"], errors="coerce")
        positive = positive[odds_numeric.isna() | (odds_numeric > -155)]
    if positive.empty:
        return '<p class="empty-msg">No positive EV game lines under -155 juice today.</p>'

    positive = positive.sort_values("ev_pct", ascending=False)

    filter_bar = """
    <div class="filter-bar">
      <select id="gl-market" class="filter-select" onchange="applyGLFilters()">
        <option value="">All Markets</option>
        <option value="Total">Totals</option>
        <option value="Spread">Spreads</option>
        <option value="Moneyline">Moneyline</option>
        <option value="F5 ML">F5 ML</option>
        <option value="F5 Total">F5 Total</option>
      </select>
      <button class="filter-reset" onclick="resetGLFilters()">Reset</button>
    </div>"""

    rows_html = ""
    for _, r in positive.iterrows():
        market    = str(r.get("market",""))
        side      = str(r.get("side",""))
        ev        = r.get("ev_pct", 0)
        rating    = str(r.get("rating","POOR"))
        matchup   = str(r.get("matchup","-"))
        game_time = str(r.get("game_time","-"))
        best_odds = fmt_odds(r.get("best_odds",""))
        pick_key  = f"{matchup}|{market}|{side}"
        try: ev_f = float(ev)
        except: ev_f = 0

        rows_html += f"""
        <tr data-market="{market}">
          <td>{rating_badge(rating)}</td>
          <td class="muted">{matchup}</td>
          <td><span class="stat-pill">{market}</span></td>
          <td><strong>{side}</strong></td>
          <td style="font-family:monospace">{best_odds}</td>
          <td style="font-family:monospace">{fmt_odds(r.get('fair_odds',''))}</td>
          <td>{fmt_pct(float(r.get('fair_prob',0))*100)}</td>
          <td>{ev_badge(ev)}</td>
          <td class="muted">{game_time}</td>
          <td>
            <div style="display:flex;align-items:center;gap:4px">
              <select id="u_{pick_key}" style="background:var(--surface2);border:1px solid var(--border);border-radius:4px;padding:0.15rem 0.3rem;color:var(--text);font-size:0.72rem;font-family:monospace">
                <option value="1">1u</option><option value="2">2u</option>
                <option value="3">3u</option><option value="4">4u</option><option value="5">5u</option>
              </select>
              <button onclick="trackBet('{side}','{market}','-','{side}','{best_odds}','{matchup}','{game_time}','MLB')"
                id="tb_{pick_key}"
                style="background:rgba(0,229,160,0.1);border:1px solid rgba(0,229,160,0.3);color:#00e5a0;border-radius:4px;padding:0.2rem 0.5rem;font-size:0.7rem;cursor:pointer;font-family:monospace;white-space:nowrap">
                Track
              </button>
            </div>
          </td>
        </tr>"""

    return filter_bar + f"""
    <p style="font-size:0.75rem;color:var(--muted);margin-bottom:0.75rem">
      Fair odds from consensus across DraftKings, FanDuel, BetMGM, Caesars. Check Bovada before betting.
    </p>
    <div class="table-wrap">
      <table class="data-table" id="gl-table"><thead><tr>
        <th>Rating</th><th>Matchup</th><th>Market</th><th>Side</th>
        <th>Best Odds</th><th>Fair Odds</th><th>Fair%</th><th>EV%</th><th>Time</th><th>Track</th>
      </tr></thead><tbody>{rows_html}</tbody></table>
    </div>"""


def build_hr_props(df, hist_df=None):
    """HR home run props panel — top most-likely HR hitters ranked by Poisson P(HR)."""
    if df.empty:
        return '<p class="empty-msg">No HR props data. Run compute_ev_mlb.py first.</p>'

    # Show all players with P(HR) >= 10%
    show = df[df["prob_hr"] >= 0.10].copy() if "prob_hr" in df.columns else df.copy()
    if show.empty:
        return '<p class="empty-msg">No HR prop data available today.</p>'

    def odds_str(v):
        try:
            v = int(float(v))
            return f"+{v}" if v > 0 else str(v)
        except: return "-"

    def ev_color(ev):
        try:
            e = float(ev)
            if e >= 10:   return "#00e5a0"
            if e >= 2:    return "#44d4a0"
            if e >= -5:   return "#f5a623"
            return "#f04e4e"
        except: return "#5a6172"

    rows_html = ""
    for _, r in show.iterrows():
        p_hr      = r.get("prob_hr_pct", round(float(r.get("prob_hr", 0)) * 100, 1))
        ev        = r.get("ev_pct", float("nan"))
        pos_ev    = str(r.get("positive_ev", "")).lower() in ("true", "1")
        row_style = 'style="background:rgba(0,229,160,0.04)"' if pos_ev else ""

        try: ev_str = f"{float(ev):+.1f}%"
        except: ev_str = "-"

        try: p_str = f"{float(p_hr):.1f}%"
        except: p_str = "-"

        form5    = r.get("form_5g", "")
        park     = r.get("park_component", "")
        starter  = str(r.get("opp_starter", "")).strip()
        hand     = str(r.get("throw_hand", "")).strip()
        matchup  = str(r.get("matchup", "-"))
        player   = str(r.get("player_name", "-"))
        odds     = odds_str(r.get("best_odds", ""))
        book     = str(r.get("best_book", "")).replace("betonlineag", "BetOnline").replace("espnbet", "ESPN BET").replace("betrivers", "BetRivers")
        implied  = r.get("implied_prob", float("nan"))

        try: imp_str = f"{float(implied)*100:.1f}%"
        except: imp_str = "-"

        try: form5_str = f"{float(form5):.2f}"
        except: form5_str = "-"

        try: park_str = f"{float(park):.3f}"
        except: park_str = "-"

        starter_label = f"{starter} ({hand})" if starter and starter not in ("nan", "") else "TBD"
        ev_col   = ev_color(ev)
        rank_col = "#00e5a0" if pos_ev else "var(--muted)"

        rows_html += f"""
        <tr {row_style}>
          <td style="font-family:monospace;color:{rank_col};font-weight:700">{p_str}</td>
          <td><strong>{player}</strong></td>
          <td class="muted" style="font-size:0.78rem">{matchup}</td>
          <td style="font-family:monospace">{odds}</td>
          <td style="font-family:monospace;font-size:0.75rem;color:var(--muted)">{imp_str}</td>
          <td style="font-family:monospace;color:{ev_col};font-weight:600">{ev_str}</td>
          <td style="font-family:monospace;color:var(--muted);font-size:0.75rem">{form5_str}</td>
          <td style="font-family:monospace;color:var(--muted);font-size:0.75rem">{park_str}</td>
          <td style="font-size:0.75rem;color:var(--muted)">{starter_label}</td>
          <td style="font-size:0.72rem;color:var(--muted)">{book}</td>
        </tr>"""

    pos_count = len(show[show["positive_ev"] == True]) if "positive_ev" in show.columns else 0

    today_section = f"""
    <p style="font-size:0.75rem;color:var(--muted);margin-bottom:0.75rem">
      P(HR) = Poisson probability of hitting &ge;1 HR · PROJ_HR incorporates park, pitcher, form, weather &amp; Statcast ·
      <span style="color:#00e5a0">Green rows = positive EV vs best available odds</span> · {pos_count} +EV picks today
    </p>
    <div class="table-wrap">
      <table class="data-table" id="hr-table"><thead><tr>
        <th><span class="tt" data-tip="Poisson probability of hitting at least 1 HR today. P = 1 - e^(-lambda) where lambda = PROJ_HR per game.">P(HR)</span></th>
        <th><span class="tt" data-tip="Batter name.">Player</span></th>
        <th><span class="tt" data-tip="Today's game (Away @ Home).">Matchup</span></th>
        <th><span class="tt" data-tip="Best available OVER odds across all books. Higher = better payout for you.">Best Odds</span></th>
        <th><span class="tt" data-tip="Implied probability baked into the best odds. P(HR) above this = positive EV.">Implied</span></th>
        <th><span class="tt" data-tip="Expected value vs best available odds. Positive = model sees edge. Formula: P(HR) x payout - P(no HR).">EV%</span></th>
        <th><span class="tt" data-tip="Average HRs per game over the last 5 games. Captures current hot streak.">Form (5g)</span></th>
        <th><span class="tt" data-tip="Ballpark HR factor from historical data. Above 1.0 = HR-friendly park, below 1.0 = pitcher-friendly.">Park Factor</span></th>
        <th><span class="tt" data-tip="Today's opposing starting pitcher and their handedness (L/R). PROJ_HR already adjusts for platoon splits.">Opp Starter</span></th>
        <th><span class="tt" data-tip="Book offering the best OVER odds for this player today.">Book</span></th>
      </tr></thead><tbody>{rows_html}</tbody></table>
    </div>"""

    # ── History section ───────────────────────────────────────────────────────
    hist_section = ""
    if hist_df is not None and not hist_df.empty:
        graded = hist_df[hist_df["result"].str.upper().isin(["WIN", "LOSS"])].copy()

        # Summary stats
        total    = len(graded)
        wins     = (graded["result"].str.upper() == "WIN").sum()
        hit_rate = wins / total if total > 0 else 0.0
        try:
            total_profit = graded["profit"].astype(float).sum()
        except Exception:
            total_profit = 0.0

        # EV picks only stats
        ev_graded = graded[graded.get("positive_ev", graded["result"].apply(lambda _: False)).astype(str).str.lower().isin(["true", "1"])]
        ev_wins   = (ev_graded["result"].str.upper() == "WIN").sum() if not ev_graded.empty else 0
        ev_total  = len(ev_graded)
        ev_rate   = ev_wins / ev_total if ev_total > 0 else 0.0
        try:
            ev_profit = ev_graded["profit"].astype(float).sum()
        except Exception:
            ev_profit = 0.0

        profit_color = "#00e5a0" if total_profit >= 0 else "#f04e4e"
        ev_profit_color = "#00e5a0" if ev_profit >= 0 else "#f04e4e"

        summary_html = f"""
        <div class="stat-strip" style="margin:2rem 0 1rem">
          <div class="stat-chip">
            <span class="chip-val">{total}</span>
            <span class="chip-lbl">Graded</span>
          </div>
          <div class="stat-chip">
            <span class="chip-val" style="color:#00e5a0">{wins}</span>
            <span class="chip-lbl">Hit HR</span>
          </div>
          <div class="stat-chip">
            <span class="chip-val">{hit_rate:.0%}</span>
            <span class="chip-lbl">Hit Rate (all)</span>
          </div>
          <div class="stat-chip">
            <span class="chip-val" style="color:{profit_color}">{total_profit:+.1f}u</span>
            <span class="chip-lbl">P&amp;L (all, 1u)</span>
          </div>
          <div class="stat-chip">
            <span class="chip-val">{ev_rate:.0%}</span>
            <span class="chip-lbl">Hit Rate (+EV only)</span>
          </div>
          <div class="stat-chip">
            <span class="chip-val" style="color:{ev_profit_color}">{ev_profit:+.1f}u</span>
            <span class="chip-lbl">P&amp;L (+EV only)</span>
          </div>
        </div>"""

        # Group by date descending
        hist_rows_html = ""
        for day, day_grp in graded.sort_values("date", ascending=False).groupby("date", sort=False):
            day_wins = (day_grp["result"].str.upper() == "WIN").sum()
            day_total = len(day_grp)
            try:
                day_profit = day_grp["profit"].astype(float).sum()
            except Exception:
                day_profit = 0.0
            day_profit_color = "#00e5a0" if day_profit >= 0 else "#f04e4e"

            day_rows = ""
            for _, hr in day_grp.sort_values("prob_hr", ascending=False).iterrows():
                result   = str(hr.get("result", "")).upper()
                hit      = result == "WIN"
                player   = str(hr.get("player_name", "-"))
                matchup  = str(hr.get("matchup", "-"))
                prob_pct = hr.get("prob_hr_pct", round(float(hr.get("prob_hr", 0)) * 100, 1))
                odds     = hr.get("best_odds", "")
                ev       = hr.get("ev_pct", float("nan"))
                profit   = hr.get("profit", float("nan"))
                is_ev    = str(hr.get("positive_ev", "")).lower() in ("true", "1")

                try: odds_disp = f"+{int(float(odds))}" if float(odds) > 0 else str(int(float(odds)))
                except: odds_disp = "-"
                try: prob_str = f"{float(prob_pct):.1f}%"
                except: prob_str = "-"
                try: ev_str = f"{float(ev):+.1f}%"
                except: ev_str = "-"
                try: profit_str = f"{float(profit):+.2f}u"
                except: profit_str = "-"

                result_badge = (
                    '<span style="background:rgba(0,229,160,0.15);color:#00e5a0;padding:2px 8px;border-radius:4px;font-size:0.72rem;font-weight:700;font-family:monospace">HR</span>'
                    if hit else
                    '<span style="background:rgba(240,78,78,0.1);color:#f04e4e;padding:2px 8px;border-radius:4px;font-size:0.72rem;font-weight:700;font-family:monospace">NO</span>'
                )
                ev_badge_html = '<span style="background:rgba(0,229,160,0.12);color:#00e5a0;padding:1px 5px;border-radius:3px;font-size:0.65rem;font-family:monospace">+EV</span> ' if is_ev else ""
                profit_color2 = "#00e5a0" if hit else "#f04e4e"

                day_rows += f"""
                <tr>
                  <td>{result_badge}</td>
                  <td>{ev_badge_html}<strong>{player}</strong></td>
                  <td class="muted" style="font-size:0.78rem">{matchup}</td>
                  <td style="font-family:monospace">{prob_str}</td>
                  <td style="font-family:monospace">{odds_disp}</td>
                  <td style="font-family:monospace;font-size:0.75rem;color:var(--muted)">{ev_str}</td>
                  <td style="font-family:monospace;color:{profit_color2};font-weight:600">{profit_str}</td>
                </tr>"""

            hist_rows_html += f"""
            <details style="margin-bottom:0.5rem">
              <summary class="day-summary">
                <span class="day-date">{day}</span>
                <span class="day-record">{day_wins}/{day_total} hit &nbsp;·&nbsp;
                  <span style="color:{day_profit_color}">{day_profit:+.1f}u</span>
                </span>
              </summary>
              <div style="padding:0.5rem 0">
                <table class="data-table mini"><thead><tr>
                  <th>Result</th><th>Player</th><th>Matchup</th>
                  <th>P(HR)</th><th>Odds</th><th>EV%</th><th>P&amp;L</th>
                </tr></thead><tbody>{day_rows}</tbody></table>
              </div>
            </details>"""

        hist_section = f"""
        <div style="margin-top:2.5rem">
          <div class="tab-header" style="margin-bottom:0.5rem">
            <div class="tab-title" style="font-size:1.1rem">HR History</div>
            <div class="tab-subtitle">Auto-graded each morning · 1 unit per prediction · +EV picks tracked separately</div>
          </div>
          {summary_html}
          <div>{hist_rows_html}</div>
        </div>"""

    return today_section + hist_section


def build_history(df):
    if df.empty:
        return '<p class="empty-msg">No graded picks yet. Results appear after games finish.</p>'

    df = df.copy()
    df["hit_result"] = df["hit_result"].astype(str).str.upper().str.strip()
    df = df[df["hit_result"].isin(["WIN","LOSS","PUSH"])].copy()

    if df.empty:
        return '<p class="empty-msg">No graded picks yet.</p>'

    if "game_date" in df.columns:
        df["display_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    else:
        df["display_date"] = pd.NaT

    df = df.sort_values("display_date", ascending=False)

    wins     = int((df["hit_result"] == "WIN").sum())
    losses   = int((df["hit_result"] == "LOSS").sum())
    pushes   = int((df["hit_result"] == "PUSH").sum())
    graded   = wins + losses
    win_rate = f"{wins/graded*100:.1f}%" if graded > 0 else "-"
    pnl      = calc_pnl(df)

    summary_html = f"""
    <div class="stat-strip">
      <div class="stat-chip"><span class="chip-val">{wins+losses+pushes}</span><span class="chip-lbl">Total Picks</span></div>
      <div class="stat-chip" style="border-color:rgba(0,229,160,0.3)"><span class="chip-val" style="color:#00e5a0">{wins}</span><span class="chip-lbl">Wins</span></div>
      <div class="stat-chip" style="border-color:rgba(240,78,78,0.3)"><span class="chip-val" style="color:#f04e4e">{losses}</span><span class="chip-lbl">Losses</span></div>
      <div class="stat-chip" style="border-color:rgba(245,166,35,0.3)"><span class="chip-val" style="color:#f5a623">{pushes}</span><span class="chip-lbl">Pushes</span></div>
      <div class="stat-chip"><span class="chip-val">{win_rate}</span><span class="chip-lbl">Win Rate</span></div>
      {pnl_chip("Live", pnl)}
    </div>
    <p style="font-size:0.75rem;color:var(--muted);margin-bottom:1.5rem">
      P&L assumes actual odds when available, -110 otherwise. Unit = $10.
    </p>"""

    filter_html = """
    <div class="filter-bar" style="margin-bottom:1rem">
      <select id="h-stat" class="filter-select" onchange="filterHistory()">
        <option value="">All Stats</option>
        <option value="hits">Hits</option><option value="runs">Runs</option>
        <option value="rbi">RBI</option><option value="hr">HR</option>
        <option value="tb">Total Bases</option><option value="k">Strikeouts</option>
        <option value="outs">Outs Recorded</option>
      </select>
      <select id="h-side" class="filter-select" onchange="filterHistory()">
        <option value="">All Sides</option>
        <option value="OVER">OVER</option><option value="UNDER">UNDER</option>
      </select>
      <select id="h-result" class="filter-select" onchange="filterHistory()">
        <option value="">All Results</option>
        <option value="WIN">WIN</option><option value="LOSS">LOSS</option>
        <option value="PUSH">PUSH</option>
      </select>
      <button class="filter-reset" onclick="resetHistFilters()">Reset</button>
    </div>"""

    # ── Daily P&L tracker ────────────────────────────────────────────────────
    dates_sorted = sorted(df["display_date"].dropna().unique())  # oldest -> newest
    daily_pnl    = []   # list of (date, day_pnl, cumulative_pnl)
    running      = 0.0
    for date in dates_sorted:
        day_df  = df[df["display_date"] == date]
        day_p   = calc_pnl(day_df)
        running += day_p
        daily_pnl.append((date, day_p, running))

    # Build chart bars (scaled to tallest absolute value)
    max_abs = max((abs(p) for _, p, _ in daily_pnl), default=1) or 1
    BAR_MAX_H = 56   # px for tallest bar

    bar_items = ""
    for date, day_p, cum_p in daily_pnl:
        h       = max(3, int(abs(day_p) / max_abs * BAR_MAX_H))
        color   = "#00e5a0" if day_p >= 0 else "#f04e4e"
        sign    = "+" if day_p >= 0 else ""
        csign   = "+" if cum_p >= 0 else ""
        label   = str(date)[5:]   # MM-DD
        tip     = f"{sign}${day_p:.2f} · cumulative {csign}${cum_p:.2f}"
        bar_items += f"""
        <div class="pnl-bar-wrap" title="{tip}">
          <div class="pnl-bar" style="height:{h}px;background:{color};{'margin-top:auto' if day_p >= 0 else 'margin-bottom:auto'}"></div>
          <div class="pnl-bar-lbl">{label}</div>
        </div>"""

    # Running-total line data for mini sparkline (SVG path)
    n = len(daily_pnl)
    if n > 1:
        min_c  = min(c for _, _, c in daily_pnl)
        max_c  = max(c for _, _, c in daily_pnl)
        span   = max_c - min_c or 1
        W, H   = 420, 50
        pts    = []
        for i, (_, _, c) in enumerate(daily_pnl):
            x = int(i / (n - 1) * W)
            y = int(H - (c - min_c) / span * H)
            pts.append(f"{x},{y}")
        zero_y = int(H - (0 - min_c) / span * H)
        zero_y = max(0, min(H, zero_y))
        sparkline_html = f"""
        <div style="margin-top:0.75rem">
          <span style="font-size:0.72rem;color:var(--muted);font-family:monospace">CUMULATIVE P&L</span>
          <svg width="{W}" height="{H+4}" viewBox="0 0 {W} {H+4}" style="display:block;overflow:visible">
            <line x1="0" y1="{zero_y}" x2="{W}" y2="{zero_y}" stroke="#ffffff18" stroke-width="1" stroke-dasharray="4,3"/>
            <polyline points="{' '.join(pts)}" fill="none" stroke="#3d8ef8" stroke-width="2" stroke-linejoin="round"/>
          </svg>
        </div>"""
    else:
        sparkline_html = ""

    # Final cumulative badge
    cum_color = "#00e5a0" if running >= 0 else "#f04e4e"
    cum_sign  = "+" if running >= 0 else ""
    tracker_html = f"""
    <div class="profit-tracker" style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:1rem 1.25rem;margin-bottom:1.5rem">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem">
        <span style="font-size:0.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;font-family:monospace">Daily P&L Tracker</span>
        <span style="font-size:1.1rem;font-weight:700;font-family:monospace;color:{cum_color}">{cum_sign}${running:.2f} cumulative</span>
      </div>
      <div class="pnl-bars" style="display:flex;align-items:flex-end;gap:4px;height:{BAR_MAX_H+4}px;overflow-x:auto;padding-bottom:2px">
        {bar_items}
      </div>
      {sparkline_html}
    </div>
    <style>
      .pnl-bar-wrap {{display:flex;flex-direction:column;align-items:center;min-width:32px;cursor:default}}
      .pnl-bar {{width:22px;border-radius:3px 3px 0 0;transition:opacity .15s}}
      .pnl-bar-wrap:hover .pnl-bar {{opacity:0.75}}
      .pnl-bar-lbl {{font-size:0.6rem;color:var(--muted);margin-top:3px;white-space:nowrap;font-family:monospace}}
    </style>"""

    # ── Per-day collapsible rows ─────────────────────────────────────────────
    dates     = list(reversed(dates_sorted))   # newest first for display
    day_pnl_map = {date: day_p for date, day_p, _ in daily_pnl}
    days_html = ""
    for i, date in enumerate(dates):
        day_df    = df[df["display_date"] == date]
        day_wins  = (day_df["hit_result"] == "WIN").sum()
        day_total = len(day_df[day_df["hit_result"].isin(["WIN","LOSS"])])
        day_wr    = f"{day_wins}/{day_total}" if day_total > 0 else "-"
        expanded  = "open" if i == 0 else ""
        dp        = day_pnl_map.get(date, 0.0)
        dp_color  = "#00e5a0" if dp >= 0 else "#f04e4e"
        dp_sign   = "+" if dp >= 0 else ""
        dp_str    = f'<span style="font-family:monospace;font-weight:700;color:{dp_color};margin-left:0.75rem">{dp_sign}${dp:.2f}</span>'

        rows_html = ""
        for _, r in day_df.iterrows():
            stat_label = STAT_LABELS.get(str(r.get("stat","")), str(r.get("stat","")).upper())
            actual     = r.get("actual_stat", np.nan)
            side       = str(r.get("side", r.get("recommendation",""))).upper()
            res        = str(r.get("hit_result","")).upper()
            ev         = r.get("ev_pct", np.nan)
            matchup    = str(r.get("matchup", f"{r.get('away_team','?')} @ {r.get('home_team','?')}"))

            rows_html += f"""
            <tr data-stat="{r.get('stat','')}" data-side="{side}" data-result="{res}">
              <td><strong>{r.get('player_name','-')}</strong></td>
              <td><span class="stat-pill">{stat_label}</span></td>
              <td>{fmt_num(r.get('line',''),1)}</td>
              <td style="color:#3d8ef8;font-weight:600">{fmt_num(r.get('projection',''),2)}</td>
              <td>{recommendation_badge(side)}</td>
              <td>{fmt_odds(r.get('bet_odds',r.get('price','')))}</td>
              <td>{fmt_num(actual,1) if pd.notna(actual) else '-'}</td>
              <td>{fmt_pct(ev) if pd.notna(ev) else '-'}</td>
              <td class="muted" style="font-size:0.75rem">{matchup}</td>
              <td>{result_badge(res)}</td>
            </tr>"""

        days_html += f"""
        <details class="day-group" {expanded}>
          <summary class="day-summary">
            <span class="day-date">{date}</span>
            <span class="day-record">{day_wr} &nbsp; {len(day_df)} picks {dp_str}</span>
          </summary>
          <div class="table-wrap" style="margin:0.5rem 0 1rem 0">
            <table class="data-table hist-table"><thead><tr>
              <th>Player</th><th>Stat</th><th>Line</th><th>Proj</th>
              <th>Pick</th><th>Odds</th><th>Actual</th><th>EV%</th><th>Matchup</th><th>Result</th>
            </tr></thead><tbody>{rows_html}</tbody></table>
          </div>
        </details>"""

    return summary_html + filter_html + tracker_html + f'<div id="hist-days">{days_html}</div>'


BOVADA_START_DATE = "2026-03-29"  # Date personal tracking started

def build_performance(df):
    if df.empty:
        return '<p class="empty-msg">No graded picks yet.</p>'

    df = df.copy()
    df["hit_result"] = df["hit_result"].astype(str).str.upper().str.strip()
    df = df[df["hit_result"].isin(["WIN","LOSS","PUSH"])].copy()

    if df.empty:
        return '<p class="empty-msg">No graded picks yet.</p>'

    # Filter to picks since personal tracking started (March 29)
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        start_date      = pd.to_datetime(BOVADA_START_DATE)
        df_clean        = df[df["game_date"] >= start_date].copy()
        df_all          = df.copy()
    else:
        df_clean = df.copy()
        df_all   = df.copy()

    def mini_table(grp_df, group_col, label):
        if grp_df.empty: return ""
        rows = ""
        for val, g in grp_df.groupby(group_col):
            w  = (g["hit_result"] == "WIN").sum()
            l  = (g["hit_result"] == "LOSS").sum()
            wr = f"{w/(w+l)*100:.1f}%" if (w+l) > 0 else "-"
            pnl = calc_pnl(g)
            sign = "+" if pnl >= 0 else ""
            color = "#00e5a0" if pnl >= 0 else "#f04e4e"
            stat_label = STAT_LABELS.get(str(val), str(val).upper()) if group_col == "stat" else str(val)
            rows += f"<tr><td><strong>{stat_label}</strong></td><td>{w+l}</td><td style='color:#00e5a0'>{w}</td><td style='color:#f04e4e'>{l}</td><td>{wr}</td><td style='color:{color};font-family:monospace'>{sign}${pnl:.2f}</td></tr>"
        return f"""
        <div class="perf-card">
          <h3 class="section-label">{label}</h3>
          <div class="table-wrap">
            <table class="data-table mini"><thead><tr>
              <th>{group_col.title()}</th><th>Picks</th><th>W</th><th>L</th><th>Win%</th><th>P&L</th>
            </tr></thead><tbody>{rows}</tbody></table>
          </div>
        </div>"""

    # Overall stats — use clean (post-Bovada) data as primary
    use_df = df_clean if not df_clean.empty else df_all
    wins   = int((use_df["hit_result"] == "WIN").sum())
    losses = int((use_df["hit_result"] == "LOSS").sum())
    pushes = int((use_df["hit_result"] == "PUSH").sum())
    graded = wins + losses
    wr     = f"{wins/graded*100:.1f}%" if graded > 0 else "-"
    pnl    = calc_pnl(use_df)

    # All-time stats for reference
    wins_all = int((df_all["hit_result"] == "WIN").sum())
    losses_all = int((df_all["hit_result"] == "LOSS").sum())
    graded_all = wins_all + losses_all
    wr_all = f"{wins_all/graded_all*100:.1f}%" if graded_all > 0 else "-"
    pnl_all = calc_pnl(df_all)
    sign_all = "+" if pnl_all >= 0 else ""

    summary = f"""
    <div style="margin-bottom:0.5rem">
      <span style="font-size:0.72rem;font-family:monospace;color:#00e5a0;text-transform:uppercase;letter-spacing:0.05em">
        Since March 29 (personal tracking start)
      </span>
    </div>
    <div class="stat-strip">
      <div class="stat-chip"><span class="chip-val">{graded}</span><span class="chip-lbl">Graded</span></div>
      <div class="stat-chip" style="border-color:rgba(0,229,160,0.3)"><span class="chip-val" style="color:#00e5a0">{wins}</span><span class="chip-lbl">Wins</span></div>
      <div class="stat-chip" style="border-color:rgba(240,78,78,0.3)"><span class="chip-val" style="color:#f04e4e">{losses}</span><span class="chip-lbl">Losses</span></div>
      <div class="stat-chip"><span class="chip-val">{wr}</span><span class="chip-lbl">Win Rate</span></div>
      {pnl_chip("Live", pnl)}
    </div>
    <p style="font-size:0.72rem;color:var(--muted);margin-bottom:0.5rem">
      All-time (incl. pre-Bovada): {graded_all} picks · {wr_all} · {sign_all}${pnl_all:.2f}
    </p>
    <p style="font-size:0.75rem;color:var(--muted);margin-bottom:1.5rem">
      Performance based on all model picks — not just bets placed.
    </p>"""

    by_stat  = mini_table(use_df, "stat", "By Stat Type")
    side_col = "side" if "side" in use_df.columns else "recommendation"
    by_side  = mini_table(use_df, side_col, "By Side (Over / Under)")

    rating_col = next((c for c in use_df.columns if c.lower() == "bet_rating"), None)
    if rating_col:
        rating_order = ["ELITE", "STRONG", "GOOD", "AVERAGE", "POOR", "BAD"]
        use_df = use_df.copy()
        use_df[rating_col] = pd.Categorical(use_df[rating_col], categories=rating_order, ordered=True)
        by_rating = mini_table(use_df, rating_col, "By Rating Tier")
    else:
        by_rating = ""

    return summary + f'<div class="perf-grid">{by_stat}{by_side}{by_rating}</div>'


def build_4game_series(df):
    if df.empty:
        return '<p class="empty-msg">No 4-game series data. Run find_4game_series_mlb.py first.</p>'

    from datetime import date
    today = date.today()

    df = df.copy()
    df["game_date"]  = pd.to_datetime(df["game_date"]).dt.date
    df["start_date"] = pd.to_datetime(df["start_date"]).dt.date
    df["end_date"]   = pd.to_datetime(df["end_date"]).dt.date

    # Work from home-team perspective only (canonical: TEAM == HOME_TEAM)
    home_df = df[df["home_away"] == "home"].copy()

    # ── Historical home win rate in 4-game series ──────────────────────────
    completed = home_df[home_df["status"] == "Final"].copy()
    if not completed.empty and "home_win" in completed.columns:
        series_results = completed.groupby("series_key").agg(
            home_wins=("home_win", lambda x: x.sum()),
            games_played=("home_win", "count")
        ).reset_index()
        full_series = series_results[series_results["games_played"] == 4]
        home_won_at_least_1 = (full_series["home_wins"] >= 1).sum()
        total_full = len(full_series)
        hist_pct = f"{home_won_at_least_1/total_full*100:.0f}%" if total_full > 0 else "N/A"
        hist_note = f"{home_won_at_least_1}/{total_full} completed series — home team won at least 1 game"
    else:
        hist_pct  = "N/A"
        hist_note = "No completed series yet this season"

    def fmt_ml(val):
        """Format American moneyline for display."""
        try:
            v = float(val)
            if pd.isna(v): return "—"
            return f"+{int(v)}" if v > 0 else str(int(v))
        except Exception:
            return "—"

    def units_html(row, col, fallback_games_done=0):
        """Return colored unit badge from a column, or compute 2^n fallback."""
        val = row.get(col)
        try:
            v = float(val)
            if not pd.isna(v):
                return f'<span style="color:#f5a623;font-family:monospace;font-weight:700">{int(v)}u</span>'
        except Exception:
            pass
        fb = 2 ** fallback_games_done
        return f'<span style="color:#f5a623;font-family:monospace;font-weight:700">{int(fb)}u</span>'

    # ── Active series (started, not over) ─────────────────────────────────
    active_series_html = ""
    in_progress = home_df[
        (home_df["start_date"] <= today) & (home_df["end_date"] >= today)
    ].copy()

    if not in_progress.empty:
        active_rows = ""
        for series_key, sg in in_progress.groupby("series_key"):
            sg = sg.sort_values("game_num")
            home_team = sg.iloc[0].get("home_team", sg.iloc[0].get("team", "?"))
            away_team = sg.iloc[0].get("away_team", sg.iloc[0].get("opponent", "?"))

            done           = sg[sg["status"] == "Final"]
            hw_count       = int(done["home_win"].sum()) if not done.empty and "home_win" in done.columns else 0
            games_done     = len(done)
            home_already_won = hw_count >= 1

            todays_game = sg[sg["game_date"] == today]
            has_today   = not todays_game.empty

            # Qualification / fatigue / travel indicators for today's or next game
            edge_parts = []
            if has_today:
                tg = todays_game.iloc[0]
                if tg.get("high_fatigue_spot"):
                    edge_parts.append('<span style="color:#f5a623;font-size:0.68rem">FATIGUE</span>')
                if tg.get("traveled_in"):
                    edge_parts.append('<span style="color:#a78bfa;font-size:0.68rem">TRAVEL</span>')
                qual = tg.get("qualified_home_bet")
                qual_reason = tg.get("qualified_reason", "")
                if str(qual).lower() in ("false", "0", "nan", ""):
                    edge_parts.append(f'<span style="color:#f04e4e;font-size:0.68rem" title="{qual_reason}">SKIP</span>')
            edge_html = " ".join(edge_parts)

            if home_already_won:
                status_badge = '<span style="background:rgba(0,229,160,0.15);color:#00e5a0;padding:0.2rem 0.5rem;border-radius:4px;font-size:0.72rem;font-family:monospace;font-weight:700">HOME WON - DONE</span>'
                action_html  = '<span style="color:var(--muted);font-size:0.75rem">No bet needed</span>'
                mg_html      = "-"
                capped_html  = "-"
            else:
                next_game = sg[sg["game_date"] >= today]
                if next_game.empty:
                    status_badge = '<span style="background:rgba(240,78,78,0.15);color:#f04e4e;padding:0.2rem 0.5rem;border-radius:4px;font-size:0.72rem;font-family:monospace;font-weight:700">SWEPT</span>'
                    action_html  = '<span style="color:#f04e4e;font-size:0.75rem">Home swept</span>'
                    mg_html      = "-"
                    capped_html  = "-"
                else:
                    next_row    = next_game.iloc[0]
                    game_num    = int(next_row.get("game_num", 0))
                    ml_str      = fmt_ml(next_row.get("home_moneyline"))
                    mg_html     = units_html(next_row, "full_martingale_units", games_done)
                    capped_html = units_html(next_row, "capped_units", min(games_done, 3))

                    qual = next_row.get("qualified_home_bet")
                    qual_ok = str(qual).lower() not in ("false", "0", "nan", "")

                    if has_today and game_num <= 4:
                        urgency     = "color:#f04e4e;font-weight:700" if game_num == 4 else "color:#f5a623;font-weight:600"
                        last_chance = " &mdash; LAST CHANCE" if game_num == 4 else ""
                        status_badge = f'<span style="background:rgba(245,166,35,0.15);{urgency};padding:0.2rem 0.5rem;border-radius:4px;font-size:0.72rem;font-family:monospace">GAME {game_num} TODAY{last_chance}</span>'
                        if qual_ok:
                            action_html = f'<span style="color:#f5a623;font-weight:600">BET on {home_team} ML {ml_str}</span>'
                        else:
                            reason = next_row.get("qualified_reason", "filtered")
                            action_html = f'<span style="color:#5a6172;font-size:0.75rem">SKIP ({reason})</span>'
                    else:
                        status_badge = f'<span style="background:rgba(90,97,114,0.15);color:var(--muted);padding:0.2rem 0.5rem;border-radius:4px;font-size:0.72rem;font-family:monospace">GAME {game_num} — {next_row["game_date"]}</span>'
                        action_html  = f'<span style="color:var(--muted);font-size:0.75rem">{home_team} ML {ml_str}</span>'

            # Score history
            score_parts = []
            for _, gr in sg.sort_values("game_num").iterrows():
                gn = int(gr["game_num"])
                if gr["status"] == "Final":
                    import math
                    hs_raw = gr.get("home_score"); hs = 0 if (hs_raw is None or (isinstance(hs_raw, float) and math.isnan(hs_raw))) else int(hs_raw)
                    as_raw = gr.get("away_score"); as_ = 0 if (as_raw is None or (isinstance(as_raw, float) and math.isnan(as_raw))) else int(as_raw)
                    won = gr.get("home_win")
                    col = "#00e5a0" if won else "#f04e4e"
                    score_parts.append(f'<span style="color:{col};font-family:monospace">G{gn}:{hs}-{as_}</span>')
                elif gr["game_date"] == today:
                    score_parts.append(f'<span style="color:#f5a623;font-family:monospace">G{gn}:Today</span>')
                else:
                    score_parts.append(f'<span style="color:var(--muted);font-family:monospace">G{gn}:—</span>')
            scores_html = " &nbsp; ".join(score_parts)

            active_rows += f"""
            <tr>
              <td><strong>{home_team}</strong><br><span style="color:var(--muted);font-size:0.75rem">vs {away_team}</span></td>
              <td>{scores_html}</td>
              <td>{status_badge}</td>
              <td class="tt" data-tip="Full Martingale: 1u-2u-4u-8u">{mg_html}</td>
              <td class="tt" data-tip="Capped: 1u-2u-3u-4u">{capped_html}</td>
              <td>{edge_html}</td>
              <td>{action_html}</td>
            </tr>"""

        active_series_html = f"""
        <div style="margin-bottom:2rem">
          <div style="font-family:'DM Mono',monospace;font-size:0.72rem;text-transform:uppercase;
                      letter-spacing:0.08em;color:#f5a623;margin-bottom:0.75rem">
            Active Series &mdash; Martingale Tracker
          </div>
          <div class="table-wrap">
            <table class="data-table"><thead><tr>
              <th>Home Team</th>
              <th class="tt" data-tip="Green = home win, Red = home loss">Results So Far</th>
              <th>Status</th>
              <th class="tt" data-tip="Full Martingale bet: 1u-2u-4u-8u. Max exposure 15u.">Full MG</th>
              <th class="tt" data-tip="Capped progression: 1u-2u-3u-4u. Max exposure 10u.">Capped</th>
              <th>Edge</th>
              <th>Action</th>
            </tr></thead><tbody>{active_rows}</tbody></table>
          </div>
        </div>"""

    # ── Upcoming series ────────────────────────────────────────────────────
    upcoming_home = home_df[home_df["start_date"] > today].copy()
    if upcoming_home.empty:
        upcoming_html = '<p class="empty-msg">No upcoming 4-game series.</p>'
        n_upcoming    = 0
        series_rows   = []
    else:
        g1_up = upcoming_home[upcoming_home["game_num"] == 1].copy()
        seen  = set()
        series_rows = []
        for _, r in g1_up.sort_values("start_date").iterrows():
            k = r["series_key"]
            if k not in seen:
                seen.add(k)
                series_rows.append(r)
        n_upcoming = len(series_rows)

        up_rows = ""
        for r in series_rows:
            home  = r.get("home_team", r.get("team", "?"))
            away  = r.get("away_team", r.get("opponent", "?"))
            start = r["start_date"]
            end   = r["end_date"]
            days  = (start - today).days
            day_label = f'In {days}d'
            if days == 1: day_label = '<span style="color:#f5a623">Tomorrow</span>'

            # Show fatigue/travel edge tags for game 1
            tags = []
            if r.get("high_fatigue_spot"):
                tags.append('<span style="color:#f5a623;font-size:0.68rem">FATIGUE</span>')
            if r.get("traveled_in"):
                tags.append('<span style="color:#a78bfa;font-size:0.68rem">TRAVEL</span>')
            tags_html = " ".join(tags) if tags else "—"

            ml_str = fmt_ml(r.get("home_moneyline"))

            up_rows += f"""
            <tr>
              <td style="font-family:monospace;font-size:0.8rem">{start} &ndash; {end}</td>
              <td>{day_label}</td>
              <td><strong>{home}</strong></td>
              <td class="muted">vs</td>
              <td>{away}</td>
              <td style="color:#f5a623;font-family:monospace;font-weight:600">{ml_str}</td>
              <td>{tags_html}</td>
              <td style="color:#f5a623;font-family:monospace;font-weight:600">1u</td>
            </tr>"""

        upcoming_html = f"""
        <div class="table-wrap">
          <table class="data-table"><thead><tr>
            <th>Dates</th><th>Starts</th><th>Home Team</th><th></th><th>Away Team</th>
            <th class="tt" data-tip="Consensus home moneyline from sharp books">Home ML</th>
            <th>Edge</th>
            <th class="tt" data-tip="Starting bet = 1 unit on home team ML. Doubles each loss.">G1 Bet</th>
          </tr></thead><tbody>{up_rows}</tbody></table>
        </div>"""

    return f"""
    <div style="background:rgba(245,166,35,0.06);border:1px solid rgba(245,166,35,0.25);border-radius:10px;
                padding:1.2rem 1.6rem;margin-bottom:2rem;display:flex;gap:3rem;flex-wrap:wrap;align-items:center">
      <div>
        <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:#f5a623">{hist_pct}</div>
        <div style="font-size:0.72rem;font-family:'DM Mono',monospace;color:var(--muted);text-transform:uppercase">Home wins at least 1 game</div>
      </div>
      <div style="flex:1;font-size:0.8rem;color:var(--muted);line-height:1.6">
        <strong style="color:var(--text)">Strategy:</strong> Bet on the home team ML each game, doubling until they win.
        Full Martingale: 1u &rarr; 2u &rarr; 4u &rarr; 8u (max 15u exposure).
        Capped: 1u &rarr; 2u &rarr; 3u &rarr; 4u (max 10u exposure).
        Stop as soon as the home team wins &mdash; you profit on every series where they win at least 1 game.<br>
        <span style="color:var(--muted);font-size:0.72rem">{hist_note} &nbsp;&middot;&nbsp;
        SKIP = qualification filter blocked the bet (price too high, dog too big, or bullpen overloaded)</span>
      </div>
    </div>

    {active_series_html}

    <div style="margin-top:2rem">
      <div style="font-family:'DM Mono',monospace;font-size:0.72rem;text-transform:uppercase;
                  letter-spacing:0.08em;color:var(--muted);margin-bottom:0.75rem">
        Upcoming 4-Game Series &mdash; {n_upcoming} remaining this season
      </div>
      {upcoming_html}
    </div>"""


def main():
    print("Generating MLB dashboard...")

    df           = safe_read(EV_FILE)
    lines_df     = safe_read(LINES_FILE)
    history      = safe_read(HISTORY_FILE)
    series_df    = safe_read(SERIES_FILE)
    hr_df        = safe_read(HR_FILE)
    hr_hist_df   = safe_read(HR_HIST_FILE)

    # Build player → team abbreviation lookup from batter projections
    player_team: dict[str, str] = {}
    try:
        bdf = pd.read_csv(BATTER_PROJ)
        bdf.columns = [c.strip().upper() for c in bdf.columns]
        for _, row in bdf[["PLAYER_NAME", "TEAM_NAME"]].dropna().iterrows():
            player_team[str(row["PLAYER_NAME"]).strip().lower()] = abbrev_team(row["TEAM_NAME"])
    except Exception:
        pass

    # Generate reason column if not already present in the CSV
    if not df.empty and "reason" not in df.columns:
        df["reason"] = df.apply(build_reason, axis=1)

    today_str    = datetime.now().strftime("%A, %B %d %Y")
    generated_at = datetime.now().strftime("%I:%M %p")
    today_html, fallback_mode = build_today(df, game_lines_df=lines_df, player_team=player_team)
    lines_html   = build_game_lines(lines_df)
    perf_html    = build_performance(history)
    series_html  = build_4game_series(series_df)
    hr_html      = build_hr_props(hr_df, hr_hist_df)
    picks_json   = build_picks_json(df)
    sim_inputs_json = build_sim_inputs(df)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MLB Model Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:#0a0c10; --surface:#111318; --surface2:#181c23; --border:#222730;
    --accent:#e05c3a; --text:#e8ecf0; --muted:#5a6172;
    --over:#00e5a0; --under:#3d8ef8; --win:#00e5a0; --loss:#f04e4e; --push:#f5a623;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:'DM Sans',sans-serif; background:var(--bg); color:var(--text); font-size:14px; min-height:100vh; }}
  .topbar {{ display:flex; align-items:center; justify-content:space-between; padding:1.1rem 2rem; border-bottom:1px solid var(--border); background:var(--surface); position:sticky; top:0; z-index:100; }}
  .logo {{ font-family:'Syne',sans-serif; font-weight:800; font-size:1.2rem; }}
  .logo span {{ color:var(--accent); }}
  .topbar-meta {{ font-family:'DM Mono',monospace; font-size:0.72rem; color:var(--muted); text-align:right; }}
  .tabs {{ display:flex; padding:0 2rem; background:var(--surface); border-bottom:1px solid var(--border); flex-wrap:wrap; }}
  .tab-btn {{ font-weight:600; font-size:0.82rem; letter-spacing:0.04em; text-transform:uppercase; padding:0.85rem 1.4rem; border:none; background:transparent; color:var(--muted); cursor:pointer; border-bottom:2px solid transparent; transition:all 0.15s; }}
  .tab-btn:hover {{ color:var(--text); }}
  .tab-btn.active {{ color:var(--accent); border-bottom-color:var(--accent); }}
  .tab-content {{ display:none; padding:2rem; }}
  .tab-content.active {{ display:block; }}
  .tab-header {{ display:flex; align-items:baseline; gap:1rem; margin-bottom:1.5rem; }}
  .tab-title {{ font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:700; }}
  .tab-subtitle {{ font-size:0.8rem; color:var(--muted); font-family:'DM Mono',monospace; }}
  .table-wrap {{ overflow-x:auto; border-radius:8px; border:1px solid var(--border); }}
  .data-table {{ width:100%; border-collapse:collapse; font-size:0.82rem; }}
  .data-table th {{ font-family:'DM Mono',monospace; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.06em; color:var(--muted); padding:0.7rem 0.9rem; background:var(--surface2); text-align:left; white-space:nowrap; border-bottom:1px solid var(--border); }}
  .data-table td {{ padding:0.65rem 0.9rem; border-bottom:1px solid var(--border); white-space:nowrap; }}
  .data-table tbody tr:last-child td {{ border-bottom:none; }}
  .data-table tbody tr:hover {{ background:var(--surface2); }}
  .data-table.mini td, .data-table.mini th {{ padding:0.6rem 1rem; white-space:nowrap; }}
  .badge {{ display:inline-block; padding:0.2rem 0.55rem; border-radius:4px; font-size:0.72rem; font-weight:600; font-family:'DM Mono',monospace; }}
  .badge-over {{ background:rgba(0,229,160,0.15); color:var(--over); }}
  .badge-under {{ background:rgba(61,142,248,0.15); color:var(--under); }}
  .badge-win {{ background:rgba(0,229,160,0.15); color:var(--win); }}
  .badge-loss {{ background:rgba(240,78,78,0.15); color:var(--loss); }}
  .badge-push {{ background:rgba(245,166,35,0.15); color:var(--push); }}
  .stat-pill {{ display:inline-block; padding:0.15rem 0.45rem; border-radius:3px; font-size:0.68rem; font-family:'DM Mono',monospace; background:var(--surface2); border:1px solid var(--border); color:var(--muted); }}
  .ev-cell {{ font-family:'DM Mono',monospace; font-weight:500; }}
  .muted {{ color:var(--muted); }}
  .stat-strip {{ display:flex; gap:0.75rem; flex-wrap:wrap; margin-bottom:1rem; }}
  .stat-chip {{ display:flex; flex-direction:column; align-items:center; padding:0.8rem 1.2rem; background:var(--surface); border:1px solid var(--border); border-radius:8px; min-width:90px; }}
  .chip-val {{ font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:700; line-height:1; }}
  .chip-lbl {{ font-size:0.65rem; font-family:'DM Mono',monospace; text-transform:uppercase; color:var(--muted); margin-top:0.3rem; }}
  .perf-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:1.5rem; margin-top:1.5rem; }}
  .perf-card {{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:1.2rem; }}
  .section-label {{ font-family:'DM Mono',monospace; font-size:0.72rem; text-transform:uppercase; color:var(--muted); margin-bottom:0.85rem; display:block; }}
  .bet-rating {{ display:inline-block; padding:0.22rem 0.6rem; border-radius:4px; font-size:0.72rem; font-weight:600; font-family:'DM Mono',monospace; }}
  .rating-elite {{ background:rgba(0,229,160,0.2); color:#00e5a0; }}
  .rating-strong {{ background:rgba(0,229,160,0.1); color:#44d4a0; }}
  .rating-good {{ background:rgba(61,142,248,0.15); color:#3d8ef8; }}
  .rating-average {{ background:rgba(245,166,35,0.12); color:#f5a623; }}
  .rating-poor {{ background:rgba(90,97,114,0.15); color:#5a6172; }}
  .rating-bad {{ background:rgba(240,78,78,0.12); color:#f04e4e; }}
  .top-plays-badge {{ display:inline-block; background:rgba(245,166,35,0.2); color:#f5a623; padding:0.2rem 0.5rem; border-radius:4px; font-size:0.68rem; font-weight:600; font-family:'DM Mono',monospace; }}
  .top-plays-box {{ background:rgba(245,166,35,0.06); border:1px solid rgba(245,166,35,0.25); border-radius:10px; padding:1.2rem 1.4rem; margin-bottom:1.5rem; }}
  .top-plays-box. {{ padding:0.9rem 1.4rem; }}
  .top-plays-header {{ font-family:'DM Mono',monospace; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em; color:#f5a623; margin-bottom:0.85rem; font-weight:600; }}
  .top-plays-tip {{ font-size:0.75rem; color:var(--muted); margin-top:0.6rem; font-style:italic; }}
  .empty-msg {{ color:var(--muted); font-style:italic; padding:2rem 0; }}
  .tt {{ cursor: help; border-bottom: 1px dashed var(--muted); position: relative; }}
  .tt:hover::after {{
    content: attr(data-tip);
    position: absolute; top: 100%; left: 50%; transform: translateX(-50%);
    background: #1a1f2e; border: 1px solid var(--border); border-radius: 6px;
    padding: 0.5rem 0.75rem; font-size: 0.72rem; color: var(--text);
    white-space: normal; width: 220px; z-index: 100; font-weight: 400;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4); line-height: 1.4;
  }}
  .filter-bar {{ display:flex; gap:0.6rem; flex-wrap:wrap; margin-bottom:1rem; align-items:center; }}
  .filter-select {{ background:var(--surface2); border:1px solid var(--border); border-radius:6px; padding:0.4rem 0.75rem; color:var(--text); font-size:0.82rem; font-family:'DM Sans',sans-serif; outline:none; cursor:pointer; }}
  .filter-select:focus {{ border-color:var(--accent); }}
  .filter-reset {{ background:transparent; border:1px solid var(--border); border-radius:6px; padding:0.4rem 0.85rem; color:var(--muted); font-size:0.82rem; cursor:pointer; }}
  .filter-reset:hover {{ border-color:var(--accent); color:var(--text); }}
  .day-group {{ border:1px solid var(--border); border-radius:8px; margin-bottom:0.75rem; overflow:hidden; }}
  .day-summary {{ display:flex; justify-content:space-between; align-items:center; padding:0.75rem 1rem; background:var(--surface2); cursor:pointer; list-style:none; user-select:none; }}
  .day-summary::-webkit-details-marker {{ display:none; }}
  .day-summary::before {{ content:'▶'; font-size:0.65rem; color:var(--muted); margin-right:0.5rem; transition:transform 0.15s; }}
  details[open] .day-summary::before {{ transform:rotate(90deg); }}
  .day-date {{ font-family:'DM Mono',monospace; font-size:0.82rem; font-weight:600; }}
  .day-record {{ font-family:'DM Mono',monospace; font-size:0.75rem; color:var(--muted); }}
  ::-webkit-scrollbar {{ width:6px; height:6px; }}
  ::-webkit-scrollbar-track {{ background:var(--bg); }}
  ::-webkit-scrollbar-thumb {{ background:var(--border); border-radius:3px; }}
</style>
</head>
<body>
<div class="topbar">
  <div class="logo">MLB<span>.</span>MODEL</div>
  <div class="topbar-meta">{today_str}<br>Updated {generated_at} PT</div>
</div>
<div class="tabs">
  <button class="tab-btn active" onclick="showTab('today',this)">Today's Plays</button>
  <button class="tab-btn" onclick="showTab('lines',this)">Game Lines</button>
  <button class="tab-btn" onclick="showTab('hrprops',this)">HR Props</button>
  <button class="tab-btn" onclick="showTab('series',this)">4-Game Series</button>
  <button class="tab-btn" onclick="showTab('history',this)">Results History</button>
  <button class="tab-btn" onclick="showTab('parlayhistory',this)">Parlay History</button>
  <button class="tab-btn" onclick="showTab('performance',this)">Performance</button>
  <button class="tab-btn" onclick="showTab('simulations',this)">Simulations</button>
</div>
<div id="today" class="tab-content active">
  <div class="tab-header">
    <div class="tab-title">Today's MLB Plays</div>
    <div class="tab-subtitle">Player props · Filter by stat, side, rating, EV%, matchup</div>
  </div>
  {today_html}
</div>
<div id="lines" class="tab-content">
  <div class="tab-header">
    <div class="tab-title">Game Lines</div>
    <div class="tab-subtitle">Spreads · Totals · Moneylines · Consensus fair odds vs Bovada</div>
  </div>
  {lines_html}
</div>
<div id="hrprops" class="tab-content">
  <div class="tab-header">
    <div class="tab-title">HR Props</div>
    <div class="tab-subtitle">Top most-likely HR hitters · Poisson model · P(≥1 HR) vs best available odds</div>
  </div>
  {hr_html}
</div>
<div id="series" class="tab-content">
  <div class="tab-header">
    <div class="tab-title">4-Game Series</div>
    <div class="tab-subtitle">True 4-game series only &middot; No off days &middot; Game 4 = high fatigue spot</div>
  </div>
  {series_html}
</div>
<div id="history" class="tab-content">
  <div class="tab-header">
    <div class="tab-title">Results History</div>
    <div class="tab-subtitle">Your tracked bets only · Auto-graded overnight · Click Track on any pick to add</div>
  </div>
  <div id="tracked-history"></div>
</div>
<div id="parlayhistory" class="tab-content">
  <div class="tab-header">
    <div class="tab-title">Parlay History</div>
    <div class="tab-subtitle">Tracked parlays · Separate from straight bet P&L</div>
  </div>
  <div id="parlay-history"></div>
</div>
<div id="performance" class="tab-content">
  <div class="tab-header">
    <div class="tab-title">Performance</div>
    <div class="tab-subtitle">All model picks · Win rate and P&L by stat type and side</div>
  </div>
  {perf_html}
</div>
<div id="simulations" class="tab-content">
  <div class="tab-header">
    <div class="tab-title">Elite Pick Simulations</div>
    <div class="tab-subtitle">Monte Carlo + Bootstrap · 10k trials each · 60/40 ensemble · Unit sizing by confidence</div>
  </div>
  <div id="sim-content"><p style="color:var(--muted);font-style:italic;padding:1rem 0">Loading simulations…</p></div>
</div>
<script>
  const TRACKED_KEY = 'mlb_tracked_bets_v1';
  const PARLAY_KEY   = 'mlb_parlay_bets_v1';

  function trackParlay(label, odds, legs, units) {{
    const bets = JSON.parse(localStorage.getItem(PARLAY_KEY) || '[]');
    const _pst = new Date(new Date().toLocaleString('en-US', {{timeZone: 'America/Los_Angeles'}}));
    const today = _pst.getFullYear()+'-'+String(_pst.getMonth()+1).padStart(2,'0')+'-'+String(_pst.getDate()).padStart(2,'0');
    const amount = (units || 1) * 10;
    bets.push({{ id: Date.now(), date: today, label, odds, legs, units: units||1, amount, result: 'pending', pnl: null }});
    localStorage.setItem(PARLAY_KEY, JSON.stringify(bets));
    alert('Parlay tracked!');
  }}

  function setParlayResult(id, result) {{
    const bets = JSON.parse(localStorage.getItem(PARLAY_KEY) || '[]');
    const bet  = bets.find(b => b.id===id);
    if (!bet) return;
    bet.result = result;
    const o = parseFloat(String(bet.odds||'').replace(/[+]/,''));
    if (result==='win')  bet.pnl = isNaN(o)||o===0 ? 0 : o>0?(o/100)*bet.amount:(100/Math.abs(o))*bet.amount;
    if (result==='loss') bet.pnl = -bet.amount;
    if (result==='push') bet.pnl = 0;
    if (result==='void') bet.pnl = 0;
    if (result==='pending') bet.pnl = null;
    localStorage.setItem(PARLAY_KEY, JSON.stringify(bets));
    renderParlayHistory();
  }}

  function removeParlayBet(id) {{
    if (!confirm('Remove this parlay?')) return;
    const bets = JSON.parse(localStorage.getItem(PARLAY_KEY) || '[]').filter(b=>b.id!==id);
    localStorage.setItem(PARLAY_KEY, JSON.stringify(bets));
    renderParlayHistory();
  }}

  function renderParlayHistory() {{
    const container = document.getElementById('parlay-history');
    if (!container) return;
    const bets = JSON.parse(localStorage.getItem(PARLAY_KEY) || '[]');
    if (!bets.length) {{
      container.innerHTML = '<p style="color:var(--muted);font-style:italic;padding:1rem 0">No parlays tracked yet.</p>';
      return;
    }}
    let totalWagered=0, totalPnl=0, wins=0, losses=0;
    bets.forEach(b => {{
      if (b.result==='void') return;  // voided bets excluded from all stats
      totalWagered += b.amount||0;
      if (b.result==='win')  {{ wins++;   totalPnl += b.pnl||0; }}
      if (b.result==='loss') {{ losses++; totalPnl += b.pnl||0; }}
    }});
    const pnlColor = totalPnl >= 0 ? '#00e5a0' : '#f04e4e';
    const pnlSign  = totalPnl >= 0 ? '+' : '';
    const cards = bets.slice().reverse().map(b => {{
      const resBadge = b.result==='win'  ? '<span style="color:#00e5a0;font-weight:700;font-size:0.8rem">WIN</span>'
                     : b.result==='loss' ? '<span style="color:#f04e4e;font-weight:700;font-size:0.8rem">LOSS</span>'
                     : b.result==='push' ? '<span style="color:#aaa;font-size:0.8rem">PUSH</span>'
                     : b.result==='void' ? '<span style="color:#5a6172;font-weight:700;font-size:0.8rem">VOID</span>'
                     : '<span style="color:#3d8ef8;font-size:0.8rem">OPEN</span>';
      const pnlStr = (b.result==='void') ? '<span style="color:var(--muted)">—</span>'
                   : b.pnl!=null ? `<span style="color:${{b.pnl>=0?'#00e5a0':'#f04e4e'}};font-family:monospace;font-weight:600">${{b.pnl>=0?'+':''}}$${{b.pnl.toFixed(2)}}</span>` : '—';
      const resBtns = b.result==='pending'
        ? `<button class="result-btn" onclick="setParlayResult(${{b.id}},'win')">W</button><button class="result-btn" onclick="setParlayResult(${{b.id}},'loss')">L</button><button class="result-btn" onclick="setParlayResult(${{b.id}},'push')">P</button><button class="result-btn" onclick="setParlayResult(${{b.id}},'void')" style="color:#5a6172">V</button>`
        : `<button class="result-btn" onclick="setParlayResult(${{b.id}},'pending')">&#8635;</button>`;
      const legs = (b.legs||[]).map(l => `<div style="font-size:0.72rem;color:var(--muted);padding:1px 0;line-height:1.4">· ${{l}}</div>`).join('');
      return `
      <div style="background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:0.75rem 1rem;display:grid;grid-template-columns:1fr auto auto;gap:0.5rem 1rem;align-items:start">
        <div>
          <div style="font-weight:700;font-size:0.88rem;margin-bottom:3px">${{b.label}}</div>
          ${{legs}}
        </div>
        <div style="text-align:center;white-space:nowrap">
          <div style="font-family:monospace;color:#f5a623;font-size:0.85rem;font-weight:600">${{b.odds}}</div>
          <div style="font-size:0.72rem;color:var(--muted);margin-top:2px">${{b.units}}u · $${{b.amount}}</div>
          <div style="margin-top:4px">${{resBadge}}</div>
        </div>
        <div style="text-align:right;white-space:nowrap">
          <div style="margin-bottom:4px">${{pnlStr}}</div>
          <div style="display:flex;gap:3px;justify-content:flex-end;flex-wrap:wrap">${{resBtns}}</div>
          <button class="btn-del" style="margin-top:6px" onclick="removeParlayBet(${{b.id}})">✕</button>
        </div>
        <div style="grid-column:1/-1;font-size:0.68rem;color:var(--muted);margin-top:2px">${{b.date}}</div>
      </div>`;
    }}).join('');
    container.innerHTML = `
      <div style="display:flex;gap:1.5rem;margin-bottom:1rem;flex-wrap:wrap">
        <div><span style="color:var(--muted);font-size:0.75rem">Tracked</span><div style="font-size:1.4rem;font-weight:700">${{bets.length}}</div></div>
        <div><span style="color:var(--muted);font-size:0.75rem">Wins</span><div style="font-size:1.4rem;font-weight:700;color:#00e5a0">${{wins}}</div></div>
        <div><span style="color:var(--muted);font-size:0.75rem">Losses</span><div style="font-size:1.4rem;font-weight:700;color:#f04e4e">${{losses}}</div></div>
        <div><span style="color:var(--muted);font-size:0.75rem">Wagered</span><div style="font-size:1.4rem;font-weight:700">$${{totalWagered.toFixed(2)}}</div></div>
        <div><span style="color:var(--muted);font-size:0.75rem">P&L</span><div style="font-size:1.4rem;font-weight:700;color:${{pnlColor}}">${{pnlSign}}$${{totalPnl.toFixed(2)}}</div></div>
      </div>
      <div style="display:flex;flex-direction:column;gap:0.6rem">${{cards}}</div>`;
  }}
  const UNIT_SIZE   = 10;

  const SIM_INPUTS = {sim_inputs_json};
  let _simDone = false;

  try {{
    const mlbPicks = {picks_json};
    if (mlbPicks.length) {{
      const existing = JSON.parse(localStorage.getItem('dk_today_picks_v1') || '[]');
      const combined = [...existing.filter(p => p.sport !== 'MLB'), ...mlbPicks];
      localStorage.setItem('dk_today_picks_v1', JSON.stringify(combined));
    }}
  }} catch(e) {{}}

  function showTab(id, btn) {{
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById(id).classList.add('active');
    btn.classList.add('active');
    if (id === 'history') renderTrackedHistory();
    if (id === 'parlayhistory') renderParlayHistory();
    if (id === 'simulations') runSimulations();
  }}

  // ── Track a bet ───────────────────────────────────────────
  function trackBet(player, prop, line, side, odds, matchup, gameTime, sport) {{
    const pickKey  = `${{player}}|${{prop}}|${{line}}|${{side}}`;
    const unitSel  = document.getElementById('u_' + pickKey);
    const units    = unitSel ? parseFloat(unitSel.value) || 1 : 1;
    const amount   = units * UNIT_SIZE;
    const _pst = new Date(new Date().toLocaleString('en-US', {{timeZone: 'America/Los_Angeles'}})); const today = _pst.getFullYear()+'-'+String(_pst.getMonth()+1).padStart(2,'0')+'-'+String(_pst.getDate()).padStart(2,'0');

    const bets = JSON.parse(localStorage.getItem(TRACKED_KEY) || '[]');

    // Prevent duplicate tracking for same pick same day
    const exists = bets.find(b => b.player===player && b.prop===prop && b.line===line && b.side===side && b.date===today);
    if (exists) {{
      alert(`${{player}} ${{side}} ${{line}} already tracked today.`);
      return;
    }}

    const pick = {{
      id: Date.now(), date: today, player, prop, line, side, odds,
      matchup, gameTime, sport, units, amount, result: 'pending', pnl: null
    }};
    bets.unshift(pick);
    localStorage.setItem(TRACKED_KEY, JSON.stringify(bets));

    // Save to local tracker server for auto-grading (silently ignored if server not running)
    fetch('http://localhost:5001/track', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify(pick)
    }}).catch(() => {{}});

    // Update button to show tracked
    const btn = document.getElementById('tb_' + pickKey);
    if (btn) {{
      btn.textContent = '✓ Tracked';
      btn.style.background = 'rgba(0,229,160,0.25)';
      btn.style.cursor = 'default';
      btn.onclick = null;
    }}
    alert(`Tracked: ${{player}} ${{side}} ${{line}} · ${{units}}u ($$${{amount}})`);
  }}

  // ── Render tracked history ────────────────────────────────
  function renderTrackedHistory() {{
    const container = document.getElementById('tracked-history');
    const bets      = JSON.parse(localStorage.getItem(TRACKED_KEY) || '[]');

    if (!bets.length) {{
      container.innerHTML = '<p style="color:var(--muted);font-style:italic;padding:2rem 0">No tracked bets yet. Click Track next to any pick to add a bet.</p>';
      return;
    }}

    // Stats — voided bets excluded from all counts
    const wins    = bets.filter(b => b.result==='win').length;
    const losses  = bets.filter(b => b.result==='loss').length;
    const pending = bets.filter(b => b.result==='pending').length;
    const voids   = bets.filter(b => b.result==='void').length;
    const graded  = wins + losses;
    const wagered = bets.filter(b=>b.result!=='void').reduce((s,b) => s+(b.amount||0), 0);
    const pnl     = bets.reduce((s,b) => {{
      if (b.result==='win')  return s + (b.pnl || calcPnl(b.odds, b.amount));
      if (b.result==='loss') return s - b.amount;
      return s;
    }}, 0);
    const units   = pnl / UNIT_SIZE;
    const wr      = graded > 0 ? `${{(wins/graded*100).toFixed(1)}}%` : '-';

    const strip = `
    <div class="stat-strip">
      <div class="stat-chip"><span class="chip-val">${{bets.length}}</span><span class="chip-lbl">Tracked</span></div>
      <div class="stat-chip" style="border-color:rgba(0,229,160,0.3)"><span class="chip-val" style="color:#00e5a0">${{wins}}</span><span class="chip-lbl">Wins</span></div>
      <div class="stat-chip" style="border-color:rgba(240,78,78,0.3)"><span class="chip-val" style="color:#f04e4e">${{losses}}</span><span class="chip-lbl">Losses</span></div>
      <div class="stat-chip" style="border-color:rgba(61,142,248,0.3)"><span class="chip-val" style="color:#3d8ef8">${{pending}}</span><span class="chip-lbl">Pending</span></div>
      ${{voids > 0 ? `<div class="stat-chip" style="border-color:rgba(90,97,114,0.3)"><span class="chip-val" style="color:#5a6172">${{voids}}</span><span class="chip-lbl">Voided</span></div>` : ''}}
      <div class="stat-chip"><span class="chip-val">${{wr}}</span><span class="chip-lbl">Win Rate</span></div>
      <div class="stat-chip"><span class="chip-val">$${{wagered.toFixed(2)}}</span><span class="chip-lbl">Wagered</span></div>
      <div class="stat-chip" style="border-color:${{pnl>=0?'rgba(0,229,160,0.3)':'rgba(240,78,78,0.3)'}}">
        <span class="chip-val" style="color:${{pnl>=0?'#00e5a0':'#f04e4e'}}">${{pnl>=0?'+':''}}$${{pnl.toFixed(2)}}</span>
        <span class="chip-lbl">P&L</span>
      </div>
      <div class="stat-chip">
        <span class="chip-val" style="color:${{units>=0?'#00e5a0':'#f04e4e'}}">${{units>=0?'+':''}}${{units.toFixed(1)}}u</span>
        <span class="chip-lbl">Units P&L</span>
      </div>
    </div>
    <p style="font-size:0.75rem;color:var(--muted);margin-bottom:1rem">
      Bets you tracked from Today's Plays · Auto-graded overnight · 1 unit = $10
    </p>`;

    // Group by date
    const byDate = {{}};
    bets.forEach(b => {{
      if (!byDate[b.date]) byDate[b.date] = [];
      byDate[b.date].push(b);
    }});

    const dates    = Object.keys(byDate).sort().reverse();

    // ── Daily P&L tracker ─────────────────────────────────────────────────
    const datesAsc = Object.keys(byDate).sort();  // oldest -> newest for chart
    const BAR_MAX_H = 56;
    const dailyPnl  = [];   // {{date, dayPnl, cumPnl}}
    let running = 0;
    datesAsc.forEach(date => {{
      const dayBets = byDate[date];
      const dayPnl  = dayBets.reduce((s, b) => {{
        if (b.result==='win')  return s + (b.pnl || calcPnl(b.odds, b.amount));
        if (b.result==='loss') return s - b.amount;
        return s;
      }}, 0);
      running += dayPnl;
      dailyPnl.push({{ date, dayPnl, cumPnl: running }});
    }});

    const maxAbs = Math.max(...dailyPnl.map(d => Math.abs(d.dayPnl)), 1);

    const barItems = dailyPnl.map(d => {{
      const h     = Math.max(3, Math.round(Math.abs(d.dayPnl) / maxAbs * BAR_MAX_H));
      const color = d.dayPnl >= 0 ? '#00e5a0' : '#f04e4e';
      const sign  = d.dayPnl >= 0 ? '+' : '';
      const csign = d.cumPnl >= 0 ? '+' : '';
      const label = d.date.slice(5);  // MM-DD
      const tip   = `${{sign}}$${{d.dayPnl.toFixed(2)}} · cumulative ${{csign}}$${{d.cumPnl.toFixed(2)}}`;
      return `<div style="display:flex;flex-direction:column;align-items:center;min-width:34px;cursor:default" title="${{tip}}">
        <div style="width:22px;height:${{h}}px;background:${{color}};border-radius:3px 3px 0 0;margin-top:auto;transition:opacity .15s" onmouseover="this.style.opacity='.7'" onmouseout="this.style.opacity='1'"></div>
        <div style="font-size:0.6rem;color:var(--muted);margin-top:3px;font-family:monospace">${{label}}</div>
      </div>`;
    }}).join('');

    // Cumulative sparkline SVG
    let sparkHtml = '';
    if (dailyPnl.length > 1) {{
      const minC = Math.min(...dailyPnl.map(d => d.cumPnl));
      const maxC = Math.max(...dailyPnl.map(d => d.cumPnl));
      const span = maxC - minC || 1;
      const W = 420, H = 50;
      const n = dailyPnl.length;
      const pts = dailyPnl.map((d, i) => {{
        const x = Math.round(i / (n-1) * W);
        const y = Math.round(H - (d.cumPnl - minC) / span * H);
        return `${{x}},${{y}}`;
      }}).join(' ');
      const zeroY = Math.max(0, Math.min(H, Math.round(H - (0 - minC) / span * H)));
      sparkHtml = `<div style="margin-top:0.75rem">
        <span style="font-size:0.72rem;color:var(--muted);font-family:monospace">CUMULATIVE P&L</span>
        <svg width="${{W}}" height="${{H+4}}" viewBox="0 0 ${{W}} ${{H+4}}" style="display:block;overflow:visible">
          <line x1="0" y1="${{zeroY}}" x2="${{W}}" y2="${{zeroY}}" stroke="#ffffff18" stroke-width="1" stroke-dasharray="4,3"/>
          <polyline points="${{pts}}" fill="none" stroke="#3d8ef8" stroke-width="2" stroke-linejoin="round"/>
        </svg>
      </div>`;
    }}

    const cumColor = running >= 0 ? '#00e5a0' : '#f04e4e';
    const cumSign  = running >= 0 ? '+' : '';
    const tracker  = `
    <div style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:1rem 1.25rem;margin-bottom:1.5rem">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem">
        <span style="font-size:0.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;font-family:monospace">Daily P&L Tracker</span>
        <span style="font-size:1.1rem;font-weight:700;font-family:monospace;color:${{cumColor}}">${{cumSign}}$${{running.toFixed(2)}} cumulative</span>
      </div>
      <div style="display:flex;align-items:flex-end;gap:4px;height:${{BAR_MAX_H+4}}px;overflow-x:auto;padding-bottom:2px">
        ${{barItems}}
      </div>
      ${{sparkHtml}}
    </div>`;

    // ── Per-day collapsible rows ───────────────────────────────────────────
    const dayPnlMap = {{}};
    dailyPnl.forEach(d => {{ dayPnlMap[d.date] = d.dayPnl; }});
    let daysHtml = '';

    dates.forEach((date, i) => {{
      const dayBets  = byDate[date];
      const dWins    = dayBets.filter(b=>b.result==='win').length;
      const dTotal   = dayBets.filter(b=>b.result==='win'||b.result==='loss').length;
      const dWR      = dTotal > 0 ? `${{dWins}}/${{dTotal}}` : '-';
      const expanded = i === 0 ? 'open' : '';
      const dp       = dayPnlMap[date] || 0;
      const dpColor  = dp >= 0 ? '#00e5a0' : '#f04e4e';
      const dpSign   = dp >= 0 ? '+' : '';
      const dpStr    = `<span style="font-family:monospace;font-weight:700;color:${{dpColor}};margin-left:0.75rem">${{dpSign}}$${{dp.toFixed(2)}}</span>`;

      const rows = dayBets.map(b => {{
        const resBadge = b.result==='win'   ? '<span class="badge badge-win">WIN</span>'
          : b.result==='loss'  ? '<span class="badge badge-loss">LOSS</span>'
          : b.result==='push'  ? '<span class="badge badge-push">PUSH</span>'
          : b.result==='void'  ? '<span class="badge" style="color:#5a6172">VOID</span>'
          : '<span class="badge badge-pending">OPEN</span>';
        const pnlVal = b.result==='win'  ? (b.pnl || calcPnl(b.odds, b.amount))
          : b.result==='loss' ? -b.amount : null;
        const pnlStr = b.result==='void' ? '<span style="color:var(--muted)">-</span>'
          : pnlVal !== null
          ? `<span style="color:${{pnlVal>=0?'#00e5a0':'#f04e4e'}};font-family:monospace">${{pnlVal>=0?'+':''}}$${{Math.abs(pnlVal).toFixed(2)}}</span>`
          : '-';
        const resBtns = b.result==='pending'
          ? `<button class="result-btn" onclick="setTrackedResult(${{b.id}},'win')">W</button>
             <button class="result-btn" onclick="setTrackedResult(${{b.id}},'loss')">L</button>
             <button class="result-btn" onclick="setTrackedResult(${{b.id}},'push')">P</button>
             <button class="result-btn" onclick="setTrackedResult(${{b.id}},'void')" style="color:#5a6172">V</button>`
          : `<button class="result-btn" onclick="setTrackedResult(${{b.id}},'pending')">&#8635;</button>`;

        return `<tr>
          <td><strong>${{b.player}}</strong></td>
          <td><span class="stat-pill">${{b.prop}}</span></td>
          <td>${{b.line}}</td>
          <td style="color:#00e5a0;font-family:monospace">${{b.side}}</td>
          <td style="font-family:monospace">${{b.odds}}</td>
          <td style="font-family:monospace">${{b.units}}u · $${{b.amount}}</td>
          <td class="muted" style="font-size:0.75rem">${{b.matchup}}</td>
          <td>${{resBadge}}<div style="margin-top:3px">${{resBtns}}</div></td>
          <td>${{pnlStr}}</td>
          <td><button class="btn-del" onclick="removeTracked(${{b.id}})">✕</button></td>
        </tr>`;
      }}).join('');

      daysHtml += `
      <details class="day-group" ${{expanded}}>
        <summary class="day-summary">
          <span class="day-date">${{date}}</span>
          <span class="day-record">${{dWR}} &nbsp; ${{dayBets.length}} bets ${{dpStr}}</span>
        </summary>
        <div class="table-wrap" style="margin:0.5rem 0 1rem 0">
          <table class="data-table"><thead><tr>
            <th>Player</th><th>Stat</th><th>Line</th><th>Pick</th>
            <th>Odds</th><th>Units</th><th>Matchup</th><th>Result</th><th>P&L</th><th></th>
          </tr></thead><tbody>${{rows}}</tbody></table>
        </div>
      </details>`;
    }});

    container.innerHTML = strip + tracker + daysHtml;
  }}

  function calcPnl(oddsStr, amount) {{
    const o = parseFloat(String(oddsStr||'').replace(/[+]/,''));
    if (isNaN(o)||o===0) return 0;
    return o>0?(o/100)*amount:(100/Math.abs(o))*amount;
  }}

  function setTrackedResult(id, result) {{
    const bets = JSON.parse(localStorage.getItem(TRACKED_KEY) || '[]');
    const bet  = bets.find(b => b.id===id);
    if (!bet) return;
    bet.result = result;
    if (result==='win')  bet.pnl = calcPnl(bet.odds, bet.amount);
    if (result==='loss') bet.pnl = -bet.amount;
    if (result==='push') bet.pnl = 0;
    if (result==='void') bet.pnl = 0;
    if (result==='pending') bet.pnl = null;
    localStorage.setItem(TRACKED_KEY, JSON.stringify(bets));
    renderTrackedHistory();
  }}

  function removeTracked(id) {{
    if (!confirm('Remove this tracked bet?')) return;
    const bets = JSON.parse(localStorage.getItem(TRACKED_KEY) || '[]').filter(b=>b.id!==id);
    localStorage.setItem(TRACKED_KEY, JSON.stringify(bets));
    fetch('http://localhost:5001/untrack/' + id, {{method: 'DELETE'}}).catch(() => {{}});
    renderTrackedHistory();
  }}

  function applyFilters() {{
    const stat        = document.getElementById('f-stat').value.toLowerCase();
    const side        = document.getElementById('f-side').value.toUpperCase();
    const rating      = document.getElementById('f-rating').value.toUpperCase();
    const evMin       = parseFloat(document.getElementById('f-ev').value) || 0;
    const fairMin     = parseFloat(document.getElementById('f-fairprob').value) || 0;
    const matchup     = document.getElementById('f-matchup').value;
    const ratings     = rating.split(',').map(r => r.trim()).filter(Boolean);
    // When Elite+Strong filter is active, hide picks over the per-stat correlation cap
    const eliteStrongOnly = ratings.length > 0 && ratings.every(r => r === 'ELITE' || r === 'STRONG');

    document.querySelectorAll('#mlb-table tbody tr').forEach(row => {{
      const rRating = row.dataset.rating ? row.dataset.rating.toUpperCase() : '';
      const ratingMatch = ratings.length === 0 || ratings.some(r => rRating.includes(r));
      const corrCapped  = row.dataset.corrCapped === '1';
      const show = (!stat    || row.dataset.stat.toLowerCase() === stat)    &&
                   (!side    || row.dataset.side.toUpperCase() === side)     &&
                   ratingMatch                                                &&
                   ((parseFloat(row.dataset.ev)||0) >= evMin)                &&
                   ((parseFloat(row.dataset.fairprob)||0) >= fairMin)        &&
                   (!matchup || row.dataset.matchup === matchup)             &&
                   !(eliteStrongOnly && corrCapped);
      row.style.display = show ? '' : 'none';
    }});
  }}

  function resetFilters() {{
    ['f-stat','f-side','f-ev','f-fairprob','f-matchup'].forEach(id => document.getElementById(id).value = '');
    document.getElementById('f-rating').value = '{"" if fallback_mode else "ELITE,STRONG"}';
    applyFilters();
  }}

  function applyGLFilters() {{
    const market = document.getElementById('gl-market').value;
    document.querySelectorAll('#gl-table tbody tr').forEach(row => {{
      row.style.display = (!market || row.dataset.market === market) ? '' : 'none';
    }});
  }}

  function resetGLFilters() {{
    document.getElementById('gl-market').value = '';
    applyGLFilters();
  }}

  function filterHistory() {{
    const stat   = document.getElementById('h-stat') ? document.getElementById('h-stat').value.toLowerCase() : '';
    const side   = document.getElementById('h-side') ? document.getElementById('h-side').value.toUpperCase() : '';
    const result = document.getElementById('h-result') ? document.getElementById('h-result').value.toUpperCase() : '';
    document.querySelectorAll('.hist-table tbody tr').forEach(row => {{
      const show = (!stat   || row.dataset.stat.toLowerCase() === stat) &&
                   (!side   || row.dataset.side.toUpperCase() === side)  &&
                   (!result || row.dataset.result.toUpperCase() === result);
      row.style.display = show ? '' : 'none';
    }});
  }}

  function resetHistFilters() {{
    ['h-stat','h-side','h-result'].forEach(id => {{ const el=document.getElementById(id); if(el) el.value=''; }});
    filterHistory();
  }}

  // On load: apply default filters so only ELITE + STRONG show
  document.addEventListener('DOMContentLoaded', function() {{
    renderTrackedHistory();
    applyFilters();
  }});

  // ── Elite Pick Simulations ──────────────────────────────────────────────

  function _poissonDraw(lam) {{
    const L = Math.exp(-Math.min(lam, 700));
    let k = 0, p = 1;
    do {{ k++; p *= Math.random(); }} while (p > L);
    return k - 1;
  }}

  function _normalDraw(mu, sigma) {{
    const u1 = Math.random() || 1e-10, u2 = Math.random();
    return mu + sigma * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }}

  function _monteCarlo(inp, N) {{
    const {{proj_mean, proj_std, line, side, stat_type}} = inp;
    let hits = 0;
    for (let i = 0; i < N; i++) {{
      const v = stat_type === 'normal' ? _normalDraw(proj_mean, proj_std) : _poissonDraw(proj_mean);
      if (side === 'OVER'  && v > line)  hits++;
      if (side === 'UNDER' && v < line)  hits++;
    }}
    return hits / N;
  }}

  function _bootstrap(inp, N) {{
    const {{samples, line, side}} = inp;
    const n = samples.length;
    let hits = 0;
    const drawn = [];
    for (let i = 0; i < N; i++) {{
      const v = samples[Math.floor(Math.random() * n)];
      if (side === 'OVER'  && v > line)  hits++;
      if (side === 'UNDER' && v < line)  hits++;
      drawn.push(v);
    }}
    drawn.sort((a, b) => a - b);
    return {{
      prob:  hits / N,
      ci_lo: drawn[Math.floor(0.025 * N)],
      ci_hi: drawn[Math.floor(0.975 * N)],
    }};
  }}

  function _decimal(odds) {{
    return odds > 0 ? odds / 100 + 1 : 100 / Math.abs(odds) + 1;
  }}

  function _ev(prob, odds) {{
    const d = _decimal(odds);
    return (prob * (d - 1) - (1 - prob)) * 100;
  }}

  function _unitTier(ensProb, ciWidth) {{
    if (ensProb >= 0.70 && ciWidth < 1.5) return 4;
    if (ensProb >= 0.65 && ciWidth < 2.5) return 3;
    if (ensProb >= 0.60 && ciWidth < 4.0) return 2;
    return 1;
  }}

  function runSimulations() {{
    if (_simDone) return;
    _simDone = true;

    const inputs = window.SIM_INPUTS || SIM_INPUTS || [];
    if (!inputs.length) {{
      document.getElementById('sim-content').innerHTML =
        '<p style="color:var(--muted);font-style:italic;padding:1rem 0">No ELITE picks today.</p>';
      return;
    }}

    const N = 10000;
    const results = inputs.map(inp => {{
      const mcProb               = _monteCarlo(inp, N);
      const {{prob: bsProb, ci_lo, ci_hi}} = _bootstrap(inp, N);
      const ensProb              = inp.has_real_samples ? 0.6 * mcProb + 0.4 * bsProb : mcProb;
      const ev                   = _ev(ensProb, inp.odds);
      const ciWidth              = ci_hi - ci_lo;
      const units                = _unitTier(ensProb, ciWidth);
      const modelEV              = _ev(inp.fair_prob, inp.odds);
      return {{...inp, mcProb, bsProb, ensProb, ev, ci_lo, ci_hi, ciWidth, units, modelEV,
               disagree: (modelEV - ev) > 5}};
    }});

    results.sort((a, b) => b.ev - a.ev);

    const totalUnits = results.reduce((s, r) => s + r.units, 0);
    const avgEV      = results.reduce((s, r) => s + r.ev, 0) / results.length;
    const avgSign    = avgEV >= 0 ? '+' : '';

    const summaryHtml = `
    <div class="stat-strip" style="margin-bottom:1.5rem">
      <div class="stat-chip"><span class="chip-val">${{results.length}}</span><span class="chip-lbl">Elite Picks</span></div>
      <div class="stat-chip" style="border-color:rgba(0,229,160,0.3)">
        <span class="chip-val" style="color:#00e5a0">${{avgSign}}${{avgEV.toFixed(1)}}%</span>
        <span class="chip-lbl">Avg Ensemble EV</span>
      </div>
      <div class="stat-chip">
        <span class="chip-val">${{totalUnits}}u</span>
        <span class="chip-lbl">Rec. Units Total</span>
      </div>
    </div>`;

    const unitColors = ['', '#5a6172', '#3d8ef8', '#00e5a0', '#e05c3a'];

    const rows = results.map(r => {{
      const sc   = r.side === 'OVER' ? '#00e5a0' : '#3d8ef8';
      const uc   = unitColors[r.units];
      const evC  = r.ev >= 5 ? '#00e5a0' : r.ev >= 0 ? '#f5a623' : '#f04e4e';
      const evS  = r.ev >= 0 ? '+' : '';
      const warn = r.disagree
        ? `<span style="font-size:0.62rem;background:rgba(245,166,35,0.2);color:#f5a623;border-radius:3px;padding:1px 5px;margin-left:4px;font-family:monospace" title="Simulation EV is 5+ pts below model EV">⚠</span>`
        : '';
      return `<tr>
        <td>
          <strong style="font-size:0.88rem">${{r.player}}</strong><br>
          <span style="font-size:0.7rem;color:var(--muted)">${{r.matchup}}</span>
        </td>
        <td>
          <span style="color:${{sc}};font-weight:700">${{r.side}}</span>
          <span style="font-family:monospace"> ${{r.line}} ${{r.stat_label}}</span>
        </td>
        <td style="font-family:monospace;color:var(--muted)">${{(r.fair_prob*100).toFixed(1)}}%</td>
        <td style="font-family:monospace;color:#00e5a0">${{(r.mcProb*100).toFixed(1)}}%</td>
        <td style="font-family:monospace;color:#3d8ef8">${{(r.bsProb*100).toFixed(1)}}%</td>
        <td style="font-family:monospace;font-weight:700">${{(r.ensProb*100).toFixed(1)}}%</td>
        <td style="font-family:monospace;color:${{evC}}">${{evS}}${{r.ev.toFixed(1)}}%${{warn}}</td>
        <td style="font-family:monospace;font-size:0.78rem">${{r.ci_lo.toFixed(2)}}–${{r.ci_hi.toFixed(2)}}</td>
        <td>
          <span style="background:${{uc}}22;color:${{uc}};border:1px solid ${{uc}}55;border-radius:4px;
                       padding:0.2rem 0.55rem;font-family:monospace;font-weight:700;font-size:0.82rem">
            ${{r.units}}u
          </span>
        </td>
      </tr>`;
    }}).join('');

    const tableHtml = `
    <div class="table-wrap">
      <table class="data-table">
        <thead><tr>
          <th>Player</th><th>Pick</th>
          <th title="Original model probability">Model Prob</th>
          <th title="Monte Carlo 10k trials from Poisson/Normal projection">MC Prob</th>
          <th title="Bootstrap 10k resamples from last 30 games">Bootstrap Prob</th>
          <th title="60% MC + 40% Bootstrap">Ensemble Prob</th>
          <th title="EV recalculated with ensemble probability">Ensemble EV</th>
          <th title="Bootstrap 2.5th–97.5th percentile of stat outcomes">95% CI</th>
          <th title="Unit tier: both ensemble prob AND CI width must qualify">Units</th>
        </tr></thead>
        <tbody>${{rows}}</tbody>
      </table>
    </div>`;

    document.getElementById('sim-content').innerHTML = summaryHtml + tableHtml;
  }}
</script>
</body>
</html>"""

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(html, encoding="utf-8")
    print(f"DONE - Dashboard saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()