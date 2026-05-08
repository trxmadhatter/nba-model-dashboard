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

ISOTONIC_FILE = _ROOT / "Data/mlb/processed/calibration_isotonic_mlb.csv"

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
        note = "! low-n" if r["low_conf"] else ""
        print(
            f"  {r['bucket']:<12} {r['n']:>5}  "
            f"{r['predicted'] * 100:>8.1f}%  {r['actual'] * 100:>6.1f}%  "
            f"{r['gap'] * 100:>+6.1f}%  {note}"
        )
    print(f"\n  Calibration grade: {calibration_grade(cal_rows)}")
    print("  (A=<3% avg gap  B=3-6%  C=6-10%  D=>10%)")


def retroactive_calibration_curve(df: pd.DataFrame, iso_csv_path: str | None = None) -> pd.DataFrame | None:
    """
    Return a copy of df with fair_prob replaced by isotonic-calibrated values.
    Uses priority: (stat, side) > (stat, BOTH) > (_global_, _GLOBAL_).
    Returns None if the isotonic CSV doesn't exist or can't be loaded.
    """
    csv_path = Path(iso_csv_path) if iso_csv_path else ISOTONIC_FILE
    if not csv_path.exists():
        return None
    try:
        iso_df = pd.read_csv(csv_path)
        if not {"stat", "side", "x", "y"}.issubset(iso_df.columns):
            return None
        iso_df["stat"] = iso_df["stat"].astype(str).str.strip().str.lower()
        iso_df["side"] = iso_df["side"].astype(str).str.strip().str.upper()
        iso_df["x"]    = pd.to_numeric(iso_df["x"], errors="coerce")
        iso_df["y"]    = pd.to_numeric(iso_df["y"], errors="coerce")
        iso_df = iso_df.dropna(subset=["x", "y"])
        iso_map: dict = {}
        for (stat, side), group in iso_df.groupby(["stat", "side"]):
            g = group.sort_values("x")
            iso_map[(str(stat), str(side))] = (g["x"].values, g["y"].values)
    except Exception:
        return None

    if not iso_map:
        return None

    result = df.copy()
    raw_col = "fair_prob_raw" if "fair_prob_raw" in df.columns else "fair_prob"

    def _interpolate(row) -> float:
        prob = float(row[raw_col]) if pd.notna(row[raw_col]) else float(row["fair_prob"])
        stat_n = str(row["stat"]).strip().lower()
        side_n = str(row["side"]).strip().upper()
        for key in [(stat_n, side_n), (stat_n, "BOTH"), ("_global_", "_GLOBAL_")]:
            if key in iso_map:
                xs, ys = iso_map[key]
                return float(np.interp(prob, xs, ys))
        return float(row["fair_prob"])

    result["fair_prob"] = result.apply(_interpolate, axis=1)
    return result


def book_roi(df: pd.DataFrame) -> list[dict]:
    if "pricing_source" not in df.columns:
        return []
    rows = []
    for source, group in df.groupby("pricing_source", dropna=True):
        graded = group[group["hit_result"].isin(["WIN", "LOSS"])]
        if graded.empty:
            continue
        wins   = (graded["hit_result"] == "WIN").sum()
        n      = len(graded)
        profit = graded.apply(pnl, axis=1).sum()
        rows.append({
            "book":     source,
            "n":        n,
            "win_rate": wins / n,
            "profit":   profit,
            "roi":      profit / (n * BET_SIZE) * 100,
        })
    return sorted(rows, key=lambda x: x["roi"], reverse=True)


def print_book_roi(book_rows: list[dict]) -> None:
    print(f"\n{'=' * 62}")
    print("  MODULE 2 — PER-BOOK ROI")
    print(f"{'=' * 62}")
    if not book_rows:
        print("  pricing_source column not present — skipping")
        return
    print(f"  {'Book':<20} {'N':>5}  {'Win%':>6}  {'Profit':>10}  {'ROI':>7}")
    print(f"  {'-' * 56}")
    for r in book_rows:
        print(
            f"  {r['book']:<20} {r['n']:>5}  "
            f"{r['win_rate'] * 100:>5.1f}%  "
            f"${r['profit']:>+9,.0f}  {r['roi']:>+6.1f}%"
        )


def direction_edge(df: pd.DataFrame) -> list[dict]:
    rows = []
    for (stat, side), group in df.groupby(["stat", "side"]):
        graded = group[group["hit_result"].isin(["WIN", "LOSS"])]
        if len(graded) < MIN_DIRECTION:
            continue
        wins   = (graded["hit_result"] == "WIN").sum()
        n      = len(graded)
        profit = graded.apply(pnl, axis=1).sum()
        rows.append({
            "stat_side": f"{stat} {side.upper()}",
            "n":         n,
            "win_rate":  wins / n,
            "profit":    profit,
            "roi":       profit / (n * BET_SIZE) * 100,
        })
    return sorted(rows, key=lambda x: x["roi"], reverse=True)


def print_direction_edge(dir_rows: list[dict]) -> None:
    print(f"\n{'=' * 62}")
    print("  MODULE 3 — DIRECTION EDGE (OVER vs UNDER by stat)")
    print(f"{'=' * 62}")
    if not dir_rows:
        print("  No stat/side combinations with >= 10 graded picks")
        return
    print(f"  {'Stat + Side':<20} {'N':>5}  {'Win%':>6}  {'Profit':>10}  {'ROI':>7}")
    print(f"  {'-' * 56}")
    for r in dir_rows:
        print(
            f"  {r['stat_side']:<20} {r['n']:>5}  "
            f"{r['win_rate'] * 100:>5.1f}%  "
            f"${r['profit']:>+9,.0f}  {r['roi']:>+6.1f}%"
        )


def build_html(
    cal_rows: list[dict],
    book_rows: list[dict],
    dir_rows: list[dict],
    date_range: str,
    total_picks: int,
    iso_cal_rows: list[dict] | None = None,
) -> str:
    grade     = calibration_grade(cal_rows)
    best_book = book_rows[0]["book"] if book_rows else "N/A"
    best_dir  = dir_rows[0]["stat_side"] if dir_rows else "N/A"
    best_book_roi = f"{book_rows[0]['roi']:+.1f}%" if book_rows else "N/A"
    best_dir_roi  = f"{dir_rows[0]['roi']:+.1f}%" if dir_rows else "N/A"

    def cal_row_html(r: dict) -> str:
        flag      = "&#9888;" if r["low_conf"] else ""
        gap_color = (
            "#e74c3c" if abs(r["gap"]) > 0.06
            else "#f39c12" if abs(r["gap"]) > 0.03
            else "#2ecc71"
        )
        return (
            f"<tr><td>{r['bucket']}</td><td>{r['n']}</td>"
            f"<td>{r['predicted'] * 100:.1f}%</td>"
            f"<td>{r['actual'] * 100:.1f}%</td>"
            f"<td style='color:{gap_color}'>{r['gap'] * 100:+.1f}%</td>"
            f"<td>{flag}</td></tr>"
        )

    def money_row_html(r: dict, key: str) -> str:
        roi_color = "#2ecc71" if r["roi"] > 0 else "#e74c3c"
        return (
            f"<tr><td>{r[key]}</td><td>{r['n']}</td>"
            f"<td>{r['win_rate'] * 100:.1f}%</td>"
            f"<td>${r['profit']:+,.0f}</td>"
            f"<td style='color:{roi_color}'>{r['roi']:+.1f}%</td></tr>"
        )

    cal_html  = "\n".join(cal_row_html(r) for r in cal_rows)
    book_html = "\n".join(money_row_html(r, "book") for r in book_rows)
    dir_html  = "\n".join(money_row_html(r, "stat_side") for r in dir_rows)

    no_data_6 = "<tr><td colspan='6' style='color:#6b7280;text-align:center'>No data</td></tr>"
    no_data_5 = "<tr><td colspan='5' style='color:#6b7280;text-align:center'>No data</td></tr>"

    iso_section = ""
    if iso_cal_rows:
        iso_grade    = calibration_grade(iso_cal_rows)
        iso_cal_html = "\n".join(cal_row_html(r) for r in iso_cal_rows)
        iso_section  = (
            '<p class="note" style="margin-top:16px;color:#f59e0b">'
            'Module 1b &mdash; Retroactive Isotonic'
            ' (in-sample fit &mdash; not out-of-sample accuracy)</p>'
            '<table>'
            '<tr><th>Bucket</th><th>N</th><th>Predicted</th><th>Actual</th><th>Gap</th><th></th></tr>'
            f'{iso_cal_html or no_data_6}'
            '</table>'
            f'<p class="note">Grade: <strong style="color:#a78bfa">{iso_grade}</strong>'
            ' &nbsp;&middot;&nbsp; [in-sample]</p>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MLB Accuracy Report</title>
<style>
body    {{background:#0f0f23;color:#e0e0e0;font-family:monospace;margin:0;padding:20px}}
h1      {{color:#a78bfa;margin-bottom:4px}}
.sub    {{color:#6b7280;font-size:.85em;margin-bottom:24px}}
.cards  {{display:flex;gap:20px;flex-wrap:wrap;margin-bottom:32px}}
.card   {{background:#1e1e3f;border:1px solid #2d2d5e;border-radius:8px;padding:16px 22px;min-width:150px}}
.label  {{color:#9ca3af;font-size:.75em;text-transform:uppercase;letter-spacing:.05em}}
.value  {{font-size:1.6em;font-weight:bold;color:#a78bfa;margin-top:4px}}
.hint   {{color:#6b7280;font-size:.75em;margin-top:2px}}
details {{margin-bottom:18px}}
summary {{background:#1e1e3f;border:1px solid #2d2d5e;border-radius:6px;padding:11px 16px;
          cursor:pointer;color:#c4b5fd;user-select:none}}
summary:hover {{background:#252550}}
.body   {{padding:16px;background:#12122a;border:1px solid #2d2d5e;border-top:none;
          border-radius:0 0 6px 6px;overflow-x:auto}}
table   {{border-collapse:collapse;width:100%;font-size:.9em}}
th      {{color:#9ca3af;font-size:.78em;text-transform:uppercase;letter-spacing:.05em;
          border-bottom:1px solid #2d2d5e;padding:8px 12px;text-align:right}}
th:first-child {{text-align:left}}
td      {{padding:7px 12px;border-bottom:1px solid #1a1a35;text-align:right}}
td:first-child {{text-align:left}}
tr:hover td {{background:#1a1a35}}
.note   {{color:#9ca3af;font-size:.82em;margin-top:10px}}
</style>
</head>
<body>
<h1>MLB Accuracy Report</h1>
<div class="sub">{date_range} &nbsp;·&nbsp; {total_picks:,} graded picks (after filters)</div>

<div class="cards">
  <div class="card">
    <div class="label">Calibration</div>
    <div class="value">{grade}</div>
    <div class="hint">model prob vs actual</div>
  </div>
  <div class="card">
    <div class="label">Best Book</div>
    <div class="value" style="font-size:1.1em;padding-top:6px">{best_book}</div>
    <div class="hint">ROI {best_book_roi}</div>
  </div>
  <div class="card">
    <div class="label">Best Edge</div>
    <div class="value" style="font-size:1.1em;padding-top:6px">{best_dir}</div>
    <div class="hint">ROI {best_dir_roi}</div>
  </div>
</div>

<details open>
  <summary>Module 1 — Calibration Curve</summary>
  <div class="body">
    <table>
      <tr><th>Bucket</th><th>N</th><th>Predicted</th><th>Actual</th><th>Gap</th><th></th></tr>
      {cal_html or no_data_6}
    </table>
    <p class="note">Grade: <strong style="color:#a78bfa">{grade}</strong>
    &nbsp;·&nbsp; A=&lt;3% avg gap &nbsp; B=3–6% &nbsp; C=6–10% &nbsp; D=&gt;10%
    &nbsp;·&nbsp; &#9888; = fewer than 20 picks in bucket</p>
    {iso_section}
  </div>
</details>

<details open>
  <summary>Module 2 — Per-Book ROI</summary>
  <div class="body">
    <table>
      <tr><th>Book</th><th>N</th><th>Win%</th><th>Profit</th><th>ROI</th></tr>
      {book_html or no_data_5}
    </table>
  </div>
</details>

<details open>
  <summary>Module 3 — Direction Edge (OVER vs UNDER by Stat)</summary>
  <div class="body">
    <table>
      <tr><th>Stat + Side</th><th>N</th><th>Win%</th><th>Profit</th><th>ROI</th></tr>
      {dir_html or no_data_5}
    </table>
    <p class="note">Only stat/side combinations with ≥10 graded picks shown. Sorted by ROI.</p>
  </div>
</details>

</body>
</html>"""


def main() -> None:
    if not GRADED_FILE.exists():
        print(f"Graded picks not found: {GRADED_FILE}")
        return

    raw = pd.read_csv(GRADED_FILE)
    raw.columns = [c.strip().lower() for c in raw.columns]
    if "fair_prob" not in raw.columns:
        raw["fair_prob"] = np.nan
    else:
        raw["fair_prob"] = pd.to_numeric(raw["fair_prob"], errors="coerce")

    df = load_and_filter(raw)

    date_col   = "game_date" if "game_date" in raw.columns else "log_date" if "log_date" in raw.columns else None
    date_range = f"{raw[date_col].min()} to {raw[date_col].max()}" if date_col else "unknown date range"
    total_picks = len(df)

    print(f"\n{'=' * 62}")
    print("  MLB ACCURACY ANALYSIS")
    print(f"  {date_range}")
    print(f"  {total_picks:,} graded picks after filters")
    print(f"{'=' * 62}")

    cal_rows  = calibration_curve(df)
    book_rows = book_roi(df)
    dir_rows  = direction_edge(df)

    print_calibration(cal_rows)

    # Module 1b — retroactive isotonic calibration (in-sample)
    iso_df = retroactive_calibration_curve(df)
    iso_cal_rows = None
    if iso_df is not None:
        iso_cal_rows = calibration_curve(iso_df)
        print(f"\n{'=' * 62}")
        print("  MODULE 1b — CALIBRATION CURVE (retroactive isotonic)")
        print("  NOTE: in-sample fit — shows Grade A potential, not out-of-sample accuracy")
        print(f"{'=' * 62}")
        print(f"  {'Bucket':<12} {'N':>5}  {'Predicted':>9}  {'Actual':>7}  {'Gap':>7}  Note")
        print(f"  {'-' * 58}")
        for r in iso_cal_rows:
            note = "! low-n" if r["low_conf"] else ""
            print(
                f"  {r['bucket']:<12} {r['n']:>5}  "
                f"{r['predicted'] * 100:>8.1f}%  {r['actual'] * 100:>6.1f}%  "
                f"{r['gap'] * 100:>+6.1f}%  {note}"
            )
        print(f"\n  Calibration grade: {calibration_grade(iso_cal_rows)}")
        print("  (A=<3% avg gap  B=3-6%  C=6-10%  D=>10%)  [in-sample]")

    print_book_roi(book_rows)
    print_direction_edge(dir_rows)

    html = build_html(cal_rows, book_rows, dir_rows, date_range, total_picks, iso_cal_rows=iso_cal_rows)
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"\n  HTML report -> {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
