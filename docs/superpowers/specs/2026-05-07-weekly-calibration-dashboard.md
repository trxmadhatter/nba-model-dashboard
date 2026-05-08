# Weekly Calibration Dashboard Panel — Design Spec
**Date:** 2026-05-07

## Problem

The Performance tab shows all-time aggregate stats but no time dimension. There's no way to tell if the model's calibration is improving, degrading, or drifting week over week.

## Solution

Add a collapsible "Weekly Calibration" panel at the bottom of the existing Performance tab in `Data/mlb/dashboard_mlb.html`. One row per calendar week (Monday–Sunday) showing pick count, win rate, model-predicted rate, gap, P&L, and a color-coded trend indicator.

## Goals

1. See calibration quality week-by-week as the season progresses
2. Spot weeks where the model was systematically overconfident or underconfident
3. Zero operational change — no new scripts, no new data sources

## Out of Scope

- Per-stat or per-side weekly breakdown
- Week-over-week delta calculations
- Alerts or notifications

---

## Design

### Data Source

`Data/mlb/history/graded_picks_mlb.csv` — already loaded in `build_performance()`. No new file reads.

### Columns Used

| Column | Use |
|--------|-----|
| `game_date` | Group by ISO week (Monday–Sunday) |
| `hit_result` | WIN/LOSS counts, win rate |
| `fair_prob_raw` | Model-predicted probability (pre-calibration) — avg per week |
| `bet_odds` | P&L calculation (flat $100 per pick) |

### Weekly Row Fields

| Field | Description |
|-------|-------------|
| Week | Mon DD Mon – Sun DD Mon label (e.g. "Apr 7 – Apr 13") |
| Picks | Count of WIN+LOSS rows that week |
| W | Win count |
| L | Loss count |
| Win% | Actual win rate (wins / graded) |
| Model% | Average `fair_prob_raw` for that week's picks |
| Gap | Win% − Model% (negative = model overconfident) |
| P&L | Flat $100 per pick profit/loss |
| Trend | Color dot: green if \|gap\| < 5%, yellow if 5–10%, red if >10% |

### Minimum Picks Threshold

Weeks with fewer than **5** graded picks show a dimmed row (opacity 0.45) with all fields populated. Weeks with 0 graded picks are omitted entirely.

### Gap Color Rules

| \|Gap\| | Color | Meaning |
|---------|-------|---------|
| < 5% | `#00e5a0` (green) | Well-calibrated |
| 5–10% | `#f59e0b` (yellow) | Moderate drift |
| > 10% | `#f04e4e` (red) | Significant overconfidence |

---

## Implementation

### File Changed

`Scripts/generate_dashboard_mlb.py` only.

### New Function: `build_weekly_calibration(df)`

```
Input:  graded picks DataFrame (same df passed to build_performance)
Output: HTML string — a <details> element with collapsible weekly table
```

Logic:
1. Filter to WIN/LOSS rows with valid `game_date` and `fair_prob_raw`
2. Parse `game_date` to datetime, compute ISO week start (Monday)
3. Group by week start date, sort ascending
4. Per group: compute picks, W, L, win_rate, model_avg, gap, pnl
5. Build one `<tr>` per week, apply dimming for thin weeks (<5 picks)
6. Wrap in `<details><summary>Weekly Calibration</summary>...</details>`

### Integration Point

At the end of `build_performance()`, append `build_weekly_calibration(df)` output to the returned HTML string.

---

## HTML Structure

```html
<details style="margin-top:1.5rem">
  <summary style="...">Weekly Calibration</summary>
  <div style="...">
    <table class="data-table mini">
      <thead>
        <tr>
          <th>Week</th><th>Picks</th><th>W</th><th>L</th>
          <th>Win%</th><th>Model%</th><th>Gap</th><th>P&L</th><th></th>
        </tr>
      </thead>
      <tbody>
        <!-- one <tr> per week -->
      </tbody>
    </table>
    <p class="note">Model% = avg fair_prob_raw (pre-calibration Poisson output).
    Gap = Win% - Model%. Dot: green &lt;5%, yellow 5-10%, red &gt;10%.</p>
  </div>
</details>
```

---

## Success Criteria

- `python Scripts/generate_dashboard_mlb.py` runs without error
- Performance tab shows "Weekly Calibration" collapsible at the bottom
- Table has one row per week from season start to today
- Thin weeks (< 5 picks) appear dimmed but not hidden
- Gap column is color-coded correctly
- All existing Performance tab content unchanged
- No new files created
