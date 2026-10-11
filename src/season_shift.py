"""Did batter behaviour change between 2025 and 2026 (first ABS season)? Development data only.

Breaking and offspeed pitches by height relative to the batter-height zone: swing rate and whiff per swing.
Writes reports/season_shift_2025_2026.md.
"""
from pathlib import Path

import pandas as pd

from features import add_batter_height_zone
from pitcher_analysis import wilson

ROOT = Path(__file__).resolve().parents[1]
BINS = [-9, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 9]


def main():
    frames = []
    for season in (2025, 2026):
        m = pd.read_parquet(ROOT / "data" / "model" / f"statcast_{season}_regular" / "model_pitches.parquet")
        m = add_batter_height_zone(m[m["model_population"]]).assign(season=season)
        frames.append(m[m["pitch_group"].isin(["breaking", "offspeed"])])
    d = pd.concat(frames, ignore_index=True)
    d["height_band"] = pd.cut(d["plate_z_abs_zone"], BINS).astype(str)
    g = d.groupby(["height_band", "season"], observed=True, sort=False)
    t = g.agg(pitches=("pitch_key", "size"), swings=("eligible_swing", "sum"), misses=("is_miss", lambda s: int((s == 1).sum())),
              swing_rate=("is_swing", "mean")).reset_index()
    t["whiff_per_swing"] = t["misses"] / t["swings"]
    t["lo"], t["hi"] = wilson(t["misses"], t["swings"])
    order = {str(b): i for i, b in enumerate(pd.cut(pd.Series([-1.0, -0.4, -0.1, 0.1, 0.4, 0.6, 0.9, 2.0]), BINS))}
    t = t.sort_values(["height_band", "season"], key=lambda s: s.map(order) if s.name == "height_band" else s)
    L = ["# Batter behaviour, 2025 vs 2026 (first ABS season)", "",
         "Breaking and offspeed pitches, model population, by vertical location relative to the batter-height ABS zone "
         "(0 = bottom at 27% of height, 1 = top at 53.5%). Development data; 2026 was the v1 test season.", "",
         t.round(3).to_markdown(index=False), "",
         "Below the zone, batters swung more often in 2026 and missed less often when they did. Above the zone, they "
         "swung less. Because the model's target is whiff **given a swing**, a change in which pitches batters choose to "
         "swing at changes the target relationship itself. Fixing a feature definition cannot remove that, which is "
         "why v2 closes only part of the 2026 calibration gap.", "",
         "This is an association across one rule change, not proof that ABS caused it. Other 2026 changes (hitter "
         "approach, pitch mix, ball) could contribute.", ""]
    (ROOT / "reports" / "season_shift_2025_2026.md").write_text("\n".join(L))
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
