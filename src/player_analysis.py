"""Pitcher comparisons and evidence-backed player reports (MLB, 2025-2026).

For every pitcher x pitch type x season: tracking summaries, peer percentiles (same pitch type and throwing hand,
mean velocity within +-2 mph, pitcher excluded), season-over-season changes with 95% intervals, and observed vs
expected whiff (expected from data/pitcher_comparison/expected.parquet, which is pitcher-disjoint and batter-adjusted).

Every flag records its comparison group, units, denominator, uncertainty and supporting rows (pitch keys in an
evidence file). Flags are observations; the report lists hypotheses separately and makes no causal claims.

Writes reports/pitcher_comparison.csv, reports/pitcher_flags.csv, reports/pitcher_comparison_report.md and
reports/players/<pitcher_id>.md (+ _evidence.csv, figure).
Usage: .venv/bin/python src/player_analysis.py [--pitcher-id ID]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from features import add_batter_height_zone, add_features
from pitcher_analysis import wilson

ROOT = Path(__file__).resolve().parents[1]
SEASONS = ("2025", "2026")
CFG = yaml.safe_load((ROOT / "config" / "pitcher_comparison.yaml").read_text())
LOW_PITCHES, LOW_SWINGS = CFG["low_sample"]["tracking_pitches"], CFG["low_sample"]["outcome_swings"]
MIN_PEERS, VELO_BAND = CFG["peers"]["min_peers"], CFG["peers"]["velocity_band_mph"]
PCT_LO, PCT_HI = CFG["peer_extreme_percentile"]
WVE = CFG["whiff_vs_expected"]
RESID_MIN_SWINGS, FDR_Q, BOOT, SEED = WVE["min_swings"], WVE["fdr_q"], WVE["bootstrap"], 20250601
FB_DIFF = ["velo_diff_fb", "ivb_diff_fb", "hmov_diff_fb"]
SHAPE = {"velocity_mph": "mph", "spin_rpm": "rpm", "ivb_in": "in", "hmov_arm_in": "in", "extension_ft": "ft",
         "velo_diff_fb": "mph", "ivb_diff_fb": "in", "hmov_diff_fb": "in"}
SPREAD = {"release_x_sd_in": "in", "release_z_sd_in": "in"}


def load_pitches():
    frames = []
    for s in SEASONS:
        m = pd.read_parquet(ROOT / "data" / "model" / f"statcast_{s}_regular" / "model_pitches.parquet")
        frames.append(m[m["model_population"]].drop(columns=["partition"]))
    p = add_batter_height_zone(add_features(pd.concat(frames, ignore_index=True)))
    p["season"] = pd.to_datetime(p["game_date"]).dt.year.astype(str)
    return p


def summarize(p, ex):
    keys = ["pitcher_id", "pitcher_name", "p_throws", "season", "pitch_type_raw"]
    g = p.groupby(keys, observed=True)
    t = g.agg(pitches=("pitch_key", "size"), games=("game_pk", "nunique"),
              velocity_mph=("velocity_mph", "mean"), velo_sd=("velocity_mph", "std"),
              spin_rpm=("spin_rpm", "mean"), ivb_in=("ivb_in", "mean"), ivb_sd=("ivb_in", "std"),
              hmov_arm_in=("hmov_arm_in", "mean"), hmov_sd=("hmov_arm_in", "std"), extension_ft=("extension_ft", "mean"),
              release_x_sd_in=("release_x_ft", lambda s: s.std() * 12), release_z_sd_in=("release_z_ft", lambda s: s.std() * 12),
              velo_diff_fb=("velo_diff_fb", "mean"), ivb_diff_fb=("ivb_diff_fb", "mean"), hmov_diff_fb=("hmov_diff_fb", "mean")).reset_index()
    tot = p.groupby(["pitcher_id", "season"]).size().rename("season_pitches")
    t = t.merge(tot, on=["pitcher_id", "season"])
    t["usage"] = t["pitches"] / t["season_pitches"]
    e = ex.groupby(["pitcher_id", "season", "pitch_type_raw"]).agg(
        swings=("y", "size"), misses=("y", "sum"), expected=("expected", "mean"), batter_index=("batter_whiff_asof", "mean")).reset_index()
    t = t.merge(e, on=["pitcher_id", "season", "pitch_type_raw"], how="left").fillna({"swings": 0, "misses": 0})
    t["whiff"] = t["misses"] / t["swings"].where(t["swings"] > 0)
    t["whiff_lo"], t["whiff_hi"] = wilson(t["misses"], t["swings"])
    t["whiff_minus_expected"] = t["whiff"] - t["expected"]
    t["low_sample_tracking"], t["low_sample_swings"] = t["pitches"] < LOW_PITCHES, t["swings"] < LOW_SWINGS
    return t


def peer_percentiles(t):
    """Percentile of each row's metric among peers: same season, pitch type and hand, >= LOW_PITCHES pitches, mean
    velocity within +-VELO_BAND (velocity itself is compared without the band), pitcher excluded."""
    t = t.copy()
    ok = t["pitches"] >= LOW_PITCHES
    for metric in list(SHAPE) + list(SPREAD):
        t[f"{metric}_pct"], t[f"{metric}_peers"] = np.nan, 0
    for _, grp in t[ok].groupby(["season", "pitch_type_raw", "p_throws"]):
        v = grp["velocity_mph"].to_numpy()
        for i, idx in enumerate(grp.index):
            others = np.arange(len(grp)) != i
            band = others & (np.abs(v - v[i]) <= VELO_BAND)
            for metric in list(SHAPE) + list(SPREAD):
                pool = others if metric == "velocity_mph" else band
                vals = grp[metric].to_numpy()[pool]
                vals = vals[~np.isnan(vals)]
                if len(vals) >= MIN_PEERS and not np.isnan(grp.at[idx, metric]):
                    t.at[idx, f"{metric}_pct"] = 100 * (np.mean(vals < grp.at[idx, metric]) + 0.5 * np.mean(vals == grp.at[idx, metric]))
                    t.at[idx, f"{metric}_peers"] = len(vals)
    return t


def season_changes(t):
    a = t[t["season"] == "2025"].set_index(["pitcher_id", "pitch_type_raw"])
    b = t[t["season"] == "2026"].set_index(["pitcher_id", "pitch_type_raw"])
    j = a.join(b, lsuffix="_25", rsuffix="_26", how="inner")
    out = pd.DataFrame(index=j.index)
    both = (j["pitches_25"] >= LOW_PITCHES) & (j["pitches_26"] >= LOW_PITCHES)
    types = j.index.get_level_values("pitch_type_raw")
    for m, sd in (("velocity_mph", "velo_sd"), ("ivb_in", "ivb_sd"), ("hmov_arm_in", "hmov_sd")):
        raw = j[f"{m}_26"] - j[f"{m}_25"]
        league = raw[both].groupby(types[both]).median()           # league-wide shift for the pitch type
        d = raw - pd.Series(types, index=j.index).map(league).fillna(0).to_numpy()
        se = np.sqrt(j[f"{sd}_25"] ** 2 / j["pitches_25"] + j[f"{sd}_26"] ** 2 / j["pitches_26"])
        out[f"d_{m}"], out[f"d_{m}_lo"], out[f"d_{m}_hi"] = d, d - 1.96 * se, d + 1.96 * se
        out[f"raw_d_{m}"] = raw
    out["d_usage"] = j["usage_26"] - j["usage_25"]
    out["pitches_25"], out["pitches_26"] = j["pitches_25"], j["pitches_26"]
    return out.reset_index()


def residual_bootstrap(ex, keys):
    """Observed minus expected whiff per group with a game-resampling 95% interval and two-sided p-value."""
    rng = np.random.default_rng(SEED)
    per_game = ex.assign(r=ex["y"] - ex["expected"]).groupby(keys + ["game_pk"]).agg(r=("r", "sum"), n=("y", "size")).reset_index()
    rows = []
    for k, g in per_game.groupby(keys):
        n = g["n"].sum()
        if n < RESID_MIN_SWINGS:
            continue
        w = rng.poisson(1.0, (BOOT, len(g)))
        draws = (w @ g["r"].to_numpy()) / np.maximum(w @ g["n"].to_numpy(), 1)
        point = g["r"].sum() / n
        p = min(1.0, 2 * min(np.mean(draws <= 0), np.mean(draws >= 0)))
        rows.append(dict(zip(keys, k if isinstance(k, tuple) else (k,)), swings=int(n), games=len(g),
                         resid=point, lo=np.percentile(draws, 2.5), hi=np.percentile(draws, 97.5), p=p))
    out = pd.DataFrame(rows)
    # Benjamini-Hochberg across all tested groups
    order = out["p"].rank(method="first")
    out["bh_threshold"] = FDR_Q * order / len(out)
    passing = out.loc[out["p"] <= out["bh_threshold"], "p"]
    out["fdr_flag"] = out["p"] <= (passing.max() if len(passing) else -1)
    return out


def build_flags(t, changes, resid, ex):
    flags = []
    cur = t[(t["season"] == "2026") & (t["pitches"] >= LOW_PITCHES)]
    for _, r in cur.iterrows():
        for m, unit in SHAPE.items():
            if m in FB_DIFF and r.pitch_type_raw in CFG["fastball_diff_metrics_exclude_types"]:
                continue
            pct, n = r.get(f"{m}_pct"), r.get(f"{m}_peers")
            if pd.notna(pct) and (pct >= PCT_HI or pct <= PCT_LO):
                group = (f"2026 {r.pitch_type_raw} from {r.p_throws}HP" +
                         ("" if m == "velocity_mph" else f", mean velocity within ±{VELO_BAND:g} mph"))
                flags.append(dict(pitcher_id=r.pitcher_id, pitch_type=r.pitch_type_raw, kind="peer extreme", metric=m,
                                  value=r[m], unit=unit, comparison_group=group, comparison_n=int(n),
                                  denominator=f"{int(r.pitches)} pitches", uncertainty=f"percentile {pct:.0f} of {int(n)} peers (rank, no interval)",
                                  evidence=f"season=2026, pitch_type={r.pitch_type_raw}"))
        for m, unit in SPREAD.items():
            pct, n = r.get(f"{m}_pct"), r.get(f"{m}_peers")
            if pd.notna(pct) and pct >= CFG["release_spread_percentile"]:
                flags.append(dict(pitcher_id=r.pitcher_id, pitch_type=r.pitch_type_raw, kind="release spread", metric=m,
                                  value=r[m], unit=unit, comparison_group=f"2026 {r.pitch_type_raw} from {r.p_throws}HP, ±{VELO_BAND:g} mph",
                                  comparison_n=int(n), denominator=f"{int(r.pitches)} pitches",
                                  uncertainty=f"percentile {pct:.0f} of {int(n)} peers", evidence=f"season=2026, pitch_type={r.pitch_type_raw}"))
    for _, r in changes.iterrows():
        if min(r.pitches_25, r.pitches_26) < LOW_PITCHES:
            continue
        for m, unit in (("velocity_mph", "mph"), ("ivb_in", "in"), ("hmov_arm_in", "in")):
            d, lo, hi, thr = r[f"d_{m}"], r[f"d_{m}_lo"], r[f"d_{m}_hi"], CFG["season_change"][m]
            if abs(d) >= thr and (lo > 0 or hi < 0):
                flags.append(dict(pitcher_id=r.pitcher_id, pitch_type=r.pitch_type_raw, kind="season change", metric=m, value=d,
                                  unit=unit, comparison_group="own 2025 pitches of the same type, net of the league's median change for that type",
                                  comparison_n=int(r.pitches_25),
                                  denominator=f"{int(r.pitches_25)} (2025) vs {int(r.pitches_26)} (2026) pitches",
                                  uncertainty=f"95% CI {lo:+.2f} to {hi:+.2f} {unit}", evidence=f"seasons=2025+2026, pitch_type={r.pitch_type_raw}"))
    for _, r in resid[resid["fdr_flag"] & (resid["season"] == "2026")].iterrows():
        flags.append(dict(pitcher_id=r.pitcher_id, pitch_type="all", kind="whiff vs expected", metric="whiff_minus_expected",
                          value=r.resid, unit="proportion", comparison_group="pitcher-disjoint model of other pitchers on similar pitches and batters",
                          comparison_n=int(r.swings), denominator=f"{int(r.swings)} swings in {int(r.games)} games",
                          uncertainty=f"95% game-bootstrap CI {r.lo:+.3f} to {r.hi:+.3f}; BH FDR q={FDR_Q}", evidence="season=2026, all swings"))
    f = pd.DataFrame(flags)
    names = t.drop_duplicates("pitcher_id").set_index("pitcher_id")["pitcher_name"]
    f.insert(1, "pitcher_name", f["pitcher_id"].map(names))
    return f


def player_report(pid, t, changes, resid, flags, p, ex):
    me = t[t["pitcher_id"] == pid]
    name, hand = me["pitcher_name"].iloc[0], me["p_throws"].iloc[0]
    out_dir = ROOT / "reports" / "players"
    out_dir.mkdir(parents=True, exist_ok=True)
    cur = me[me["season"] == "2026"].sort_values("pitches", ascending=False)
    rep = cur[["pitch_type_raw", "pitches", "usage", "velocity_mph", "spin_rpm", "ivb_in", "hmov_arm_in", "extension_ft",
               "release_x_sd_in", "release_z_sd_in", "swings", "misses", "whiff", "whiff_lo", "whiff_hi", "expected"]].copy()
    pct = cur[["pitch_type_raw"] + [f"{m}_pct" for m in SHAPE] + ["velocity_mph_peers", "ivb_in_peers"]].round(0)
    ch = changes[changes["pitcher_id"] == pid].round(2)
    rs = resid[resid["pitcher_id"] == pid].round(4)
    fl = flags[flags["pitcher_id"] == pid]

    # Evidence: the pitch keys behind each flag
    ev = []
    for i, f in fl.reset_index(drop=True).iterrows():
        q = p[(p["pitcher_id"] == pid)]
        if f.pitch_type != "all":
            q = q[q["pitch_type_raw"] == f.pitch_type]
        if "2025" not in f.evidence:
            q = q[q["season"] == "2026"]
        if f.kind == "whiff vs expected":
            q = ex[(ex["pitcher_id"] == pid) & (ex["season"] == "2026")]
        ev.append(pd.DataFrame({"flag_id": i + 1, "pitch_key": q["pitch_key"].values}))
    if ev:
        pd.concat(ev).to_csv(out_dir / f"{pid}_evidence.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, season in zip(axes, SEASONS):
        q = p[(p["pitcher_id"] == pid) & (p["season"] == season)]
        for tname, d in q.groupby("pitch_type_raw"):
            if len(d) >= LOW_PITCHES:
                ax.scatter(d["hmov_arm_in"], d["ivb_in"], s=4, alpha=0.4, label=f"{tname} (n={len(d)})")
        ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
        ax.set_title(f"{season}", fontsize=10); ax.set_xlabel("arm-side horizontal movement (in)"); ax.set_ylabel("induced vertical break (in)")
        ax.legend(fontsize=7, markerscale=3)
    fig.suptitle(f"{name}: movement by season"); fig.tight_layout()
    fig.savefig(out_dir / f"{pid}_movement.png", dpi=120); plt.close(fig)

    L = [f"# {name} ({hand}HP, MLBAM {pid}) — player report", "",
         "MLB regular seasons 2025–2026, model-population pitches. Generated by `src/player_analysis.py`. Every number traces to "
         f"`data/model/statcast_202x_regular/model_pitches.parquet` rows with `pitcher_id == {pid}`; flag evidence lists the pitch keys "
         f"(`reports/players/{pid}_evidence.csv`).", "",
         "**Units.** mph, rpm, inches (movement: IVB, arm-side HB), ft (extension), release spread = SD in inches. "
         "Whiff = misses / ordinary swings (Wilson 95% CI). Expected = mean pitcher-disjoint, batter-adjusted model expectation on the same swings.", "",
         "## 2026 repertoire", "", rep.round(3).to_markdown(index=False), "",
         "## 2026 peer percentiles", "",
         f"Peers: other {hand}HP pitchers' 2026 pitches of the same type (≥{LOW_PITCHES} pitches each); shape metrics restricted to mean velocity "
         f"within ±{VELO_BAND:g} mph. 50 = typical; percentiles need ≥{MIN_PEERS} peers.", "",
         pct.to_markdown(index=False), "",
         "## Change from 2025 to 2026 (same pitch type)", "", ch.drop(columns=["pitcher_id"]).to_markdown(index=False) if len(ch) else "No overlapping pitch types.", "",
         f"![movement]({pid}_movement.png)", "",
         "## Whiff vs expected (all swings)", "",
         (rs.to_markdown(index=False) if len(rs) else f"Fewer than {RESID_MIN_SWINGS} swings in each season."), "",
         "Expected comes from a model that never saw this pitcher and accounts for batters faced (their earlier-game whiff rates). "
         "A positive difference means more misses than other pitchers got on similar pitches to similar batters. It can reflect "
         "deception, sequencing, catcher framing of swings, unmeasured pitch traits or chance. It is **not** attributed to any one cause.", "",
         "## Flags (observations)", "",
         (fl.drop(columns=["pitcher_id", "pitcher_name"]).assign(flag_id=range(1, len(fl) + 1)).round(3).to_markdown(index=False)
          if len(fl) else "No flags met the thresholds."), "",
         "## Hypotheses to test (not findings)", "",
         *([f"- Flag {i + 1} ({f.kind}, {f.pitch_type} {f.metric}): is it stable across the next 30 days of outings, and does it "
            "coincide with changes in swing rate as well as whiff rate?" for i, f in fl.reset_index(drop=True).iterrows()] or ["- None generated."]), "",
         "## Limits", "",
         "- Associations only; nothing here shows that changing a pitch would change outcomes (a practice experiment is needed).",
         "- 2026 is the first ABS season; batter behaviour on low pitches changed (`reports/season_shift_2025_2026.md`).",
         "- Release spread mixes true inconsistency with tracking noise and pitch-type labelling.",
         "- Statcast pitch types are assigned by an algorithm. A season change, especially with a large change in pitch count for "
         "that type, may partly be a relabelled pitch (e.g. slider vs sweeper) rather than a changed pitch.", ""]
    (out_dir / f"{pid}.md").write_text("\n".join(L))


def main(pitcher_id=None):
    p = load_pitches()
    ex = pd.read_parquet(ROOT / "data" / "pitcher_comparison" / "expected.parquet")
    t = peer_percentiles(summarize(p, ex))
    changes = season_changes(t)
    resid = residual_bootstrap(ex, ["pitcher_id", "season"])
    flags = build_flags(t, changes, resid, ex)
    t.round(5).to_csv(ROOT / "reports" / "pitcher_comparison.csv", index=False)
    flags.round(5).to_csv(ROOT / "reports" / "pitcher_flags.csv", index=False)

    if pitcher_id is None:  # coverage rule: most 2026 pitches among pitchers who also pitched in 2025
        both = t.groupby("pitcher_id")["season"].nunique().pipe(lambda s: s[s == 2].index)
        tot = t[(t["season"] == "2026") & t["pitcher_id"].isin(both)].groupby("pitcher_id")["pitches"].sum()
        pitcher_id = int(tot.idxmax())
    player_report(pitcher_id, t, changes, resid, flags, p, ex)

    imp = pd.read_csv(ROOT / "data" / "pitcher_comparison" / "permutation_importance.csv")
    r26 = resid[resid["season"] == "2026"]
    calib = ex.groupby("season").agg(observed=("y", "mean"), expected=("expected", "mean"), swings=("y", "size"))
    L = ["# Pitcher comparisons and explanations", "",
         "MLB 2025–2026 regular seasons. Outputs: `reports/pitcher_comparison.csv` (every pitcher × pitch type × season), "
         "`reports/pitcher_flags.csv` (evidence-backed observations), and per-pitcher reports in `reports/players/` "
         "(`src/player_analysis.py --pitcher-id ID`).", "",
         "## Expected whiff (pitcher-disjoint, batter-adjusted)", "",
         "5-fold cross-fitting with folds grouped by pitcher over 2025 + 2026 swings; inputs are the v2 features, season, and the "
         f"batter's shrunken whiff rate from earlier games (prior weight {100} swings). Overall agreement:", "",
         calib.round(4).to_markdown(), "",
         "## What the model relies on (permutation importance, August 2026 validation, v2 development model)", "",
         "Increase in log loss when one feature is shuffled. Correlated features share credit (e.g. velocity and velo_diff_fb), so "
         "small values do not mean irrelevant. This is predictive reliance, not a causal effect.", "",
         imp.round(5).to_markdown(index=False), "",
         "## Whiff vs expected across pitchers (2026)", "",
         f"{len(r26)} pitchers with ≥{RESID_MIN_SWINGS} swings. Benjamini–Hochberg at q = {FDR_Q}: **{int(r26['fdr_flag'].sum())}** flagged "
         f"(about {FDR_Q * max(int(r26['fdr_flag'].sum()), 0):.0f} of these may be false discoveries). Without correction, "
         f"{int((r26['p'] < 0.05).sum())} would pass p < 0.05 (≈{0.05 * len(r26):.0f} expected by chance alone).", "",
         "## Flags", "",
         flags.groupby("kind").size().rename("flags").to_frame().to_markdown(), "",
         f"Rules (`config/pitcher_comparison.yaml`): peer percentile ≤ {PCT_LO} or ≥ {PCT_HI} among ≥ {MIN_PEERS} peers (≥ {LOW_PITCHES} pitches each; "
         f"fastball-difference metrics not ranked for fastballs); release spread ≥ {CFG['release_spread_percentile']}th percentile; season change, "
         f"net of the league's median change for the pitch type, ≥ {CFG['season_change']['velocity_mph']} mph or ≥ {CFG['season_change']['ivb_in']} in "
         f"with a 95% interval excluding 0; whiff vs expected passing BH FDR (≥ {RESID_MIN_SWINGS} swings). Each row names its comparison group, "
         "units, denominator, uncertainty and evidence filter. An earlier draft used 10/90 percentiles and fixed 1 mph / 2 in changes and "
         "produced a median of 8 flags per pitcher, which flagged ordinary variation.", "",
         f"## Example player report: `reports/players/{pitcher_id}.md`", "",
         "Selection rule: most 2026 pitches among pitchers who also pitched in 2025 (coverage, not quality).", "",
         "## Limits", "",
         "- Peer percentiles describe where a pitch sits, not whether it is good; pitch shape value depends on the whole arsenal.",
         "- Observed − expected includes chance; intervals and FDR control are reported, and flags are prompts for review.",
         "- MLB peers only. College players are not compared to MLB norms.", ""]
    (ROOT / "reports" / "pitcher_comparison_report.md").write_text("\n".join(L))
    print(f"comparison rows {len(t):,}; flags {len(flags):,} ({flags.groupby('kind').size().to_dict()}); report pitcher {pitcher_id}")
    print("2026 pitchers tested", len(r26), "FDR flagged", int(r26["fdr_flag"].sum()), "| p<0.05", int((r26["p"] < 0.05).sum()))
    print(calib.round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pitcher-id", type=int)
    main(**vars(ap.parse_args()))
