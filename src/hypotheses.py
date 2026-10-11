"""Constrained, coach-reviewable hypotheses (rules in config/hypotheses.yaml).

Three kinds, all restricted to what the pitcher has demonstrated:
  own_history  - the same pitch type with its 2025 shape vs its 2026 shape. Observed outcome differences are net of the
                 league's change for that pitch type; the model estimate re-scores his 2026 pitches (same locations,
                 counts, batter sides) with the 2025 shape.
  sensitivity  - one shape feature moved by a small step, only where his own pitches show that value is within his range
                 and jointly supported. Labelled MODEL SENSITIVITY, not an instruction.
  release      - his pitches released far from his usual (game-median) release point vs the rest; the model estimate
                 moves release to the game median, and the observed difference is the rest minus the far quarter.
Tiers follow config/hypotheses.yaml: model-only evidence (every sensitivity record) never ranks above C; A and B need the
pitcher's own observed run value to agree in sign with the model estimate.
Each is scored on four outcomes: P(swing), P(whiff | swing), swinging-strike probability (their product, an
intermediate outcome) and expected run value (runs saved per 100 pitches). Location is never perturbed.

Writes reports/hypotheses.csv, reports/hypotheses_report.md and reports/players/<id>_hypotheses.md.
Usage: .venv/bin/python src/hypotheses.py [--pitcher-id ID]
"""
import argparse
import datetime as dt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config" / "hypotheses.yaml").read_text())
SUP = CFG["support"]
SHAPE = SUP["shape_features"]
DIFF_OF = {"velocity_mph": "velo_diff_fb", "ivb_in": "ivb_diff_fb", "hmov_arm_in": "hmov_diff_fb"}
FASTBALLS = {"FF", "SI", "FA"}
SEED, BOOT = 20250601, 500
UNIT = {"velocity_mph": "mph", "spin_rpm": "rpm", "ivb_in": "in", "hmov_arm_in": "in"}


class Scorer:
    def __init__(self):
        m = joblib.load(ROOT / "models" / "outcome_models_v2" / "outcome_models.joblib")
        self.swing, self.whiff, self.rv, self.X, self.XS = m["swing"], m["whiff"], m["run_value"], m["features"], m["swing_features"]

    def score(self, df):
        X = df[self.X]
        s = self.swing.predict_proba(df[self.XS])[:, 1]
        w = self.whiff.predict_proba(X)[:, 1]
        return pd.DataFrame({"p_swing": s, "p_whiff": w, "swstr": s * w, "rv": self.rv.predict(X)}, index=df.index)


def shifted(df, feature, step, pitch_type):
    out = df.copy()
    out[feature] = out[feature] + step
    if feature in DIFF_OF and pitch_type not in FASTBALLS:   # reference fastball unchanged, so the difference moves too
        out[DIFF_OF[feature]] = out[DIFF_OF[feature]] + step
    return out


def supported(own, feature, target_mean, scale):
    """Marginal: target inside his own 5-95% range; joint: enough of his own pitches near the shifted mean shape."""
    lo, hi = np.percentile(own[feature].dropna(), SUP["own_pitch_percentiles"])
    if not (lo <= target_mean <= hi):
        return False, f"{feature} {target_mean:.1f} outside his 5-95% range {lo:.1f}-{hi:.1f}"
    centre = own[SHAPE].mean()
    centre[feature] = target_mean
    z = ((own[SHAPE] - centre) / scale).dropna()
    n = int((np.sqrt((z ** 2).sum(axis=1)) <= SUP["neighbour_radius_sd"] * np.sqrt(len(SHAPE))).sum())
    return n >= SUP["min_neighbours"], f"{n} of his own pitches near the shifted shape"


def boot_mean_diff(delta, games, rng):
    """Mean per-pitch difference with a game-resampling 95% interval."""
    g = pd.DataFrame({"d": delta, "g": games}).groupby("g")["d"].agg(["sum", "size"])
    w = rng.poisson(1.0, (BOOT, len(g)))
    draws = (w @ g["sum"].to_numpy()) / np.maximum(w @ g["size"].to_numpy(), 1)
    return float(delta.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def model_estimate(base, scen, games, rng):
    """Scenario minus base on the same pitches, for each outcome; whiff|swing aggregated with swing weights."""
    out = {}
    for k, scale in (("p_swing", 100), ("swstr", 100), ("rv", 100)):
        m, lo, hi = boot_mean_diff((scen[k] - base[k]).to_numpy() * scale, games, rng)
        out[k] = (m, lo, hi)
    wb = (base["p_swing"] * base["p_whiff"]).sum() / base["p_swing"].sum()
    ws = (scen["p_swing"] * scen["p_whiff"]).sum() / scen["p_swing"].sum()
    out["p_whiff"] = (100 * (ws - wb), np.nan, np.nan)
    return out


def swstr_flag(df):
    """1 = swinging strike, 0 = anything else. is_miss is missing on takes, so it is filled before averaging over pitches."""
    return df["is_miss"].eq(1).fillna(False).astype(float)


def observed_outcomes(df):
    sw = df["swing"] == 1
    return {"p_swing": df["swing"].mean(), "p_whiff": df.loc[sw, "is_miss"].mean() if sw.any() else np.nan,
            "swstr": swstr_flag(df).mean(), "rv": df["run_value"].mean()}


def observed_did(a, b, change, rng, paired=False):
    """(b - a) minus `change` (the league's change for the pitch type, or zeros), per 100 pitches, with a game bootstrap.

    Independent resampling of each group's games by default; paired=True resamples the same games for both groups
    (a and b are pitches from the same games).
    """
    res = {}
    for k in ("p_swing", "swstr", "rv"):
        col = {"p_swing": "swing", "swstr": "_swstr", "rv": "run_value"}[k]
        ga, gb = (d.groupby("game_pk")[col].agg(["sum", "count"]) for d in (a, b))
        if paired:
            games = ga.index.union(gb.index)
            ga, gb = ga.reindex(games, fill_value=0), gb.reindex(games, fill_value=0)
            w = rng.poisson(1.0, (BOOT, len(games)))
            ws = (w, w)
        else:
            ws = (rng.poisson(1.0, (BOOT, len(ga))), rng.poisson(1.0, (BOOT, len(gb))))
        da, db = ((w @ g["sum"].to_numpy()) / np.where(w @ g["count"].to_numpy() > 0, w @ g["count"].to_numpy(), np.nan)
                  for w, g in zip(ws, (ga, gb)))
        diff = 100 * (db - da - change[k])
        point = 100 * (b[col].mean() - a[col].mean() - change[k])
        res[k] = (point, float(np.nanpercentile(diff, 2.5)), float(np.nanpercentile(diff, 97.5)))
    return res


def tier(kind, est, obs=None):
    """Tier rules in config/hypotheses.yaml `ranking`. Model-only records (obs is None) are always C."""
    rv_m = est["rv"][0]
    tradeoff = bool(np.sign(est["swstr"][0]) != np.sign(rv_m) and abs(est["swstr"][0]) > 0.1)
    if obs is None or tradeoff:
        return "C", tradeoff
    o_m, o_lo, o_hi = obs["rv"]
    if rv_m == 0 or np.sign(o_m) != np.sign(rv_m):
        return "C", tradeoff
    if kind == "own_history" and (o_lo > 0 or o_hi < 0):
        return "A", tradeoff
    return "B", tradeoff


def fmt(t, unit="pp"):
    m, lo, hi = t
    return f"{m:+.2f} {unit}" + ("" if np.isnan(lo) else f" (95% CI {lo:+.2f} to {hi:+.2f})")


def record(kind, r, est, obs, finding, evidence, support_note, tier_, tradeoff, n):
    review = CFG["review_after_pitches"]
    o_rv = (np.nan, np.nan, np.nan) if obs is None else obs["rv"]
    return {
        "pitcher_id": r["pitcher_id"], "pitcher_name": r["pitcher_name"], "pitch_type": r["pitch_type"], "kind": kind, "tier": tier_,
        "perturbed_features": ", ".join(r["perturbed"]),
        "finding": finding, "supporting_evidence": evidence, "support_check": support_note,
        "hypothesis_to_test": r["hypothesis"],
        "model_estimate": (f"P(swing) {fmt(est['p_swing'])}; P(whiff | swing) {fmt(est['p_whiff'])}; swinging-strike prob. {fmt(est['swstr'])}; "
                           f"run value {fmt(est['rv'], 'runs saved / 100 pitches')}"),
        "observed_difference": ("" if obs is None else
                                f"{r['obs_label']}: swing rate {fmt(obs['p_swing'])}; SwStr {fmt(obs['swstr'])}; run value {fmt(obs['rv'], 'runs / 100')}"),
        "trade_off": ("swinging strikes and run value move in opposite directions" if tradeoff else "none detected in modelled outcomes"),
        "rv_estimate_per100": est["rv"][0], "rv_lo": est["rv"][1], "rv_hi": est["rv"][2], "swstr_estimate_pp": est["swstr"][0],
        "obs_rv_per100": o_rv[0], "obs_rv_lo": o_rv[1], "obs_rv_hi": o_rv[2],
        "pitches_used": n,
        "limitations": ("Associations from MLB 2025-2026 models and observations; not evidence that changing the pitch causes the change. "
                        "2026 batter behaviour on low pitches differs from 2025. Model uncertainty is not in the interval (it covers game sampling only)."),
        "coach_selected_experiment": "PENDING COACH SELECTION",
        "suggested_designs": ("Alternate blocks of bullpen/live sessions with and without the change (counterbalanced), same pitch-count limits; "
                              "a no-batter bullpen can test repeatability of the shape, not whiffs."),
        "success_metric": (f"Primary: run value per 100 {r['pitch_type']} pitches vs his 2026 baseline, 95% CI. Secondary: SwStr and whiff per swing. "
                           "Guardrails: ball rate and in-zone rate must not worsen by more than their 95% CI half-width."),
        "review_point": f"after {review} {r['pitch_type']} pitches under the change, or 30 days (whichever first)",
        "status": "proposed", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }


def main(pitcher_id=None):
    rng = np.random.default_rng(SEED)
    p = pd.read_parquet(ROOT / "data" / "outcomes" / "pitches_all.parquet")
    p["_swstr"] = swstr_flag(p)
    scorer = Scorer()
    league = {t: {k: (observed_outcomes(g[g["season"] == "2026"])[k] - observed_outcomes(g[g["season"] == "2025"])[k])
                  for k in ("p_swing", "swstr", "rv")}
              for t, g in p.groupby("pitch_type_raw") if g["season"].nunique() == 2}
    scale = p.groupby("pitch_type_raw")[SHAPE].std()
    flags6 = pd.read_csv(ROOT / "reports" / "pitcher_flags.csv")
    release_flagged = set(zip(flags6.loc[flags6["kind"] == "release spread", "pitcher_id"], flags6.loc[flags6["kind"] == "release spread", "pitch_type"]))
    records, rejected = [], []

    for (pid, ptype), g in p.groupby(["pitcher_id", "pitch_type_raw"]):
        a, b = g[g["season"] == "2025"], g[g["season"] == "2026"]
        name = g["pitcher_name"].iloc[0]
        if len(b) < CFG["min_pitches_per_type"]:
            continue
        base = scorer.score(b)

        # own history
        if len(a) >= CFG["min_pitches_per_profile"] and ptype in league:
            scen = b.copy()
            for f in SHAPE:
                scen = shifted(scen, f, a[f].mean() - b[f].mean(), ptype) if f in DIFF_OF else scen.assign(**{f: scen[f] + a[f].mean() - b[f].mean()})
            est = model_estimate(scorer.score(scen), base, b["game_pk"].to_numpy(), rng)   # 2026 shape minus 2025 shape
            obs = observed_did(a, b, league[ptype], rng)
            shape_change = ", ".join(f"{f.replace('_', ' ')} {b[f].mean() - a[f].mean():+.1f} {UNIT[f]}" for f in SHAPE)
            t, tradeoff = tier("own_history", est, obs)
            records.append(record("own_history", {"pitcher_id": pid, "pitcher_name": name, "pitch_type": ptype,
                                  "perturbed": SHAPE, "obs_label": "net of league change",
                                  "hypothesis": f"His {ptype} with its 2025 shape vs its 2026 shape: does the shape difference go with a difference in run value when thrown to the same locations and counts?"},
                                  est, obs, f"{ptype} shape changed 2025→2026 ({shape_change}); {len(a)} pitches in 2025, {len(b)} in 2026.",
                                  f"seasons 2025+2026, pitch_type={ptype}; model re-scores his 2026 pitches with the 2025 mean shape (estimate = 2026 shape − 2025 shape).",
                                  "both profiles thrown by him (≥ min pitches each)", t, tradeoff, len(b)))

        # sensitivity
        for f, step in CFG["sensitivity_steps"].items():
            for sgn in (+1, -1):
                target = b[f].mean() + sgn * step
                ok, note = supported(b, f, target, scale.loc[ptype])
                if not ok:
                    rejected.append({"pitcher_id": pid, "pitch_type": ptype, "feature": f, "step": sgn * step, "reason": note})
                    continue
                est = model_estimate(base, scorer.score(shifted(b, f, sgn * step, ptype)), b["game_pk"].to_numpy(), rng)
                t, tradeoff = tier("sensitivity", est)
                records.append(record("sensitivity", {"pitcher_id": pid, "pitcher_name": name, "pitch_type": ptype, "perturbed": [f],
                                      "hypothesis": f"MODEL SENSITIVITY: his {ptype} with {f.replace('_', ' ')} {sgn * step:+g} {UNIT[f]} (a value he already throws) — is the modelled run-value difference visible in his own outcomes?"},
                                      est, None, f"Model sensitivity for {ptype} {f} {sgn * step:+g} {UNIT[f]}, other features and locations fixed.",
                                      f"season 2026, pitch_type={ptype}, {len(b)} pitches", note, t, tradeoff, len(b)))

        # release consistency (only where the pitcher comparison flagged a large release spread)
        if (pid, ptype) in release_flagged:
            med = b.groupby("game_pk")[["release_x_arm_ft", "release_z_ft"]].transform("median")
            dist = np.sqrt(((b[["release_x_arm_ft", "release_z_ft"]] - med) ** 2).sum(axis=1)) * 12
            far = dist >= dist.quantile(0.75)
            scen = b.copy()
            scen.loc[far, ["release_x_arm_ft", "release_z_ft"]] = med.loc[far].to_numpy()
            est = model_estimate(base[far], scorer.score(scen[far]), b.loc[far, "game_pk"].to_numpy(), rng)
            o_far, o_near = observed_outcomes(b[far]), observed_outcomes(b[~far])
            obs = observed_did(b[far], b[~far], {"p_swing": 0.0, "swstr": 0.0, "rv": 0.0}, rng, paired=True)
            t, tradeoff = tier("release", est, obs)
            records.append(record("release", {"pitcher_id": pid, "pitcher_name": name, "pitch_type": ptype,
                                  "perturbed": ["release_x_arm_ft", "release_z_ft"], "obs_label": "rest minus far-release quarter (observed)",
                                  "hypothesis": f"Are his {ptype} pitches released ≥ {dist.quantile(0.75):.1f} in from his game-median release point less effective, and does a tighter release change that?"},
                                  est, obs,
                                  f"Release spread flagged in the pitcher comparison. Far-release quarter vs rest (observed): run value {100 * o_far['rv']:+.2f} vs {100 * o_near['rv']:+.2f} runs/100; SwStr {100 * o_far['swstr']:.1f}% vs {100 * o_near['swstr']:.1f}%.",
                                  f"season 2026, pitch_type={ptype}, {int(far.sum())} far-release pitches", "release moved to his own game median (demonstrated)",
                                  t, tradeoff, int(far.sum())))

    h = pd.DataFrame(records)
    h["rank_score"] = h["rv_estimate_per100"].abs() / ((h["rv_hi"] - h["rv_lo"]) / 3.92).replace(0, np.nan)
    h = h.sort_values(["tier", "rank_score"], ascending=[True, False]).reset_index(drop=True)
    h.insert(0, "hypothesis_id", [f"H{i + 1:05d}" for i in range(len(h))])
    h.to_csv(ROOT / "reports" / "hypotheses.csv", index=False)
    rej = pd.DataFrame(rejected)

    if pitcher_id is None:
        pitcher_id = 645261 if (h["pitcher_id"] == 645261).any() else int(h["pitcher_id"].iloc[0])
    mine = h[h["pitcher_id"] == pitcher_id]
    name = mine["pitcher_name"].iloc[0]
    out = ROOT / "reports" / "players"
    out.mkdir(parents=True, exist_ok=True)
    rec_cols = ["finding", "supporting_evidence", "support_check", "hypothesis_to_test", "model_estimate", "observed_difference",
                "trade_off", "limitations", "coach_selected_experiment", "suggested_designs", "success_metric", "review_point", "status"]
    L = [f"# {name} (MLBAM {pitcher_id}) — hypotheses for coach review", "",
         "These are **hypotheses to test**, not instructions. Estimates are associations from MLB 2025–2026 models; they do not show that "
         "changing a pitch will change results. Location changes are deliberately not proposed. Each record follows a standard form.", "",
         "Outcomes: P(swing); P(whiff | swing); swinging-strike probability = their product (an intermediate outcome, not pitch value); "
         "run value = runs saved per 100 pitches (positive favours the pitcher; includes called strikes, balls and contact damage).", "",
         f"Tiers: A — {CFG['ranking']['tier_A']}. B — {CFG['ranking']['tier_B']}. C — {CFG['ranking']['tier_C']}.", ""]
    for _, r in mine.head(8).iterrows():
        L += [f"## {r.hypothesis_id} · {r.pitch_type} · {r.kind} · tier {r.tier}", ""]
        L += [f"- **{c.replace('_', ' ').capitalize()}:** {r[c]}" for c in rec_cols if isinstance(r[c], str) and r[c]]
        L += [""]
    L += [f"{len(mine)} hypotheses in total for this pitcher; full list in `reports/hypotheses.csv` "
          f"(filter `pitcher_id == {pitcher_id}`). Scenarios rejected for lack of support: "
          f"{int((rej['pitcher_id'] == pitcher_id).sum()) if len(rej) else 0}.", ""]
    (out / f"{pitcher_id}_hypotheses.md").write_text("\n".join(L))

    gate = {
        "every record has a measurable success metric and review point": bool(h["success_metric"].str.contains("95% CI").all() and h["review_point"].notna().all()),
        "no record claims a guaranteed improvement": not h[rec_cols].apply(lambda c: c.astype(str).str.contains(r"\bwill (?:improve|increase|reduce)|guarantee", case=False)).any().any(),
        "no location perturbations": not h["perturbed_features"].str.contains("plate_").any(),
        "no model-only record ranks above tier C": bool((h.loc[h["obs_rv_per100"].isna(), "tier"] == "C").all()),
        "every sensitivity record is labelled as model sensitivity": bool(h.loc[h["kind"] == "sensitivity", "hypothesis_to_test"].str.startswith("MODEL SENSITIVITY").all()),
        "every record awaits coach selection": bool((h["coach_selected_experiment"] == "PENDING COACH SELECTION").all()),
    }
    S = ["# Constrained coaching hypotheses", "",
         "Generated by `src/hypotheses.py` with rules in `config/hypotheses.yaml`; outcome models in `reports/outcome_models.md`.", "",
         "## Counts", "", h.groupby(["kind", "tier"]).size().unstack(fill_value=0).to_markdown(), "",
         f"Scenarios rejected because the pitcher has not demonstrated them (outside his 5–95% range or too few of his own pitches "
         f"nearby): **{len(rej):,}**. Location perturbations: none, by rule.", "",
         "## How often swinging strikes and run value disagree", "",
         f"{int((h['trade_off'] != 'none detected in modelled outcomes').sum()):,} of {len(h):,} hypotheses "
         f"({(h['trade_off'] != 'none detected in modelled outcomes').mean():.0%}) have modelled swinging-strike and run-value changes in "
         "opposite directions. Ranking on swinging strikes alone would mis-rank these, which is why run value decides the tier.", "",
         "## Top tier-A hypotheses (all pitchers)", "",
         "`rv_estimate_per100` is the model estimate; `obs_rv_*` is his observed run-value difference net of the league's change "
         "(runs saved per 100 pitches, 95% CI). Order within a tier is by the model estimate relative to its sampling interval.", "",
         h[h["tier"] == "A"].head(10)[["hypothesis_id", "pitcher_name", "pitch_type", "kind", "rv_estimate_per100", "obs_rv_per100", "obs_rv_lo", "obs_rv_hi", "swstr_estimate_pp", "pitches_used"]]
           .round(3).to_markdown(index=False), "",
         "## Gate checks", "", pd.DataFrame({"check": list(gate), "pass": list(gate.values())}).to_markdown(index=False), "",
         f"## Example pitcher: `reports/players/{pitcher_id}_hypotheses.md`", "",
         "## Limits", "",
         "- Every estimate is associational. Own-history comparisons mix the shape change with anything else that changed between seasons "
         "(batters, health, role, catcher); netting out the league's change removes only the league-wide part.",
         "- Intervals cover game-to-game sampling, not model error; treat small estimates as noise.",
         "- No hypothesis becomes advice until a coach selects it, the experiment is run, and the pre-stated metric is reviewed.", ""]
    (ROOT / "reports" / "hypotheses_report.md").write_text("\n".join(S))
    print(h.groupby(["kind", "tier"]).size().unstack(fill_value=0).to_string())
    print("rejected:", len(rej), "| trade-offs:", int((h["trade_off"] != "none detected in modelled outcomes").sum()), "| gate:", gate)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pitcher-id", type=int)
    main(**vars(ap.parse_args()))
