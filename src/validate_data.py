"""Audit the normalized Statcast sample.

Writes reports/data_audit.md, reports/source_inventory.csv (every raw column and its role)
and reports/data_dictionary.csv (the canonical contract). Exits non-zero if a hard check fails.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
MEASURES = ["velocity_mph", "spin_rpm", "extension_ft", "release_x_ft", "release_z_ft",
            "hmov_ft", "ivb_ft", "plate_x_ft", "plate_z_ft", "arm_angle_deg", "spin_axis_deg"]


def md(df, **kw):
    return df.to_markdown(**kw)


def resumed_games(manifest):
    """Games the schedule lists on more than one date (suspended and resumed)."""
    path = ROOT / manifest.get("schedule_file", "")
    if not manifest.get("schedule_file") or not path.exists():
        return {}
    dates = {}
    for d in json.loads(path.read_text()).get("dates", []):
        for g in d["games"]:
            dates.setdefault(g["gamePk"], set()).add(d["date"])
    return {pk: sorted(ds) for pk, ds in dates.items() if len(ds) > 1}


def sequence_gaps(df):
    """Structural completeness: missing at-bat numbers within games, non-contiguous pitch numbers within PAs."""
    ab = df.groupby("game_pk")["at_bat_number"].agg(lambda s: len(set(range(1, int(s.max()) + 1)) - set(s)))
    pa = df.groupby(["game_pk", "at_bat_number"])["pitch_number"].agg(
        lambda s: s.min() == 1 and s.max() == len(s) and s.is_unique)
    return ab, pa


def explain_ab_gaps(ab_gaps, cache_dir):
    """Look up missing at-bat numbers in MLB play-by-play (cached). Returns rows: game, at-bat, event, pitches."""
    from ingest import get_with_retries
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pk in ab_gaps[ab_gaps > 0].index:
        path = cache_dir / f"playbyplay_{int(pk)}.json"
        if not path.exists():
            path.write_bytes(get_with_retries(f"https://statsapi.mlb.com/api/v1/game/{int(pk)}/playByPlay", {}).content)
        plays = json.loads(path.read_text())["allPlays"]
        seen = set(ab_gaps.attrs["present"][pk])
        for ab in sorted(set(range(1, max(seen) + 1)) - seen):
            play = plays[ab - 1]
            rows.append({"game_pk": int(pk), "at_bat_number": ab, "event": play["result"].get("event"),
                         "pitches_in_play_by_play": sum(1 for e in play["playEvents"] if e.get("isPitch"))})
    return pd.DataFrame(rows)


def main(source="statcast_sample"):
    ds = yaml.safe_load((ROOT / "config" / "data_sources.yaml").read_text())[source]
    if ds.get("sealed"):
        raise SystemExit(f"{source} is sealed")
    contract = yaml.safe_load((ROOT / "config" / "field_mapping.yaml").read_text())
    manifest = json.loads((ROOT / "data" / "manifests" / f"{ds['source_id']}.json").read_text())
    df = pd.read_parquet(ROOT / ds["processed_path"])
    report_path = ROOT / "reports" / ("data_audit.md" if source == "statcast_sample" else f"data_audit_{ds['source_id']}.md")
    ab_gaps, pa_ok = sequence_gaps(df)
    ab_gaps.attrs["present"] = df.groupby("game_pk")["at_bat_number"].agg(lambda s: sorted(set(s))).to_dict()
    gap_rows = explain_ab_gaps(ab_gaps, ROOT / "data" / "raw" / "statsapi" / "playbyplay")
    unexplained = gap_rows[gap_rows["pitches_in_play_by_play"] > 0] if len(gap_rows) else gap_rows

    resumed = resumed_games(manifest)
    batches = pd.DataFrame(manifest["batches"])
    if len(batches) > 14:  # season: summarize by month, list problem days
        batches["month"] = batches["date"].str[:7]
        coverage = batches.groupby("month").agg(days=("date", "size"), rows=("rows", "sum"), games_in_data=("games_in_data", "sum"),
                                                games_final_in_schedule=("games_final_in_schedule", "sum"), bytes=("bytes", "sum")).reset_index()
    else:
        coverage = batches[["date", "rows", "games_in_data", "games_final_in_schedule", "bytes", "sha256"]].assign(sha256=lambda x: x["sha256"].str[:12] + "…")
    problem_days = batches[batches["games_in_data"] != batches["games_final_in_schedule"]]["date"].tolist()
    sc = contract["statcast"]

    # Hard checks
    checks = {
        "row count equals manifest total": len(df) == manifest["total_rows"],
        "every row has a status": df["status"].isin(["retained", "excluded", "quarantined"]).all(),
        "pitch_key unique": not df["pitch_key"].duplicated().any(),
        "all scheduled final games present": not manifest["final_games_missing_from_window"],
        "schema identical across days": all(b["columns_match_first_day"] for b in manifest["batches"]),
        "balls in 0-3": df["balls"].between(0, 3).all(),
        "strikes in 0-2": df["strikes"].between(0, 2).all(),
        "p_throws/stand in {L,R}": df["p_throws"].isin(["L", "R"]).all() and df["stand"].isin(["L", "R"]).all(),
        "no unknown outcomes": not df["outcome_category"].eq("unknown").any(),
        "denylisted columns absent from processed table": not set(sc["leakage_denylist"]) & set(df.columns),
        "every missing at-bat has zero pitches in MLB play-by-play": len(unexplained) == 0,
    }

    # Column inventory: every raw column and its role
    used = {src: spec["internal"] for sec in ("identifiers", "measurements", "context", "label_only")
            for src, spec in sc[sec].items()}
    inv = pd.DataFrame({"raw_column": manifest["columns"]})
    inv["role"] = inv["raw_column"].map(
        lambda c: "label_only" if c in sc["label_only"] else
        "leakage_denylist" if c in sc["leakage_denylist"] else
        "deprecated" if c.endswith("_deprecated") or c.endswith("_legacy") else
        "contract" if c in used else "unused_available")
    inv["internal_name"] = inv["raw_column"].map(used)
    inv.to_csv(ROOT / "reports" / "source_inventory.csv", index=False)

    dd = [{"raw_column": src, "internal_name": spec["internal"], "section": sec,
           "unit": spec.get("unit", ""), "type": spec.get("type", ""), "convention": spec.get("convention", ""),
           "status": spec.get("status", ""), "note": spec.get("note", ""),
           "missing_value_behavior": "NaN kept; never zero-filled" if sec == "measurements" else "row quarantined" if sec == "identifiers" else ""}
          for sec in ("identifiers", "measurements", "context", "label_only") for src, spec in sc[sec].items()]
    derived = [
        ("hmov_arm_in", "pfx_x * (-1 RHP / +1 LHP) * 12", "in", "arm-side positive"),
        ("ivb_in", "pfx_z * 12", "in", "induced vertical break (gravity removed)"),
        ("release_x_arm_ft", "release_pos_x * (-1 RHP / +1 LHP)", "ft", "arm-side positive"),
        ("in_zone", "|plate_x| <= 0.83 ft and sz_bot - 0.12 <= plate_z <= sz_top + 0.12", "bool", "rulebook zone + ball radius; NA without location"),
        ("is_miss", "outcome_mapping.yaml", "0/1/NA", "1 miss, 0 contact, NA otherwise"),
    ]
    dd += [{"raw_column": "(derived)", "internal_name": n, "section": "derived", "unit": u, "convention": c,
            "status": "verified", "note": f, "missing_value_behavior": "NA if inputs missing"} for n, f, u, c in derived]
    pd.DataFrame(dd).to_csv(ROOT / "reports" / "data_dictionary.csv", index=False)

    ret = df[df["status"] == "retained"]
    L = [f"# Data audit — {ds['source_id']}", "",
         f"Source: {manifest['source']}. Requested dates {manifest['requested_dates'][0]} to {manifest['requested_dates'][1]} "
         f"(regular season). Generated by `src/validate_data.py` from `{ds['processed_path']}`.", "",
         ("This one-week sample tests the pipeline. It is **not** evidence of general model quality."
          if source == "statcast_sample" else f"Role: {ds.get('role', '')}."), "",
         "## Hard checks", "", md(pd.DataFrame({"check": list(checks), "pass": list(checks.values())}), index=False), "",
         "## Retrieval and coverage", "",
         md(coverage, index=False), "",
         f"Per-day sha256 checksums are in the manifest. Days where games in data ≠ final games on that official date: "
         f"{problem_days or 'none'}.", "",
         f"Games in data: **{manifest['games_in_data']}** of **{manifest['final_games_in_schedule']}** final games on the "
         f"MLB schedule for the window. Missing: {manifest['final_games_missing_from_window'] or 'none'}.", "",
         f"Suspended/resumed games (listed on more than one schedule date): {len(resumed)}"
         + (f" — {', '.join(f'{pk} ({chr(8594).join(d)})' for pk, d in list(resumed.items())[:10])}" if resumed else "") + ". "
         "Savant files every pitch of such a game under its official (start) date, so `game_date` is the official game "
         "date, not necessarily the day each pitch was thrown.", "",
         "### Structural completeness", "",
         f"- Games with missing at-bat numbers: {int((ab_gaps > 0).sum())} of {len(ab_gaps):,} "
         f"(missing at-bats in total: {int(ab_gaps.sum())}).",
         f"- Plate appearances whose pitch numbers are not 1..n: {int((~pa_ok).sum())} of {len(pa_ok):,}.",
         "- Each missing at-bat is looked up in MLB play-by-play (cached under `data/raw/statsapi/playbyplay/`). A plate "
         "appearance with no pitch (inning ends on a pickoff or caught stealing) legitimately has no Statcast row; "
         "anything else would mean the export dropped pitches.", "",
         (md(gap_rows.groupby(["event", "pitches_in_play_by_play"]).size().rename("at-bats").reset_index(), index=False)
          if len(gap_rows) else "No gaps."), "",
         "## Row accounting", "",
         md(df.groupby(["status", "status_reason"]).size().rename("rows").reset_index(), index=False), "",
         f"Total {len(df):,} = retained {len(ret):,} + excluded {(df.status == 'excluded').sum():,} + quarantined "
         f"{(df.status == 'quarantined').sum():,}. Row-level log: exclusion log beside `{ds['processed_path']}`.", "",
         "## Inventory (retained rows)", "",
         md(pd.DataFrame({"pitches": [len(ret)], "games": [ret.game_pk.nunique()], "pitchers": [ret.pitcher_id.nunique()],
                          "batters": [ret.batter_id.nunique()]}), index=False), "",
         "### Outcomes", "",
         md(df.groupby(["outcome_category", "description_raw"]).size().rename("pitches").reset_index(), index=False), "",
         f"Eligible ordinary swings: **{int(ret.eligible_swing.sum()):,}** "
         f"(misses {int((ret.is_miss == 1).sum()):,}, contact {int((ret.is_miss == 0).sum()):,}); "
         f"pooled whiff rate {ret.is_miss.mean():.3f}. Bunts ({int(ret.outcome_category.eq('bunt').sum())}) and takes are excluded "
         "from the swing task, not relabelled. Checked swings ruled as swings cannot be distinguished from full swings.", "",
         "### Pitch types", "",
         md(ret.groupby(["pitch_group", "pitch_type_raw"], dropna=False)
            .agg(pitches=("pitch_key", "size"), pitchers=("pitcher_id", "nunique"), swings=("eligible_swing", "sum"))
            .reset_index(), index=False), "",
         "## Missing measurements (retained rows)", "",
         "Share of retained rows with each measurement missing, overall and by pitch type.", "",
         md(ret.groupby(ret["pitch_type_raw"].fillna("(none)"))[MEASURES].agg(lambda s: s.isna().mean())
            .pipe(lambda t: pd.concat([ret[MEASURES].isna().mean().rename("ALL").to_frame().T, t]))
            .mul(100).round(2)), "",
         f"Rows with no tracking at all: {int(ret.review_flags.str.contains('no_tracking').sum())}; "
         f"by outcome: {ret[ret.review_flags.str.contains('no_tracking')].outcome_category.value_counts().to_dict()}. "
         "Absent tracking is distinct from a missing outcome: no retained row lacks an outcome.", "",
         "Games with the highest share of rows missing velocity:", "",
         md(ret.groupby("game_pk").agg(pitches=("pitch_key", "size"), missing_velocity=("velocity_mph", lambda s: s.isna().sum()))
            .assign(share=lambda t: (t.missing_velocity / t.pitches).round(3)).sort_values("share", ascending=False).head(5)), "",
         "Pitchers with the highest share of rows missing velocity (≥ 20 pitches):", "",
         md(ret.groupby(["pitcher_id", "pitcher_name"]).agg(pitches=("pitch_key", "size"), missing_velocity=("velocity_mph", lambda s: s.isna().sum()))
            .query("pitches >= 20").assign(share=lambda t: (t.missing_velocity / t.pitches).round(3))
            .sort_values("share", ascending=False).head(5).reset_index(), index=False), "",
         "## Review flags (kept, not removed)", "",
         "Plausibility ranges are in `config/field_mapping.yaml`. Flags are for human review.", "",
         md(ret.review_flags.str.split(";").explode().loc[lambda s: s != ""].value_counts().rename("rows").to_frame()), "",
         "Flagged rows by pitcher (top 10):", "",
         md(ret[ret.review_flags.str.contains("out_of_range")]
            .groupby(["pitcher_name", "pitch_type_raw", "review_flags"]).agg(rows=("pitch_key", "size"),
                    velo=("velocity_mph", "median"), rel_z=("release_z_ft", "median"), ext=("extension_ft", "median"))
            .sort_values("rows", ascending=False).head(10).reset_index(), index=False), "",
         "Interpretation for review: concentrated flags from a single pitcher usually indicate an unusual but genuine "
         "delivery (e.g. a submarine release near 1.3 ft) or a position player pitching (≈50 mph, high release), not "
         "a tracking error. In this sample the concentrated flags are Tyler Rogers (submariner) and position players "
         "pitching (Capra, Herrera, Wilson, Bride); the remaining single-pitch location flags are extreme non-competitive misses. "
         "**Modeling note:** position-player pitches (EP/KN/FA and sub-60 mph) should be excluded from the whiff model's "
         "training population by rule when the model dataset is built; they stay in the descriptive tables.", "",
         "## Columns", "",
         md(inv.groupby("role").size().rename("columns").to_frame()), "",
         "Full list with roles: `reports/source_inventory.csv`. Canonical fields, units and conventions: "
         "`reports/data_dictionary.csv`.", "",
         "**Leakage denylist** (never predictors): " + ", ".join(f"`{c}`" for c in sc["leakage_denylist"]) + ". "
         "`miss_distance` is populated on ~99% of misses and 0% of contacts; the bat-tracking fields exist only "
         "when a swing occurred and describe the batter, not the pitch.", "",
         "## Fields required by each analysis", "",
         "| Analysis | Required fields |", "|---|---|",
         "| Repertoire / usage | pitch_type_raw, pitcher_id |",
         "| Velocity, spin, extension | velocity_mph, spin_rpm, extension_ft |",
         "| Movement | hmov_ft, ivb_ft, p_throws (arm-side transform) |",
         "| Release consistency | release_x_ft, release_z_ft, game_pk |",
         "| Location / zone rate | plate_x_ft, plate_z_ft, sz_top_ft, sz_bot_ft |",
         "| Whiff rate | description_raw (via outcome_mapping.yaml) |",
         ]
    report_path.write_text("\n".join(L) + "\n")
    failed = [k for k, v in checks.items() if not v]
    print(f"wrote {report_path.relative_to(ROOT)};", "FAILED: " + ", ".join(failed) if failed else "all hard checks pass")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="statcast_sample")
    main(**vars(ap.parse_args()))
