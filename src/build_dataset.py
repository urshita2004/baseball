"""Dataset build: model population, swing table, chronological splits and label checks.

Keeps every pitch (takes included) in model_pitches.parquet with its partition and model-population
status; derives swings.parquet (the whiff-model rows). Writes the split manifest to
data/splits/<source_id>.json and reports/dataset_<source_id>.md.

Usage: .venv/bin/python src/build_dataset.py [--source statcast_2025|statcast_sample]
"""
import argparse
import datetime as dt
import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import yaml

from ingest import read_raw_bytes
from pitcher_analysis import wilson
from players import PITCHER_POSITIONS, primary_positions

ROOT = Path(__file__).resolve().parents[1]
PARTS = ["train", "validation", "calibration"]
STRIKEOUT_PHRASES = r"(strikes out (?:swinging|on a foul tip|on a missed bunt|on a foul bunt)|called out on strikes)"
EXPECTED_PHRASE = {"swinging_strike": "strikes out swinging", "swinging_strike_blocked": "strikes out swinging",
                   "foul_tip": "strikes out on a foul tip", "called_strike": "called out on strikes",
                   "missed_bunt": "strikes out on a missed bunt", "foul_bunt": "strikes out on a foul bunt",
                   "bunt_foul_tip": "strikes out on a foul bunt"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assign_partitions(df, cuts):
    """Partition whole dates so cumulative game share crosses each cut at a date boundary."""
    games = df.groupby("game_pk")["game_date"].first().sort_values()
    per_date = games.groupby(games).size()
    share = per_date.cumsum() / per_date.sum()
    bounds = [cuts["train"], cuts["validation"]]
    part_of_date = pd.Series(pd.cut(share, [-1, *bounds, 2], labels=PARTS), index=share.index).astype(str)
    return df["game_date"].map(part_of_date), part_of_date


def strikeout_label_check(manifest):
    """Independent check: Statcast label of each strikeout pitch vs the play-by-play text."""
    frames = []
    for b in manifest["batches"]:
        if b["rows"]:
            raw = read_raw_bytes(ROOT / b["file"])
            frames.append(pd.read_csv(io.BytesIO(raw), usecols=["description", "events", "des"], low_memory=False))
    d = pd.concat(frames, ignore_index=True)
    k = d[d["events"].isin(["strikeout", "strikeout_double_play"])].copy()
    k["phrase"] = k["des"].str.extract(STRIKEOUT_PHRASES)[0]
    k["expected"] = k["description"].map(EXPECTED_PHRASE)
    table = pd.crosstab(k["description"], k["phrase"].fillna("(no phrase)"))
    mismatches = int((k["phrase"].notna() & (k["phrase"] != k["expected"])).sum())
    return table, mismatches, int(k["phrase"].isna().sum()), len(k)


def apply_model_population(df, split_cfg, cache_name):
    """Model-population flags; the same function is applied to the sealed season by final_test.py."""
    df = df.copy()
    positions, people_meta = primary_positions(df["pitcher_id"].dropna().unique(), cache_name)
    df["pitcher_position"] = df["pitcher_id"].map(positions)
    reason = pd.Series("", index=df.index)
    reason = reason.where(df["status"] == "retained", "status:" + df["status_reason"])
    if split_cfg["model_population"]["exclude_no_tracking"]:
        reason = reason.where((reason != "") | ~df["review_flags"].str.contains("no_tracking"), "no_tracking")
    if split_cfg["model_population"]["exclude_position_players"]:
        pp = ~df["pitcher_position"].isin(PITCHER_POSITIONS)
        reason = reason.where((reason != "") | ~pp, "position_player_pitching")
    reason = reason.where((reason != "") | (df["p_throws"].isin(["L", "R"]) & df["stand"].isin(["L", "R"])), "handedness_unknown")
    df["model_exclusion"] = reason
    df["model_population"] = reason.eq("")
    return df, people_meta


def main(source="statcast_2025"):
    cfg_all = yaml.safe_load((ROOT / "config" / "data_sources.yaml").read_text())
    ds = cfg_all[source]
    if ds.get("sealed"):
        raise SystemExit(f"{source} is sealed")
    split_cfg = yaml.safe_load((ROOT / "config" / "splits.yaml").read_text())
    manifest = json.loads((ROOT / "data" / "manifests" / f"{ds['source_id']}.json").read_text())
    in_path = ROOT / ds["processed_path"]
    df = pd.read_parquet(in_path)
    out_dir = ROOT / "data" / "model" / ds["source_id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Model population
    df, people_meta = apply_model_population(df, split_cfg, ds["source_id"])

    # Partitions over the whole regular season (all games, so boundaries do not depend on the filters above)
    df["partition"], part_of_date = assign_partitions(df, split_cfg["partitions"])
    swings = df[df["model_population"] & df["eligible_swing"]].copy()

    # Checks
    game_parts = df.groupby("game_pk")["partition"].nunique()
    per_part = swings.groupby("partition").agg(games=("game_pk", "nunique"), swings=("pitch_key", "size"),
                                               misses=("is_miss", lambda s: int((s == 1).sum())),
                                               pitchers=("pitcher_id", "nunique")).reindex(PARTS)
    per_part["whiff_rate"] = (per_part["misses"] / per_part["swings"]).round(4)
    per_part["lo"], per_part["hi"] = wilson(per_part["misses"], per_part["swings"])
    ko_table, ko_mismatch, ko_unparsed, ko_n = strikeout_label_check(manifest)
    train_pitchers = set(swings.loc[swings["partition"] == "train", "pitcher_id"])
    unseen = {p: swings[(swings["partition"] == p) & ~swings["pitcher_id"].isin(train_pitchers)] for p in PARTS[1:]}
    train_types = swings[swings["partition"] == "train"].groupby("pitch_type_raw").size()
    rare = sorted(train_types[train_types < split_cfg["rare_pitch_type_min_train_swings"]].index)
    checks = {
        "every game in exactly one partition": bool((game_parts == 1).all()),
        "partitions ordered in time (train < validation < calibration)": bool(
            df.groupby("partition")["game_date"].max()["train"] < df.groupby("partition")["game_date"].min()["validation"]
            and df.groupby("partition")["game_date"].max()["validation"] < df.groupby("partition")["game_date"].min()["calibration"]),
        "each partition has both classes": bool(((per_part["misses"] > 0) & (per_part["misses"] < per_part["swings"])).all()),
        "strikeout labels agree with play-by-play text": ko_mismatch == 0,
        "swing table has no bunts, takes or unknowns": bool(swings["outcome_category"].isin(["miss", "contact"]).all()),
        "every pitch kept (takes included)": len(df) == manifest["total_rows"],
    }

    # Outputs
    keep = [c for c in df.columns if c != "source_file"]
    df[keep].to_parquet(out_dir / "model_pitches.parquet", index=False)
    swings[keep].to_parquet(out_dir / "swings.parquet", index=False)
    date_ranges = {p: [str(df.loc[df.partition == p, "game_date"].min()), str(df.loc[df.partition == p, "game_date"].max())] for p in PARTS}
    split_manifest = {
        "source_id": ds["source_id"], "version": split_cfg["version"],
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "rule": split_cfg["unit"], "cuts": split_cfg["partitions"], "date_ranges": date_ranges,
        "game_pks": {p: sorted(int(g) for g in df.loc[df.partition == p, "game_pk"].unique()) for p in PARTS},
        "swing_counts": per_part[["games", "swings", "misses", "pitchers"]].astype(int).to_dict(orient="index"),
        "rare_pitch_types_pooled_to_group": rare,
        "people_snapshot": people_meta,
        "inputs_sha256": {str(in_path.relative_to(ROOT)): sha(in_path),
                          **{f"config/{n}": sha(ROOT / "config" / n) for n in ("outcome_mapping.yaml", "field_mapping.yaml", "features.yaml", "splits.yaml")}},
        "outputs_sha256": {f"data/model/{ds['source_id']}/{n}": sha(out_dir / n) for n in ("model_pitches.parquet", "swings.parquet")},
        "checks": checks,
    }
    (ROOT / "data" / "splits").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "splits" / f"{ds['source_id']}.json").write_text(json.dumps(split_manifest, indent=2))

    pop = df.groupby("model_exclusion").size().rename("pitches").reset_index()
    pop["model_exclusion"] = pop["model_exclusion"].replace("", "(in model population)")
    pp_rows = df[df["model_exclusion"] == "position_player_pitching"]
    L = [f"# Model dataset — {ds['source_id']}", "",
         f"Generated by `src/build_dataset.py`. Input `{in_path.relative_to(ROOT)}` ({len(df):,} pitches, "
         f"{df.game_pk.nunique():,} games, {manifest['population']['start_date']} to {manifest['population']['end_date']}, "
         f"game_type {manifest['population']['game_type']}).", "",
         "## Checks", "", pd.DataFrame({"check": list(checks), "pass": list(checks.values())}).to_markdown(index=False), "",
         "## Model population", "",
         "Every pitch stays in `model_pitches.parquet` (takes included, for later swing-probability work). "
         "The whiff model uses `swings.parquet`: model population AND ordinary (non-bunt) swing.", "",
         pop.to_markdown(index=False), "",
         f"Position players pitching: {pp_rows.pitcher_id.nunique()} players, {len(pp_rows):,} pitches "
         f"({people_meta['rule']}; snapshot {people_meta['retrieved_at_utc']}).", "",
         "## Partitions", "",
         f"Rule: {split_cfg['unit']}. Cuts at {split_cfg['partitions']['train']:.0%} / {split_cfg['partitions']['validation']:.0%} of games.", "",
         per_part.assign(date_range=[" to ".join(date_ranges[p]) for p in PARTS], lo=per_part["lo"].round(3), hi=per_part["hi"].round(3))
                 .reset_index().to_markdown(index=False), "",
         "Seasonal drift is expected (weather, call-ups, fatigue), which is one reason calibration has its own late-season period "
         "and the final test is a different season.", "",
         "### Pitchers unseen in training", "",
         "\n".join(f"- {p}: {u.pitcher_id.nunique()} pitchers with no training swings, {len(u):,} swings "
                   f"({len(u) / max(1, (swings.partition == p).sum()):.1%} of the partition). Reported as a separate subgroup "
                   "for new-pitcher claims." for p, u in unseen.items()), "",
         "## Label check against play-by-play text", "",
         f"For all {ko_n:,} strikeout pitches, the outcome label is compared with the phrase in MLB's play description "
         f"(an independent field). Mismatches: **{ko_mismatch}**; descriptions without a recognised phrase: {ko_unparsed}.", "",
         ko_table.to_markdown(), "",
         "## Pitch types in training swings", "",
         f"Codes with fewer than {split_cfg['rare_pitch_type_min_train_swings']} training swings are pooled to their "
         f"pitch group when baselines are fitted: {rare or 'none'}.", "",
         train_types.sort_values(ascending=False).rename("train swings").to_frame().to_markdown(), ""]
    report = ROOT / "reports" / f"dataset_{ds['source_id']}.md"
    report.write_text("\n".join(L))
    failed = [k for k, v in checks.items() if not v]
    print(per_part.to_string())
    print(f"wrote {report.relative_to(ROOT)};", "FAILED: " + ", ".join(failed) if failed else "all checks pass")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="statcast_2025")
    main(**vars(ap.parse_args()))
