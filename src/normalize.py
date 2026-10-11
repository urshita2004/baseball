"""Apply the canonical data contract to the raw Statcast sample.

Every raw row is kept and given a status:
  retained    - usable for analysis (may still carry review flags)
  excluded    - outside the task by rule (pitchouts, automatic balls/strikes, excluded pitch types)
  quarantined - cannot be trusted as a distinct pitch (duplicate key, missing identifiers)
Implausible measurements are flagged in `review_flags`, never removed.

Writes the source's `processed_path` (config/data_sources.yaml) and an exclusion log beside it.
Usage: .venv/bin/python src/normalize.py [--source statcast_sample|statcast_2025]
"""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ingest import read_raw_bytes
from labels import outcome_category, whiff_label

ROOT = Path(__file__).resolve().parents[1]
KEY = ["game_pk", "at_bat_number", "pitch_number"]


def load_raw(manifest_path):
    """Load raw day files, verifying each against the manifest checksum."""
    manifest = json.loads(Path(manifest_path).read_text())
    frames = []
    for b in manifest["batches"]:
        raw = read_raw_bytes(ROOT / b["file"])
        if hashlib.sha256(raw).hexdigest() != b["sha256"]:
            raise SystemExit(f"checksum mismatch for {b['file']}")
        frames.append(pd.read_csv(io.BytesIO(raw), low_memory=False) if b["rows"] else pd.DataFrame())
    files = [b["file"] for b in manifest["batches"]]
    raw = pd.concat(frames, ignore_index=True).copy()
    raw["source_file"] = np.repeat(files, [len(f) for f in frames])
    return raw, manifest


def review_flags(df, plausibility):
    """Semicolon-joined review flags per row. Flags are for human review, never removals."""
    flags = pd.Series("", index=df.index)
    for col, (lo, hi) in plausibility.items():
        bad = df[col].notna() & ~df[col].between(lo, hi)
        flags = flags.where(~bad, flags + f"{col}_out_of_range;")
    no_tracking = df[["velocity_mph", "ivb_in", "plate_x_ft"]].isna().all(axis=1)
    flags = flags.where(~no_tracking, flags + "no_tracking;")
    return flags.where(df["spin_rpm"].notna() | no_tracking, flags + "no_spin;")


def assign_status(rules, index):
    """First matching (mask, status, reason) rule wins; unmatched rows are retained."""
    status = pd.Series("retained", index=index)
    reason = pd.Series("", index=index)
    for mask, s, r in rules:
        hit = mask & (status == "retained")
        status, reason = status.where(~hit, s), reason.where(~hit, r)
    return status, reason


def normalize(raw: pd.DataFrame, contract: dict, source_id: str) -> pd.DataFrame:
    sc = contract["statcast"]
    rename = {src: spec["internal"]
              for section in ("identifiers", "measurements", "context", "label_only")
              for src, spec in sc[section].items()}
    missing = set(rename) - set(raw.columns)
    if missing:
        raise SystemExit(f"contract columns absent from export: {sorted(missing)}")
    df = raw[list(rename) + ["source_file"]].rename(columns=rename).copy()
    df.insert(0, "source_id", source_id)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    for col in df.columns:
        if col.endswith(("_ft", "_mph", "_rpm", "_deg")):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["pitch_key"] = (df["game_pk"].astype("Int64").astype(str) + "-" + df["at_bat_number"].astype("Int64").astype(str)
                       + "-" + df["pitch_number"].astype("Int64").astype(str))
    # Source-scoped player IDs: identical numbers across sources do not establish identity.
    df["pitcher_key"] = "statcast:" + df["pitcher_id"].astype("Int64").astype(str)

    # Arm-side horizontal convention (verified: arm-side = -pfx_x for RHP, +pfx_x for LHP).
    arm_sign = df["p_throws"].map({"R": -1.0, "L": 1.0})
    df["hmov_arm_in"] = df["hmov_ft"] * arm_sign * 12
    df["ivb_in"] = df["ivb_ft"] * 12
    df["release_x_arm_ft"] = df["release_x_ft"] * arm_sign
    # Location relative to the batter: + = away (RHB stand on the 3B side, i.e. negative catcher's-view x).
    df["plate_x_away_ft"] = df["plate_x_ft"] * df["stand"].map({"R": 1.0, "L": -1.0})
    zone_height = (df["sz_top_ft"] - df["sz_bot_ft"]).where(lambda h: h > 0)
    df["plate_z_zone"] = (df["plate_z_ft"] - df["sz_bot_ft"]) / zone_height

    groups = {code: g for g, codes in contract["pitch_type_groups"].items() for code in codes}
    df["pitch_type_raw"] = df["pitch_type_raw"].astype("string")
    df["pitch_group"] = df["pitch_type_raw"].map(groups).fillna("unknown").where(df["pitch_type_raw"].notna(), "unknown")

    df["outcome_category"] = outcome_category(df["description_raw"])
    df["is_miss"] = whiff_label(df["outcome_category"])
    df["is_swing"] = df["outcome_category"].isin(["miss", "contact", "bunt"])
    df["eligible_swing"] = df["is_miss"].notna()

    # Rulebook zone: plate is 17 in wide; add one ball radius (~1.45 in) to each edge.
    half_width = (17 / 2 + 1.45) / 12
    has_loc = df[["plate_x_ft", "plate_z_ft", "sz_top_ft", "sz_bot_ft"]].notna().all(axis=1)
    in_zone = ((df["plate_x_ft"].abs() <= half_width)
               & (df["plate_z_ft"] <= df["sz_top_ft"] + 1.45 / 12)
               & (df["plate_z_ft"] >= df["sz_bot_ft"] - 1.45 / 12))
    df["in_zone"] = in_zone.astype("boolean").where(has_loc)

    df["review_flags"] = review_flags(df, contract["plausibility"])
    df["status"], df["status_reason"] = assign_status([
        (df[["game_pk", "at_bat_number", "pitch_number", "pitcher_id", "batter_id"]].isna().any(axis=1),
         "quarantined", "missing_identifier"),
        (df.duplicated("pitch_key", keep=False), "quarantined", "duplicate_pitch_key"),
        (df["outcome_category"].eq("non_competitive"), "excluded", "non_competitive_pitch"),
        (df["pitch_type_raw"].isin(contract["pitch_type_groups"]["excluded"]), "excluded", "excluded_pitch_type"),
    ], df.index)
    return df


def main(source="statcast_sample"):
    ds = yaml.safe_load((ROOT / "config" / "data_sources.yaml").read_text())[source]
    if ds.get("sealed"):
        raise SystemExit(f"{source} is sealed; it is processed only by the frozen final-evaluation procedure")
    contract = yaml.safe_load((ROOT / "config" / "field_mapping.yaml").read_text())
    raw, manifest = load_raw(ROOT / "data" / "manifests" / f"{ds['source_id']}.json")
    df = normalize(raw, contract, ds["source_id"])
    assert len(df) == manifest["total_rows"], "row count drifted from manifest"
    out = ROOT / ds["processed_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log = df[df["status"] != "retained"][["pitch_key", "source_file", "pitcher_id", "pitch_type_raw",
                                          "description_raw", "status", "status_reason"]]
    log.to_csv(out.with_name(out.stem.replace("pitches", "exclusion_log") + ".csv"), index=False)
    print(df["status"].value_counts().to_string())
    print(f"wrote {len(df)} rows; {len(log)} in exclusion log; flagged rows: {(df['review_flags'] != '').sum()}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="statcast_sample")
    main(**vars(ap.parse_args()))
