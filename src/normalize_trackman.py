"""Apply the canonical contract to the college TrackMan file (analysed independently of Statcast).

Output columns match data/processed/pitches.parquet where the measurement exists, in the same
catcher's-view convention, so the descriptive code can run on either source. The two sources are
never concatenated into one training table.

Writes data/processed/trackman_pitches.parquet and data/processed/trackman_exclusion_log.csv.
"""
import json
from pathlib import Path

import pandas as pd
import yaml

from labels import outcome_category, whiff_label
from normalize import assign_status, review_flags

ROOT = Path(__file__).resolve().parents[1]
BAD_DATE = "1/4/1900"


def reconstruct_games(raw):
    """Game sequence from file order: new game when Date changes or PitchNo stops increasing."""
    date = raw["Date"].where(raw["Date"] != BAD_DATE).ffill()
    new_game = (date != date.shift()) | (raw["PitchNo"] <= raw["PitchNo"].shift())
    return new_game.cumsum(), date


def normalize(raw, contract, cfg):
    tm = contract["trackman"]
    df = pd.DataFrame(index=raw.index)
    df["source_id"] = cfg["source_id"]
    game_seq, date = reconstruct_games(raw)
    df["game_seq"] = game_seq
    df["game_pk"] = "tm:" + game_seq.astype(str)  # source-scoped surrogate, not an MLB game_pk
    df["game_date"] = pd.to_datetime(date, format="%m/%d/%Y").dt.date
    df["date_repaired"] = raw["Date"].eq(BAD_DATE)
    df["pitch_number_in_game"] = raw["PitchNo"]
    df["inning"], df["outs"] = raw["Inning"], raw["Outs"]
    df["balls"], df["strikes"] = raw["Balls"], raw["Strikes"]
    df["pitcher_id"], df["batter_id"] = raw["PitcherId"], raw["BatterId"]
    df["pitcher_name"] = "TM " + raw["PitcherId"].astype(str)  # anonymized; no names exist
    df["pitcher_team"], df["batter_team"] = raw["PitcherTeam"], raw["BatterTeam"]
    df["p_throws"] = raw["PitcherThrows"].map(tm["context"]["PitcherThrows"]["map"]).astype("string")
    df["stand"] = raw["BatterSide"].map(tm["context"]["BatterSide"]["map"]).astype("string")

    # Measurements, converted to the Statcast catcher's-view convention (see field_mapping.yaml).
    df["velocity_mph"] = raw["RelSpeed"]
    df["spin_rpm"] = raw["SpinRate"]
    df["spin_axis_deg"] = raw["SpinAxis"]
    df["extension_ft"] = raw["Extension"]
    df["release_z_ft"] = raw["RelHeight"]
    df["release_x_ft"] = -raw["RelSide"]
    df["hmov_ft"] = -raw["HorzBreak"] / 12
    df["ivb_ft"] = raw["InducedVertBreak"] / 12
    df["plate_x_ft"] = -raw["PlateLocSide"]
    df["plate_z_ft"] = raw["PlateLocHeight"]

    arm_sign = df["p_throws"].map({"R": -1.0, "L": 1.0}).astype(float)
    df["hmov_arm_in"] = df["hmov_ft"] * arm_sign * 12
    df["ivb_in"] = df["ivb_ft"] * 12
    df["release_x_arm_ft"] = df["release_x_ft"] * arm_sign

    df["pitch_type_raw"] = raw["TaggedPitchType"].astype("string")
    df["pitch_type_auto"] = raw["AutoPitchType"].astype("string")
    groups = {c: g for g, codes in contract["trackman_pitch_type_groups"].items() for c in codes}
    df["pitch_group"] = df["pitch_type_raw"].map(groups).fillna("unknown")
    df["pitch_group_auto"] = df["pitch_type_auto"].map(groups).fillna("unknown")

    df["description_raw"] = raw["PitchCall"]
    cat = outcome_category(raw["PitchCall"], mapping=load_trackman_mapping())
    df["outcome_category"] = cat.where(~(raw["PitchCall"].eq("InPlay") & raw["TaggedHitType"].eq("Bunt")), "bunt")
    df["is_miss"] = whiff_label(df["outcome_category"])
    df["is_swing"] = df["outcome_category"].isin(["miss", "contact", "bunt"])
    df["eligible_swing"] = df["is_miss"].notna()

    # No per-batter strike zone in this file: fixed 1.5-3.5 ft (documented difference from Statcast).
    half_width = (17 / 2 + 1.45) / 12
    has_loc = df[["plate_x_ft", "plate_z_ft"]].notna().all(axis=1)
    in_zone = (df["plate_x_ft"].abs() <= half_width) & df["plate_z_ft"].between(1.5 - 1.45 / 12, 3.5 + 1.45 / 12)
    df["in_zone"] = in_zone.astype("boolean").where(has_loc)

    df["pitch_key"] = df["game_pk"] + "-" + df["pitch_number_in_game"].astype(str)
    df["pitcher_key"] = "trackman_college:" + df["pitcher_id"].astype(str)

    flags = review_flags(df, contract["plausibility"])
    flags = flags.where(df["p_throws"].notna(), flags + "pitcher_hand_undefined;")
    flags = flags.where(~df["date_repaired"], flags + "date_repaired_from_block;")
    df["review_flags"] = flags
    df["status"], df["status_reason"] = assign_status([
        (df[["pitcher_id", "batter_id", "pitch_number_in_game"]].isna().any(axis=1), "quarantined", "missing_identifier"),
        (df.duplicated("pitch_key", keep=False), "quarantined", "duplicate_pitch_key"),
    ], df.index)
    return df


def load_trackman_mapping():
    from labels import load_mapping
    return load_mapping("trackman")


def main():
    cfg = yaml.safe_load((ROOT / "config" / "data_sources.yaml").read_text())["trackman_college"]
    contract = yaml.safe_load((ROOT / "config" / "field_mapping.yaml").read_text())
    manifest = json.loads((ROOT / "data" / "manifests" / f"{cfg['source_id']}.json").read_text())
    raw = pd.read_csv(ROOT / manifest["file"], low_memory=False)
    df = normalize(raw, contract, cfg)
    assert len(df) == manifest["rows"], "row count drifted from manifest"
    out = ROOT / "data" / "processed"
    df.to_parquet(out / "trackman_pitches.parquet", index=False)
    log = df[(df["status"] != "retained") | (df["review_flags"] != "")][
        ["pitch_key", "pitcher_id", "pitch_type_raw", "description_raw", "status", "status_reason", "review_flags"]]
    log.to_csv(out / "trackman_exclusion_log.csv", index=False)
    print(df["status"].value_counts().to_string())
    print(f"{df['game_pk'].nunique()} games; {len(df)} rows; flagged rows: {(df['review_flags'] != '').sum()}")


if __name__ == "__main__":
    main()
