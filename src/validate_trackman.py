"""Audit the college TrackMan file on its own terms. Writes reports/trackman_audit.md."""
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def label_sensitivity(tm):
    """Whiff rate by pitch group under operator tags vs TrackMan auto labels (same swings)."""
    def tab(col):
        g = tm[tm["eligible_swing"]].groupby(col).agg(swings=("pitch_key", "size"), misses=("is_miss", lambda s: int((s == 1).sum())))
        g["whiff"] = (g["misses"] / g["swings"]).round(3)
        return g
    out = pd.concat({"operator tag (primary)": tab("pitch_group"), "TrackMan auto": tab("pitch_group_auto")}, axis=1)
    out.columns = [f"{src.split(' (')[0]}: {stat}" for src, stat in out.columns]
    out.index.name = "pitch group"
    return out


def main():
    cfg = yaml.safe_load((ROOT / "config" / "data_sources.yaml").read_text())["trackman_college"]
    manifest = json.loads((ROOT / "data" / "manifests" / f"{cfg['source_id']}.json").read_text())
    tm = pd.read_parquet(ROOT / "data" / "processed" / "trackman_pitches.parquet")
    sc = pd.read_parquet(ROOT / "data" / "processed" / "pitches.parquet").query("status == 'retained'")

    games = tm.groupby("game_seq").agg(date=("game_date", "first"), pitches=("pitch_key", "size"),
                                       first_pitch=("pitch_number_in_game", "min"), last_pitch=("pitch_number_in_game", "max"),
                                       innings=("inning", "max"), batting_teams=("batter_team", lambda s: "/".join(sorted(s.unique()))),
                                       pitchers=("pitcher_id", "nunique"))
    gaps = tm.groupby("game_seq")["pitch_number_in_game"].apply(lambda s: int((s.diff().dropna() != 1).sum())).sum()
    checks = {
        "sha256 matches Mendeley-published value": manifest["sha256_matches_published"],
        "row count = 3,344 (publication)": len(tm) == 3344,
        "reconstructed games = 21 (publication)": tm["game_pk"].nunique() == 21,
        "PitchNo contiguous within every game": gaps == 0,
        "pitch_key unique": not tm["pitch_key"].duplicated().any(),
        "no unknown outcomes": not tm["outcome_category"].eq("unknown").any(),
    }
    sides = tm.groupby("pitcher_id")["pitcher_team"].nunique()
    fb_tm = tm[tm["pitch_type_auto"] == "Four-Seam"]
    fb_sc = sc[sc["pitch_type_raw"] == "FF"]
    conv = pd.DataFrame({
        "check": ["RHP four-seam release side (catcher's view, ft)", "RHP four-seam horizontal movement (catcher's view, in)",
                  "RHB hit-by-pitch plate side (catcher's view, ft)", "four-seam arm-side movement, RHP / LHP (in)"],
        "Statcast": [fb_sc.query("p_throws == 'R'")["release_x_ft"].median(), (fb_sc.query("p_throws == 'R'")["hmov_ft"] * 12).median(),
                     sc.query("description_raw == 'hit_by_pitch' and stand == 'R'")["plate_x_ft"].median(),
                     " / ".join(f"{fb_sc.query('p_throws == @h')['hmov_arm_in'].median():.1f}" for h in "RL")],
        "TrackMan (converted)": [fb_tm.query("p_throws == 'R'")["release_x_ft"].median(), (fb_tm.query("p_throws == 'R'")["hmov_ft"] * 12).median(),
                                 tm.query("description_raw == 'HitByPitch' and stand == 'R'")["plate_x_ft"].median(),
                                 " / ".join(f"{fb_tm.query('p_throws == @h')['hmov_arm_in'].median():.1f}" for h in "RL")],
    })
    for col in ("Statcast", "TrackMan (converted)"):
        conv[col] = conv[col].map(lambda v: v if isinstance(v, str) else f"{v:.2f}")
    dist = pd.concat({
        "MLB (Statcast wk)": sc.groupby("pitch_group")[["velocity_mph", "spin_rpm", "ivb_in", "hmov_arm_in", "extension_ft"]].median(),
        "College (TrackMan, auto type)": tm.groupby("pitch_group_auto")[["velocity_mph", "spin_rpm", "ivb_in", "hmov_arm_in", "extension_ft"]].median(),
    }).round(1).loc[lambda t: t.index.get_level_values(1).isin(["fastball", "cutter", "breaking", "offspeed"])]
    swing = pd.DataFrame({
        "source": ["MLB (Statcast wk)", "College (TrackMan)"],
        "eligible swings": [int(sc["eligible_swing"].sum()), int(tm["eligible_swing"].sum())],
        "misses": [int((sc["is_miss"] == 1).sum()), int((tm["is_miss"] == 1).sum())],
    }).assign(whiff_rate=lambda t: (t["misses"] / t["eligible swings"]).round(3))

    L = [f"# College TrackMan audit — {cfg['source_id']}", "",
         f"**Source.** {manifest['citation']}", "",
         f"**License.** {manifest['license']}. Attribution above must accompany any reuse; this project transforms "
         "the data (sign conventions, unit conversions, reconstructed game IDs) as documented in `config/field_mapping.yaml`.", "",
         f"**File.** `{manifest['file']}`, {manifest['bytes']:,} bytes, sha256 `{manifest['sha256'][:16]}…` (matches the value Mendeley publishes), "
         f"{manifest['rows']:,} rows × {len(manifest['columns'])} columns. Tracking system: TrackMan V3 (all rows).", "",
         "## Hard checks", "", pd.DataFrame({"check": list(checks), "pass": list(checks.values())}).to_markdown(index=False), "",
         "## Game reconstruction", "",
         "The file has no game identifier. Games are reconstructed from file order (new game when the date changes or "
         "`PitchNo` stops increasing). Two dates hold two separate games each, so date + teams is not a key. "
         f"{int(tm['date_repaired'].sum())} rows carried the placeholder date 1/4/1900; each sits at the correct `PitchNo` inside "
         "a game block and takes that game's date (flagged `date_repaired_from_block`).", "",
         games.reset_index().assign(date=lambda t: t["date"].astype(str)).to_markdown(index=False), "",
         "Dates run Oct 2022 – Feb 2023 (fall scrimmages and preseason). The publication says dates are anonymized; whether "
         "their order is real is **unverified**, so this report makes no time-trend claims.", "",
         "## Population structure — the main limitation", "",
         f"- Pitchers: {tm['pitcher_id'].nunique()}; batters: {tm['batter_id'].nunique()}; pitching teams: "
         f"{sorted(tm['pitcher_team'].unique())}; batting teams: {sorted(tm['batter_team'].unique())}.",
         f"- **{int((sides > 1).sum())} of {len(sides)} pitchers appear on both T_A and T_B**: these are intrasquad scrimmages "
         "of one program, plus two games against outside opponents (T_C, T_D) who only bat.",
         "- Pitchers mostly face their own teammates, repeatedly. Whiff rates here describe one program's pitchers "
         "against one program's hitters in exhibition settings, not college baseball in general.",
         "- Consequence for the transfer evaluation: any transfer evaluation must be grouped by game **and** reported as single-program "
         "evidence. Pitcher-disjoint evaluation is barely possible with 16 pitchers ≥ 100 pitches.", "",
         "## Outcomes", "",
         tm.groupby(["outcome_category", "description_raw"]).size().rename("pitches").reset_index().to_markdown(index=False), "",
         "Bunts are identifiable only when put in play (`TaggedHitType == Bunt`, 25). Foul and missed bunts, and foul "
         "tips, cannot be separated, so the swing task is slightly noisier than Statcast's.", "",
         swing.to_markdown(index=False), "",
         "## Pitch types: operator tag vs TrackMan auto", "",
         pd.crosstab(tm["pitch_type_raw"], tm["pitch_type_auto"].fillna("(none)")).to_markdown(), "",
         "The two classifications disagree substantially (e.g. operator \"Fastball\" includes TrackMan four-seams, "
         "sinkers and ~300 changeups; operator \"Slider\" includes ~260 TrackMan curveballs). Neither is ground truth. "
         "Group-level comparisons with MLB below use the auto type, which is closer to Statcast's algorithmic labels.", "",
         "### Label sensitivity: do conclusions change with the label source?", "",
         "Operator tags are the primary reporting labels (documented trained operator); TrackMan auto labels are kept. "
         "Neither is ground truth. The same swings grouped both ways:", "",
         label_sensitivity(tm).to_markdown(), "",
         f"Pitches whose group differs between the two sources: {int((tm['pitch_group'] != tm['pitch_group_auto']).sum()):,} "
         f"of {len(tm):,} ({(tm['pitch_group'] != tm['pitch_group_auto']).mean():.0%}). Any pitch-type-level college finding "
         "should be shown under both labels; a finding that holds under only one is reported as label-dependent. Example: "
         "offspeed whiff rate is markedly higher under operator tags than auto labels, because the auto classifier calls "
         "~300 operator-tagged fastballs changeups; fastball and breaking-ball rates barely move.", "",
         "## Coordinate conventions (verified against Statcast)", "",
         "TrackMan reports horizontal values from the pitcher's view; they are negated on import.", "",
         conv.to_markdown(index=False), "",
         "TrackMan's own `pfxx`/`pfxz` are **not** Statcast's `pfx_x`/`pfx_z` (ratios ≈ −1.66 and 1.78 to HorzBreak/IVB) "
         "and are not used.", "",
         "## Distribution comparison (medians; descriptive only)", "", dist.to_markdown(), "",
         "Differences combine level of play (college vs MLB), measurement system (TrackMan V3, described by the publication "
         "as optical tracking, vs MLB Hawk-Eye) and pitch-type labelling. A unit conversion alone does not make the "
         "measurements equivalent. Two gaps are too large to attribute to level of play without checking: extension "
         "(~5.6 vs 6.5 ft, which depends on how each system locates the rubber) and offspeed IVB (~10 vs ~4 in, partly "
         "the auto-classifier labelling firm fastball-like pitches as changeups). Treat both as measurement questions for the transfer evaluation.", "",
         "## Missing measurements and review flags", "",
         (tm[["velocity_mph", "spin_rpm", "extension_ft", "release_x_ft", "release_z_ft", "hmov_ft", "ivb_ft", "plate_x_ft", "plate_z_ft"]]
          .isna().sum().rename("missing rows").to_frame().T.to_markdown(index=False)), "",
         tm["review_flags"].str.split(";").explode().loc[lambda s: s != ""].value_counts().rename("rows").to_frame().to_markdown(), "",
         "Pitcher 205 has `PitcherThrows = Undefined` on all 95 pitches; arm-side features are NA for those rows rather than guessed. "
         "Extension flags are concentrated in one pitcher (691205, ≈3.95 ft), consistent with a short-stride delivery.", "",
         "Absent for this source: per-batter strike zone (`sz_top`/`sz_bot`; a fixed 1.5–3.5 ft zone is used), arm angle, real game IDs, player names.", "",
         "## Supported use", "",
         "- **Supported now:** descriptive pitcher analysis (repertoire, velocity, spin, movement, release consistency, location) for these 32 pitchers.",
         "- **Experimental only:** applying an MLB-trained whiff model here. With 1,407 swings from one program, any AUC will have wide, game-clustered uncertainty and describes this program only.",
         "- **Not supported:** college norms, MLB percentiles for college players, time trends, or pooling with Statcast in one training table.", ""]
    (ROOT / "reports" / "trackman_audit.md").write_text("\n".join(L))
    failed = [k for k, v in checks.items() if not v]
    print("wrote reports/trackman_audit.md;", "FAILED: " + ", ".join(failed) if failed else "all hard checks pass")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
