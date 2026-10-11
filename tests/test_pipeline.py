"""Checks for the parts most likely to invalidate the project.

Fixture tests run from committed real-row samples in tests/fixtures/. Tests marked `processed`
also check the full processed tables when they exist locally (raw data is not committed).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import normalize  # noqa: E402
import normalize_trackman  # noqa: E402
from labels import load_mapping, outcome_category, whiff_label  # noqa: E402
from pitcher_analysis import summarize, wilson  # noqa: E402

CONTRACT = yaml.safe_load((ROOT / "config" / "field_mapping.yaml").read_text())
FIX = ROOT / "tests" / "fixtures"
PROCESSED = ROOT / "data" / "processed"


@pytest.fixture(scope="module")
def statcast():
    raw = pd.read_csv(FIX / "statcast_sample.csv", low_memory=False).assign(source_file="fixture").copy()
    return normalize.normalize(raw, CONTRACT, "fixture")


@pytest.fixture(scope="module")
def trackman():
    raw = pd.read_csv(FIX / "trackman_sample.csv", low_memory=False)
    return normalize_trackman.normalize(raw, CONTRACT, {"source_id": "fixture"})


# Labels -------------------------------------------------------------------

@pytest.mark.parametrize("desc,cat,label", [
    ("swinging_strike", "miss", 1), ("swinging_strike_blocked", "miss", 1),
    ("foul", "contact", 0), ("foul_tip", "contact", 0), ("hit_into_play", "contact", 0),
    ("foul_bunt", "bunt", pd.NA), ("missed_bunt", "bunt", pd.NA),
    ("ball", "take", pd.NA), ("called_strike", "take", pd.NA),
    ("automatic_ball", "non_competitive", pd.NA), ("pitchout", "non_competitive", pd.NA),
])
def test_statcast_label_cases(desc, cat, label):
    c = outcome_category(pd.Series([desc]))
    assert c.iloc[0] == cat
    got = whiff_label(c).iloc[0]
    assert (pd.isna(got) and pd.isna(label)) or got == label


def test_missing_description_is_unknown_not_contact():
    c = outcome_category(pd.Series([None, "foul"]))
    assert c.tolist() == ["unknown", "contact"]
    assert pd.isna(whiff_label(c).iloc[0])


def test_unmapped_description_raises():
    with pytest.raises(ValueError, match="unmapped"):
        outcome_category(pd.Series(["some_new_outcome"]))


def test_trackman_mapping_and_in_play_bunt(trackman):
    assert load_mapping("trackman")["StrikeSwinging"] == "miss"
    raw = pd.DataFrame({"PitchCall": ["InPlay", "InPlay"], "TaggedHitType": ["Bunt", "GroundBall"]})
    cat = outcome_category(raw["PitchCall"], load_mapping("trackman"))
    cat = cat.where(~(raw["PitchCall"].eq("InPlay") & raw["TaggedHitType"].eq("Bunt")), "bunt")
    assert cat.tolist() == ["bunt", "contact"]


# Normalization and status --------------------------------------------------

def test_every_fixture_row_has_status_and_key_is_unique(statcast):
    assert statcast["status"].isin(["retained", "excluded", "quarantined"]).all()
    assert not statcast["pitch_key"].duplicated().any()


def test_non_competitive_rows_excluded_not_deleted(statcast):
    nc = statcast[statcast["outcome_category"] == "non_competitive"]
    assert len(nc) >= 2 and (nc["status"] == "excluded").all()


def test_duplicate_key_is_quarantined():
    raw = pd.read_csv(FIX / "statcast_sample.csv", low_memory=False).head(3)
    raw = pd.concat([raw, raw.head(1)], ignore_index=True)
    raw["source_file"] = "fixture"
    df = normalize.normalize(raw, CONTRACT, "fixture")
    assert (df["status_reason"] == "duplicate_pitch_key").sum() == 2


def test_implausible_values_flagged_not_removed(statcast):
    rogers = statcast[statcast["pitcher_name"].str.startswith("Rogers, Tyler")]
    assert len(rogers) and rogers["review_flags"].str.contains("release_z_ft_out_of_range").all()
    assert (rogers["status"] == "retained").all()


def test_missing_measurement_never_zero_filled():
    raw = pd.read_csv(FIX / "statcast_sample.csv", low_memory=False).head(2)
    raw.loc[0, "release_spin_rate"] = np.nan
    raw["source_file"] = "fixture"
    df = normalize.normalize(raw, CONTRACT, "fixture")
    assert pd.isna(df.loc[0, "spin_rpm"]) and "no_spin" in df.loc[0, "review_flags"]


# Conventions ----------------------------------------------------------------

def test_arm_side_transform_matches_savant(statcast):
    raw = pd.read_csv(FIX / "statcast_sample.csv", low_memory=False)
    expected = raw["api_break_x_arm"] * 12  # Savant's arm-side value, feet
    assert np.allclose(statcast["hmov_arm_in"], expected, atol=0.05, equal_nan=True)


def test_trackman_signs_converted_to_catcher_view(trackman):
    raw = pd.read_csv(FIX / "trackman_sample.csv", low_memory=False)
    assert np.allclose(trackman["plate_x_ft"], -raw["PlateLocSide"], equal_nan=True)
    assert np.allclose(trackman["hmov_ft"] * 12, -raw["HorzBreak"], equal_nan=True)


@pytest.mark.skipif(not (PROCESSED / "pitches.parquet").exists(), reason="processed data not present")
@pytest.mark.parametrize("path,col,ff", [("pitches.parquet", "pitch_type_raw", "FF"),
                                         ("trackman_pitches.parquet", "pitch_type_auto", "Four-Seam")])
def test_processed_four_seam_arm_side_positive_for_both_hands(path, col, ff):
    df = pd.read_parquet(PROCESSED / path)
    med = df[df[col] == ff].groupby("p_throws")["hmov_arm_in"].median()
    assert (med[["L", "R"]] > 4).all()


# TrackMan game reconstruction -------------------------------------------------

def test_trackman_games_split_on_pitchno_reset_and_bad_date_repaired(trackman):
    assert trackman["game_pk"].nunique() == 3  # two games on 10/26/2022 + one on 2/9/2023
    fixed = trackman[trackman["date_repaired"]]
    assert len(fixed) == 1 and str(fixed["game_date"].iloc[0]) == "2023-02-09"
    assert not trackman["pitch_key"].duplicated().any()


# Leakage ---------------------------------------------------------------------

def test_denylisted_columns_absent_from_processed_fixture(statcast, trackman):
    deny = set(CONTRACT["statcast"]["leakage_denylist"]) | set(CONTRACT["trackman"]["leakage_denylist"])
    assert not deny & set(statcast.columns)
    assert not deny & set(trackman.columns)


def test_miss_distance_is_denylisted():
    assert "miss_distance" in CONTRACT["statcast"]["leakage_denylist"]
    for c in ("bat_speed", "swing_length", "attack_angle", "swing_path_tilt"):
        assert c in CONTRACT["statcast"]["leakage_denylist"]


# Denominators -------------------------------------------------------------------

def test_summary_counts_and_denominators(statcast):
    ret = statcast[statcast["status"] == "retained"]
    s = summarize(ret.assign(k=1), ["k"]).iloc[0]
    assert s["swings"] == ret["is_miss"].notna().sum()
    assert s["misses"] == (ret["is_miss"] == 1).sum()
    assert s["whiff_rate"] == pytest.approx(s["misses"] / s["swings"])
    assert s["swstr_rate"] == pytest.approx(s["misses"] / s["classified"])
    assert s["swings"] < s["classified"]  # whiff and SwStr use different denominators


def test_wilson_interval():
    lo, hi = wilson(np.array([0, 5, 3]), np.array([0, 10, 3]))
    assert np.isnan(lo[0]) and np.isnan(hi[0])
    assert lo[1] < 0.5 < hi[1] and hi[2] == pytest.approx(1.0)


# Allowlist, sealed test, partitions -----------------------------------

FEATURES = yaml.safe_load((ROOT / "config" / "features.yaml").read_text())


def test_allowlist_never_overlaps_denylist():
    allowed = set(FEATURES["allow"]["numeric"]) | set(FEATURES["allow"]["categorical"]) | set(FEATURES["allow"]["stage4_derived"])
    sources = {v["source"] for sec in ("numeric", "categorical") for v in FEATURES["allow"][sec].values()}
    deny = set(CONTRACT["statcast"]["leakage_denylist"]) | {"miss_distance", "description", "type", "events"}
    assert not allowed & deny
    assert not {s.strip() for src in sources for s in src.split(",")} & deny
    assert not allowed & set(FEATURES["never"]["identifiers"])


def test_allowlisted_row_features_exist_in_processed_fixture(statcast):
    for sec in ("numeric", "categorical"):
        for name in FEATURES["allow"][sec]:
            assert name in statcast.columns, name


def test_only_ingest_and_final_test_touch_sealed_data():
    allowed = {"ingest.py", "final_test.py", "final_test_v2.py"}
    for f in (ROOT / "src").glob("*.py"):
        text = f.read_text()
        if f.name not in allowed:
            assert "data/sealed" not in text and "statcast_2027" not in text, f.name


def test_normalize_refuses_sealed_source():
    with pytest.raises(SystemExit, match="sealed"):
        normalize.main(source="statcast_2027_sealed")


def test_partitions_whole_dates_and_ordered():
    import build_dataset
    dates = pd.to_datetime(pd.Series(["2025-04-01"] * 3 + ["2025-04-02"] * 3 + ["2025-04-03"] * 2 + ["2025-04-04"] * 2)).dt.date
    df = pd.DataFrame({"game_pk": range(10), "game_date": dates})
    part, by_date = build_dataset.assign_partitions(df, {"train": 0.6, "validation": 0.8})
    assert df.assign(p=part).groupby("game_date")["p"].nunique().eq(1).all()
    assert list(by_date) == ["train", "train", "validation", "calibration"]


def test_protocol_lock_matches_files():
    lock_path = ROOT / "data" / "splits" / "protocol_lock.json"
    if not lock_path.exists():
        pytest.skip("protocol not locked")
    import hashlib
    import json
    lock = json.loads(lock_path.read_text())
    changed = [f for f, h in lock["sha256"].items() if hashlib.sha256((ROOT / f).read_bytes()).hexdigest() != h]
    assert not changed, f"locked files changed without amendment/re-lock: {changed}"


# As-of features, pitch-type levels, metrics ----------------------------

def _synthetic_pitches(n=40, seed=0):
    rng = np.random.default_rng(seed)
    types = (["FF"] * 15 + ["SL"] * 10 + ["SI"] * 5 + ["FF"] * 10)[:n]
    return pd.DataFrame({
        "pitcher_id": 1, "game_date": pd.to_datetime(["2025-04-01"] * 20 + ["2025-04-07"] * 20).date,
        "game_pk": [10] * 20 + [11] * 20, "at_bat_number": np.repeat(np.arange(1, 9), 5), "pitch_number": np.tile(np.arange(1, 6), 8),
        "pitch_type_raw": pd.array(types, dtype="string"), "velocity_mph": rng.normal(94, 1, n),
        "ivb_in": rng.normal(16, 1, n), "hmov_arm_in": rng.normal(8, 1, n)})


def test_fastball_reference_uses_only_earlier_pitches():
    from features import fastball_reference
    df = _synthetic_pitches()
    ref = fastball_reference(df)
    assert ref["fb_ref_missing"].iloc[:10].eq(1).all()            # fewer than 10 prior fastballs
    i = 25
    prior = df.iloc[:i]
    assert ref["fb_ref_velo"].iloc[i] == pytest.approx(prior.loc[prior.pitch_type_raw == "FF", "velocity_mph"].mean())
    changed = df.copy()
    changed.loc[30:, "velocity_mph"] += 50                       # perturb only later pitches
    assert np.allclose(fastball_reference(changed)["fb_ref_velo"].iloc[:31], ref["fb_ref_velo"].iloc[:31], equal_nan=True)


def test_fastball_reference_is_order_independent_of_input_row_order():
    from features import fastball_reference
    df = _synthetic_pitches()
    shuffled = df.sample(frac=1, random_state=1)
    assert np.allclose(fastball_reference(shuffled).loc[df.index, "fb_ref_velo"], fastball_reference(df)["fb_ref_velo"], equal_nan=True)


def test_pitch_type_levels_pool_rare_and_fall_back_for_unseen():
    from features import apply_pitch_type_levels, pitch_type_levels
    train = pd.DataFrame({"pitch_type_raw": ["FF", "SL", "SC"], "pitch_group": ["fastball", "breaking", "breaking"]})
    levels = pitch_type_levels(train, rare=["SC"])
    new = pd.DataFrame({"pitch_type_raw": ["FF", "SC", "SV"], "pitch_group": ["fastball", "breaking", "breaking"]})
    assert apply_pitch_type_levels(new, levels).tolist() == ["FF", "group:breaking", "group:breaking"]


def test_metrics_and_calibration_on_known_inputs():
    from evaluate import metrics
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.6, 20000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    m = metrics(y, p)
    assert abs(m["calib_intercept"]) < 0.05 and abs(m["calib_slope"] - 1) < 0.1
    assert np.isnan(metrics(y, np.full(y.size, y.mean()))["calib_slope"])


def test_baseline_model_card_records_reproducible_pooled_model():
    import json
    card_path = ROOT / "models" / "baselines_v1" / "model_card.json"
    if not card_path.exists():
        pytest.skip("baselines not run")
    card = json.loads(card_path.read_text())
    assert card["reload_reproduces_predictions"]
    deny = set(CONTRACT["statcast"]["leakage_denylist"])
    assert not deny & set(card["features_in_order"])
    assert "validation" in card and "calibration" not in json.dumps(card["training_coverage"])


# Calibration ---------------------------------------------------------

def test_calibrators_behave():
    from calibrate import Calibrator
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.6, 5000)
    y = (rng.uniform(size=p.size) < np.clip(p + 0.05, 0, 1)).astype(int)   # model under-predicts by 5 points
    assert np.array_equal(Calibrator("none").fit(p, y).transform(p), p)
    shifted = Calibrator("intercept").fit(p, y).transform(p)
    assert shifted.mean() == pytest.approx(y.mean(), abs=1e-6)
    assert np.all(np.diff(Calibrator("sigmoid").fit(p, y).transform(np.sort(p))) >= 0)   # monotone


def test_calibration_decision_follows_amendment_rule():
    import json
    path = ROOT / "models" / "calibration_v1" / "decision.json"
    if not path.exists():
        pytest.skip("calibration not run")
    d = json.loads(path.read_text())
    assert d["fitted_on"]["partition"] == "calibration" and d["judged_on"]["partition"] == "validation"
    assert d["chosen"] in {"none", "intercept", "sigmoid", "isotonic"}


def test_sealed_test_scored_exactly_once_and_rescoring_refused():
    import json
    log_path = ROOT / "data" / "sealed" / "manifests" / "test_open_log.json"
    if not log_path.exists():
        pytest.skip("sealed test not run")
    log = json.loads(log_path.read_text())
    assert sum(1 for e in log if "criteria" in e) == 1          # one scoring; later entries are labelled diagnostics
    import final_test
    with pytest.raises(SystemExit, match="already scored"):
        final_test.main(dry_run=False)
    assert json.loads(log_path.read_text()) == log               # refused before anything was logged or read


def test_protocol_lock_v2_matches_files():
    lock_path = ROOT / "data" / "splits" / "protocol_lock_v2.json"
    if not lock_path.exists():
        pytest.skip("v2 protocol not locked")
    import hashlib
    import json
    lock = json.loads(lock_path.read_text())
    changed = [f for f, h in lock["sha256"].items() if hashlib.sha256((ROOT / f).read_bytes()).hexdigest() != h]
    assert not changed, f"v2 locked files changed without amendment/re-lock: {changed}"


def test_drift_gate_flags_definition_change_but_not_stable_features():
    from drift_check import gate
    rng = np.random.default_rng(0)
    n = 4000
    ref = pd.DataFrame({"batter_id": rng.integers(0, 100, n), "x": rng.normal(0, 1, n)})
    ref["sz_top_ft"] = 3.4 + rng.normal(0, 0.07, n)
    ref["sz_bot_ft"] = 1.6 + rng.normal(0, 0.06, n)
    ref["plate_z_zone"] = rng.normal(0.4, 0.5, n)
    tgt = ref.copy()
    tgt["sz_top_ft"] = 3.2 + tgt["batter_id"] * 0.001      # fixed per batter
    tgt["sz_bot_ft"] = 1.6 + tgt["batter_id"] * 0.001
    _, _, flagged = gate(ref, tgt, ["x", "plate_z_zone"])
    assert flagged == ["plate_z_zone"]
    _, _, flagged = gate(ref, tgt, ["x"])
    assert flagged == []


def test_batter_height_zone_definition():
    from features import ABS_BOTTOM, ABS_TOP
    from players import parse_height_ft
    assert parse_height_ft('6\' 2"') == pytest.approx(6 + 2 / 12)
    assert parse_height_ft(None) is None
    h = 6.0
    assert (ABS_BOTTOM * h - ABS_BOTTOM * h) / ((ABS_TOP - ABS_BOTTOM) * h) == 0
    assert (ABS_TOP * h - ABS_BOTTOM * h) / ((ABS_TOP - ABS_BOTTOM) * h) == pytest.approx(1)


# Pitcher comparison ------------------------------------------------------------------------

def test_batter_context_uses_only_earlier_games():
    from expected_whiff import batter_context
    sw = pd.DataFrame({"batter_id": [1, 1, 1, 1], "game_date": pd.to_datetime(["2025-04-01", "2025-04-01", "2025-04-02", "2025-04-03"]).date,
                       "game_pk": [1, 1, 2, 3], "y": [1, 1, 0, 0]})
    out = batter_context(sw, league_rate=0.25)
    assert out.loc[0, "batter_whiff_asof"] == pytest.approx(0.25)                    # first game: league prior only
    assert out.loc[2, "batter_whiff_asof"] == pytest.approx((2 + 100 * 0.25) / (2 + 100))
    assert (out.loc[0:1, "batter_whiff_asof"] == out.loc[0, "batter_whiff_asof"]).all()  # same game, same value


def test_pitcher_flags_carry_required_evidence():
    path = ROOT / "reports" / "pitcher_flags.csv"
    if not path.exists():
        pytest.skip("pitcher comparison not run")
    f = pd.read_csv(path)
    required = ["pitcher_id", "kind", "metric", "value", "unit", "comparison_group", "comparison_n", "denominator", "uncertainty", "evidence"]
    assert f[required].notna().all().all()
    fb = f[(f["metric"].isin(["velo_diff_fb", "ivb_diff_fb", "hmov_diff_fb"])) & f["pitch_type"].isin(["FF", "SI", "FA"])]
    assert fb.empty


def test_example_player_report_has_evidence_for_every_flag():
    import re
    reports = list((ROOT / "reports" / "players").glob("*_evidence.csv"))
    if not reports:
        pytest.skip("no player reports")
    ev = pd.read_csv(reports[0])
    pid = int(re.match(r"(\d+)_", reports[0].name).group(1))
    flags = pd.read_csv(ROOT / "reports" / "pitcher_flags.csv")
    assert ev["flag_id"].nunique() == (flags["pitcher_id"] == pid).sum()
    assert ev.groupby("flag_id").size().min() > 0


# College transfer -------------------------------------------------------

def test_protocol_lock_transfer_matches_files():
    lock_path = ROOT / "data" / "splits" / "protocol_lock_transfer.json"
    if not lock_path.exists():
        pytest.skip("transfer protocol not locked")
    import hashlib
    import json
    lock = json.loads(lock_path.read_text())
    changed = [f for f, h in lock["sha256"].items() if hashlib.sha256((ROOT / f).read_bytes()).hexdigest() != h]
    assert not changed


def test_transfer_feature_sets_exclude_unavailable_and_leaky_inputs():
    cfg = yaml.safe_load((ROOT / "config" / "trackman_transfer.yaml").read_text())
    unavailable = set(cfg["feature_sets"]["unavailable_in_college"])
    deny = set(CONTRACT["statcast"]["leakage_denylist"]) | set(CONTRACT["trackman"]["leakage_denylist"])
    for k in ("core", "full"):
        cols = set(cfg["feature_sets"][k]["numeric"])
        assert not cols & unavailable and not cols & deny
    assert not {"extension_ft", "ivb_in"} & set(cfg["feature_sets"]["core"]["numeric"])   # open questions stay out of core


def test_other_game_fastball_reference_never_uses_the_same_game():
    from transfer import other_game_fastball_reference
    rows = []
    for game, velo in ((1, 90.0), (2, 94.0)):
        rows += [{"pitcher_id": 7, "game_pk": game, "pitch_type_sc": "FF", "velocity_mph": velo, "ivb_in": 15.0, "hmov_arm_in": 8.0}] * 12
    ref = other_game_fastball_reference(pd.DataFrame(rows)).set_index("game_pk")
    assert ref.loc[1, "ref_velo"] == pytest.approx(94.0) and ref.loc[2, "ref_velo"] == pytest.approx(90.0)


def test_transfer_decision_matches_rule():
    import json
    path = ROOT / "models" / "transfer_reduced_v1" / "decision.json"
    if not path.exists():
        pytest.skip("transfer not run")
    d = json.loads(path.read_text())
    assert d["status"].startswith("validated") == all(d["rule"].values())
    assert "this program" in d["status"] or d["status"] == "experimental"


# Hypotheses ----------------------------------------------------------------

def test_batter_swing_context_uses_only_earlier_games():
    from outcome_models import SWING_CFG, batter_swing_context
    k = SWING_CFG["shrink_pitches"]
    p = pd.DataFrame({"batter_id": 1, "game_date": ["2025-04-01"] * 4 + ["2025-04-02"] * 2, "game_pk": [10] * 4 + [11] * 2,
                      "swing": [1, 1, 0, 1, 0, 0], "plate_x_away_ft": [0.0, 0.0, 2.0, 2.0, 0.0, 2.0], "plate_z_abs_zone": 0.5})
    out = batter_swing_context(p, pd.Series(True, index=p.index))
    first, second = out[out["game_pk"] == 10], out[out["game_pk"] == 11]
    assert (first["bat_prior_pitches"] == 0).all() and first["bat_swing_asof"].nunique() == 1    # nothing earlier: league rate only
    assert first["bat_swing_asof"].iloc[0] == pytest.approx(0.5)
    assert (second["bat_prior_pitches"] == 4).all()                                               # game 11's own pitches not counted
    assert second["bat_swing_asof"].iloc[0] == pytest.approx((3 + k * 0.5) / (4 + k))
    assert second["bat_zswing_asof"].iloc[0] == pytest.approx((2 + k * 2 / 3) / (2 + k))
    assert second["bat_chase_asof"].iloc[0] == pytest.approx((1 + k * 1 / 3) / (2 + k))


def test_shifted_moves_fastball_difference_only_for_non_fastballs():
    from hypotheses import shifted
    df = pd.DataFrame({"velocity_mph": [85.0], "velo_diff_fb": [-9.0]})
    assert shifted(df, "velocity_mph", 1.0, "SL")["velo_diff_fb"].iloc[0] == -8.0
    assert shifted(df, "velocity_mph", 1.0, "FF")["velo_diff_fb"].iloc[0] == -9.0


def test_support_rejects_values_the_pitcher_has_not_thrown():
    from hypotheses import supported
    rng = np.random.default_rng(0)
    own = pd.DataFrame({"velocity_mph": rng.normal(85, 0.5, 300), "spin_rpm": rng.normal(2400, 50, 300),
                        "ivb_in": rng.normal(2, 1, 300), "hmov_arm_in": rng.normal(-6, 1, 300)})
    scale = pd.Series({"velocity_mph": 2.0, "spin_rpm": 200.0, "ivb_in": 3.0, "hmov_arm_in": 3.0})
    assert supported(own, "velocity_mph", 85.3, scale)[0]
    assert not supported(own, "velocity_mph", 89.0, scale)[0]          # outside his own range


def test_tier_rules_keep_model_only_evidence_at_c():
    from hypotheses import tier
    est = {"rv": (0.4, 0.3, 0.5), "swstr": (0.5, 0.3, 0.7)}
    clear, noisy, opposite = {"rv": (2.0, 0.5, 3.5)}, {"rv": (2.0, -0.5, 4.5)}, {"rv": (-2.0, -3.5, -0.5)}
    assert tier("sensitivity", est)[0] == "C"                          # model-only, however tight its interval
    assert tier("own_history", est, clear)[0] == "A"
    assert tier("own_history", est, noisy)[0] == "B"
    assert tier("release", est, clear)[0] == "B"                       # release never reaches A
    assert tier("own_history", est, opposite)[0] == "C"                # observed and model disagree
    assert tier("own_history", {"rv": (0.4, 0.3, 0.5), "swstr": (-0.5, -0.7, -0.3)}, clear) == ("C", True)


def test_observed_swinging_strike_rate_is_per_pitch_and_interval_covers_point():
    from hypotheses import observed_did, observed_outcomes, swstr_flag
    rng = np.random.default_rng(0)
    def pitches(n, miss_rate):
        swing = rng.random(n) < 0.5
        d = pd.DataFrame({"game_pk": rng.integers(0, 30, n), "swing": swing.astype(int), "run_value": rng.normal(0, 0.2, n),
                          "is_miss": pd.array(np.where(swing, rng.random(n) < miss_rate, 0), dtype="Int8")})
        d.loc[~swing, "is_miss"] = pd.NA                                # takes carry no miss label
        return d.assign(_swstr=lambda x: swstr_flag(x))
    a, b = pitches(4000, 0.2), pitches(4000, 0.4)
    assert observed_outcomes(a)["swstr"] == pytest.approx((a["is_miss"] == 1).sum() / len(a))
    for paired in (False, True):
        point, lo, hi = observed_did(a, b, {"p_swing": 0, "swstr": 0, "rv": 0}, rng, paired=paired)["swstr"]
        assert lo < point < hi and 5 < point < 15                      # about +10 pp per pitch, not +20 pp per swing


def test_hypothesis_records_follow_standard_form_and_gate():
    path = ROOT / "reports" / "hypotheses.csv"
    if not path.exists():
        pytest.skip("hypotheses not run")
    h = pd.read_csv(path)
    required = ["finding", "supporting_evidence", "hypothesis_to_test", "model_estimate", "limitations",
                "coach_selected_experiment", "success_metric", "review_point", "tier"]
    assert h[required].notna().all().all()
    assert (h["coach_selected_experiment"] == "PENDING COACH SELECTION").all()
    assert h.loc[h["kind"] == "sensitivity", "hypothesis_to_test"].str.startswith("MODEL SENSITIVITY").all()
    assert not h["hypothesis_to_test"].str.contains("plate_", case=False).any()
    assert not h["perturbed_features"].str.contains("plate_").any()
    assert set(h["tier"]) <= {"A", "B", "C"}
    # revised tier rules (config/hypotheses.yaml): model-only evidence never ranks above C; A and B need agreeing observations
    assert (h.loc[h["kind"] == "sensitivity", "tier"] == "C").all()
    ab = h[h["tier"].isin(["A", "B"])]
    assert (np.sign(ab["obs_rv_per100"]) == np.sign(ab["rv_estimate_per100"])).all()
    assert not ab["trade_off"].str.contains("opposite").any()
    a = h[h["tier"] == "A"]
    assert (a["kind"] == "own_history").all() and ((a["obs_rv_lo"] > 0) | (a["obs_rv_hi"] < 0)).all()
    assert h["observed_difference"].isna().eq(h["obs_rv_per100"].isna()).all()
