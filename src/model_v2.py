"""Model v2 (reports/evaluation_protocol_v2.md): replace plate_z_zone with the season-invariant plate_z_abs_zone.

E1 cross-season check (2025 -> all 2026; post-hoc development evidence) and E2 within-development validation
(train -> August 2026), each with v1 and v2 features and the same model specification. Then the Amendment 1
calibration rule on September/August 2026, the final fit on train + validation, and the frozen artifact.

Writes data/model/v2/ (feature table, out-of-sample 2026 predictions for the pitcher comparison), models/final_v2/ and
reports/model_v2_report.md.
Usage: .venv/bin/python src/model_v2.py
"""
import datetime as dt
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from calibrate import KINDS, CalibratedModel, Calibrator
from evaluate import metrics, two_way_bootstrap
from features import FB_DERIVED, add_batter_height_zone, add_features, apply_pitch_type_levels, pitch_type_levels

ROOT = Path(__file__).resolve().parents[1]
SEED = 20250601
CFG = yaml.safe_load((ROOT / "config" / "splits_v2.yaml").read_text())
CAT = ["pitch_type_level", "p_throws", "stand", "count"]
NUM_V1 = list(yaml.safe_load((ROOT / "config" / "features.yaml").read_text())["allow"]["numeric"]) + FB_DERIVED + ["fb_ref_missing"]
NUM_V2 = list(yaml.safe_load((ROOT / "config" / "features_v2.yaml").read_text())["allow"]["numeric"]) + FB_DERIVED + ["fb_ref_missing"]
OUT_DATA, OUT_MODEL = ROOT / "data" / "model" / "v2", ROOT / "models" / "final_v2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(num):
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)
    pre = ColumnTransformer([("num", "passthrough", num), ("cat", enc, CAT)], verbose_feature_names_out=False)
    clf = HistGradientBoostingClassifier(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, max_iter=800,
                                         early_stopping=False, categorical_features=[False] * len(num) + [True] * len(CAT),
                                         random_state=SEED)
    return Pipeline([("pre", pre), ("clf", clf)])


def load_swings():
    frames = []
    for source in CFG["seasons"]:
        sid = yaml.safe_load((ROOT / "config" / "data_sources.yaml").read_text())[source]["source_id"]
        m = pd.read_parquet(ROOT / "data" / "model" / sid / "model_pitches.parquet")
        frames.append(m[m["model_population"]].drop(columns=["partition"]))
    pitches = add_batter_height_zone(add_features(pd.concat(frames, ignore_index=True)))  # as-of refs carry across seasons
    sw = pitches[pitches["eligible_swing"]].copy()
    sw["y"] = sw["is_miss"].astype(int)
    sw["count"] = sw["balls"].astype(int).astype(str) + "-" + sw["strikes"].astype(int).astype(str)
    for c in ["p_throws", "stand"]:
        sw[c] = sw[c].astype(str)
    d = pd.to_datetime(sw["game_date"])
    sw["partition"] = "unassigned"
    for p, (a, b) in CFG["partitions"].items():
        sw.loc[d.between(pd.Timestamp(a), pd.Timestamp(b)), "partition"] = p
    return sw


def in_window(sw, window):
    d = pd.to_datetime(sw["game_date"])
    return sw[d.between(pd.Timestamp(window[0]), pd.Timestamp(window[1]))]


def compare(fit_rows, score_rows, label):
    """Fit with v1 and v2 numeric features on the same rows; score the same rows; bootstrap v2 - v1."""
    preds, models = {}, {}
    for name, num in (("v1 features", NUM_V1), ("v2 features", NUM_V2)):
        models[name] = build(num).fit(fit_rows[num + CAT], fit_rows["y"].to_numpy())
        preds[name] = models[name].predict_proba(score_rows[num + CAT])[:, 1]
        print(f"{label} {name}: {metrics(score_rows['y'], preds[name])['log_loss']:.5f}", flush=True)
    y = score_rows["y"].to_numpy()
    table = pd.DataFrame({k: metrics(y, v) for k, v in preds.items()}).T
    boot = two_way_bootstrap(y, preds, score_rows["game_pk"], score_rows["pitcher_id"], reference="v1 features", seed=SEED)
    by_group = pd.DataFrame({k: pd.Series(v, index=score_rows.index).groupby(score_rows["pitch_group"].astype(str)).mean()
                             for k, v in preds.items()}).assign(observed=score_rows.groupby(score_rows["pitch_group"].astype(str))["y"].mean(),
                                                                swings=score_rows.groupby(score_rows["pitch_group"].astype(str)).size())
    return table, boot, by_group, models, preds


def main():
    sw = load_swings()
    rare_types = lambda rows: sorted(t for t, n in rows.groupby("pitch_type_raw").size().items() if n < CFG["rare_pitch_type_min_train_swings"])
    train = sw[sw["partition"] == "train"]
    levels = pitch_type_levels(train, rare_types(train))
    sw["pitch_type_level"] = apply_pitch_type_levels(sw, levels).astype(str)
    train, val, cal = (sw[sw["partition"] == p] for p in ("train", "validation", "calibration"))

    # E1: cross-season check
    e1_fit, e1_score = in_window(sw, CFG["cross_season_check"]["fit"]), in_window(sw, CFG["cross_season_check"]["score"])
    e1_levels = pitch_type_levels(e1_fit, rare_types(e1_fit))
    e1_fit, e1_score = (d.assign(pitch_type_level=apply_pitch_type_levels(d, e1_levels).astype(str)) for d in (e1_fit, e1_score))
    e1_table, e1_boot, e1_groups, e1_models, e1_preds = compare(e1_fit, e1_score, "E1")
    # E2: within development
    e2_table, e2_boot, e2_groups, e2_models, _ = compare(train, val, "E2")

    # Calibration rule (Amendment 1 of v1, unchanged)
    base = e2_models["v2 features"]
    p_cal, p_val = base.predict_proba(cal[NUM_V2 + CAT])[:, 1], base.predict_proba(val[NUM_V2 + CAT])[:, 1]
    fitted = {k: Calibrator(k).fit(p_cal, cal["y"].to_numpy()) for k in KINDS}
    cal_preds = {k: c.transform(p_val) for k, c in fitted.items()}
    cal_table = pd.DataFrame({k: metrics(val["y"], v) for k, v in cal_preds.items()}).T
    cal_boot = two_way_bootstrap(val["y"].to_numpy(), cal_preds, val["game_pk"], val["pitcher_id"], reference="none", seed=SEED)
    ll = cal_boot[cal_boot["metric"] == "log_loss"].set_index("model")
    ok = [k for k in KINDS[1:] if ll.loc[k, "hi95"] < 0]
    chosen = min(ok, key=lambda k: cal_table.loc[k, "log_loss"]) if ok else "none"

    # Final fit and freeze
    fit_rows = pd.concat([train, val])
    pipeline = build(NUM_V2).fit(fit_rows[NUM_V2 + CAT], fit_rows["y"].to_numpy())
    calibrator = Calibrator(chosen)
    if chosen != "none":
        calibrator.fit(pipeline.predict_proba(cal[NUM_V2 + CAT])[:, 1], cal["y"].to_numpy())
    final = CalibratedModel(pipeline, calibrator)
    OUT_MODEL.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, OUT_MODEL / "final_model.joblib")
    probe = cal.sample(2000, random_state=SEED)[NUM_V2 + CAT]
    reproduces = bool((joblib.load(OUT_MODEL / "final_model.joblib").predict_proba(probe) == final.predict_proba(probe)).all())
    sept = metrics(cal["y"], final.predict_proba(cal[NUM_V2 + CAT])[:, 1])
    card = {"model": "hgb_iter800", "version": "final_v2", "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "change_from_v1": "plate_z_zone replaced by plate_z_abs_zone (batter-height ABS zone, same definition every season)",
            "features_in_order": NUM_V2 + CAT, "pitch_type_levels": levels,
            "fit_on": {"dates": [CFG["partitions"]["train"][0], CFG["partitions"]["validation"][1]], "swings": int(len(fit_rows)),
                       "misses": int(fit_rows["y"].sum()), "pitchers": int(fit_rows["pitcher_id"].nunique())},
            "calibration": {"decision": chosen, "calibrator": calibrator.describe()},
            "september_2026_check": {k: float(v) for k, v in sept.items()},
            "reload_reproduces_predictions": reproduces, "artifact_sha256": sha(OUT_MODEL / "final_model.joblib"),
            "protocol_lock_v2_sha256": sha(ROOT / "data" / "splits" / "protocol_lock_v2.json"),
            "sealed_test": "2027 regular season, not yet available; drift-check gate before scoring (protocol v2 §5)",
            "environment": {"python": platform.python_version(), "scikit-learn": sklearn.__version__},
            "intended_use": "Retrospective pitch analysis of MLB swings. Not for college players, not causal, not a coaching prescription."}
    (OUT_MODEL / "model_card.json").write_text(json.dumps(card, indent=2, default=str))

    # Data for the pitcher comparison: feature table and out-of-sample 2026 predictions (E1 v2 model, fitted on 2025 only)
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    sw.drop(columns=["source_file"], errors="ignore").to_parquet(OUT_DATA / "swings_v2.parquet", index=False)
    pd.DataFrame({"pitch_key": e1_score["pitch_key"].values, "game_date": e1_score["game_date"].values,
                  "pred_oos": e1_preds["v2 features"]}).to_parquet(OUT_DATA / "pred_2026_from_2025_model.parquet", index=False)
    joblib.dump(e1_models["v2 features"], OUT_DATA / "model_2025_only_v2.joblib")
    joblib.dump(e2_models["v2 features"], OUT_DATA / "model_train_only_v2.joblib")

    cols = ["mean_pred", "prevalence", "log_loss", "brier", "roc_auc", "calib_intercept", "calib_slope"]
    L = ["# Model v2 — season-invariant vertical location", "",
         "Protocol: `reports/evaluation_protocol_v2.md` (locked before fitting). Only the vertical-location feature changes; "
         "the model specification is v1's gradient boosting, not re-tuned. **No result here is a clean test**: 2026 outcomes "
         "were seen in the v1 test. The clean test is the 2027 regular season.", "",
         "## E1 — cross-season check: fit 2025 Mar–Aug, score all of 2026 (post-hoc evidence)", "",
         e1_table[cols].round(5).to_markdown(), "",
         "v2 minus v1 (two-way game × pitcher bootstrap):", "", e1_boot.round(5).to_markdown(index=False), "",
         "Mean prediction vs observed by pitch group (2026):", "", e1_groups.round(4).to_markdown(), "",
         "For reference, the frozen v1 model on 2026 (one-time test): calibration intercept −0.112, slope 0.854.", "",
         "## E2 — within development: fit 2025-03-18 to 2026-07-31, score August 2026", "",
         e2_table[cols].round(5).to_markdown(), "", e2_boot.round(5).to_markdown(index=False), "",
         "## Calibration decision (v1 Amendment 1 rule, fitted on September 2026, judged on August 2026)", "",
         cal_table[["mean_pred", "log_loss", "calib_intercept", "calib_slope"]].round(5).to_markdown(), "",
         cal_boot[cal_boot["metric"] == "log_loss"].round(5).to_markdown(index=False), "",
         f"**Decision: {chosen}.**", "",
         "## Frozen v2", "",
         f"`models/final_v2/final_model.joblib` (sha256 `{card['artifact_sha256'][:16]}…`), fitted on {len(fit_rows):,} swings "
         f"({card['fit_on']['dates'][0]} to {card['fit_on']['dates'][1]}). Reload reproduces: {reproduces}. "
         f"September 2026 (not used in fitting{' except the calibrator' if chosen != 'none' else ''}): AUC {sept['roc_auc']:.4f}, "
         f"intercept {sept['calib_intercept']:+.3f}, slope {sept['calib_slope']:.3f}.", ""]
    (ROOT / "reports" / "model_v2_report.md").write_text("\n".join(L))
    print(e1_table[cols].round(4).to_string()); print(e1_groups.round(4).to_string())
    print(e2_table[cols].round(4).to_string()); print("calibration:", chosen, "| reproduces", reproduces)
    print("Sept 2026:", {k: round(sept[k], 4) for k in ("roc_auc", "calib_intercept", "calib_slope")})


if __name__ == "__main__":
    main()
