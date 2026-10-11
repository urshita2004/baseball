"""Outcome models: P(swing | pitch) and expected pitcher run value per pitch (plus the existing whiff model).

All three use the v2 features and the v2 development partitions: fitted on train (2025-03-18 to 2026-07-31) and
checked on August 2026. The swing model also uses the batter's own swing tendencies from his earlier games
(config/hypotheses.yaml `swing_model`); September 2026 is scored as a confirmation. Run value is Statcast's delta_pitcher_run_exp (runs saved; positive = good for the pitcher),
used here only as an outcome, never as a predictor.

Writes data/outcomes/pitches_all.parquet, models/outcome_models_v2/ and reports/outcome_models.md.
Usage: .venv/bin/python src/outcome_models.py
"""
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from evaluate import metrics
from features import add_batter_height_zone, add_features, apply_pitch_type_levels
from ingest import read_raw_bytes
from model_v2 import CAT, CFG, NUM_V2, SEED

ROOT = Path(__file__).resolve().parents[1]
OUT_MODELS, OUT_DATA = ROOT / "models" / "outcome_models_v2", ROOT / "data" / "outcomes"
SWING_CFG = yaml.safe_load((ROOT / "config" / "hypotheses.yaml").read_text())["swing_model"]
BATTER = SWING_CFG["batter_features"]


def batter_swing_context(p, fit_rows):
    """As-of batter swing tendencies from strictly earlier games: overall, in-zone and out-of-zone (chase) swing rates.

    Each rate is shrunk toward the league rate of `fit_rows` (a boolean mask: the training period). The zone is the
    season-invariant one (plate_z_abs_zone), not sz_top/sz_bot.
    """
    k = SWING_CFG["shrink_pitches"]
    d = p[["batter_id", "game_date", "game_pk", "swing"]].copy()
    d["z"] = ((p["plate_x_away_ft"].abs() <= SWING_CFG["zone_half_width_ft"]) & p["plate_z_abs_zone"].between(0, 1)).astype(int)
    d["o"] = 1 - d["z"]
    d["zs"], d["os"], d["n"] = d["z"] * d["swing"], d["o"] * d["swing"], 1
    f = d[fit_rows.to_numpy()]
    league = {"n": f["swing"].mean(), "z": f["zs"].sum() / f["z"].sum(), "o": f["os"].sum() / f["o"].sum()}
    g = (d.groupby(["batter_id", "game_date", "game_pk"])[["n", "swing", "z", "zs", "o", "os"]].sum()
           .reset_index().sort_values(["batter_id", "game_date", "game_pk"]))
    prior = g.groupby("batter_id")[["n", "swing", "z", "zs", "o", "os"]].cumsum() - g[["n", "swing", "z", "zs", "o", "os"]]
    g["bat_swing_asof"] = (prior["swing"] + k * league["n"]) / (prior["n"] + k)
    g["bat_zswing_asof"] = (prior["zs"] + k * league["z"]) / (prior["z"] + k)
    g["bat_chase_asof"] = (prior["os"] + k * league["o"]) / (prior["o"] + k)
    g["bat_prior_pitches"] = prior["n"]
    return p.merge(g[["batter_id", "game_pk"] + BATTER], on=["batter_id", "game_pk"], how="left")


def run_values():
    """Pitch-level delta_pitcher_run_exp from the raw files (checksummed manifests), keyed like pitch_key."""
    frames = []
    for season in (2025, 2026):
        m = json.loads((ROOT / "data" / "manifests" / f"statcast_{season}_regular.json").read_text())
        for b in m["batches"]:
            if b["rows"]:
                frames.append(pd.read_csv(io.BytesIO(read_raw_bytes(ROOT / b["file"])),
                                          usecols=["game_pk", "at_bat_number", "pitch_number", "delta_pitcher_run_exp"]))
    rv = pd.concat(frames, ignore_index=True)
    rv["pitch_key"] = rv["game_pk"].astype(str) + "-" + rv["at_bat_number"].astype(str) + "-" + rv["pitch_number"].astype(str)
    return rv[["pitch_key", "delta_pitcher_run_exp"]].rename(columns={"delta_pitcher_run_exp": "run_value"})


def load_all_pitches():
    frames = [pd.read_parquet(ROOT / "data" / "model" / f"statcast_{s}_regular" / "model_pitches.parquet") for s in (2025, 2026)]
    p = pd.concat([f[f["model_population"]].drop(columns=["partition"]) for f in frames], ignore_index=True)
    p = add_batter_height_zone(add_features(p))
    p = p[p["outcome_category"] != "bunt"].copy()
    p["swing"] = p["outcome_category"].isin(["miss", "contact"]).astype(int)
    p["count"] = p["balls"].astype(int).astype(str) + "-" + p["strikes"].astype(int).astype(str)
    for c in ["p_throws", "stand"]:
        p[c] = p[c].astype(str)
    card = json.loads((ROOT / "models" / "final_v2" / "model_card.json").read_text())
    p["pitch_type_level"] = apply_pitch_type_levels(p, card["pitch_type_levels"]).astype(str)
    d = pd.to_datetime(p["game_date"])
    p["partition"] = "unassigned"
    for part, (a, b) in CFG["partitions"].items():
        p.loc[d.between(pd.Timestamp(a), pd.Timestamp(b)), "partition"] = part
    p["season"] = d.dt.year.astype(str)
    p = batter_swing_context(p, p["partition"] == "train")
    return p.merge(run_values(), on="pitch_key", how="left")


def pre(num=NUM_V2):
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)
    return ColumnTransformer([("num", "passthrough", num), ("cat", enc, CAT)], verbose_feature_names_out=False)


def cat_mask(num=NUM_V2):
    return [False] * len(num) + [True] * len(CAT)


def main():
    p = load_all_pitches()
    X = NUM_V2 + CAT
    XS = NUM_V2 + BATTER + CAT
    tr, va, sep = p[p["partition"] == "train"], p[p["partition"] == "validation"], p[p["partition"] == "calibration"]

    def fit_swing(num):
        return Pipeline([("pre", pre(num)), ("clf", HistGradientBoostingClassifier(
            learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, max_iter=800, early_stopping=False,
            categorical_features=cat_mask(num), random_state=SEED))]).fit(tr[num + CAT], tr["swing"])
    swing, swing_pitch_only = fit_swing(NUM_V2 + BATTER), fit_swing(NUM_V2)
    trv = tr.dropna(subset=["run_value"])
    rv = Pipeline([("pre", pre()), ("reg", HistGradientBoostingRegressor(
        learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, max_iter=800, early_stopping=False,
        categorical_features=cat_mask(), random_state=SEED))]).fit(trv[X], trv["run_value"])
    whiff = joblib.load(ROOT / "data" / "model" / "v2" / "model_train_only_v2.joblib")

    sw_m = metrics(va["swing"], swing.predict_proba(va[XS])[:, 1])
    rows = []
    for label, d in (("August 2026 (chosen on)", va), ("September 2026 (confirmation)", sep)):
        for name, model, cols in (("pitch features + batter tendencies (saved)", swing, XS), ("pitch features only (v1)", swing_pitch_only, X)):
            m = metrics(d["swing"], model.predict_proba(d[cols])[:, 1])
            rows.append({"period": label, "model": name, "n": len(d), "accuracy": float(((model.predict_proba(d[cols])[:, 1] >= 0.5) == d["swing"]).mean()),
                         "roc_auc": m["roc_auc"], "log_loss": m["log_loss"], "calib_slope": m["calib_slope"]})
        rows.append({"period": label, "model": "always the more common outcome", "n": len(d), "accuracy": float(max(d["swing"].mean(), 1 - d["swing"].mean()))})
    acc_table = pd.DataFrame(rows)
    vrv = va.dropna(subset=["run_value"])
    pred_rv = rv.predict(vrv[X])
    count_mean = trv.groupby("count")["run_value"].mean()
    base_rv = vrv["count"].map(count_mean).fillna(trv["run_value"].mean())
    mse = lambda a, b: float(np.mean((a - b) ** 2))
    rv_table = pd.DataFrame({"model": ["run-value model", "count-only mean", "constant"],
                             "mse": [mse(vrv["run_value"], pred_rv), mse(vrv["run_value"], base_rv), mse(vrv["run_value"], trv["run_value"].mean())]})
    rv_table["r2_vs_constant"] = 1 - rv_table["mse"] / rv_table.loc[2, "mse"]
    dec = pd.DataFrame({"pred": pred_rv, "obs": vrv["run_value"].to_numpy()})
    dec["decile"] = pd.qcut(dec["pred"], 10, labels=False, duplicates="drop")
    dec = dec.groupby("decile").agg(n=("obs", "size"), mean_pred=("pred", "mean"), mean_obs=("obs", "mean"))

    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    joblib.dump({"swing": swing, "whiff": whiff, "run_value": rv, "features": X, "swing_features": XS}, OUT_MODELS / "outcome_models.joblib")
    keep = [c for c in p.columns if c != "source_file"]
    p[keep].to_parquet(OUT_DATA / "pitches_all.parquet", index=False)

    L = ["# Outcome models", "",
         "Fitted on v2 train (2025-03-18 to 2026-07-31), checked on August 2026. Features: v2 allowlist. Associational models: "
         "they describe outcomes of pitches like these, not what would happen if a pitcher changed something.", "",
         "## P(swing | pitch) — all non-bunt pitches", "",
         "Features: the v2 allowlist plus the batter's own swing tendencies from his earlier games only (overall, in-zone and chase "
         "swing rates, shrunk toward the league rate; `config/hypotheses.yaml`). The whiff and run-value models do not use them.", "",
         pd.Series(sw_m).round(4).to_frame("August 2026").T[["n", "prevalence", "mean_pred", "log_loss", "roc_auc", "calib_intercept", "calib_slope"]].to_markdown(), "",
         "Accuracy = share of pitches where the predicted side of 0.5 matches what the batter did. The batter features were chosen on "
         "August; September was not used to fit or choose anything in this model.", "",
         acc_table.round(4).to_markdown(index=False), "",
         "## Expected pitcher run value per pitch", "",
         "Target: Statcast `delta_pitcher_run_exp` (runs saved; positive favours the pitcher). Per-pitch run value is very noisy, so R² is "
         "small by nature; the comparison that matters is against the count-only mean, because count drives most of it.", "",
         rv_table.round(6).to_markdown(index=False), "",
         "Calibration by predicted decile (runs per pitch):", "", dec.round(4).to_markdown(), "",
         "## P(miss | swing)", "", "Reused: the v2 development model (`data/model/v2/model_train_only_v2.joblib`).", ""]
    (ROOT / "reports" / "outcome_models.md").write_text("\n".join(L))
    print(acc_table.round(4).to_string())
    print(pd.Series(sw_m).round(4).to_string()); print(rv_table.round(6).to_string()); print(dec.round(4).to_string())


if __name__ == "__main__":
    main()
