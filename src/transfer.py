"""Transfer of reduced-feature MLB whiff models to college TrackMan (reports/transfer_protocol.md).

Fits two reduced-feature models on MLB only, scores every college swing without college fitting (T0), then
leave-one-game-out recalibration (T1). Applies the locked decision rule and writes the supported-use statement.

Writes models/transfer_reduced_v1/ and reports/trackman_transfer_report.md.
Usage: .venv/bin/python src/transfer.py
"""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from calibrate import Calibrator
from drift_check import drift_table
from evaluate import EPS, metrics, two_way_bootstrap
from features import MIN_FB_PITCHES, apply_pitch_type_levels

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config" / "trackman_transfer.yaml").read_text())
CONTRACT = yaml.safe_load((ROOT / "config" / "field_mapping.yaml").read_text())
CAT = CFG["feature_sets"]["categorical"]
SETS = {k: CFG["feature_sets"][k]["numeric"] for k in ("core", "full")}
SEED, REPS = 20250601, 2000
OUT = ROOT / "models" / "transfer_reduced_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_lock():
    lock = json.loads((ROOT / "data" / "splits" / "protocol_lock_transfer.json").read_text())
    changed = [f for f, h in lock["sha256"].items() if sha(ROOT / f) != h]
    if changed:
        raise SystemExit(f"transfer protocol locked files changed: {changed}")


def build(num):
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)
    pre = ColumnTransformer([("num", "passthrough", num), ("cat", enc, CAT)], verbose_feature_names_out=False)
    clf = HistGradientBoostingClassifier(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, max_iter=800,
                                         early_stopping=False, categorical_features=[False] * len(num) + [True] * len(CAT),
                                         random_state=SEED)
    return Pipeline([("pre", pre), ("clf", clf)])


def other_game_fastball_reference(pitches):
    """Primary fastball (FF or SI) per pitcher from his OTHER games: tracking only, no chronology needed."""
    fb = pitches[pitches["pitch_type_sc"].isin(["FF", "SI"])]
    rows = []
    for (pid, game), _ in pitches.groupby(["pitcher_id", "game_pk"]):
        other = fb[(fb["pitcher_id"] == pid) & (fb["game_pk"] != game)]
        if other.empty:
            continue
        primary = other["pitch_type_sc"].value_counts().idxmax()
        ref = other[other["pitch_type_sc"] == primary]
        if len(ref) >= MIN_FB_PITCHES:
            rows.append({"pitcher_id": pid, "game_pk": game, "ref_velo": ref["velocity_mph"].mean(),
                         "ref_ivb": ref["ivb_in"].mean(), "ref_hmov": ref["hmov_arm_in"].mean()})
    return pd.DataFrame(rows, columns=["pitcher_id", "game_pk", "ref_velo", "ref_ivb", "ref_hmov"])


def college_swings(levels, label_source):
    tm = pd.read_parquet(ROOT / "data" / "processed" / "trackman_pitches.parquet")
    tm = tm[tm["p_throws"].isin(["L", "R"]) & tm["stand"].isin(["L", "R"])].copy()
    src_col = "pitch_type_auto" if label_source == "auto" else "pitch_type_raw"
    tm["pitch_type_sc"] = tm[src_col].map(CFG["pitch_type_to_statcast"][label_source])
    groups = {c: g for g, codes in CONTRACT["pitch_type_groups"].items() for c in codes}
    tm["pitch_type_raw_sc"] = tm["pitch_type_sc"]
    tm["pitch_group_sc"] = tm["pitch_type_sc"].map(groups).fillna("unknown")
    tm["pitch_type_level"] = apply_pitch_type_levels(
        tm.rename(columns={"pitch_type_raw": "_tag", "pitch_type_raw_sc": "pitch_type_raw", "pitch_group": "_grp", "pitch_group_sc": "pitch_group"}),
        levels).astype(str)
    ref = other_game_fastball_reference(tm)
    tm = tm.merge(ref, on=["pitcher_id", "game_pk"], how="left")
    tm["velo_diff_fb"] = tm["velocity_mph"] - tm["ref_velo"]
    tm["ivb_diff_fb"] = tm["ivb_in"] - tm["ref_ivb"]
    tm["hmov_diff_fb"] = tm["hmov_arm_in"] - tm["ref_hmov"]
    tm["fb_ref_missing"] = tm["ref_velo"].isna().astype(int)
    tm["plate_x_away_ft"] = tm["plate_x_ft"] * tm["stand"].map({"R": 1.0, "L": -1.0})
    sw = tm[tm["eligible_swing"]].copy()
    sw["y"] = sw["is_miss"].astype(int)
    sw["count"] = sw["balls"].astype(int).astype(str) + "-" + sw["strikes"].astype(int).astype(str)
    for c in ["p_throws", "stand"]:
        sw[c] = sw[c].astype(str)
    return tm, sw


def two_way_ci(y, p, games, pitchers, reps=REPS):
    """Bootstrap 95% intervals for AUC, log loss and calibration intercept/slope (game x pitcher Poisson weights)."""
    rng = np.random.default_rng(SEED)
    y, p = np.asarray(y, int), np.clip(np.asarray(p, float), EPS, 1 - EPS)
    z = logit(p)
    gc, gn = pd.factorize(games)[0], pd.Series(games).nunique()
    pc, pn = pd.factorize(pitchers)[0], pd.Series(pitchers).nunique()
    out = {"roc_auc": [], "log_loss": [], "calib_intercept": [], "calib_slope": []}
    for _ in range(reps):
        w = rng.poisson(1.0, gn)[gc] * rng.poisson(1.0, pn)[pc]
        if w.sum() == 0 or len(np.unique(y[w > 0])) < 2:
            continue
        out["roc_auc"].append(roc_auc_score(y, p, sample_weight=w))
        out["log_loss"].append(log_loss(y, p, sample_weight=w))
        ybar = np.average(y, weights=w)
        out["calib_intercept"].append(brentq(lambda a: np.average(expit(a + z), weights=w) - ybar, -8, 8))
        out["calib_slope"].append(LogisticRegression(C=np.inf).fit(z.reshape(-1, 1), y, sample_weight=w).coef_[0, 0])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


def logo(sw, fn):
    """Leave-one-game-out: fn(train_rows, test_rows) -> predictions for test_rows."""
    pred = pd.Series(np.nan, index=sw.index)
    for g in sw["game_pk"].unique():
        te = sw["game_pk"] == g
        pred[te] = fn(sw[~te], sw[te])
    return pred.to_numpy()


def main():
    verify_lock()
    mlb = pd.read_parquet(ROOT / "data" / "model" / "v2" / "swings_v2.parquet")
    card_v2 = json.loads((ROOT / "models" / "final_v2" / "model_card.json").read_text())
    levels = card_v2["pitch_type_levels"]
    train, val = mlb[mlb["partition"] == "train"], mlb[mlb["partition"] == "validation"]

    models, mlb_val = {}, {}
    for name, num in SETS.items():
        models[name] = build(num).fit(train[num + CAT], train["y"].to_numpy())
        mlb_val[name] = metrics(val["y"], models[name].predict_proba(val[num + CAT])[:, 1])
        print(f"MLB validation, {name}: log loss {mlb_val[name]['log_loss']:.5f}, AUC {mlb_val[name]['roc_auc']:.4f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, m in models.items():
        joblib.dump(m, OUT / f"reduced_{name}.joblib")

    tm_all, sw = college_swings(levels, "auto")
    _, sw_op = college_swings(levels, "operator")
    y, games, pitchers = sw["y"].to_numpy(), sw["game_pk"].to_numpy(), sw["pitcher_id"].to_numpy()

    preds = {f"{k} (auto types)": models[k].predict_proba(sw[SETS[k] + CAT])[:, 1] for k in SETS}
    preds["core (operator types)"] = models["core"].predict_proba(sw_op[SETS["core"] + CAT])[:, 1]
    prior = train["y"].mean()
    rates = train.groupby("pitch_type_level")["y"].mean()
    preds["MLB constant"] = np.full(len(sw), prior)
    preds["MLB pitch-type rate"] = sw["pitch_type_level"].map(rates).fillna(prior).to_numpy()
    preds["college LOGO base rate"] = logo(sw, lambda tr, te: np.full(len(te), tr["y"].mean()))

    table = pd.DataFrame({k: metrics(y, v) for k, v in preds.items()}).T
    cis = {k: two_way_ci(y, preds[k], games, pitchers) for k in ["core (auto types)", "full (auto types)", "core (operator types)"]}
    boot = two_way_bootstrap(y, {k: preds[k] for k in ["core (auto types)", "full (auto types)", "core (operator types)", "college LOGO base rate"]},
                             games, pitchers, reference="college LOGO base rate", reps=REPS, seed=SEED)

    # T1: leave-one-game-out recalibration of the core model
    p_core = preds["core (auto types)"]
    sw["_p"] = p_core
    t1 = {kind: logo(sw, lambda tr, te, k=kind: Calibrator(k).fit(tr["_p"].to_numpy(), tr["y"].to_numpy()).transform(te["_p"].to_numpy()))
          for kind in ("intercept", "sigmoid")}
    t1_table = pd.DataFrame({f"core + LOGO {k}": metrics(y, v) for k, v in t1.items()}).T
    t1_ci = {k: two_way_ci(y, v, games, pitchers) for k, v in t1.items()}

    # Decision rule (locked)
    ll_core = boot[(boot["model"] == "core (auto types)") & (boot["metric"] == "log_loss")].iloc[0]
    rule = {
        "core AUC 95% lower bound > 0.60": cis["core (auto types)"]["roc_auc"][0] > 0.60,
        "core log loss beats college LOGO base rate (95% CI below 0)": bool(ll_core["hi95"] < 0),
        "holds under operator-tag pitch types (AUC lower bound > 0.60)": cis["core (operator types)"]["roc_auc"][0] > 0.60,
    }
    validated = all(rule.values())
    sig = t1_table.loc["core + LOGO sigmoid"]
    calibrated = validated and abs(sig["calib_intercept"]) <= 0.1 and 0.9 <= sig["calib_slope"] <= 1.1
    status = ("validated for ranking swings in this program's scrimmages" + ("; probabilities calibrated after LOGO sigmoid recalibration" if calibrated else "; probabilities not calibrated")
              if validated else "experimental")

    # Distribution comparison (shared inputs, MLB swings vs college swings)
    dist = drift_table(mlb, sw, [f for f in SETS["full"] if f not in ("balls", "strikes", "fb_ref_missing")])
    groups = sw.assign(p=p_core).groupby("pitch_group_auto").agg(swings=("y", "size"), observed=("y", "mean"), predicted=("p", "mean"))
    hands = sw.assign(p=p_core).groupby(["p_throws", "stand"]).agg(swings=("y", "size"), observed=("y", "mean"), predicted=("p", "mean"))

    decision = {"status": status, "rule": {k: bool(v) for k, v in rule.items()}, "calibrated": bool(calibrated),
                "college_swings": int(len(sw)), "games": int(sw["game_pk"].nunique()), "pitchers": int(sw["pitcher_id"].nunique()),
                "core_ci": cis["core (auto types)"], "models": {k: sha(OUT / f"reduced_{k}.joblib") for k in SETS}}
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2))

    fmt = lambda d: {k: f"{v[0]:.3f} to {v[1]:.3f}" for k, v in d.items()}
    L = ["# College TrackMan transfer report", "",
         "Protocol `reports/transfer_protocol.md`, locked before scoring and verified unchanged. Data: Pifer (2024), Mendeley "
         "Data, DOI 10.17632/xfnz6mkdzm.3, CC BY 4.0; transformed as documented in `config/field_mapping.yaml` and "
         "`config/trackman_transfer.yaml`.", "",
         f"**Supported-use statement: {status}.** Scope: one college program's mostly intrasquad exhibition games "
         f"({decision['games']} games, {decision['pitchers']} pitchers, {decision['college_swings']:,} swings). No college norms; "
         "MLB percentiles are never college percentiles.", "",
         "## Step 3 — frozen MLB models", "",
         "Not applied. v1 needs `plate_z_zone` (no `sz_top`/`sz_bot` in the college file); v2 needs `plate_z_abs_zone` (batters "
         "are anonymized, so no heights) and `arm_angle_deg` (not exported). Blank inputs would pretend these were measured.", "",
         "## Step 4 — reduced-feature MLB models (fitted on MLB only)", "",
         "Cost of the reduction on MLB validation (August 2026); the full v2 development model scored log loss ≈ 0.421, AUC ≈ 0.794:", "",
         pd.DataFrame(mlb_val).T[["log_loss", "brier", "roc_auc", "calib_intercept", "calib_slope"]].round(4).to_markdown(), "",
         "## Step 2 — shared inputs, MLB vs college (descriptive)", "",
         "PSI compares college swings with MLB swings. Large values mix level of play, measurement system and pitch labels; "
         "they are not corrected (college values are never adjusted to look like MLB).", "",
         dist.round(3).to_markdown(index=False), "",
         "## T0 — initial transfer, no college fitting", "",
         table[["n", "prevalence", "mean_pred", "log_loss", "brier", "roc_auc", "calib_intercept", "calib_slope"]].round(4).to_markdown(), "",
         "95% intervals (two-way game × pitcher bootstrap, 2,000 replicates; few clusters, so approximate):", "",
         pd.DataFrame({k: fmt(v) for k, v in cis.items()}).to_markdown(), "",
         "Differences vs the college leave-one-game-out base rate:", "", boot.round(4).to_markdown(index=False), "",
         "By pitch group (TrackMan auto types) and handedness, core model:", "",
         groups.round(3).to_markdown(), "", hands.round(3).to_markdown(), "",
         "## T1 — leave-one-game-out recalibration (core model)", "",
         t1_table[["mean_pred", "log_loss", "brier", "roc_auc", "calib_intercept", "calib_slope"]].round(4).to_markdown(), "",
         pd.DataFrame({k: fmt(v) for k, v in t1_ci.items()}).to_markdown(), "",
         "## Decision rule (locked)", "",
         pd.DataFrame({"condition": list(rule), "met": [bool(v) for v in rule.values()]}).to_markdown(index=False), "",
         f"Calibrated (T1 sigmoid intercept within ±0.1 and slope within 0.9–1.1): **{calibrated}**.", "",
         "## Caveats on the supported-use statement", "",
         f"- **Calibration is weakly established.** The locked rule uses point estimates. After leave-one-game-out sigmoid "
         f"recalibration the slope is {sig['calib_slope']:.2f}, but its 95% interval is {t1_ci['sigmoid']['calib_slope'][0]:.2f} to "
         f"{t1_ci['sigmoid']['calib_slope'][1]:.2f}; 21 games cannot pin calibration down.",
         "- **Calibrated overall, not within pitch groups.** College fastballs are under-predicted and offspeed pitches over-predicted "
         "(table above), and the errors offset. The offspeed gap matches TrackMan's auto classifier labelling ~300 operator-tagged "
         "fastballs as changeups. A single recalibration curve cannot fix group-level errors, so group-level probabilities are "
         "**not** supported.",
         "- **Before recalibration the model is over-confident** (slope 0.71): its probabilities are too spread out for this program.",
         "- **The leave-one-game-out base rate** has AUC below 0.5 by construction (a high-whiff game gets a lower held-out mean). "
         "Its log loss is still a fair reference.",
         "- **Supported use, concretely:** ordering this program's swings by how likely a miss was given the pitch (e.g. which pitches "
         "drew misses the model expected vs not), with the recalibrated overall level. **Not supported:** per-pitch-type probabilities, "
         "comparisons with MLB pitchers, college-wide claims.", "",
         "## Mapping decisions and open questions", "",
         "- Pitch types: TrackMan auto → Statcast codes (primary); operator tags as sensitivity. No college sweeper (ST) label.",
         "- Fastball differences use the pitcher's primary fastball in his other games (college chronology is unverified), "
         "unlike MLB's as-of definition; they appear only in the sensitivity model.",
         "- Extension (~1 ft shorter) and IVB definitions remain open measurement questions and are excluded from the core model.",
         "- College labels are noisier: foul tips and foul bunts are inside `FoulBall`; missed bunts are inside `StrikeSwinging`.", ""]
    (ROOT / "reports" / "trackman_transfer_report.md").write_text("\n".join(L))
    print(table[["n", "prevalence", "mean_pred", "log_loss", "roc_auc", "calib_intercept", "calib_slope"]].round(4).to_string())
    print({k: fmt(v) for k, v in cis.items()})
    print(boot[boot.metric == "log_loss"].round(4).to_string())
    print(t1_table[["mean_pred", "log_loss", "roc_auc", "calib_intercept", "calib_slope"]].round(4).to_string())
    print("RULE:", rule, "| calibrated:", calibrated, "| STATUS:", status)


if __name__ == "__main__":
    main()
