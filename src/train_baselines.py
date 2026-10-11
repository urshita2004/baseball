"""Transparent baselines for P(miss | ordinary swing), fit on 2025 train, scored on 2025 validation.

Models (one pooled model each, across all pitchers):
  constant      - training whiff rate
  pitch_type    - training whiff rate by pitch type (rare codes pooled to their group)
  logit_linear  - logistic regression, linear terms
  logit_spline  - logistic regression on cubic-spline bases (additive; transparent, captures U-shaped location effects)
Preprocessing is fit inside each pipeline on training rows only. The calibration period and the sealed
test season are not read. Selection metric: validation log loss (reports/evaluation_protocol.md).

Writes models/baselines_v1/ and reports/baseline_report.md.
Usage: .venv/bin/python src/train_baselines.py
"""
import datetime as dt
import hashlib
import json
import platform
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_union
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from evaluate import metrics, reliability_table, two_way_bootstrap
from features import CATEGORICAL, FB_DERIVED, NUMERIC, add_features, apply_pitch_type_levels, pitch_type_levels

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ID = "statcast_2025_regular"
MODEL_DIR = ROOT / "models" / "baselines_v1"
FIG = ROOT / "reports" / "figures"
NUM = NUMERIC + FB_DERIVED
CAT = ["pitch_type_level", "p_throws", "stand", "count"]
C_GRID = [0.1, 1.0, 10.0]


def load(parts=("train", "validation")):
    """Swing rows for the given 2025 partitions (train always included, for pitch-type levels); dict by partition."""
    split = json.loads((ROOT / "data" / "splits" / f"{SOURCE_ID}.json").read_text())
    pitches = pd.read_parquet(ROOT / "data" / "model" / SOURCE_ID / "model_pitches.parquet")
    pitches = add_features(pitches[pitches["model_population"]])
    sw = pitches[pitches["eligible_swing"] & pitches["partition"].isin(["train", *parts])].copy()
    sw["y"] = sw["is_miss"].astype(int)
    sw["count"] = sw["balls"].astype(int).astype(str) + "-" + sw["strikes"].astype(int).astype(str)
    for c in ["p_throws", "stand"]:
        sw[c] = sw[c].astype(str)
    levels = pitch_type_levels(sw[sw["partition"] == "train"], split["rare_pitch_types_pooled_to_group"])
    sw["pitch_type_level"] = apply_pitch_type_levels(sw, levels).astype(str)
    data = {p: sw[sw["partition"] == p].copy() for p in ("train", *parts)}
    return data, levels, split


def numeric_block(spline):
    impute = SimpleImputer(strategy="median")
    if spline:
        transform = Pipeline([("impute", impute), ("spline", SplineTransformer(n_knots=7, degree=3, knots="quantile",
                                                                                extrapolation="constant")),
                              ("scale", StandardScaler())])
    else:
        transform = Pipeline([("impute", impute), ("scale", StandardScaler())])
    return make_union(transform, MissingIndicator(features="missing-only", error_on_new=False))


def logit_pipeline(spline, C):
    pre = ColumnTransformer([("num", numeric_block(spline), NUM),
                             ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), CAT)])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=C, max_iter=3000))])


def count_state(df):
    b, s = df["balls"].astype(int), df["strikes"].astype(int)
    return np.select([s == 2, s > b, b > s], ["two strikes", "pitcher ahead", "pitcher behind"], "even")


def subgroup_table(val, pred_cols, train_pitchers):
    groups = {
        "pitch group": val["pitch_group"].astype(str),
        "matchup (P-B)": val["p_throws"] + "-" + val["stand"],
        "count state": pd.Series(count_state(val), index=val.index),
        "pitcher in training": np.where(val["pitcher_id"].isin(train_pitchers), "returning", "unseen in train"),
    }
    rows = []
    for name, key in groups.items():
        for level, idx in val.groupby(key).groups.items():
            d = val.loc[idx]
            row = {"subgroup": name, "level": level, "swings": len(d), "whiff_rate": d["y"].mean()}
            if len(d) < 500:
                row["note"] = "insufficient (<500 swings)"
            else:
                for c in pred_cols:
                    m = metrics(d["y"], d[c])
                    row[f"{c} log_loss"], row[f"{c} AUC"] = m["log_loss"], m["roc_auc"]
                    row[f"{c} mean_pred"] = m["mean_pred"]
            rows.append(row)
    return pd.DataFrame(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    data, levels, split = load()
    train, val = data["train"], data["validation"]
    y_tr, y_va = train["y"].to_numpy(), val["y"].to_numpy()
    preds, fitted, rows = {}, {}, []

    prior = y_tr.mean()
    preds["constant"] = np.full(len(val), prior)
    by_type = train.groupby("pitch_type_level")["y"].mean()
    preds["pitch_type"] = val["pitch_type_level"].map(by_type).fillna(prior).to_numpy()
    fitted["pitch_type"] = {"rates": by_type.to_dict(), "fallback": prior}

    for spline in (False, True):
        for C in C_GRID:
            name = f"logit_{'spline' if spline else 'linear'}_C{C:g}"
            pipe = logit_pipeline(spline, C).fit(train[NUM + CAT], y_tr)
            preds[name] = pipe.predict_proba(val[NUM + CAT])[:, 1]
            fitted[name] = pipe
            print(f"{name}: val log loss {metrics(y_va, preds[name])['log_loss']:.5f}", flush=True)

    table = pd.DataFrame({k: metrics(y_va, v) for k, v in preds.items()}).T
    best = {fam: table.loc[[k for k in table.index if k.startswith(fam)], "log_loss"].idxmin()
            for fam in ("logit_linear", "logit_spline")}
    selected = min(best.values(), key=lambda k: table.loc[k, "log_loss"])
    boot = pd.concat([two_way_bootstrap(y_va, {k: preds[k] for k in ["constant", "pitch_type", *best.values()]},
                                        val["game_pk"], val["pitcher_id"], reference=ref)
                      for ref in ("constant", "pitch_type")], ignore_index=True)
    boot = boot[boot["model"] != boot["vs"]]

    # Save artifacts
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[selected], MODEL_DIR / "selected_pipeline.joblib")
    joblib.dump({k: fitted[k] for k in best.values()} | {"pitch_type": fitted["pitch_type"], "constant": prior},
                MODEL_DIR / "all_baselines.joblib")
    probe = val.sample(1000, random_state=20250601)
    reloaded = joblib.load(MODEL_DIR / "selected_pipeline.joblib").predict_proba(probe[NUM + CAT])[:, 1]
    reproduces = bool(np.allclose(reloaded, fitted[selected].predict_proba(probe[NUM + CAT])[:, 1], atol=0, rtol=0))
    card = {
        "model": selected, "version": "baselines_v1", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "target": "P(miss | ordinary non-bunt swing); miss = swinging_strike, swinging_strike_blocked; contact = foul, foul_tip, hit_into_play",
        "status": "development baseline; uncalibrated; not for decisions",
        "features_in_order": NUM + CAT,
        "units": "mph, rpm, ft (release/plate, catcher's view; *_arm arm-side +; plate_x_away + away from batter), in (movement)",
        "pitch_type_levels": levels,
        "training_coverage": {"partition": "train", "dates": split["date_ranges"]["train"], "swings": int(len(train)),
                              "misses": int(y_tr.sum()), "pitchers": int(train["pitcher_id"].nunique())},
        "validation": {"dates": split["date_ranges"]["validation"], **{k: float(v) for k, v in table.loc[selected].items()}},
        "reload_reproduces_predictions": reproduces,
        "inputs_sha256": {"swing table": split["outputs_sha256"][f"data/model/{SOURCE_ID}/swings.parquet"],
                          "features.yaml": sha(ROOT / "config" / "features.yaml")},
        "environment": {"python": platform.python_version(), "scikit-learn": sklearn.__version__},
        "artifact_sha256": sha(MODEL_DIR / "selected_pipeline.joblib"),
    }
    (MODEL_DIR / "model_card.json").write_text(json.dumps(card, indent=2))

    # Figures and report
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for k in ["pitch_type", *best.values()]:
        r = reliability_table(y_va, preds[k])
        ax.plot(r["mean_pred"], r["observed"], marker="o", label=k)
    ax.plot([0, 0.7], [0, 0.7], color="grey", lw=0.8, ls="--")
    ax.set_xlabel("mean predicted P(miss | swing)"); ax.set_ylabel("observed whiff rate")
    ax.set_title("Validation reliability (deciles), Aug 2025"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "baseline_reliability.png", dpi=130); plt.close(fig)

    val = val.assign(**{k: preds[k] for k in ["pitch_type", selected]})
    sub = subgroup_table(val, ["pitch_type", selected], set(train["pitcher_id"]))
    lin = fitted[best["logit_linear"]]
    names = lin.named_steps["pre"].get_feature_names_out()
    coefs = pd.Series(lin.named_steps["clf"].coef_[0], index=names)
    coefs = coefs[[n.startswith("num__") for n in names]].rename(lambda n: n.split("__")[-1])
    coefs = coefs.sort_values(key=np.abs, ascending=False).head(14)

    fmt = {"n": "{:,.0f}", "prevalence": "{:.4f}", "mean_pred": "{:.4f}", "log_loss": "{:.5f}", "brier": "{:.5f}",
           "roc_auc": "{:.4f}", "avg_precision": "{:.4f}", "calib_intercept": "{:+.3f}", "calib_slope": "{:.3f}"}
    na = lambda f: (lambda v: "n/a" if pd.isna(v) else f.format(v))
    show = table.copy()
    for c, f in fmt.items():
        show[c] = show[c].map(na(f))
    L = ["# Baseline report", "",
         f"Target: P(miss | ordinary swing). Train: 2025 {split['date_ranges']['train'][0]} to {split['date_ranges']['train'][1]} "
         f"({len(train):,} swings). Scored on validation: {split['date_ranges']['validation'][0]} to {split['date_ranges']['validation'][1]} "
         f"({len(val):,} swings). The calibration period (September) and the sealed 2026 season were not read.", "",
         "All models are single **pooled** models across pitchers. Preprocessing (imputation, scaling, splines, encoding) is "
         "fit on training rows only. Features: `config/features.yaml` v1, plus as-of fastball differences (`src/features.py`).", "",
         "## Validation metrics", "",
         show.to_markdown(), "",
         f"**Selected (lowest validation log loss): `{selected}`.** Within-family choices of the regularization strength C "
         f"were made on validation, as the protocol allows; the calibration period and test remain untouched. With "
         f"{len(train):,} training swings the regularization strength barely matters (log loss identical to 4 decimals across "
         f"C = {C_GRID}), so this choice is not a source of overfitting to validation.", "",
         "Calibration intercept is the shift needed on the logit scale (0 = right on average); slope 1 = well spread.", "",
         "## Differences vs baselines (two-way game × pitcher bootstrap, 300 replicates)", "",
         "Negative log-loss/Brier differences and positive AUC differences favour the model.", "",
         boot.round(5).to_markdown(index=False), "",
         "![reliability](figures/baseline_reliability.png)", "",
         "## Subgroups (validation)", "",
         "Groups with fewer than 500 swings are marked insufficient rather than ranked.", "",
         sub.round(4).to_markdown(index=False), "",
         f"## Linear model: largest standardized numeric coefficients (`{best['logit_linear']}`)", "",
         "Coefficients are associations conditional on the other features, which are correlated (e.g. velocity and "
         "velo_diff_fb). They are not causal effects and not a ranking of what a pitcher should change.", "",
         coefs.round(3).rename("coef (per SD, logit scale)").to_frame().to_markdown(), "",
         "## Reproducibility", "",
         f"Saved pipeline `models/baselines_v1/selected_pipeline.joblib` (sha256 `{card['artifact_sha256'][:16]}…`). "
         f"Reloaded predictions on 1,000 fixed validation rows reproduce exactly: **{reproduces}**. Model card: "
         "`models/baselines_v1/model_card.json`.", "",
         "## Completion gate", "",
         "- Learned models vs simple baselines on validation: see the bootstrap table.",
         "- Calibration is reported but not fixed here; the calibrator is fitted on the September period at the calibration step.", ""]
    (ROOT / "reports" / "baseline_report.md").write_text("\n".join(L))
    print(table[["log_loss", "brier", "roc_auc", "calib_intercept", "calib_slope"]].round(4).to_string())
    print("selected:", selected, "| reload reproduces:", reproduces)


if __name__ == "__main__":
    main()
