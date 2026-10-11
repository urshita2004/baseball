"""Tree-ensemble candidates, replacement decision and learning curve (protocol Amendment 2).

All fits use 2025 train; all scores use 2025 validation. September and the sealed season are not read.
Writes models/selection_v1/ (selection.json, train-only selected model) and reports/model_selection_report.md.
Usage: .venv/bin/python src/model_selection.py
"""
import datetime as dt
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from evaluate import metrics, two_way_bootstrap
from train_baselines import CAT, NUM, load, logit_pipeline

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "selection_v1"
SEED = 20250601
REFERENCE = "logit_spline_C1"
SPECS = {
    "logit_spline_C1": {"family": "logit_spline", "C": 1.0},
    "rf_leaf50": {"family": "rf", "min_samples_leaf": 50},
    "rf_leaf200": {"family": "rf", "min_samples_leaf": 200},
    "hgb_iter300": {"family": "hgb", "max_iter": 300},
    "hgb_iter800": {"family": "hgb", "max_iter": 800},
}
WINDOWS_WEEKS = [4, 8, 16, None]  # None = all training data


def tree_pre():
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)
    return ColumnTransformer([("num", "passthrough", NUM), ("cat", enc, CAT)], verbose_feature_names_out=False)


def build(name):
    """Unfitted pipeline for a named specification (shared with final_fit.py)."""
    spec = SPECS[name]
    if spec["family"] == "logit_spline":
        return logit_pipeline(spline=True, C=spec["C"])
    if spec["family"] == "rf":
        clf = RandomForestClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=spec["min_samples_leaf"],
                                     n_jobs=-1, random_state=SEED)
    else:
        cat_mask = [False] * len(NUM) + [True] * len(CAT)
        clf = HistGradientBoostingClassifier(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                             max_iter=spec["max_iter"], early_stopping=False,
                                             categorical_features=cat_mask, random_state=SEED)
    return Pipeline([("pre", tree_pre()), ("clf", clf)])


def main():
    data, levels, split = load()
    train, val = data["train"], data["validation"]
    X_tr, y_tr, X_va, y_va = train[NUM + CAT], train["y"].to_numpy(), val[NUM + CAT], val["y"].to_numpy()

    preds, fitted, timing = {}, {}, {}
    for name in SPECS:
        t = time.time()
        fitted[name] = build(name).fit(X_tr, y_tr)
        preds[name] = fitted[name].predict_proba(X_va)[:, 1]
        timing[name] = round(time.time() - t, 1)
        print(f"{name}: val log loss {metrics(y_va, preds[name])['log_loss']:.5f} ({timing[name]}s)", flush=True)

    table = pd.DataFrame({k: metrics(y_va, v) for k, v in preds.items()}).T
    boot = two_way_bootstrap(y_va, preds, val["game_pk"], val["pitcher_id"], reference=REFERENCE, seed=SEED)
    ll = boot[boot["metric"] == "log_loss"].set_index("model")
    best = table["log_loss"].idxmin()
    replaces = best != REFERENCE and ll.loc[best, "hi95"] < 0
    selected = best if replaces else REFERENCE
    reason = (f"`{best}` has the lowest validation log loss and its improvement over `{REFERENCE}` has a 95% interval below zero"
              if replaces else f"no candidate beat `{REFERENCE}` with a 95% interval below zero; the simpler model stays")

    # Learning curve for the selected specification
    end = pd.Timestamp(split["date_ranges"]["train"][1])
    curve, curve_preds = [], {}
    for w in WINDOWS_WEEKS:
        rows = train if w is None else train[pd.to_datetime(train["game_date"]) > end - pd.Timedelta(weeks=w)]
        label = "all training" if w is None else f"last {w} weeks"
        model = fitted[selected] if w is None else build(selected).fit(rows[NUM + CAT], rows["y"].to_numpy())
        curve_preds[label] = model.predict_proba(X_va)[:, 1]
        curve.append({"window": label, "swings": len(rows), **{k: metrics(y_va, curve_preds[label])[k] for k in ("log_loss", "brier", "roc_auc")}})
        print(f"learning curve {label}: {curve[-1]['log_loss']:.5f}", flush=True)
    curve = pd.DataFrame(curve)
    cb = two_way_bootstrap(y_va, {"all training": curve_preds["all training"], "last 16 weeks": curve_preds["last 16 weeks"]},
                           val["game_pk"], val["pitcher_id"], reference="last 16 weeks", seed=SEED)
    full_vs_16 = cb[cb["metric"] == "log_loss"].iloc[0]
    add_2024 = bool(full_vs_16["difference"] <= -0.001 and full_vs_16["hi95"] < 0)

    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[selected], OUT / "selected_train_only.joblib")
    selection = {"selected": selected, "spec": SPECS[selected], "reason": reason, "reference": REFERENCE,
                 "add_2024": add_2024, "full_vs_16_weeks_log_loss": {k: float(full_vs_16[k]) for k in ("difference", "lo95", "hi95")},
                 "validation_metrics": {k: float(v) for k, v in table.loc[selected].items()},
                 "train_only_artifact": str((OUT / "selected_train_only.joblib").relative_to(ROOT)),
                 "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    (OUT / "selection.json").write_text(json.dumps(selection, indent=2))

    L = ["# Model selection (protocol Amendment 2)", "",
         f"Fitted on 2025 train ({len(train):,} swings), scored on validation {split['date_ranges']['validation'][0]} to "
         f"{split['date_ranges']['validation'][1]} ({len(val):,} swings). September and 2026 were not read. Grids and thresholds "
         "were fixed in Amendment 2 before fitting.", "",
         "## Candidates", "",
         table[["log_loss", "brier", "roc_auc", "avg_precision", "calib_intercept", "calib_slope"]].assign(fit_seconds=pd.Series(timing)).round(5).to_markdown(), "",
         f"### Difference vs `{REFERENCE}` (two-way game × pitcher bootstrap, 300 replicates)", "",
         boot.round(5).to_markdown(index=False), "",
         f"## Decision: **{selected}**", "", f"Reason: {reason}.", "",
         "## Learning curve (selected specification, same validation period)", "",
         curve.round(5).to_markdown(index=False), "",
         f"All training vs last 16 weeks: log-loss difference {full_vs_16['difference']:+.5f} "
         f"(95% interval {full_vs_16['lo95']:+.5f} to {full_vs_16['hi95']:+.5f}). Rule: add 2024 only if ≤ −0.001 with the "
         f"interval below zero. **Add 2024: {'yes' if add_2024 else 'no'}.**", "",
         "Note: shorter windows are also closer in time to validation, so the curve mixes sample size with recency; "
         "a flat curve means more 2025 rows are not helping, so older-season rows are unlikely to either.", ""]
    (ROOT / "reports" / "model_selection_report.md").write_text("\n".join(L))
    print(table[["log_loss", "brier", "roc_auc", "calib_slope"]].round(5).to_string())
    print(ll.round(5).to_string()); print("SELECTED:", selected, "|", reason); print(curve.round(5).to_string()); print("add 2024:", add_2024)


if __name__ == "__main__":
    main()
