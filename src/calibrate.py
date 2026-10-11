"""Calibration decision (protocol Amendment 1).

Candidates are fitted on the train-only model's September (calibration-period) scores and judged on August
(validation), which none of them saw. A calibrator is adopted only if it beats *none* on validation log loss
with a 95% two-way bootstrap interval below zero; otherwise *none* is used. The sealed 2026 season is not read.

Writes models/calibration_v1/ (decision + calibrated candidate) and reports/calibration_report.md.
Usage: .venv/bin/python src/calibrate.py
"""
import datetime as dt
import hashlib
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from evaluate import EPS, metrics, reliability_table, two_way_bootstrap
from train_baselines import CAT, NUM, load

ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "models" / "selection_v1" / "selection.json"
OUT = ROOT / "models" / "calibration_v1"


def base_model_path():
    """Train-only selected model: the tree-model selection if it exists, else the baseline selection."""
    if SELECTION.exists():
        return ROOT / json.loads(SELECTION.read_text())["train_only_artifact"]
    return ROOT / "models" / "baselines_v1" / "selected_pipeline.joblib"
SEED = 20250601


class Calibrator:
    """Maps model probabilities to calibrated probabilities. kind: none | intercept | sigmoid | isotonic."""

    def __init__(self, kind):
        self.kind = kind

    def fit(self, p, y):
        z = logit(np.clip(p, EPS, 1 - EPS))
        y = np.asarray(y, int)
        if self.kind == "intercept":
            self.a_ = brentq(lambda a: expit(a + z).mean() - y.mean(), -5, 5)
        elif self.kind == "sigmoid":
            self.lr_ = LogisticRegression(C=np.inf).fit(z.reshape(-1, 1), y)
        elif self.kind == "isotonic":
            self.iso_ = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip").fit(p, y)
        return self

    def transform(self, p):
        z = logit(np.clip(p, EPS, 1 - EPS))
        if self.kind == "none":
            return np.asarray(p, float)
        if self.kind == "intercept":
            return expit(z + self.a_)
        if self.kind == "sigmoid":
            return self.lr_.predict_proba(z.reshape(-1, 1))[:, 1]
        return self.iso_.predict(p)

    def describe(self):
        if self.kind == "intercept":
            return f"logit shift {self.a_:+.4f}"
        if self.kind == "sigmoid":
            return f"intercept {self.lr_.intercept_[0]:+.4f}, slope {self.lr_.coef_[0, 0]:.4f} on logit"
        return self.kind


class CalibratedModel:
    """Frozen pipeline followed by a calibrator; predict_proba matches the sklearn convention."""

    def __init__(self, pipeline, calibrator):
        self.pipeline, self.calibrator = pipeline, calibrator

    def predict_proba(self, X):
        p = self.calibrator.transform(self.pipeline.predict_proba(X)[:, 1])
        return np.column_stack([1 - p, p])


KINDS = ["none", "intercept", "sigmoid", "isotonic"]


def september_cv(p, y, games, folds=5):
    """Game-grouped CV within September: mean held-out log loss per candidate."""
    out = {k: [] for k in KINDS}
    for tr, te in GroupKFold(n_splits=folds).split(p, y, groups=games):
        for k in KINDS:
            q = np.clip(Calibrator(k).fit(p[tr], y[tr]).transform(p[te]), EPS, 1 - EPS)
            out[k].append(-np.mean(y[te] * np.log(q) + (1 - y[te]) * np.log(1 - q)))
    return {k: float(np.mean(v)) for k, v in out.items()}


def main():
    data, levels, split = load(parts=("validation", "calibration"))
    val, cal = data["validation"], data["calibration"]
    BASE = base_model_path()
    model = joblib.load(BASE)
    p_val = model.predict_proba(val[NUM + CAT])[:, 1]
    p_cal = model.predict_proba(cal[NUM + CAT])[:, 1]
    y_val, y_cal = val["y"].to_numpy(), cal["y"].to_numpy()

    fitted = {k: Calibrator(k).fit(p_cal, y_cal) for k in KINDS}
    val_preds = {k: c.transform(p_val) for k, c in fitted.items()}
    table = pd.DataFrame({k: metrics(y_val, v) for k, v in val_preds.items()}).T
    boot = two_way_bootstrap(y_val, val_preds, val["game_pk"], val["pitcher_id"], reference="none", seed=SEED)
    cv = september_cv(p_cal, y_cal, cal["game_pk"].to_numpy())
    sept_uncal = metrics(y_cal, p_cal)

    # Decision rule (Amendment 1)
    ll = boot[boot["metric"] == "log_loss"].set_index("model")
    candidates = [k for k in KINDS[1:] if ll.loc[k, "hi95"] < 0]
    if "isotonic" in candidates and cv["isotonic"] >= cv["sigmoid"]:
        candidates.remove("isotonic")
    chosen = min(candidates, key=lambda k: table.loc[k, "log_loss"]) if candidates else "none"
    reason = (f"`{chosen}` improves validation log loss with a 95% interval below zero" if chosen != "none" else
              "no candidate improved validation log loss with a 95% interval below zero; the simpler option wins")

    OUT.mkdir(parents=True, exist_ok=True)
    calibrated = CalibratedModel(model, fitted[chosen])
    joblib.dump(calibrated, OUT / "calibrated_candidate.joblib")
    decision = {"chosen": chosen, "reason": reason, "calibrator": fitted[chosen].describe(),
                "fitted_on": {"partition": "calibration", "dates": split["date_ranges"]["calibration"], "swings": int(len(cal))},
                "judged_on": {"partition": "validation", "dates": split["date_ranges"]["validation"], "swings": int(len(val))},
                "base_model_sha256": hashlib.sha256(BASE.read_bytes()).hexdigest(),
                "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "note": "Decision rule: evaluation_protocol.md Amendment 1. Chosen type is refit on the final (train+validation) model in the final fit."}
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, (title, y, preds) in zip(axes, [("September (fit period), train-only model", y_cal, {"none": p_cal, **{k: fitted[k].transform(p_cal) for k in ["intercept", "sigmoid"]}}),
                                            ("August validation (held out)", y_val, {k: val_preds[k] for k in ["none", "intercept", "sigmoid", "isotonic"]})]):
        for k, p in preds.items():
            r = reliability_table(y, p)
            ax.plot(r["mean_pred"], r["observed"], marker="o", ms=3, label=k)
        ax.plot([0, 0.75], [0, 0.75], color="grey", lw=0.8, ls="--")
        ax.set_title(title, fontsize=9); ax.set_xlabel("mean predicted"); ax.set_ylabel("observed whiff rate"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(ROOT / "reports" / "figures" / "calibration.png", dpi=130); plt.close(fig)

    show = table[["mean_pred", "prevalence", "log_loss", "brier", "roc_auc", "calib_intercept", "calib_slope"]].copy()
    L = ["# Calibration decision (protocol Amendment 1)", "",
         f"Base model: `{BASE.relative_to(ROOT)}` (fitted on train only). "
         f"Calibrators fitted on September ({split['date_ranges']['calibration'][0]} to {split['date_ranges']['calibration'][1]}, "
         f"{len(cal):,} swings) and judged on August validation ({len(val):,} swings). The 2026 sealed season was not read.", "",
         "## What September looks like to the uncalibrated model", "",
         f"Observed whiff rate {sept_uncal['prevalence']:.4f} vs mean prediction {sept_uncal['mean_pred']:.4f}; calibration "
         f"intercept {sept_uncal['calib_intercept']:+.3f}, slope {sept_uncal['calib_slope']:.3f}. "
         "So the model under-predicts September slightly: some of September's higher whiff rate is not explained by the pitch features.", "",
         "## Candidates judged on August (held out from every calibrator)", "",
         show.round(5).to_markdown(), "",
         "Calibration changes probabilities, not ranking: AUC is identical for monotone calibrators (isotonic can tie ranks).", "",
         "### Difference vs *none* (two-way game × pitcher bootstrap)", "",
         boot.round(5).to_markdown(index=False), "",
         "### Game-grouped 5-fold CV within September (held-out log loss)", "",
         pd.Series(cv).rename("log loss").round(5).to_frame().to_markdown(), "",
         "Within September every calibrator helps, as expected: that is the period they are fitted to. The decision uses August "
         "because the 2026 test is a full season, not a September.", "",
         f"## Decision: **{chosen}**", "", f"Reason: {reason}.", "",
         f"Calibrator: {fitted[chosen].describe()}. Saved as `models/calibration_v1/calibrated_candidate.joblib`; "
         "decision record `models/calibration_v1/decision.json`.", "",
         "![calibration](figures/calibration.png)", ""]
    (ROOT / "reports" / "calibration_report.md").write_text("\n".join(L))
    print(show.round(5).to_string()); print(boot[boot.metric == "log_loss"].round(5).to_string()); print("CV", {k: round(v, 5) for k, v in cv.items()})
    print("September uncalibrated:", {k: round(sept_uncal[k], 4) for k in ["prevalence", "mean_pred", "calib_intercept", "calib_slope"]})
    print("CHOSEN:", chosen, "|", reason)


if __name__ == "__main__":
    main()
