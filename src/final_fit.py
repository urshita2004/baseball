"""Final fit: refit the selected specification on 2025 train + validation, apply the calibration decision,
and freeze the artifact (protocol §5 step 5, Amendments 1-2).

Refuses to run if the calibration decision was not made on the selected model.
Writes models/final_v1/final_model.joblib and models/final_v1/model_card.json.
Usage: .venv/bin/python src/final_fit.py
"""
import datetime as dt
import hashlib
import json
import platform
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from calibrate import Calibrator, CalibratedModel
from evaluate import metrics
from model_selection import build
from train_baselines import CAT, NUM, load

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "final_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    selection = json.loads((ROOT / "models" / "selection_v1" / "selection.json").read_text())
    decision = json.loads((ROOT / "models" / "calibration_v1" / "decision.json").read_text())
    if decision["base_model_sha256"] != sha(ROOT / selection["train_only_artifact"]):
        raise SystemExit("calibration decision was made on a different model; rerun src/calibrate.py")
    data, levels, split = load(parts=("validation", "calibration"))
    fit_rows = pd.concat([data["train"], data["validation"]])
    cal = data["calibration"]

    pipeline = build(selection["selected"]).fit(fit_rows[NUM + CAT], fit_rows["y"].to_numpy())
    calibrator = Calibrator(decision["chosen"])
    if decision["chosen"] != "none":
        calibrator.fit(pipeline.predict_proba(cal[NUM + CAT])[:, 1], cal["y"].to_numpy())
    model = CalibratedModel(pipeline, calibrator)

    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT / "final_model.joblib")
    reloaded = joblib.load(OUT / "final_model.joblib")
    probe = cal.sample(2000, random_state=20250601)[NUM + CAT]
    reproduces = bool((reloaded.predict_proba(probe) == model.predict_proba(probe)).all())
    # September was not used to fit the model; when no calibrator is fitted it is a fully held-out later period.
    sept = metrics(cal["y"].to_numpy(), model.predict_proba(cal[NUM + CAT])[:, 1])
    lock = json.loads((ROOT / "data" / "splits" / "protocol_lock.json").read_text())
    card = {
        "model": selection["selected"], "spec": selection["spec"], "version": "final_v1",
        "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "target": "P(miss | ordinary non-bunt swing); miss = swinging_strike, swinging_strike_blocked; contact = foul, foul_tip, hit_into_play",
        "features_in_order": NUM + CAT, "pitch_type_levels": levels,
        "units": "mph, rpm, ft (release/plate, catcher's view; *_arm arm-side +; plate_x_away + away from batter), in (movement)",
        "fit_on": {"partitions": ["train", "validation"], "dates": [split["date_ranges"]["train"][0], split["date_ranges"]["validation"][1]],
                   "swings": int(len(fit_rows)), "misses": int(fit_rows["y"].sum()), "pitchers": int(fit_rows["pitcher_id"].nunique())},
        "calibration": {"decision": decision["chosen"], "reason": decision["reason"], "calibrator": calibrator.describe()},
        "september_check": {k: float(v) for k, v in sept.items()},
        "september_check_note": ("September was not used in fitting (no calibrator), so this is a later held-out period"
                                 if decision["chosen"] == "none" else "September was used to fit the calibrator; not held out"),
        "reload_reproduces_predictions": reproduces,
        "artifact_sha256": sha(OUT / "final_model.joblib"),
        "protocol_lock_sha256": sha(ROOT / "data" / "splits" / "protocol_lock.json"),
        "protocol_sha256": lock["sha256"]["reports/evaluation_protocol.md"],
        "environment": {"python": platform.python_version(), "scikit-learn": sklearn.__version__},
        "intended_use": "Retrospective pitch analysis of MLB swings. Not for college players, not causal, not a coaching prescription.",
    }
    (OUT / "model_card.json").write_text(json.dumps(card, indent=2))
    print(f"froze {selection['selected']} (calibration: {decision['chosen']}); sha {card['artifact_sha256'][:16]}; reproduces {reproduces}")
    print("September check:", {k: round(v, 4) for k, v in sept.items() if k in ("log_loss", "brier", "roc_auc", "calib_intercept", "calib_slope", "prevalence", "mean_pred")})


if __name__ == "__main__":
    main()
