"""One-time evaluation of frozen model v2 on the sealed 2027 regular season (reports/evaluation_protocol_v2.md §5).

Order: verify the v2 lock and artifact -> label-free drift gate (2026 vs target; refuses to score if any v2 input is
flagged) -> log the opening -> score once. As-of fastball references carry pitcher history from 2025-2026, as in
training. Metrics, baselines, bootstrap, subgroups and criteria are those of protocol v1 §6-§7 (shared code).

  --dry-run   run every step with 2026 (development) as the target; writes reports/final_test_v2_dry_run.md
Usage: .venv/bin/python src/final_test_v2.py --dry-run
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml

import normalize
from build_dataset import apply_model_population
from drift_check import gate
from evaluate import metrics, two_way_bootstrap
from features import add_batter_height_zone, add_features, apply_pitch_type_levels
from final_test import REPS, SEED, criteria, subgroups
from model_v2 import CAT, NUM_V2

ROOT = Path(__file__).resolve().parents[1]
SEALED_SOURCE, DEV_TARGET, REFERENCE = "statcast_2027_sealed", "statcast_2026", "statcast_2026"
OPEN_LOG = ROOT / "data" / "sealed" / "manifests" / "test_open_log_v2.json"
NUMERIC_INPUTS = list(yaml.safe_load((ROOT / "config" / "features_v2.yaml").read_text())["allow"]["numeric"])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(card):
    lock = json.loads((ROOT / "data" / "splits" / "protocol_lock_v2.json").read_text())
    changed = [f for f, h in lock["sha256"].items() if sha(ROOT / f) != h]
    if changed:
        raise SystemExit(f"v2 locked files changed: {changed}")
    if sha(ROOT / "models" / "final_v2" / "final_model.joblib") != card["artifact_sha256"]:
        raise SystemExit("frozen v2 artifact does not match its model card")


def target_population(source):
    cfg = yaml.safe_load((ROOT / "config" / "data_sources.yaml").read_text())[source]
    contract = yaml.safe_load((ROOT / "config" / "field_mapping.yaml").read_text())
    split_cfg = yaml.safe_load((ROOT / "config" / "splits.yaml").read_text())
    raw, manifest = normalize.load_raw(ROOT / cfg.get("manifest_dir", "data/manifests") / f"{cfg['source_id']}.json")
    df = normalize.normalize(raw, contract, cfg["source_id"])
    df, people = apply_model_population(df, split_cfg, cfg["source_id"])
    return df, manifest, people


def main(dry_run=False, override_reason=None):
    card = json.loads((ROOT / "models" / "final_v2" / "model_card.json").read_text())
    verify(card)
    source = DEV_TARGET if dry_run else SEALED_SOURCE
    target, manifest, people = target_population(source)

    # Drift gate: inputs only, before any outcome is read or any score computed.
    ref_cfg = yaml.safe_load((ROOT / "config" / "data_sources.yaml").read_text())[REFERENCE]
    ref = pd.read_parquet(ROOT / ref_cfg["processed_path"]).query("status == 'retained'")
    cache = "batters" if dry_run else f"batters_{source}"
    if not dry_run:
        from players import people_snapshot
        people_snapshot(target["batter_id"].dropna().unique(), cache, refresh=True)
    _, _, flagged = gate(add_batter_height_zone(ref), add_batter_height_zone(target[target["status"] == "retained"], cache), NUMERIC_INPUTS)
    if flagged:
        raise SystemExit(f"drift gate: flagged inputs {flagged}; investigate (no outcomes read) and amend the protocol before scoring")

    log = json.loads(OPEN_LOG.read_text()) if OPEN_LOG.exists() else []
    if not dry_run:
        if any("criteria" in e for e in log) and not override_reason:
            raise SystemExit("sealed v2 test already scored; refusing without --override-reason")
        log.append({"opened_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                    "model_sha256": card["artifact_sha256"], "override_reason": override_reason, "drift_gate": "passed"})
        OPEN_LOG.write_text(json.dumps(log, indent=2))

    history = []
    for s in ("statcast_2025_regular", "statcast_2026_regular"):
        m = pd.read_parquet(ROOT / "data" / "model" / s / "model_pitches.parquet")
        if s.replace("_regular", "") != source:  # the target season is not its own history
            history.append(m[m["model_population"]].drop(columns=["partition"]))
    tgt = target[target["model_population"]].assign(_target=True)
    pitches = add_features(pd.concat([*(h.assign(_target=False) for h in history), tgt], ignore_index=True))
    pitches = add_batter_height_zone(pitches[pitches["_target"]], cache)
    sw = pitches[pitches["eligible_swing"]].copy()
    sw["y"] = sw["is_miss"].astype(int)
    sw["count"] = sw["balls"].astype(int).astype(str) + "-" + sw["strikes"].astype(int).astype(str)
    for c in ["p_throws", "stand"]:
        sw[c] = sw[c].astype(str)
    sw["pitch_type_level"] = apply_pitch_type_levels(sw, card["pitch_type_levels"]).astype(str)

    model = joblib.load(ROOT / "models" / "final_v2" / "final_model.joblib")
    dev = pd.read_parquet(ROOT / "data" / "model" / "v2" / "swings_v2.parquet",
                          columns=["partition", "pitch_type_level", "y", "pitcher_id"])
    fit = dev[dev["partition"].isin(["train", "validation"])]
    sw["final"] = model.predict_proba(sw[NUM_V2 + CAT])[:, 1]
    sw["constant"] = fit["y"].mean()
    sw["pitch_type"] = sw["pitch_type_level"].map(fit.groupby("pitch_type_level")["y"].mean()).fillna(fit["y"].mean())
    y = sw["y"].to_numpy()
    table = pd.DataFrame({k: metrics(y, sw[k]) for k in ("constant", "pitch_type", "final")}).T
    preds = {k: sw[k].to_numpy() for k in ("constant", "pitch_type", "final")}
    boot = pd.concat([two_way_bootstrap(y, preds, sw["game_pk"], sw["pitcher_id"], reference=r, reps=REPS, seed=SEED)
                      for r in ("constant", "pitch_type")], ignore_index=True)
    boot = boot[boot["model"] == "final"]
    crit = criteria(boot, table.loc["final"])
    sub = subgroups(sw, set(dev["pitcher_id"]))

    tag = "final_test_v2_dry_run" if dry_run else "final_test_v2"
    head = ("# DRY RUN — v2 final-test code with 2026 as target (2026 is development data; NOT a test result)" if dry_run
            else "# Sealed test — model v2 on the 2027 MLB regular season")
    L = [head, "", f"Model v2 (`models/final_v2`, sha256 `{card['artifact_sha256'][:16]}…`), calibration {card['calibration']['decision']}. "
         f"Drift gate vs {REFERENCE}: passed. {manifest['games_in_data']:,} games; {len(sw):,} model-population swings.", "",
         "## Metrics", "", table.round(5).to_markdown(), "",
         "## vs baselines (two-way bootstrap)", "", boot.round(5).to_markdown(index=False), "",
         "## Criteria (protocol v1 §7)", "", pd.DataFrame({"criterion": list(crit), "met": list(crit.values())}).to_markdown(index=False), "",
         "## Subgroups", "", sub.round(4).to_markdown(index=False), ""]
    report = ROOT / "reports" / f"{tag}.md"
    report.write_text("\n".join(L))
    if not dry_run:
        log[-1].update({"report": str(report.relative_to(ROOT)), "report_sha256": sha(report), "criteria": crit,
                        "metrics": {k: float(v) for k, v in table.loc["final"].items()}})
        OPEN_LOG.write_text(json.dumps(log, indent=2))
    print(table.round(5).to_string()); print(crit)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--override-reason")
    main(**vars(ap.parse_args()))
