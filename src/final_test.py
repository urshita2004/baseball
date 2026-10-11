"""One-time evaluation of the frozen model on the sealed 2026 regular season (protocol §6-§8).

Before scoring it verifies the protocol lock (every locked file unchanged) and the frozen artifact checksum.
The sealed season is processed with the same code as development (normalize, model population, features).

  --dry-run   run every step on the 2025 development season instead (results are in-sample and meaningless;
              the point is to find defects without touching 2026). Writes reports/final_test_dry_run.md.
  (default)   score 2026 once. Each opening is logged in data/sealed/manifests/test_open_log.json; a second
              scoring is refused unless --override-reason "<why>" is given, and the reason is printed in the report.

Usage: .venv/bin/python src/final_test.py --dry-run
       .venv/bin/python src/final_test.py
"""
import argparse
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
import yaml

import normalize
from build_dataset import apply_model_population
from evaluate import metrics, reliability_table, two_way_bootstrap
from features import add_features, apply_pitch_type_levels
from train_baselines import CAT, NUM, count_state

ROOT = Path(__file__).resolve().parents[1]
SEALED_SOURCE = "statcast_2026_sealed_v1_retired"  # v1 test (scored 2026-10-11); kept for the record
DEV_SOURCE = "statcast_2025"
OPEN_LOG = ROOT / "data" / "sealed" / "manifests" / "test_open_log.json"
SEED, REPS, MIN_GROUP = 20250601, 1000, 500


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_locks(card):
    lock = json.loads((ROOT / "data" / "splits" / "protocol_lock.json").read_text())
    changed = [f for f, h in lock["sha256"].items() if sha(ROOT / f) != h]
    if changed:
        raise SystemExit(f"locked files changed: {changed}")
    if sha(ROOT / "models" / "final_v1" / "final_model.joblib") != card["artifact_sha256"]:
        raise SystemExit("frozen model artifact does not match its model card")
    if sha(ROOT / "data" / "splits" / "protocol_lock.json") != card["protocol_lock_sha256"]:
        raise SystemExit("protocol lock changed after the model was frozen")


def prepare(source):
    """Raw manifest -> normalized pitches -> model population -> as-of features -> swing rows."""
    cfg_all = yaml.safe_load((ROOT / "config" / "data_sources.yaml").read_text())
    ds = cfg_all[source]
    contract = yaml.safe_load((ROOT / "config" / "field_mapping.yaml").read_text())
    split_cfg = yaml.safe_load((ROOT / "config" / "splits.yaml").read_text())
    manifest_path = ROOT / ds.get("manifest_dir", "data/manifests") / f"{ds['source_id']}.json"
    raw, manifest = normalize.load_raw(manifest_path)
    df = normalize.normalize(raw, contract, ds["source_id"])
    assert len(df) == manifest["total_rows"], "row count drifted from manifest"
    df, people_meta = apply_model_population(df, split_cfg, ds["source_id"])
    pitches = add_features(df[df["model_population"]])
    sw = pitches[pitches["eligible_swing"]].copy()
    sw["y"] = sw["is_miss"].astype(int)
    sw["count"] = sw["balls"].astype(int).astype(str) + "-" + sw["strikes"].astype(int).astype(str)
    for c in ["p_throws", "stand"]:
        sw[c] = sw[c].astype(str)
    population = df["model_exclusion"].replace("", "(in model population)").value_counts()
    return sw, manifest, people_meta, population


def criteria(boot, test_metrics):
    def beats(ref, m):
        r = boot[(boot["vs"] == ref) & (boot["model"] == "final") & (boot["metric"] == m)].iloc[0]
        return bool(r["hi95"] < 0)
    return {
        "beats constant on log loss (95% CI below 0)": beats("constant", "log_loss"),
        "beats constant on Brier (95% CI below 0)": beats("constant", "brier"),
        "beats pitch-type baseline on log loss (95% CI below 0)": beats("pitch_type", "log_loss"),
        "beats pitch-type baseline on Brier (95% CI below 0)": beats("pitch_type", "brier"),
        "calibration slope within 0.9-1.1": bool(0.9 <= test_metrics["calib_slope"] <= 1.1),
        "calibration intercept within ±0.1": bool(abs(test_metrics["calib_intercept"]) <= 0.1),
    }


def subgroups(sw, returning_pitchers):
    keys = {
        "pitch group": sw["pitch_group"].astype(str),
        "matchup (P-B)": sw["p_throws"] + "-" + sw["stand"],
        "count state": pd.Series(count_state(sw), index=sw.index),
        "month": pd.to_datetime(sw["game_date"]).dt.strftime("%Y-%m"),
        "pitcher history": pd.Series(np.where(sw["pitcher_id"].isin(returning_pitchers), "returning (2025 swings)", "new (no 2025 swings)"), index=sw.index),
    }
    rows = []
    for name, key in keys.items():
        for level, idx in sw.groupby(key).groups.items():
            d = sw.loc[idx]
            row = {"subgroup": name, "level": level, "swings": len(d), "whiff_rate": d["y"].mean()}
            if len(d) < MIN_GROUP:
                row["note"] = f"insufficient (<{MIN_GROUP} swings)"
            else:
                for c in ("final", "pitch_type"):
                    m = metrics(d["y"], d[c])
                    row[f"{c} log_loss"], row[f"{c} AUC"] = m["log_loss"], m["roc_auc"]
                row["final mean_pred"] = d["final"].mean()
                row["final calib_slope"] = metrics(d["y"], d["final"])["calib_slope"]
            rows.append(row)
    return pd.DataFrame(rows)


def main(dry_run=False, override_reason=None):
    card = json.loads((ROOT / "models" / "final_v1" / "model_card.json").read_text())
    verify_locks(card)
    source = DEV_SOURCE if dry_run else SEALED_SOURCE
    log = json.loads(OPEN_LOG.read_text()) if OPEN_LOG.exists() else []
    if not dry_run:
        if log and not override_reason:
            raise SystemExit(f"sealed test already scored {len(log)} time(s); refusing without --override-reason")
        log.append({"opened_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                    "model_sha256": card["artifact_sha256"], "override_reason": override_reason})
        OPEN_LOG.write_text(json.dumps(log, indent=2))  # logged before any 2026 row is read

    model = joblib.load(ROOT / "models" / "final_v1" / "final_model.joblib")
    baselines = joblib.load(ROOT / "models" / "baselines_v1" / "all_baselines.joblib")
    sw, manifest, people_meta, population = prepare(source)
    sw["pitch_type_level"] = apply_pitch_type_levels(sw, card["pitch_type_levels"]).astype(str)
    sw["final"] = model.predict_proba(sw[NUM + CAT])[:, 1]
    sw["constant"] = baselines["constant"]
    pt = baselines["pitch_type"]
    sw["pitch_type"] = sw["pitch_type_level"].map(pt["rates"]).fillna(pt["fallback"])

    y = sw["y"].to_numpy()
    table = pd.DataFrame({k: metrics(y, sw[k]) for k in ("constant", "pitch_type", "final")}).T
    preds = {k: sw[k].to_numpy() for k in ("constant", "pitch_type", "final")}
    boot = pd.concat([two_way_bootstrap(y, preds, sw["game_pk"], sw["pitcher_id"], reference=ref, reps=REPS, seed=SEED)
                      for ref in ("constant", "pitch_type")], ignore_index=True)
    boot = boot[boot["model"] == "final"]
    crit = criteria(boot, table.loc["final"])
    dev_swings = pd.read_parquet(ROOT / "data" / "model" / "statcast_2025_regular" / "swings.parquet", columns=["pitcher_id"])
    sub = subgroups(sw, set(dev_swings["pitcher_id"]))

    tag = "final_test_dry_run" if dry_run else "final_test"
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for k in ("pitch_type", "final"):
        r = reliability_table(y, sw[k])
        ax.plot(r["mean_pred"], r["observed"], marker="o", label=k)
    ax.plot([0, 0.75], [0, 0.75], color="grey", lw=0.8, ls="--")
    ax.set_xlabel("mean predicted P(miss | swing)"); ax.set_ylabel("observed whiff rate")
    ax.set_title(f"{'DRY RUN (2025, in-sample)' if dry_run else 'Sealed test: 2026 regular season'}", fontsize=10); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ROOT / "reports" / "figures" / f"{tag}_reliability.png", dpi=130); plt.close(fig)

    head = ("# DRY RUN — final-test code on 2025 (in-sample; numbers are NOT a test result)" if dry_run
            else "# Sealed test — 2026 MLB regular season")
    L = [head, "",
         f"Model `{card['model']}` (`models/final_v1`, sha256 `{card['artifact_sha256'][:16]}…`, frozen {card['frozen_utc']}), "
         f"calibration: {card['calibration']['decision']}. Protocol: `reports/evaluation_protocol.md` (locked; verified unchanged before scoring).", "",
         f"Population: {manifest['population']['game_type']} games {manifest['population']['start_date']} to "
         f"{manifest['population']['end_date']}; {manifest['games_in_data']:,} games, {manifest['total_rows']:,} pitches; "
         f"{len(sw):,} model-population ordinary swings. Position snapshot {people_meta['retrieved_at_utc']}.", "",
         population.rename("pitches").to_frame().to_markdown(), "",
         *( [f"**Override reason for re-scoring:** {override_reason}", ""] if override_reason else []),
         "## Metrics", "", table.round(5).to_markdown(), "",
         f"## Final model vs baselines (two-way game × pitcher bootstrap, {REPS} replicates, seed {SEED})", "",
         boot.round(5).to_markdown(index=False), "",
         "## Pre-stated success criteria (§7)", "",
         pd.DataFrame({"criterion": list(crit), "met": list(crit.values())}).to_markdown(index=False), "",
         f"ROC-AUC (aspirational reference 0.70, not required): {table.loc['final', 'roc_auc']:.4f}.", "",
         f"![reliability](figures/{tag}_reliability.png)", "",
         "## Subgroups", "", f"Groups under {MIN_GROUP} swings are reported as insufficient.", "",
         sub.round(4).to_markdown(index=False), "",
         "## Scope", "",
         "Future-season performance for MLB swings, retrospective (uses the measured pitch). Not evidence about college "
         "players or about the effect of changing a pitch.", ""]
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
