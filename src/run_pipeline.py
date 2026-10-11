"""Run the development pipeline end to end and write reports/run_record.json.

The default run uses the one-week sample and college TrackMan; --season adds the dataset build on the full 2025
regular season (download, normalize, audit, model population, splits) and the baselines. The sealed test season is never
touched here: it is downloaded once with ingest.py and read only by the final-evaluation script.

Usage: .venv/bin/python src/run_pipeline.py [--season] [--comparison] [--refresh]
Uses raw files already on disk unless --refresh is given; checksums are verified either way.
"""
import argparse
import datetime as dt
import hashlib
import importlib.metadata as md
import json
import platform
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    ["src/ingest.py"], ["src/ingest_trackman.py"],
    ["src/normalize.py"], ["src/normalize_trackman.py"],
    ["src/validate_data.py"], ["src/validate_trackman.py"],
    ["src/pitcher_analysis.py"],
]
SEASON_STEPS = [
    ["src/ingest.py", "--source", "statcast_2025"],
    ["src/normalize.py", "--source", "statcast_2025"],
    ["src/validate_data.py", "--source", "statcast_2025"],
    ["src/build_dataset.py", "--source", "statcast_2025"],
    ["src/train_baselines.py"],
    ["src/calibrate.py"],
]
# Pitcher comparison on 2025 + 2026 (2026 is development data since the v1 test). model_v2.py and final_fit.py are one-time
# freeze scripts and are deliberately not rerun here.
COMPARISON_STEPS = [
    ["src/ingest.py", "--source", "statcast_2026"],
    ["src/normalize.py", "--source", "statcast_2026"],
    ["src/validate_data.py", "--source", "statcast_2026"],
    ["src/build_dataset.py", "--source", "statcast_2026"],
    ["src/drift_check.py", "--features", "v1"],
    ["src/drift_check.py", "--features", "v2"],
    ["src/season_shift.py"],
    ["src/expected_whiff.py"],
    ["src/player_analysis.py"],
    ["src/transfer.py"],
]
PACKAGES = ["pandas", "numpy", "pyarrow", "matplotlib", "scikit-learn", "scipy", "joblib", "PyYAML", "requests", "pytest"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(season=False, comparison=False, refresh=False):
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    steps = STEPS + (SEASON_STEPS if season or comparison else []) + (COMPARISON_STEPS if comparison else [])
    for step in steps:
        cmd = [sys.executable, *step] + (["--refresh"] if refresh and "ingest" in step[0] else [])
        print("$", " ".join(step))
        subprocess.run(cmd, cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT, check=True)
    cfg = yaml.safe_load((ROOT / "config" / "data_sources.yaml").read_text())
    record = {
        "started_utc": started,
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "packages": {p: md.version(p) for p in PACKAGES},
        "random_seed": cfg["random_seed"],
        "config_sha256": {p.name: sha(p) for p in sorted((ROOT / "config").glob("*.yaml"))},
        "manifests": {p.name: sha(p) for p in sorted((ROOT / "data" / "manifests").glob("*.json"))},
        "outputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sorted(
            [*(ROOT / "data" / "processed").rglob("*.parquet"), *(ROOT / "data" / "model").rglob("*.parquet")])},
        "split_manifests": {p.name: sha(p) for p in sorted((ROOT / "data" / "splits").glob("*.json"))},
        "steps": [" ".join(s) for s in steps] + ["pytest tests"],
    }
    (ROOT / "reports" / "run_record.json").write_text(json.dumps(record, indent=2))
    print("wrote reports/run_record.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", action="store_true", help="also build the model dataset on the full 2025 season")
    ap.add_argument("--comparison", action="store_true", help="also run 2026 processing, drift checks and the pitcher comparison (implies --season)")
    ap.add_argument("--refresh", action="store_true")
    main(**vars(ap.parse_args()))
