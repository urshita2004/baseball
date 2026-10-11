"""Download the college TrackMan file and verify it against the checksum Mendeley publishes.

Writes data/raw/trackman_college/Track_Combo.csv (unchanged) and data/manifests/<source_id>.json.
Usage: .venv/bin/python src/ingest_trackman.py [--refresh]
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from ingest import get_with_retries

ROOT = Path(__file__).resolve().parents[1]


def main(refresh=False):
    cfg = yaml.safe_load((ROOT / "config" / "data_sources.yaml").read_text())["trackman_college"]
    f = cfg["file"]
    path = ROOT / "data" / "raw" / "trackman_college" / f["name"]
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "data" / "manifests" / f"{cfg['source_id']}.json"
    retrieved_at = None
    if refresh or not path.exists():
        path.write_bytes(get_with_retries(f["download_url"], {}).content)
        retrieved_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    elif manifest_path.exists():
        retrieved_at = json.loads(manifest_path.read_text())["retrieved_at_utc"]
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != f["sha256"]:
        raise SystemExit(f"{path.name}: sha256 {digest} does not match published {f['sha256']}")
    df = pd.read_csv(path, low_memory=False)
    manifest = {
        "source_id": cfg["source_id"], "title": cfg["title"], "citation": " ".join(cfg["citation"].split()),
        "doi": cfg["doi"], "version": cfg["version"], "license": cfg["license"],
        "file": str(path.relative_to(ROOT)), "download_url": f["download_url"], "retrieved_at_utc": retrieved_at,
        "bytes": len(raw), "sha256": digest, "sha256_matches_published": True,
        "rows": int(len(df)), "columns": list(df.columns),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"{path.name}: {len(df)} rows x {len(df.columns)} columns; checksum matches published value")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    main(**vars(ap.parse_args()))
