"""Retrieve Statcast day by day and preserve each response unchanged.

Sources are defined in config/data_sources.yaml. For each day in the window this writes the exact
bytes Savant returned (gzip-compressed when `compress: true`; the recorded sha256 is of the
uncompressed response), fetches the MLB schedule once for the window, and writes a manifest that
reconciles games in the data against completed games on the schedule.

Existing raw files are reused (resumable) unless --refresh is given.

Usage: .venv/bin/python src/ingest.py [--source statcast_sample|statcast_2025|statcast_2026_sealed] [--refresh]
"""
import argparse
import datetime as dt
import gzip
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
SEASONS_API = "https://statsapi.mlb.com/api/v1/seasons/{season}"


def get_with_retries(url, params, attempts=5, timeout=240):
    for i in range(attempts):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if i == attempts - 1:
                raise
            wait = 10 * 2**i
            print(f"  retry {i + 1} after {e!r}; sleeping {wait}s", file=sys.stderr)
            time.sleep(wait)


def read_raw_bytes(path: Path) -> bytes:
    """The original response bytes, whether stored plain or gzip-compressed."""
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == ".gz" else data


def source_config(name):
    cfg = yaml.safe_load((ROOT / "config" / "data_sources.yaml").read_text())
    return cfg[name], cfg["schedule_check"]


def window(src):
    """Start/end dates: explicit in config, or the official regular season from the MLB seasons API."""
    if "season" in src:
        s = get_with_retries(SEASONS_API.format(season=src["season"]), {"sportId": 1}).json()["seasons"][0]
        return s["regularSeasonStartDate"], s["regularSeasonEndDate"], s
    return str(src["start_date"]), str(src["end_date"]), None


def final_games_by_official_date(schedule_json):
    """{gamePk: officialDate} for completed games. Suspended games appear on start and resume dates
    with the same officialDate; Savant files every pitch under that official date."""
    out = {}
    for d in schedule_json.get("dates", []):
        for g in d["games"]:
            state = g["status"]["detailedState"]
            if state.startswith(("Final", "Completed", "Game Over")):
                out[g["gamePk"]] = g["officialDate"]
    return out


def main(source="statcast_sample", refresh=False):
    src, sched_cfg = source_config(source)
    raw_dir = ROOT / src["raw_dir"]
    manifest_dir = ROOT / src.get("manifest_dir", "data/manifests")
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    start, end, season_meta = window(src)

    sched_path = raw_dir / f"schedule_{start}_{end}.json"
    if refresh or not sched_path.exists():
        r = get_with_retries(sched_cfg["endpoint"], {"sportId": 1, "startDate": start, "endDate": end,
                                                     "gameType": src["game_type"]})
        sched_path.write_bytes(r.content)
    final = final_games_by_official_date(json.loads(sched_path.read_bytes()))

    d0, d1 = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    days = [(d0 + dt.timedelta(i)).isoformat() for i in range((d1 - d0).days + 1)]
    manifest_path = manifest_dir / f"{src['source_id']}.json"
    old = {}
    if manifest_path.exists():
        old = {b["date"]: b for b in json.loads(manifest_path.read_text())["batches"]}

    batches, columns, all_got = [], None, set()
    for day in days:
        path = raw_dir / (f"{day}.csv.gz" if src.get("compress") else f"{day}.csv")
        params = dict(src["params"], game_date_gt=day, game_date_lt=day, hfGT=f"{src['game_type']}|")
        retrieved_at = old.get(day, {}).get("retrieved_at_utc")
        if refresh or not path.exists():
            r = get_with_retries(src["endpoint"], params)
            path.write_bytes(gzip.compress(r.content, mtime=0) if src.get("compress") else r.content)
            retrieved_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
            time.sleep(src.get("request_delay_s", 0))
        raw = read_raw_bytes(path)
        digest = hashlib.sha256(raw).hexdigest()
        if day in old and old[day]["sha256"] != digest and not refresh:
            raise SystemExit(f"{path} changed since it was recorded in the manifest")
        df = pd.read_csv(io.BytesIO(raw), low_memory=False) if raw.strip() else pd.DataFrame()
        if len(df) and columns is None:
            columns = list(df.columns)
        got = set(df["game_pk"].unique()) if len(df) else set()
        all_got |= got
        expected = {pk for pk, od in final.items() if od == day}
        batches.append({
            "date": day, "file": str(path.relative_to(ROOT)), "request_params": params,
            "retrieved_at_utc": retrieved_at, "bytes": len(raw), "stored_bytes": path.stat().st_size,
            "sha256": digest, "rows": int(len(df)),
            "games_in_data": len(got), "games_final_in_schedule": len(expected),
            "final_games_missing_from_data": sorted(int(x) for x in expected - got),
            "column_count": len(df.columns),
            "columns_match_first_day": (not len(df)) or list(df.columns) == columns,
        })
        print(f"  {day}: {len(df):>6} rows, {len(got):>2} games (schedule final, official date: {len(expected)})", flush=True)

    window_missing = sorted(int(x) for x in set(final) - all_got)
    manifest = {
        "source_id": src["source_id"],
        "source": "MLB Statcast via Baseball Savant CSV export",
        "endpoint": src["endpoint"],
        "license_note": " ".join(src["license_note"].split()),
        "population": {"game_type": src["game_type"], "start_date": start, "end_date": end,
                       "date_rule": "official regular-season dates (MLB seasons API)" if season_meta else "explicit in config",
                       "season_api_record": season_meta},
        "sealed": bool(src.get("sealed")),
        "schedule_file": str(sched_path.relative_to(ROOT)),
        "requested_dates": [days[0], days[-1]],
        "total_rows": sum(b["rows"] for b in batches),
        "columns": columns,
        "games_in_data": len(all_got),
        "final_games_in_schedule": len(final),
        "final_games_missing_from_window": window_missing,
        "games_in_data_not_final_in_window": sorted(int(x) for x in all_got - set(final)),
        "batches": batches,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    problems = [b["date"] for b in batches if not b["columns_match_first_day"]
                or (b["rows"] == 0 and b["games_final_in_schedule"] > 0)]
    if window_missing:
        problems.append(f"final games missing: {window_missing}")
    print(f"wrote {manifest_path.relative_to(ROOT)}: {manifest['total_rows']} rows, "
          f"{len(all_got)}/{len(final)} scheduled final games")
    if problems:
        print(f"COVERAGE/SCHEMA PROBLEMS: {problems} - see manifest", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="statcast_sample")
    ap.add_argument("--refresh", action="store_true", help="re-download days already on disk")
    main(**vars(ap.parse_args()))
