"""Primary positions from the MLB Stats API, used to identify position players pitching.

The raw responses are cached under data/raw/statsapi/ with a checksum; a cached file is reused
(the position snapshot date is recorded) unless refresh=True.
"""
import datetime as dt
import hashlib
import json
from pathlib import Path

from ingest import get_with_retries

ROOT = Path(__file__).resolve().parents[1]
PEOPLE_API = "https://statsapi.mlb.com/api/v1/people"
PITCHER_POSITIONS = {"P", "TWP"}  # pitchers and two-way players (e.g. Ohtani) are not position players


def people_snapshot(player_ids, cache_name, refresh=False, batch=150):
    """Cached MLB Stats API person records for the given ids: ({id: record}, provenance)."""
    path = ROOT / "data" / "raw" / "statsapi" / f"people_{cache_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = sorted({int(i) for i in player_ids})
    if refresh or not path.exists():
        people = []
        for i in range(0, len(ids), batch):
            r = get_with_retries(PEOPLE_API, {"personIds": ",".join(map(str, ids[i:i + batch]))})
            people += r.json()["people"]
        snapshot = {"retrieved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                    "endpoint": PEOPLE_API, "people": people}
        path.write_text(json.dumps(snapshot))
    snapshot = json.loads(path.read_text())
    people = {p["id"]: p for p in snapshot["people"]}
    missing = set(ids) - set(people)
    if missing:
        raise SystemExit(f"{len(missing)} ids absent from cached people snapshot {path.name}; rerun with refresh")
    meta = {"file": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "retrieved_at_utc": snapshot["retrieved_at_utc"]}
    return people, meta


def primary_positions(player_ids, cache_name, refresh=False):
    """{player_id: position abbreviation} plus provenance for the cached snapshot."""
    people, meta = people_snapshot(player_ids, cache_name, refresh)
    pos = {i: p.get("primaryPosition", {}).get("abbreviation") for i, p in people.items()}
    meta["rule"] = f"position player pitching = primary position not in {sorted(PITCHER_POSITIONS)}"
    return pos, meta


def parse_height_ft(text):
    """'6\' 2"' -> 6.1667 ft; None if unparseable."""
    try:
        feet, inches = text.replace('"', "").split("'")
        return int(feet) + int(inches.strip() or 0) / 12
    except (AttributeError, ValueError):
        return None


def batter_heights(batter_ids, cache_name="batters"):
    """{batter_id: height in ft} from the cached people snapshot (height as listed now, not per season)."""
    people, meta = people_snapshot(batter_ids, cache_name)
    return {i: parse_height_ft(p.get("height")) for i, p in people.items()}, meta
